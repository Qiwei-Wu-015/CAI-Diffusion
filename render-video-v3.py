#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import argparse
from pathlib import Path
import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from decord import VideoReader, cpu
from tqdm import tqdm

# =============================
# 默认路径
# =============================
DEFAULT_PROJECT_ROOT = "/media/Hulu/面部反应生成/baseline_react2025-main-1"
DEFAULT_DATASET_ROOT = "/data2/REACT2024-VIS"
DEFAULT_SPLIT = "test"

# 默认读取 META 和 预测结果的文件
DEFAULT_META_PT = "/media/Hulu/面部反应生成/baseline_react2025-main-1/outputs/motion_diffusion/react_2025/online/260316164839_o6xzfea2/3dmm.pt"
DEFAULT_REFERENCE_IMAGE = "/media/Hulu/面部反应生成/baseline_react2025-main-1/img/2.png"

def load_image_as_tensor(image_path: Path, image_transform):
    image_path = Path(image_path)
    if not image_path.is_file():
        raise FileNotFoundError(f"找不到参考图片: {image_path}")
    img = Image.open(str(image_path)).convert("RGB")
    img = image_transform(img)
    return img

def add_project_to_syspath(project_root: str):
    project_root = str(Path(project_root).resolve())
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

def obtain_seq_index(index, num_frames, semantic_radius=13):
    seq = list(range(index - semantic_radius, index + semantic_radius + 1))
    seq = [min(max(item, 0), num_frames - 1) for item in seq]
    return seq

def transform_semantic(semantic: torch.Tensor) -> torch.Tensor:
    semantic_list = []
    for i in range(semantic.shape[0]):
        index = obtain_seq_index(i, semantic.shape[0])
        semantic_item = semantic[index, :].unsqueeze(0)
        semantic_list.append(semantic_item)
    semantic = torch.cat(semantic_list, dim=0)
    return semantic.transpose(1, 2)

def resolve_video_path(dataset_root: str, split: str, rel_no_suffix: str) -> Path:
    rel = Path(rel_no_suffix).with_suffix(".mp4")
    return Path(dataset_root) / split / "video-face-crop" / rel

def resolve_coeff_path(dataset_root: str, split: str, rel_no_suffix: str) -> Path:
    rel = Path(rel_no_suffix).with_suffix(".npy")
    return Path(dataset_root) / split / "coefficients" / rel

def load_reference_frame_as_tensor(video_path: Path, image_transform, mode="middle", exact_idx=0):
    vr = VideoReader(str(video_path), ctx=cpu(0))
    if len(vr) == 0:
        raise RuntimeError(f"空视频: {video_path}")

    if mode == "first":
        frame_idx = 0
    elif mode == "middle":
        frame_idx = len(vr) // 2
    elif mode == "one_third":
        frame_idx = len(vr) // 3
    elif mode == "exact":
        frame_idx = exact_idx
    else:
        raise ValueError(f"未知 mode: {mode}")
        
    frame_idx = max(0, min(frame_idx, len(vr) - 1))
    frame = vr[frame_idx].asnumpy()
    img = Image.fromarray(frame)
    img = image_transform(img)
    return img

def load_video_chunk_uint8(video_path: Path, start: int, end: int, target_hw=(224, 224)):
    vr = VideoReader(str(video_path), ctx=cpu(0))
    total = len(vr)

    start = max(0, start)
    end = min(end, total)

    if start >= end:
        return np.zeros((0, target_hw[0], target_hw[1], 3), dtype=np.uint8)

    idx = list(range(start, end))
    frames = vr.get_batch(idx).asnumpy()

    crop_h, crop_w = target_hw
    # 【已恢复】：按照你的要求，保持256，与原项目中心裁剪逻辑完全一致
    target_size = 256 

    out = []
    for f in frames:
        img = Image.fromarray(f)
        w, h = img.size
        short, long = (w, h) if w <= h else (h, w)
        if short == target_size:
            new_w, new_h = w, h
        else:
            new_w = int(w * target_size / short)
            new_h = int(h * target_size / short)
        
        img = img.resize((new_w, new_h), resample=Image.BILINEAR)

        left = int(round((new_w - crop_w) / 2.))
        top = int(round((new_h - crop_h) / 2.))
        right = int(round((new_w + crop_w) / 2.))
        bottom = int(round((new_h + crop_h) / 2.))
        
        img = img.crop((left, top, right, bottom))
        out.append(np.array(img).astype(np.uint8))

    return np.stack(out, axis=0)

def load_coeff_npy(coeff_path: Path) -> torch.Tensor:
    if not coeff_path.is_file():
        raise FileNotFoundError(f"找不到真实系数文件: {coeff_path}")

    x = np.load(coeff_path)
    x = torch.from_numpy(x).float()

    if x.ndim == 2 and x.shape[-1] == 58:
        return x
    if x.ndim == 3:
        if x.shape[0] == 1 and x.shape[2] == 58:
            x = x.squeeze(0)
        elif x.shape[1] == 1 and x.shape[2] == 58:
            x = x.squeeze(1)
        else:
            raise ValueError(f"系数形状异常: {tuple(x.shape)}")
        return x
    if x.ndim == 4:
        if x.shape[0] == 1 and x.shape[1] == 1 and x.shape[-1] == 58:
            x = x.squeeze(0).squeeze(0)
            return x

    raise ValueError(f"系数形状异常: {tuple(x.shape)}")

class FourPanelRenderer:
    def __init__(self, project_root: str, device="cuda", coeff_is_normalized=False, transform_reverse="standard"):
        self.project_root = Path(project_root).resolve()
        self.device = torch.device(device)
        self.coeff_is_normalized = coeff_is_normalized
        self.transform_reverse = transform_reverse

        add_project_to_syspath(str(self.project_root))

        from dataset.tools.util import Transform
        from utils.util import torch_img_to_np2
        from external.FaceVerse import get_faceverse
        from external.PIRender import FaceGenerator

        self.torch_img_to_np2 = torch_img_to_np2
        self.image_transform = Transform(224, 224)

        faceverse_dir = self.project_root / "external" / "FaceVerse"
        self.faceverse, _ = get_faceverse(device=self.device, img_size=224)
        self.faceverse.init_coeff_tensors()

        self.id_tensor = torch.from_numpy(
            np.load(faceverse_dir / "reference_full.npy")
        ).float().view(1, -1)[:, :150]

        pirender_ckpt = self.project_root / "external" / "PIRender" / "cur_model_fold.pth"
        if not pirender_ckpt.is_file():
            raise FileNotFoundError(f"找不到 PIRender 权重: {pirender_ckpt}")

        self.pi_render = FaceGenerator().to(self.device)
        self.pi_render.eval()

        checkpoint = torch.load(str(pirender_ckpt), map_location=self.device)
        state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint

        clean_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                clean_state_dict[k[7:]] = v
            else:
                clean_state_dict[k] = v

        self.pi_render.load_state_dict(clean_state_dict, strict=True)

        self.mean_face = torch.FloatTensor(
            np.load(faceverse_dir / "mean_face.npy").astype(np.float32)
        ).view(1, 1, -1).to(self.device)

        self.std_face = torch.FloatTensor(
            np.load(faceverse_dir / "std_face.npy").astype(np.float32)
        ).view(1, 1, -1).to(self.device)

        if transform_reverse == "zero_center":
            self._reverse_transform_3dmm = transforms.Lambda(lambda e: e + self.mean_face)
        elif transform_reverse == "standard":
            self._reverse_transform_3dmm = transforms.Lambda(lambda e: e * self.std_face + self.mean_face)
        else:
            raise ValueError(f"Unknown transform_reverse: {transform_reverse}")

        # 修复：永久移除了引发拉伸的额外 _transform Z 轴扣减逻辑

    @torch.no_grad()
    def render_video(
        self,
        listener_vectors: torch.Tensor,
        speaker_video_path: Path,
        listener_video_path: Path,
        output_video_path: Path,
        output_meta_path: Path,
        reference_image_path: Path = None,
        faceverse_chunk: int = 512,
        pirender_chunk: int = 128,
        fps: int = 25,
    ):
        output_video_path.parent.mkdir(parents=True, exist_ok=True)
        output_meta_path.parent.mkdir(parents=True, exist_ok=True)

        if listener_vectors.ndim == 3:
            listener_vectors = listener_vectors.squeeze(0)

        assert listener_vectors.ndim == 2 and listener_vectors.shape[1] == 58, \
            f"listener_vectors 应为 [T,58]，实际是 {tuple(listener_vectors.shape)}"

        if reference_image_path is not None:
            listener_reference = load_image_as_tensor(reference_image_path, self.image_transform).to(self.device)
        else:
            rot_norm = torch.norm(listener_vectors[:, 52:55], dim=1)
            best_frontal_idx = int(torch.argmin(rot_norm).item())
            listener_reference = load_reference_frame_as_tensor(
                listener_video_path, self.image_transform, mode="exact", exact_idx=best_frontal_idx
            ).to(self.device)

        listener_vectors = listener_vectors.to(self.device).float()

        if self.coeff_is_normalized:
            listener_vectors = listener_vectors.unsqueeze(0)
            listener_vectors = self._reverse_transform_3dmm(listener_vectors)[0]

        speaker_vr = VideoReader(str(speaker_video_path), ctx=cpu(0))
        listener_vr = VideoReader(str(listener_video_path), ctx=cpu(0))
        T = min(listener_vectors.shape[0], len(speaker_vr), len(listener_vr))
        if T <= 0:
            raise RuntimeError("有效帧数为 0，无法渲染")

        listener_vectors = listener_vectors[:T]
        semantics_all = transform_semantic(listener_vectors.detach()).cpu()

        writer = cv2.VideoWriter(
            str(output_video_path),
            cv2.VideoWriter_fourcc(*"MJPG"),
            fps,
            (224 * 4, 224),
        )

        meta_dump = {
            "speaker_video_path": str(speaker_video_path),
            "listener_video_path": str(listener_video_path),
            "output_video_path": str(output_video_path),
            "num_frames": int(T),
            "fps": int(fps),
            "coeff_is_normalized": bool(self.coeff_is_normalized),
            "transform_reverse": self.transform_reverse,
            "layout": ["speaker_real", "listener_real", "listener_mesh", "listener_2d"],
        }

        with open(output_meta_path, "w", encoding="utf-8") as f:
            json.dump(meta_dump, f, ensure_ascii=False, indent=2)

        frames_dir = output_video_path.parent / f"{output_video_path.stem}_frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        global_frame_idx = 0

        for start in tqdm(range(0, T, faceverse_chunk), desc=f"render {output_video_path.stem}", leave=False):
            end = min(start + faceverse_chunk, T)
            vec_chunk = listener_vectors[start:end]
            sem_chunk = semantics_all[start:end].to(self.device)
            L = vec_chunk.shape[0]

            self.faceverse.batch_size = L
            self.faceverse.init_coeff_tensors()

            self.faceverse.exp_tensor = vec_chunk[:, :52]
            self.faceverse.rot_tensor = vec_chunk[:, 52:55]
            self.faceverse.trans_tensor = vec_chunk[:, 55:]
            self.faceverse.id_tensor = self.id_tensor.reshape(1, 150).repeat(L, 1).to(self.device)

            pred_dict = self.faceverse(self.faceverse.get_packed_tensors(), render=True, texture=False)
            rendered_img_r = np.clip(pred_dict["rendered_img"].detach().cpu().numpy(), 0, 255)[:, :, :, :3].astype(np.uint8)[:, :, ::-1, :]

            fake_chunks = []
            for pstart in range(0, L, pirender_chunk):
                pend = min(pstart + pirender_chunk, L)
                sub_len = pend - pstart
                ref_batch = listener_reference.unsqueeze(0).repeat(sub_len, 1, 1, 1)
                sem_batch = sem_chunk[pstart:pend]

                out_dict = self.pi_render(ref_batch, sem_batch)
                fake_videos = self.torch_img_to_np2(out_dict["fake_image"])[:, :, :, ::-1]
                fake_chunks.append(fake_videos)

            listener_videos = np.concatenate(fake_chunks, axis=0)[:, :, ::-1, :]

            speaker_frames = load_video_chunk_uint8(speaker_video_path, start, end, target_hw=(224, 224))
            listener_real_frames = load_video_chunk_uint8(listener_video_path, start, end, target_hw=(224, 224))

            L2 = min(rendered_img_r.shape[0], listener_videos.shape[0], speaker_frames.shape[0], listener_real_frames.shape[0])

            for i in range(L2):
                combined_img = np.zeros((224, 224 * 4, 3), dtype=np.uint8)
                combined_img[:, 0:224] = speaker_frames[i]
                combined_img[:, 224:448] = listener_real_frames[i]
                combined_img[:, 448:672] = rendered_img_r[i]
                combined_img[:, 672:896] = listener_videos[i]
                
                writer.write(cv2.cvtColor(combined_img, cv2.COLOR_RGB2BGR))
                cv2.imwrite(str(frames_dir / f"frame_{global_frame_idx:04d}_2d.jpg"), cv2.cvtColor(listener_videos[i], cv2.COLOR_RGB2BGR))
                global_frame_idx += 1

        writer.release()


def main():
    parser = argparse.ArgumentParser(description="面部反应生成渲染器 (支持真实 GT 或预测 pt 数据)")

    parser.add_argument("--reference-image", type=str, default=DEFAULT_REFERENCE_IMAGE)
    parser.add_argument("--project-root", type=str, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--dataset-root", type=str, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--split", type=str, default=DEFAULT_SPLIT)
    parser.add_argument("--meta-pt", type=str, default=DEFAULT_META_PT)
    
    # 【新增开关】：使用预测数据还是真实数据
    parser.add_argument("--use-pred", action="store_true", help="开启后，使用 3dmm.pt 里面的预测值进行渲染，并将其放大3000倍；关闭则读取真实 npy")

    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--coeff-is-normalized", action="store_true", help="如果传入的系数是去均值化的，开启此项来加上均值")
    parser.add_argument("--transform-reverse", type=str, default="standard", choices=["standard", "zero_center"])
    parser.add_argument("--faceverse-chunk", type=int, default=512)
    parser.add_argument("--pirender-chunk", type=int, default=128)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--first-n", type=int, default=3)
    parser.add_argument("--sample-idx", type=int, default=1)
    parser.add_argument("--output-dir", type=str, default=None)

    args = parser.parse_args()
    add_project_to_syspath(args.project_root)

    meta_pt_path = Path(args.meta_pt).resolve()
    if not meta_pt_path.is_file():
        raise FileNotFoundError(f"找不到 meta pt: {meta_pt_path}")

    data = torch.load(str(meta_pt_path), map_location="cpu")
    if "META" not in data:
        raise KeyError(f"{meta_pt_path} 中没有 META")

    meta_list = data["META"]
    num_meta = len(meta_list)
    
    # 【新增解析】：如果使用了 --use-pred 且存在预测数据
    pred_data_list = None
    if args.use_pred:
        if "PRED_3DMM" in data:
            pred_data_list = data["PRED_3DMM"]
        elif "PRED" in data:
            pred_data_list = data["PRED"]
        elif "3dmm_coeff" in data:
            pred_data_list = data["3dmm_coeff"]
        else:
            # 有些模型直接把 tensor 保存在文件里，没用字典包装
            if isinstance(data, torch.Tensor):
                pred_data_list = data
            else:
                raise KeyError(f"{meta_pt_path} 中找不到预测数据，请确保有 PRED 或直接为 Tensor")

    if args.sample_idx is not None:
        if args.sample_idx < 0 or args.sample_idx >= num_meta:
            raise IndexError(f"--sample-idx 越界: {args.sample_idx}，有效范围是 [0, {num_meta - 1}]")
        indices = [args.sample_idx]
    else:
        indices = list(range(min(args.first_n, num_meta)))

    if args.output_dir is None:
        prefix = "rendered_pred" if args.use_pred else "rendered_real"
        if args.sample_idx is not None:
            output_dir = meta_pt_path.parent / f"{prefix}_idx{args.sample_idx:03d}"
        else:
            output_dir = meta_pt_path.parent / f"{prefix}_first{len(indices)}"
    else:
        output_dir = Path(args.output_dir).resolve()
    
    output_dir.mkdir(parents=True, exist_ok=True)

    renderer = FourPanelRenderer(
        project_root=args.project_root,
        device=args.device,
        coeff_is_normalized=args.coeff_is_normalized,
        transform_reverse=args.transform_reverse,
    )

    total_jobs = sum(min(3, len(meta_list[idx]["listener_paths"])) for idx in indices)

    print("========== Batch Render Config ==========")
    print("Mode            :", "Predict (from pt)" if args.use_pred else "Real GT (from npy)")
    print("meta pt         :", meta_pt_path)
    print("output_dir      :", output_dir)
    print("total_jobs      :", total_jobs)

    job_id = 0
    for idx in indices:
        meta = meta_list[idx]
        speaker_rel = meta["speaker_path"]
        listener_paths = meta["listener_paths"]

        speaker_video_path = resolve_video_path(args.dataset_root, args.split, speaker_rel)

        for sample_idx, listener_rel in enumerate(listener_paths[:3]):
            job_id += 1
            listener_video_path = resolve_video_path(args.dataset_root, args.split, listener_rel)
            coeff_path = resolve_coeff_path(args.dataset_root, args.split, listener_rel)
            
            # 【核心逻辑分发】：读取预测值还是真实值
            if args.use_pred:
                pred_item = pred_data_list[idx]
                
                # 兼容不同形状的预测数据 [Sample_num, T, 58] 或直接 [T, 58]
                if isinstance(pred_item, torch.Tensor):
                    if pred_item.ndim == 3:
                        if sample_idx < pred_item.shape[0]:
                            listener_vectors = pred_item[sample_idx]
                        else:
                            listener_vectors = pred_item[0]
                    elif pred_item.ndim == 2:
                        listener_vectors = pred_item
                    else:
                        raise ValueError(f"无法解析预测数据形状: {pred_item.shape}")
                else:
                    raise TypeError("预测数据格式不是 torch.Tensor")

            else:
                # 读取真实 npy
                listener_vectors = load_coeff_npy(coeff_path)

            output_name = f"idx{idx:03d}_sample{sample_idx:02d}"
            output_video_path = output_dir / f"{output_name}.avi"
            output_meta_path = output_dir / f"{output_name}.json"

            print(f"\n[{job_id}/{total_jobs}] Rendering {output_name}")
            print("speaker        :", speaker_rel)
            print("listener       :", listener_rel)
            if not args.use_pred:
                print("coeff_path     :", coeff_path)
            print("coeff_shape    :", tuple(listener_vectors.shape))

            renderer.render_video(
                listener_vectors=listener_vectors,
                speaker_video_path=speaker_video_path,
                listener_video_path=listener_video_path,
                output_video_path=output_video_path,
                output_meta_path=output_meta_path,
                reference_image_path=None, 
                faceverse_chunk=args.faceverse_chunk,
                pirender_chunk=args.pirender_chunk,
                fps=args.fps,
            )

    print("\n[OK] 全部渲染完成")

if __name__ == "__main__":
    main()