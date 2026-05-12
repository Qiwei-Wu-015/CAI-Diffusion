#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import json
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from decord import VideoReader, cpu
from tqdm import tqdm


DEFAULT_PROJECT_ROOT = "/media/Hulu/面部反应生成/baseline_react2025-main-1"
DEFAULT_DATASET_ROOT = "/data2/REACT2024-VIS"
DEFAULT_SPLIT = "test"
DEFAULT_META_PT = "/media/Hulu/面部反应生成/baseline_react2025-main-1/outputs/motion_diffusion/react_2025/online/260322173609_fb8qojg8/3dmm.pt"


def add_project_to_syspath(project_root: str):
    project_root = str(Path(project_root).resolve())
    if project_root not in sys.path:
        sys.path.insert(0, project_root)


def resolve_video_path(dataset_root: str, split: str, rel_no_suffix: str) -> Path:
    return Path(dataset_root) / split / "video-face-crop" / Path(rel_no_suffix).with_suffix(".mp4")


def resolve_coeff_path(dataset_root: str, split: str, rel_no_suffix: str) -> Path:
    return Path(dataset_root) / split / "coefficients" / Path(rel_no_suffix).with_suffix(".npy")


def load_coeff_npy(coeff_path: Path) -> torch.Tensor:
    if not coeff_path.is_file():
        raise FileNotFoundError(f"找不到真实系数文件: {coeff_path}")

    x = torch.from_numpy(np.load(coeff_path)).float()

    if x.ndim == 2 and x.shape[-1] == 58:
        return x
    if x.ndim == 3 and x.shape[-1] == 58:
        if x.shape[0] == 1:
            return x.squeeze(0)
        if x.shape[1] == 1:
            return x.squeeze(1)
    if x.ndim == 4 and x.shape[0] == 1 and x.shape[1] == 1 and x.shape[-1] == 58:
        return x.squeeze(0).squeeze(0)

    raise ValueError(f"系数形状异常: {tuple(x.shape)}")


def extract_pred_item(pred_data_list, idx: int, sample_idx: int) -> torch.Tensor:
    pred_item = pred_data_list[idx]

    if not isinstance(pred_item, torch.Tensor):
        raise TypeError("预测数据格式不是 torch.Tensor")

    if pred_item.ndim == 3:   # [N, T, 58]
        return pred_item[sample_idx] if sample_idx < pred_item.shape[0] else pred_item[0]
    if pred_item.ndim == 2:   # [T, 58]
        return pred_item

    raise ValueError(f"无法解析预测数据形状: {tuple(pred_item.shape)}")


def load_video_frame_as_tensor(video_path: Path, image_transform, frame_idx: int):
    vr = VideoReader(str(video_path), ctx=cpu(0))
    if len(vr) == 0:
        raise RuntimeError(f"空视频: {video_path}")

    frame_idx = max(0, min(frame_idx, len(vr) - 1))
    frame = vr[frame_idx].asnumpy()
    return image_transform(Image.fromarray(frame))


def load_video_chunk_uint8(video_path: Path, start: int, end: int, target_hw=(224, 224)):
    vr = VideoReader(str(video_path), ctx=cpu(0))
    total = len(vr)

    start = max(0, start)
    end = min(end, total)
    if start >= end:
        return np.zeros((0, target_hw[0], target_hw[1], 3), dtype=np.uint8)

    frames = vr.get_batch(list(range(start, end))).asnumpy()

    crop_h, crop_w = target_hw
    resize_short = 256

    out = []
    for f in frames:
        img = Image.fromarray(f)
        w, h = img.size
        short = min(w, h)

        if short != resize_short:
            new_w = int(w * resize_short / short)
            new_h = int(h * resize_short / short)
            img = img.resize((new_w, new_h), resample=Image.BILINEAR)
        else:
            new_w, new_h = w, h

        left = int(round((new_w - crop_w) / 2.0))
        top = int(round((new_h - crop_h) / 2.0))
        img = img.crop((left, top, left + crop_w, top + crop_h))
        out.append(np.array(img).astype(np.uint8))

    return np.stack(out, axis=0)


def obtain_seq_index(index, num_frames, semantic_radius=13):
    seq = list(range(index - semantic_radius, index + semantic_radius + 1))
    seq = [min(max(i, 0), num_frames - 1) for i in seq]
    return seq


def transform_semantic(semantic: torch.Tensor) -> torch.Tensor:
    semantic_list = []
    for i in range(semantic.shape[0]):
        idx = obtain_seq_index(i, semantic.shape[0])
        semantic_item = semantic[idx, :].unsqueeze(0)
        semantic_list.append(semantic_item)
    semantic = torch.cat(semantic_list, dim=0)
    return semantic.transpose(1, 2)


class Renderer:
    def __init__(
        self,
        project_root: str,
        device="cuda",
        exp_scale: float = 20,
        rot_scale: float = 10,
        trans_scale: float = 10,
    ):
        self.project_root = Path(project_root).resolve()
        self.device = torch.device(device)

        self.exp_scale = exp_scale
        self.rot_scale = rot_scale
        self.trans_scale = trans_scale

        add_project_to_syspath(str(self.project_root))

        from dataset.tools.util import Transform
        from utils.util import torch_img_to_np2
        from external.FaceVerse import get_faceverse
        from external.PIRender import FaceGenerator

        self.image_transform = Transform(224, 224)
        self.torch_img_to_np2 = torch_img_to_np2

        faceverse_dir = self.project_root / "external" / "FaceVerse"
        pirender_ckpt = self.project_root / "external" / "PIRender" / "cur_model_fold.pth"

        self.faceverse, _ = get_faceverse(device=self.device, img_size=224)
        self.faceverse.init_coeff_tensors()

        self.id_tensor = torch.from_numpy(
            np.load(faceverse_dir / "reference_full.npy")
        ).float().view(1, -1)[:, :150].to(self.device)

        self.mean_face = torch.from_numpy(
            np.load(faceverse_dir / "mean_face.npy").astype(np.float32)
        ).view(1, 1, -1).to(self.device)

        self.pi_render = FaceGenerator().to(self.device)
        self.pi_render.eval()

        checkpoint = torch.load(str(pirender_ckpt), map_location=self.device)
        state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
        clean_state_dict = {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}
        self.pi_render.load_state_dict(clean_state_dict, strict=True)

    def prepare_render_coeff(self, vectors: torch.Tensor, mode: str) -> torch.Tensor:
        if vectors.ndim == 3:
            vectors = vectors.squeeze(0)

        if vectors.ndim != 2 or vectors.shape[1] != 58:
            raise ValueError(f"listener_vectors 应为 [T,58]，实际是 {tuple(vectors.shape)}")

        vectors = vectors.float().to(self.device)

        if mode == "pred":
            vectors = vectors.unsqueeze(0).clone()
            vectors[:, :, :52] *= self.exp_scale
            vectors[:, :, 52:55] *= self.rot_scale
            vectors[:, :, 55:58] *= self.trans_scale
            vectors = (self.mean_face + vectors)[0]
        elif mode == "real":
            pass
        else:
            raise ValueError(f"未知 mode: {mode}")

        return vectors

    def pick_reference_frame(
        self,
        listener_video_path: Path,
        reference_vectors: torch.Tensor,
        reference_image_path: str = None,
    ):
        if reference_image_path is not None:
            ref_path = Path(reference_image_path)
            if not ref_path.is_file():
                raise FileNotFoundError(f"找不到参考图片: {ref_path}")
            img = Image.open(str(ref_path)).convert("RGB")
            return self.image_transform(img).to(self.device), -1

        if reference_vectors.ndim == 3:
            reference_vectors = reference_vectors.squeeze(0)

        if reference_vectors.ndim != 2 or reference_vectors.shape[1] != 58:
            raise ValueError(f"reference_vectors 应为 [T,58]，实际是 {tuple(reference_vectors.shape)}")

        rot = reference_vectors[:, 52:55].float()
        rot_norm = torch.norm(rot, dim=1)
        best_idx = int(torch.argmin(rot_norm).item())

        ref = load_video_frame_as_tensor(
            listener_video_path, self.image_transform, best_idx
        ).to(self.device)

        return ref, best_idx

    @torch.no_grad()
    def render_video(
        self,
        render_vectors: torch.Tensor,
        reference_vectors: torch.Tensor,
        mode: str,
        speaker_video_path: Path,
        listener_video_path: Path,
        output_video_path: Path,
        output_meta_path: Path,
        reference_image_path: str = None,
        faceverse_chunk: int = 512,
        pirender_chunk: int = 128,
        fps: int = 25,
    ):
        output_video_path.parent.mkdir(parents=True, exist_ok=True)
        output_meta_path.parent.mkdir(parents=True, exist_ok=True)

        render_vectors = self.prepare_render_coeff(render_vectors, mode)
        listener_reference, ref_frame_idx = self.pick_reference_frame(
            listener_video_path=listener_video_path,
            reference_vectors=reference_vectors,
            reference_image_path=reference_image_path,
        )

        speaker_vr = VideoReader(str(speaker_video_path), ctx=cpu(0))
        listener_vr = VideoReader(str(listener_video_path), ctx=cpu(0))

        T = min(render_vectors.shape[0], len(speaker_vr), len(listener_vr))
        if T <= 0:
            raise RuntimeError("有效帧数为 0，无法渲染")

        render_vectors = render_vectors[:T]
        semantics_all = transform_semantic(render_vectors.detach()).cpu()

        writer = cv2.VideoWriter(
            str(output_video_path),
            cv2.VideoWriter_fourcc(*"MJPG"),
            fps,
            (224 * 4, 224),
        )

        meta_dump = {
            "mode": mode,
            "speaker_video_path": str(speaker_video_path),
            "listener_video_path": str(listener_video_path),
            "output_video_path": str(output_video_path),
            "reference_image_path": reference_image_path,
            "reference_frame_idx": int(ref_frame_idx),
            "num_frames": int(T),
            "fps": int(fps),
            "exp_scale": float(self.exp_scale),
            "rot_scale": float(self.rot_scale),
            "trans_scale": float(self.trans_scale),
            "layout": ["speaker_real", "listener_real", "listener_mesh", "listener_2d"],
        }
        with open(output_meta_path, "w", encoding="utf-8") as f:
            json.dump(meta_dump, f, ensure_ascii=False, indent=2)

        for start in tqdm(range(0, T, faceverse_chunk), desc=f"render {output_video_path.stem}", leave=False):
            end = min(start + faceverse_chunk, T)
            vec_chunk = render_vectors[start:end]
            sem_chunk = semantics_all[start:end].to(self.device)
            L = vec_chunk.shape[0]

            self.faceverse.batch_size = L
            self.faceverse.init_coeff_tensors()
            self.faceverse.exp_tensor = vec_chunk[:, :52]
            self.faceverse.rot_tensor = vec_chunk[:, 52:55]
            self.faceverse.trans_tensor = vec_chunk[:, 55:]
            self.faceverse.id_tensor = self.id_tensor.repeat(L, 1)

            pred_dict = self.faceverse(
                self.faceverse.get_packed_tensors(),
                render=True,
                texture=False
            )
            rendered_img = np.clip(
                pred_dict["rendered_img"].detach().cpu().numpy(), 0, 255
            )[:, :, :, :3].astype(np.uint8)

            fake_chunks = []
            for pstart in range(0, L, pirender_chunk):
                pend = min(pstart + pirender_chunk, L)
                sub_len = pend - pstart

                ref_batch = listener_reference.unsqueeze(0).repeat(sub_len, 1, 1, 1)
                sem_batch = sem_chunk[pstart:pend]

                out_dict = self.pi_render(ref_batch, sem_batch)
                fake_videos = self.torch_img_to_np2(out_dict["fake_image"])
                fake_chunks.append(fake_videos)

            listener_2d = np.concatenate(fake_chunks, axis=0)

            speaker_frames = load_video_chunk_uint8(speaker_video_path, start, end, target_hw=(224, 224))
            listener_real_frames = load_video_chunk_uint8(listener_video_path, start, end, target_hw=(224, 224))

            L2 = min(
                rendered_img.shape[0],
                listener_2d.shape[0],
                speaker_frames.shape[0],
                listener_real_frames.shape[0],
            )

            for i in range(L2):
                canvas = np.zeros((224, 224 * 4, 3), dtype=np.uint8)

                speaker_bgr = cv2.cvtColor(speaker_frames[i], cv2.COLOR_RGB2BGR)
                listener_real_bgr = cv2.cvtColor(listener_real_frames[i], cv2.COLOR_RGB2BGR)
                mesh_bgr = cv2.cvtColor(rendered_img[i], cv2.COLOR_RGB2BGR)

                # 第四列保持原样
                listener_2d_bgr = listener_2d[i]

                canvas[:, 0:224] = speaker_bgr
                canvas[:, 224:448] = listener_real_bgr
                canvas[:, 448:672] = mesh_bgr
                canvas[:, 672:896] = listener_2d_bgr

                writer.write(canvas)

        writer.release()


def main():
    parser = argparse.ArgumentParser(description="REACT 3DMM 渲染器（支持真实值和预测值）")
    parser.add_argument("--project-root", type=str, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--dataset-root", type=str, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--split", type=str, default=DEFAULT_SPLIT)
    parser.add_argument("--meta-pt", type=str, default=DEFAULT_META_PT)
    parser.add_argument("--reference-image", type=str, default=None,
                        help="可选，手动指定参考图；不指定时默认使用真实3DMM的rot自动选帧")

    parser.add_argument("--mode", type=str, default="pred", choices=["real", "pred"])
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--faceverse-chunk", type=int, default=512)
    parser.add_argument("--pirender-chunk", type=int, default=128)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--sample-idx", type=int, default=55)
    parser.add_argument("--max-listeners", type=int, default=-1)
    parser.add_argument("--output-dir", type=str, default=None)

    parser.add_argument("--exp-scale", type=float, default=1000)
    parser.add_argument("--rot-scale", type=float, default=1000)
    parser.add_argument("--trans-scale", type=float, default=1000)

    args = parser.parse_args()
    add_project_to_syspath(args.project_root)

    meta_pt_path = Path(args.meta_pt).resolve()
    if not meta_pt_path.is_file():
        raise FileNotFoundError(f"找不到 meta pt: {meta_pt_path}")

    data = torch.load(str(meta_pt_path), map_location="cpu")
    if "META" not in data:
        raise KeyError(f"{meta_pt_path} 中没有 META")

    meta_list = data["META"]
    if args.sample_idx < 0 or args.sample_idx >= len(meta_list):
        raise IndexError(f"--sample-idx 越界: {args.sample_idx}，有效范围是 [0, {len(meta_list)-1}]")

    pred_data_list = None
    if args.mode == "pred":
        if "PRED_3DMM" in data:
            pred_data_list = data["PRED_3DMM"]
        elif "PRED" in data:
            pred_data_list = data["PRED"]
        elif "3dmm_coeff" in data:
            pred_data_list = data["3dmm_coeff"]
        elif isinstance(data, torch.Tensor):
            pred_data_list = data
        else:
            raise KeyError(f"{meta_pt_path} 中找不到预测数据")

    if args.output_dir is None:
        output_dir = meta_pt_path.parent / f"render_{args.mode}_idx{args.sample_idx:03d}"
    else:
        output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    renderer = Renderer(
        project_root=args.project_root,
        device=args.device,
        exp_scale=args.exp_scale,
        rot_scale=args.rot_scale,
        trans_scale=args.trans_scale,
    )

    meta = meta_list[args.sample_idx]
    speaker_rel = meta["speaker_path"]

    if args.max_listeners < 0:
        listener_paths = meta["listener_paths"]
    else:
        listener_paths = meta["listener_paths"][:args.max_listeners]
    
    speaker_video_path = resolve_video_path(args.dataset_root, args.split, speaker_rel)

    print("========== Render Config ==========")
    print("mode           :", args.mode)
    print("sample_idx     :", args.sample_idx)
    print("output_dir     :", output_dir)
    print("exp_scale      :", args.exp_scale)
    print("rot_scale      :", args.rot_scale)
    print("trans_scale    :", args.trans_scale)

    for i, listener_rel in enumerate(listener_paths):
        listener_video_path = resolve_video_path(args.dataset_root, args.split, listener_rel)
        real_coeff_path = resolve_coeff_path(args.dataset_root, args.split, listener_rel)
        real_vectors = load_coeff_npy(real_coeff_path)

        if args.mode == "real":
            render_vectors = real_vectors
        else:
            render_vectors = extract_pred_item(pred_data_list, args.sample_idx, i)

        output_name = f"idx{args.sample_idx:03d}_sample{i:02d}"
        output_video_path = output_dir / f"{output_name}.avi"
        output_meta_path = output_dir / f"{output_name}.json"

        print(f"\n[{i+1}/{len(listener_paths)}] Rendering {output_name}")
        print("listener       :", listener_rel)
        print("real_shape     :", tuple(real_vectors.shape))
        print("render_shape   :", tuple(render_vectors.shape))

        renderer.render_video(
            render_vectors=render_vectors,
            reference_vectors=real_vectors,
            mode=args.mode,
            speaker_video_path=speaker_video_path,
            listener_video_path=listener_video_path,
            output_video_path=output_video_path,
            output_meta_path=output_meta_path,
            reference_image_path=args.reference_image,
            faceverse_chunk=args.faceverse_chunk,
            pirender_chunk=args.pirender_chunk,
            fps=args.fps,
        )

    print("\n[OK] 全部渲染完成")


if __name__ == "__main__":
    main()