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
DEFAULT_DATASET_ROOT = "/data2/REACT2024-NEW"
DEFAULT_SPLIT = "test"

# 这里只是为了读取 META，不再使用其中的 PRED 3DMM
DEFAULT_META_PT = "/media/Hulu/面部反应生成/baseline_react2025-main-1/outputs/motion_diffusion/react_2025/online/260314202155_tb83tebt/3dmm.pt"


def add_project_to_syspath(project_root: str):
    project_root = str(Path(project_root).resolve())
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

#给PIRender准备输入的语义序列索引，默认语义半径为13，即每帧使用前后13帧共27帧的语义信息
def obtain_seq_index(index, num_frames, semantic_radius=13):
    seq = list(range(index - semantic_radius, index + semantic_radius + 1))
    seq = [min(max(item, 0), num_frames - 1) for item in seq]
    return seq


def transform_semantic(semantic: torch.Tensor) -> torch.Tensor:
    """
    semantic: [T, 58]
    return:   [T, 58, 27]
    """
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


def load_first_frame_as_tensor(video_path: Path, image_transform):
    vr = VideoReader(str(video_path), ctx=cpu(0))
    if len(vr) == 0:
        raise RuntimeError(f"空视频: {video_path}")
    frame = vr[0].asnumpy()
    img = Image.fromarray(frame)
    img = image_transform(img)
    return img

def load_reference_frame_as_tensor(video_path: Path, image_transform, mode="middle"):
    vr = VideoReader(str(video_path), ctx=cpu(0))
    if len(vr) == 0:
        raise RuntimeError(f"空视频: {video_path}")

    if mode == "first":
        frame_idx = 0
    elif mode == "middle":
        frame_idx = len(vr) // 2
    elif mode == "one_third":
        frame_idx = len(vr) // 3
    else:
        raise ValueError(f"未知 mode: {mode}")

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

    out = []
    for f in frames:
        if f.shape[0] != target_hw[0] or f.shape[1] != target_hw[1]:
            f = cv2.resize(f, (target_hw[1], target_hw[0]))
        out.append(f.astype(np.uint8))

    return np.stack(out, axis=0)


def load_coeff_npy(coeff_path: Path) -> torch.Tensor:
    if not coeff_path.is_file():
        raise FileNotFoundError(f"找不到真实系数文件: {coeff_path}")

    x = np.load(coeff_path)
    x = torch.from_numpy(x).float()

    # 支持多种常见格式：
    # [T, 58]
    # [1, T, 58]
    # [T, 1, 58]
    # [1, 1, T, 58]（保险）
    if x.ndim == 2 and x.shape[-1] == 58:
        return x

    if x.ndim == 3:
        if x.shape[0] == 1 and x.shape[2] == 58:   # [1, T, 58]
            x = x.squeeze(0)
        elif x.shape[1] == 1 and x.shape[2] == 58: # [T, 1, 58]
            x = x.squeeze(1)
        else:
            raise ValueError(f"系数形状异常: {tuple(x.shape)}，无法解析为 [T,58]")
        return x

    if x.ndim == 4:
        # 极少数情况做保险处理
        if x.shape[0] == 1 and x.shape[1] == 1 and x.shape[-1] == 58:
            x = x.squeeze(0).squeeze(0)
            return x

    raise ValueError(f"系数形状异常: {tuple(x.shape)}，期望 [T,58] / [1,T,58] / [T,1,58]")


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
        self.image_transform = Transform(256, 224)

        # FaceVerse
        faceverse_dir = self.project_root / "external" / "FaceVerse"
        faceverse_model_path = faceverse_dir / "data" / "faceverse_simple_v2.npy"

        self.faceverse, _ = get_faceverse(
            path=str(faceverse_model_path),
            device=self.device,
            img_size=224
        )
        self.faceverse.init_coeff_tensors()

        self.id_tensor = torch.from_numpy(
            np.load(faceverse_dir / "reference_full.npy")
        ).float().view(1, -1)[:, :150]

        # PIRender
        pirender_ckpt = self.project_root / "external" / "PIRender" / "cur_model_fold.pth"
        if not pirender_ckpt.is_file():
            raise FileNotFoundError(f"找不到 PIRender 权重: {pirender_ckpt}")

        self.pi_render = FaceGenerator().to(self.device)
        self.pi_render.eval()

        checkpoint = torch.load(str(pirender_ckpt), map_location=self.device)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        clean_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                clean_state_dict[k[7:]] = v
            else:
                clean_state_dict[k] = v

        self.pi_render.load_state_dict(clean_state_dict, strict=True)

        # mean/std
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

        # 保持和你原项目一致
        self._transform = transforms.Lambda(
            lambda e: (lambda tmp: tmp.__setitem__(
                (slice(None), -1),
                e[:, -1] - self.mean_face[0, 0, -1]
            ) or tmp)(e.clone())
        )

    @torch.no_grad()
    def render_video(
        self,
        listener_vectors: torch.Tensor,   # [T,58]
        speaker_video_path: Path,
        listener_video_path: Path,
        output_video_path: Path,
        output_meta_path: Path,
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

        # 用真实 listener 第一帧作为 reference
        # listener_reference = load_first_frame_as_tensor(
        #     listener_video_path,
        #     self.image_transform
        # ).to(self.device)

        listener_reference = load_reference_frame_as_tensor(
            listener_video_path,
            self.image_transform,
            mode="middle"
        ).to(self.device)

        listener_vectors = listener_vectors.to(self.device).float()

        # 如果输入的是已经标准化后的 3dmm，才反归一化；
        # 现在我们直接读 dataset 里的真实 coefficients，所以默认 coeff_is_normalized=False
        if self.coeff_is_normalized:
            listener_vectors = listener_vectors.unsqueeze(0)
            listener_vectors = self._reverse_transform_3dmm(listener_vectors)[0]

        # 和项目原来的渲染逻辑一致，始终做这个平移修正
        listener_vectors = self._transform(listener_vectors)

        # 长度对齐
        speaker_vr = VideoReader(str(speaker_video_path), ctx=cpu(0))
        listener_vr = VideoReader(str(listener_video_path), ctx=cpu(0))
        T = min(listener_vectors.shape[0], len(speaker_vr), len(listener_vr))
        if T <= 0:
            raise RuntimeError("有效帧数为 0，无法渲染")

        listener_vectors = listener_vectors[:T]
        semantics_all = transform_semantic(listener_vectors.detach()).cpu()  # [T,58,27]

        writer = cv2.VideoWriter(
            str(output_video_path),
            cv2.VideoWriter_fourcc(*"MJPG"),
            fps,
            (224 * 4, 224),   # 896 x 224
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

        for start in tqdm(
            range(0, T, faceverse_chunk),
            desc=f"render {output_video_path.stem}",
            leave=False
        ):
            end = min(start + faceverse_chunk, T)
            vec_chunk = listener_vectors[start:end]               # [L,58]
            sem_chunk = semantics_all[start:end].to(self.device)  # [L,58,27]
            L = vec_chunk.shape[0]

            # ---------- 3D mesh ----------
            self.faceverse.batch_size = L
            self.faceverse.init_coeff_tensors()

            self.faceverse.exp_tensor = vec_chunk[:, :52]
            self.faceverse.rot_tensor = vec_chunk[:, 52:55]
            self.faceverse.trans_tensor = vec_chunk[:, 55:]
            self.faceverse.id_tensor = self.id_tensor.reshape(1, 150).repeat(L, 1).to(self.device)

            pred_dict = self.faceverse(
                self.faceverse.get_packed_tensors(),
                render=True,
                texture=False
            )
            rendered_img_r = pred_dict["rendered_img"]
            rendered_img_r = np.clip(rendered_img_r.detach().cpu().numpy(), 0, 255)
            rendered_img_r = rendered_img_r[:, :, :, :3].astype(np.uint8)  # [L,224,224,3]

            # ---------- 2D animation ----------
            fake_chunks = []
            for pstart in range(0, L, pirender_chunk):
                pend = min(pstart + pirender_chunk, L)
                sub_len = pend - pstart

                ref_batch = listener_reference.unsqueeze(0).repeat(sub_len, 1, 1, 1)
                sem_batch = sem_chunk[pstart:pend]

                out_dict = self.pi_render(ref_batch, sem_batch)
                fake_videos = out_dict["fake_image"]
                fake_videos = self.torch_img_to_np2(fake_videos)  # [sub,224,224,3]
                fake_chunks.append(fake_videos)

            listener_videos = np.concatenate(fake_chunks, axis=0)  # [L,224,224,3]

            # ---------- 真实 speaker / listener ----------
            speaker_frames = load_video_chunk_uint8(
                speaker_video_path, start, end, target_hw=(224, 224)
            )
            listener_real_frames = load_video_chunk_uint8(
                listener_video_path, start, end, target_hw=(224, 224)
            )

            L2 = min(
                rendered_img_r.shape[0],
                listener_videos.shape[0],
                speaker_frames.shape[0],
                listener_real_frames.shape[0],
            )

            rendered_img_r = rendered_img_r[:L2]
            listener_videos = listener_videos[:L2]
            speaker_frames = speaker_frames[:L2]
            listener_real_frames = listener_real_frames[:L2]

            for i in range(L2):
                combined_img = np.zeros((224, 224 * 4, 3), dtype=np.uint8)
                combined_img[:, 0:224] = speaker_frames[i]
                combined_img[:, 224:448] = listener_real_frames[i]
                combined_img[:, 448:672] = rendered_img_r[i]
                combined_img[:, 672:896] = listener_videos[i]
                writer.write(combined_img)

        writer.release()


def main():
    parser = argparse.ArgumentParser(
        description="使用真实 coefficients 渲染前 N 个视频的全部 listener 样本。"
    )
    parser.add_argument("--project-root", type=str, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--dataset-root", type=str, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--split", type=str, default=DEFAULT_SPLIT)

    # 这里只为了读取 META
    parser.add_argument("--meta-pt", type=str, default=DEFAULT_META_PT)

    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--coeff-is-normalized", action="store_true",
                        help="如果读取的 coeff 已经是标准化后的，则开启；默认关闭，因为 dataset/coefficients 下是原始真实系数。")
    parser.add_argument("--transform-reverse", type=str, default="standard",
                        choices=["standard", "zero_center"])
    parser.add_argument("--faceverse-chunk", type=int, default=512)
    parser.add_argument("--pirender-chunk", type=int, default=128)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--first-n", type=int, default=3)
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

    if args.output_dir is None:
        output_dir = meta_pt_path.parent / "rendered_real_coeff_first3"
    else:
        output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    renderer = FourPanelRenderer(
        project_root=args.project_root,
        device=args.device,
        coeff_is_normalized=args.coeff_is_normalized,
        transform_reverse=args.transform_reverse,
    )

    total_indices = min(args.first_n, len(meta_list))
    total_jobs = 0
    for idx in range(total_indices):
        meta = meta_list[idx]
        if "listener_paths" not in meta:
            raise KeyError(f"META[{idx}] 缺少 listener_paths")
        total_jobs += len(meta["listener_paths"])

    print("========== Batch Render Config ==========")
    print("meta pt         :", meta_pt_path)
    print("dataset_root    :", args.dataset_root)
    print("split           :", args.split)
    print("device          :", args.device)
    print("coeff normalized:", args.coeff_is_normalized)
    print("reverse         :", args.transform_reverse)
    print("first_n         :", total_indices)
    print("total_jobs      :", total_jobs)
    print("output_dir      :", output_dir)

    job_id = 0
    for idx in range(total_indices):
        meta = meta_list[idx]

        if "speaker_path" not in meta or "listener_paths" not in meta:
            raise KeyError(f"META[{idx}] 缺少 speaker_path 或 listener_paths")

        speaker_rel = meta["speaker_path"]
        listener_paths = meta["listener_paths"]

        speaker_video_path = resolve_video_path(args.dataset_root, args.split, speaker_rel)
        if not speaker_video_path.is_file():
            raise FileNotFoundError(f"找不到 speaker 视频: {speaker_video_path}")

        for sample_idx, listener_rel in enumerate(listener_paths):
            job_id += 1

            listener_video_path = resolve_video_path(args.dataset_root, args.split, listener_rel)
            coeff_path = resolve_coeff_path(args.dataset_root, args.split, listener_rel)

            if not listener_video_path.is_file():
                raise FileNotFoundError(f"找不到 listener 视频: {listener_video_path}")
            if not coeff_path.is_file():
                raise FileNotFoundError(f"找不到 listener coefficients: {coeff_path}")

            listener_vectors = load_coeff_npy(coeff_path)   # [T,58]，真实系数

            output_name = f"idx{idx:03d}_sample{sample_idx:02d}"
            output_video_path = output_dir / f"{output_name}.avi"
            output_meta_path = output_dir / f"{output_name}.json"

            print(f"\n[{job_id}/{total_jobs}] Rendering {output_name}")
            print("speaker        :", speaker_rel)
            print("listener       :", listener_rel)
            print("coeff_path     :", coeff_path)
            print("coeff_shape    :", tuple(listener_vectors.shape))

            renderer.render_video(
                listener_vectors=listener_vectors,
                speaker_video_path=speaker_video_path,
                listener_video_path=listener_video_path,
                output_video_path=output_video_path,
                output_meta_path=output_meta_path,
                faceverse_chunk=args.faceverse_chunk,
                pirender_chunk=args.pirender_chunk,
                fps=args.fps,
            )

    print("\n[OK] 全部渲染完成")


if __name__ == "__main__":
    main()