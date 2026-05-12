#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import torch
import importlib.util
from pathlib import Path
from tqdm import tqdm


# =============================
# 路径
# =============================
RNN_PY = "/media/Hulu/面部反应生成/baseline_react2025-main-1/framework/motion_diffusion/diffusion/rnn.py"
CKPT = "/media/Hulu/面部反应生成/All_VAEv2_W50/checkpoint_999.pth"
RESULT_PT = "/media/Hulu/面部反应生成/baseline_react2025-main-1/outputs/motion_diffusion/react_2025/online/260322173609_fb8qojg8/results.pt"

DEVICE = "cpu"
CHUNK_SIZE = 16384
# =============================


def load_module_from_path(py_file_path):
    py_file_path = str(Path(py_file_path).resolve())
    project_root = str(Path(py_file_path).resolve().parents[3])

    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    spec = importlib.util.spec_from_file_location("rnn_module", py_file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_model(device):
    module = load_module_from_path(RNN_PY)
    LatentEmbedder = module.LatentEmbedder

    model = LatentEmbedder(
        hidden_dim=128,
        z_dim=128,
        emb_dims=[64, 64],
        num_layers=2,
        rnn_type="gru",
        dropout=0.0,
        emotion_dim=25,
        coeff_3dmm_dim=58,
    )

    ckpt = torch.load(CKPT, map_location="cpu")
    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt

    clean = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            clean[k[7:]] = v
        else:
            clean[k] = v

    model.load_state_dict(clean, strict=True)
    model.eval()
    model.to(device)

    return model


@torch.no_grad()
def convert_tensor_chunked(model, x, device, chunk_size):

    orig_shape = x.shape
    x_flat = x.reshape(-1, 25).float()

    outs = []
    total = x_flat.shape[0]

    for start in tqdm(
        range(0, total, chunk_size),
        leave=False,
        desc="chunks",
    ):
        end = min(start + chunk_size, total)

        chunk = x_flat[start:end].to(device)
        out = model.decode_coeff(chunk)

        outs.append(out.cpu())

    y = torch.cat(outs, dim=0)
    return y.reshape(*orig_shape[:-1], 58)


@torch.no_grad()
def convert_results(model, data, device, chunk_size):

    new_data = {}

    for key, value in data.items():

        if isinstance(value, list):

            new_list = []

            for item in tqdm(
                value,
                desc=f"Converting {key}",
                total=len(value),
            ):

                if isinstance(item, torch.Tensor) and item.shape[-1] == 25:

                    converted = convert_tensor_chunked(
                        model,
                        item,
                        device,
                        chunk_size,
                    )

                    new_list.append(converted)

                else:
                    new_list.append(item)

            new_data[key] = new_list

        else:
            new_data[key] = value

    return new_data


def validate(src, dst):

    print("\n========== 验证 ==========")

    for key in ["GT", "PRED"]:

        src_list = src[key]
        dst_list = dst[key]

        assert len(src_list) == len(dst_list)

        for s, d in zip(src_list, dst_list):

            assert s.shape[0] == d.shape[0]
            assert s.shape[1] == d.shape[1]

            assert s.shape[2] == 25
            assert d.shape[2] == 58

        print(f"[OK] {key} verified")

    print("\nSample shapes:")
    print("GT :", src["GT"][0].shape, "->", dst["GT"][0].shape)
    print("PRED:", src["PRED"][0].shape, "->", dst["PRED"][0].shape)


def main():

    result_path = Path(RESULT_PT)
    out_path = result_path.parent / "3dmm.pt"

    device = DEVICE
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    print("device:", device)

    print("\nLoading model...")
    model = load_model(device)

    print("\nLoading results.pt...")
    src_data = torch.load(result_path, map_location="cpu")

    print("\nConverting emotion -> 3dmm")
    dst_data = convert_results(
        model,
        src_data,
        device,
        CHUNK_SIZE,
    )

    print("\nSaving...")
    torch.save(dst_data, out_path)

    print("Saved:", out_path)

    validate(src_data, dst_data)


if __name__ == "__main__":
    main()