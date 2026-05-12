import os
import sys
import torch
import importlib.util
from collections import OrderedDict


# =========================
# 你自己的路径
# =========================
RNN_PY_PATH = "/media/Hulu/面部反应生成/baseline_react2025-main-1/framework/motion_diffusion/diffusion/rnn.py"
CKPT_PATH = "/media/Hulu/面部反应生成/All_VAEv2_W50/checkpoint_999.pth"


# =========================
# 如果训练时参数不同，请改这里
# =========================
MODEL_KWARGS = dict(
    hidden_dim=128,
    z_dim=128,
    emb_dims=[64, 64],
    num_layers=2,
    rnn_type="gru",
    dropout=0.0,
    emotion_dim=25,
    coeff_3dmm_dim=58,
)


def load_module_from_path(py_file_path, module_name="user_rnn_module"):
    """
    从指定 py 文件动态导入模块
    """
    if not os.path.isfile(py_file_path):
        raise FileNotFoundError(f"找不到 rnn.py: {py_file_path}")

    module_dir = os.path.dirname(py_file_path)
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)

    spec = importlib.util.spec_from_file_location(module_name, py_file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法从路径加载模块: {py_file_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def find_latent_embedder_class(module):
    """
    在模块里找到 LatentEmbedder 类
    """
    if not hasattr(module, "LatentEmbedder"):
        raise AttributeError("在 rnn.py 中没有找到 LatentEmbedder 类")
    return getattr(module, "LatentEmbedder")


def extract_state_dict(ckpt):
    """
    自动从 checkpoint 中提取 state_dict
    """
    if isinstance(ckpt, dict):
        possible_keys = [
            "state_dict",
            "model",
            "model_state_dict",
            "net",
            "generator",
        ]
        for k in possible_keys:
            if k in ckpt and isinstance(ckpt[k], dict):
                print(f"[Info] 检测到 checkpoint 使用 key: {k}")
                return ckpt[k]

        # 如果本身看起来就像 state_dict
        if all(isinstance(v, torch.Tensor) for v in ckpt.values()):
            print("[Info] checkpoint 本身就是 state_dict")
            return ckpt

    raise ValueError("无法自动从 checkpoint 中提取 state_dict，请手动检查 checkpoint 结构")


def remove_module_prefix(state_dict):
    """
    去掉 DataParallel / DDP 保存时常见的 module. 前缀
    """
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        nk = k[7:] if k.startswith("module.") else k
        new_state_dict[nk] = v
    return new_state_dict


def summarize_checkpoint(ckpt):
    print("\n========== Checkpoint 顶层结构 ==========")
    if isinstance(ckpt, dict):
        for k, v in ckpt.items():
            if isinstance(v, dict):
                print(f"{k}: dict, len={len(v)}")
            elif isinstance(v, torch.Tensor):
                print(f"{k}: Tensor, shape={tuple(v.shape)}")
            else:
                print(f"{k}: {type(v)}")
    else:
        print(f"checkpoint 类型: {type(ckpt)}")


def compare_model_and_ckpt(model, state_dict):
    model_dict = model.state_dict()

    model_keys = set(model_dict.keys())
    ckpt_keys = set(state_dict.keys())

    missing_keys = sorted(model_keys - ckpt_keys)
    unexpected_keys = sorted(ckpt_keys - model_keys)

    shape_mismatch = []
    matched_count = 0
    for k in sorted(model_keys & ckpt_keys):
        if model_dict[k].shape != state_dict[k].shape:
            shape_mismatch.append((k, tuple(model_dict[k].shape), tuple(state_dict[k].shape)))
        else:
            matched_count += 1

    print("\n========== 参数匹配结果 ==========")
    print(f"模型参数总数: {len(model_keys)}")
    print(f"权重参数总数: {len(ckpt_keys)}")
    print(f"同名且 shape 一致: {matched_count}")
    print(f"缺失参数数目: {len(missing_keys)}")
    print(f"多余参数数目: {len(unexpected_keys)}")
    print(f"shape 不一致数目: {len(shape_mismatch)}")

    if missing_keys:
        print("\n--- Missing keys (模型里有，权重里没有) ---")
        for k in missing_keys:
            print(k)

    if unexpected_keys:
        print("\n--- Unexpected keys (权重里有，模型里没有) ---")
        for k in unexpected_keys:
            print(k)

    if shape_mismatch:
        print("\n--- Shape mismatch ---")
        for k, model_shape, ckpt_shape in shape_mismatch:
            print(f"{k}: model={model_shape}, ckpt={ckpt_shape}")

    return missing_keys, unexpected_keys, shape_mismatch


def try_strict_load(model, state_dict):
    print("\n========== strict=True 加载测试 ==========")
    try:
        model.load_state_dict(state_dict, strict=True)
        print("结论：完全匹配，strict=True 可以成功加载。")
        return True
    except RuntimeError as e:
        print("结论：不完全匹配，strict=True 加载失败。")
        print("\nPyTorch 报错如下：")
        print(str(e))
        return False


def main():
    print("开始检查 LatentEmbedder 和 checkpoint 是否匹配...\n")
    print(f"rnn.py 路径: {RNN_PY_PATH}")
    print(f"ckpt 路径  : {CKPT_PATH}")

    # 1. 导入模块
    module = load_module_from_path(RNN_PY_PATH)

    # 2. 获取类
    LatentEmbedder = find_latent_embedder_class(module)

    # 3. 实例化模型
    print("\n========== 实例化模型参数 ==========")
    for k, v in MODEL_KWARGS.items():
        print(f"{k}: {v}")

    model = LatentEmbedder(**MODEL_KWARGS)
    model.eval()

    # 4. 读取 checkpoint
    if not os.path.isfile(CKPT_PATH):
        raise FileNotFoundError(f"找不到 checkpoint: {CKPT_PATH}")

    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    summarize_checkpoint(ckpt)

    # 5. 提取 state_dict
    state_dict = extract_state_dict(ckpt)
    state_dict = remove_module_prefix(state_dict)

    # 6. 对比
    compare_model_and_ckpt(model, state_dict)

    # 7. 尝试 strict=True
    try_strict_load(model, state_dict)


if __name__ == "__main__":
    main()