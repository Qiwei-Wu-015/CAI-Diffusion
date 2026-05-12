#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import numpy as np
import argparse
from pathlib import Path

def analyze_tensor(tensor, name="Tensor"):
    """打印 Tensor 的基础统计信息"""
    print(f"\n[{name}] 分析:")
    print(f"  形状 (Shape) : {tuple(tensor.shape)}")
    print(f"  全局极值     : Min = {tensor.min().item():.4f}, Max = {tensor.max().item():.4f}, Mean = {tensor.mean().item():.4f}")
    
    # 假设最后一维是 58，我们按物理意义拆分查看
    if tensor.shape[-1] == 58:
        # 表情 (0~52)
        exp = tensor[..., :52]
        print(f"  -> 表情(Exp) [0:52]  : Min = {exp.min().item():.6f}, Max = {exp.max().item():.6f}, Mean = {exp.mean().item():.6f}")
        
        # 旋转 (52~55)
        rot = tensor[..., 52:55]
        print(f"  -> 旋转(Rot) [52:55] : Min = {rot.min().item():.6f}, Max = {rot.max().item():.6f}, Mean = {rot.mean().item():.6f}")
        
        # 平移 (55~58)，尤其是 Z 轴深度 [57] 非常关键
        trans = tensor[..., 55:58]
        z_axis = tensor[..., 57]
        print(f"  -> 平移(Trans)[55:58]: Min = {trans.min().item():.6f}, Max = {trans.max().item():.6f}, Mean = {trans.mean().item():.6f}")
        print(f"  -> 深度(Z轴)  [57]   : Min = {z_axis.min().item():.6f}, Max = {z_axis.max().item():.6f}, Mean = {z_axis.mean().item():.6f}")
        
def main():
    parser = argparse.ArgumentParser(description="3DMM 预测文件 (.pt) 诊断工具")
    # 默认路径填入你之前提供的那个
    parser.add_argument("--pt-path", type=str, 
                        default="/media/Hulu/面部反应生成/baseline_react2025-main-1/outputs/motion_diffusion/react_2025/online/260316164839_o6xzfea2/3dmm.pt",
                        help="3dmm.pt 文件的路径")
    
    args = parser.parse_args()
    pt_path = Path(args.pt_path)
    
    if not pt_path.exists():
        print(f"❌ 找不到文件: {pt_path}")
        return

    print(f"正在加载文件: {pt_path} ...")
    data = torch.load(str(pt_path), map_location="cpu")
    
    if isinstance(data, dict):
        print(f"✅ 文件是一个字典，包含以下 Keys: {list(data.keys())}")
        
        # 尝试寻找预测数据
        pred_keys = ["PRED_3DMM", "PRED", "3dmm_coeff"]
        pred_data = None
        for k in pred_keys:
            if k in data:
                pred_data = data[k]
                print(f"✅ 找到预测数据，所在 Key 为: '{k}'")
                break
                
        if pred_data is None:
            print("❌ 在字典中没有找到标准的预测数据 Key (PRED_3DMM, PRED 等)")
            return
            
    elif isinstance(data, torch.Tensor):
        print("✅ 文件直接是一个 Tensor")
        pred_data = data
    elif isinstance(data, list):
        print(f"✅ 文件是一个 List，长度为 {len(data)}")
        pred_data = data
    else:
        print(f"❌ 未知的数据类型: {type(data)}")
        return

    # 取出第一个样本进行详细分析
    first_sample = pred_data[0]
    
    if isinstance(first_sample, torch.Tensor):
        print(f"\n====== 第 1 个预测样本诊断 ======")
        # 兼容 [多样性, 序列长度, 58] 的形状
        if first_sample.ndim == 3:
            analyze_tensor(first_sample[0], name="样本0 的 第1个生成序列")
        else:
            analyze_tensor(first_sample, name="样本0 的 序列")
            
        print("\n💡 【诊断指南】")
        print("1. 如果 '深度(Z轴)' 的值在 0.000X 左右，说明数据是**去过均值**的。必须在渲染时使用 `--coeff-is-normalized --transform-reverse zero_center` 将平均脸加回来！")
        print("2. 如果 '深度(Z轴)' 的值是 10~20 左右，说明这是真实的物理空间坐标，不需要加平均脸。")
        print("3. 如果任何数值（比如表情）出现了 > 10.0 或者 < -10.0 的情况，说明**你的模型输出了爆炸的异常值**，这是导致满屏尖刺的罪魁祸首！")
    else:
        print(f"无法解析内部数据结构，类型为: {type(first_sample)}")

if __name__ == "__main__":
    main()