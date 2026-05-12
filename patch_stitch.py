from framework.motion_diffusion.diffusion.stitch_encoder2 import ResampleModalityAdapter

with open('/media/Hulu/面部反应生成/baseline_react2025-main-1/framework/motion_diffusion/diffusion/stitch_encoder2.py', 'r') as f:
    content = f.read()

old_code = """            else:
                # 下采样 (Downsample): 60 -> 10 (使用 Adaptive Avg Pool 比线性插值损失更小)
                out = F.adaptive_avg_pool1d(out, self.dst_len)"""

new_code = """            else:
                # 下采样 (Downsample)
                stride = self.src_len // self.dst_len
                if self.src_len % self.dst_len == 0:
                    out = F.avg_pool1d(out, kernel_size=stride, stride=stride)
                else:
                    out = F.adaptive_avg_pool1d(out, self.dst_len)"""

if old_code in content:
    content = content.replace(old_code, new_code)
    with open('/media/Hulu/面部反应生成/baseline_react2025-main-1/framework/motion_diffusion/diffusion/stitch_encoder2.py', 'w') as f:
        f.write(content)
    print("Patched!")
else:
    print("Not found")

