import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------------------------------------------------------------------
# 1. 可变长模态适配器 (Resample Modality Adapter) - 修改版
# 作用：不仅做特征映射，还负责在 Source 和 Target 长度不一致时进行 上采样/下采样
# ------------------------------------------------------------------------------------------
class ResampleModalityAdapter(nn.Module):
    def __init__(self, dim, src_len, dst_len, reduction=16): 
        super().__init__()
        self.src_len = src_len
        self.dst_len = dst_len
        
        hidden_dim = max(8, dim // reduction)
        
        # 基础的 MLP 变换
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim)
        )
        
        # 初始化为0
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        # x shape: [Batch, Seq_Len_Src, Dim]
        
        # 1. 先做特征变换
        out = self.net(x)
        
        # 2. 如果长度不一致，进行重采样
        if self.src_len != self.dst_len:
            # [B, L, D] -> [B, D, L]
            out = out.transpose(1, 2)
            
            if self.dst_len > self.src_len:
                # 上采样 (Upsample): 10 -> 60
                out = F.interpolate(out, size=self.dst_len, mode='linear', align_corners=True)
            else:
                # 下采样 (Downsample): 60 -> 10 (使用 Adaptive Avg Pool 比线性插值损失更小)
                out = F.adaptive_avg_pool1d(out, self.dst_len)
                
            # [B, D, L] -> [B, L, D]
            out = out.transpose(1, 2)
            
        return out

# FeedForward 类保持不变...
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class StitchBlock(nn.Module):
    def __init__(self, dim, num_heads, modal_lengths, mlp_ratio=4., drop=0.1):
        """
        modal_lengths: list, 例如 [60, 60, 60, 10]
        """
        super().__init__()
        self.num_modals = len(modal_lengths)
        self.modal_lengths = modal_lengths
        
        # 假设最后一个模态 (index=3) 是 Future Emotion，它是“被动”的
        # 你也可以把这个 index 作为参数传进来，这里为了方便直接写死或判定长度
        self.passive_modal_idx = 3 
        
        # 3.1 Intra-Modal Self-Attention 
        self.norms_attn = nn.ModuleList([nn.LayerNorm(dim) for _ in range(self.num_modals)])
        self.attns = nn.ModuleList([
            nn.MultiheadAttention(dim, num_heads, dropout=drop, batch_first=True) 
            for _ in range(self.num_modals)
        ])
        
        # 3.2 First Stitching Layer (Attn Level)
        self.stitch_att = nn.ModuleDict()
        for src in range(self.num_modals):
            for dst in range(self.num_modals):
                if src == dst:
                    continue
                
                # --- [修改点] 核心逻辑 ---
                # 如果 Source 是 Future(3)，并且 Target 不是 Future(3)
                # 即：试图从 Future 传向 别人 -> 禁止！
                if src == self.passive_modal_idx and dst != self.passive_modal_idx:
                    continue 
                # -----------------------

                self.stitch_att[f'{src}_{dst}'] = ResampleModalityAdapter(
                    dim, src_len=modal_lengths[src], dst_len=modal_lengths[dst]
                )

        # 3.3 Feed Forward Network
        self.ffns = nn.ModuleList([
            FeedForward(dim, int(dim * mlp_ratio), dropout=drop) 
            for _ in range(self.num_modals)
        ])

        # 3.4 Second Stitching Layer (FFN Level)
        self.stitch_mlp = nn.ModuleDict()
        for src in range(self.num_modals):
            for dst in range(self.num_modals):
                if src == dst:
                    continue
                
                # --- [修改点] 核心逻辑 (同上) ---
                if src == self.passive_modal_idx and dst != self.passive_modal_idx:
                    continue
                # -----------------------------

                self.stitch_mlp[f'{src}_{dst}'] = ResampleModalityAdapter(
                    dim, src_len=modal_lengths[src], dst_len=modal_lengths[dst]
                )

    def forward(self, x_list):
        # --- Stage 1: Intra-Modal MHSA ---
        x_post_attn = []
        for i, x in enumerate(x_list):
            res = x
            x_norm = self.norms_attn[i](x)
            x_attn, _ = self.attns[i](x_norm, x_norm, x_norm)
            x_post_attn.append(res + x_attn) 
        
        # --- Stage 2: Cross-Modal Stitching 1 ---
        x_stitched_1 = [x.clone() for x in x_post_attn]
        current_state = [x.clone() for x in x_post_attn] 
        
        for src in range(self.num_modals):
            for dst in range(self.num_modals):
                # 如果是自己，或者是被禁止的连接（Future->别人），直接跳过
                key = f'{src}_{dst}'
                if key not in self.stitch_att:
                    continue
                
                adapter = self.stitch_att[key]
                feat = adapter(current_state[src])
                x_stitched_1[dst] = x_stitched_1[dst] + feat
        
        # --- Stage 3: FFN ---
        x_post_ffn = []
        for i, x in enumerate(x_stitched_1):
            res = x
            x_ffn = self.ffns[i](x)
            x_post_ffn.append(res + x_ffn)

        # --- Stage 4: Cross-Modal Stitching 2 ---
        x_final = [x.clone() for x in x_post_ffn]
        current_state = [x.clone() for x in x_post_ffn]
        
        for src in range(self.num_modals):
            for dst in range(self.num_modals):
                # 同样的判断逻辑
                key = f'{src}_{dst}'
                if key not in self.stitch_mlp:
                    continue

                adapter = self.stitch_mlp[key]
                feat = adapter(current_state[src])
                x_final[dst] = x_final[dst] + feat
        
        return x_final

# ------------------------------------------------------------------------------------------
# 4. 缝合编码器 (StitchEncoder) - 最终版
# ------------------------------------------------------------------------------------------
class StitchEncoder(nn.Module):
    def __init__(self, 
                 input_dims=[768, 58, 25, 25], 
                 modal_lengths=[60, 60, 60, 10], # 明确指定每个模态的长度
                 latent_dim=512, 
                 num_layers=2, 
                 num_heads=8):
        super().__init__()
        
        self.num_modals = len(input_dims)
        self.modal_lengths = modal_lengths
        
        # 1. 投影层
        self.projections = nn.ModuleList([
            nn.Linear(in_dim, latent_dim) for in_dim in input_dims
        ])
        
        # 2. 位置编码
        # 我们只需要初始化一次，forward的时候根据各自长度切片即可
        self.pos_embed = nn.Parameter(torch.zeros(1, 1000, latent_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        # 3. 堆叠 StitchBlocks (传入长度配置)
        self.blocks = nn.ModuleList([
            StitchBlock(latent_dim, num_heads, modal_lengths=modal_lengths)
            for _ in range(num_layers)
        ])
        
        # 4. 最终 Norm
        self.final_norms = nn.ModuleList([nn.LayerNorm(latent_dim) for _ in range(self.num_modals)])

    def forward(self, inputs):
        # inputs: list of tensors, 长度对应 modal_lengths
        assert len(inputs) == self.num_modals
        
        # 1. 投影 & 加位置编码
        x_list = []
        for i, x in enumerate(inputs):
            # Projection
            x = self.projections[i](x) # [B, L_i, Latent]
            
            # Pos Embed (根据当前模态的实际长度切片)
            cur_len = self.modal_lengths[i]
            # 安全检查：确保输入长度符合预期 (可选)
            # assert x.size(1) == cur_len, f"Input {i} length mismatch"
            
            x = x + self.pos_embed[:, :cur_len, :]
            x_list.append(x)
            
        # 2. 穿过缝合块
        for block in self.blocks:
            x_list = block(x_list)
            
        # 3. 最终输出
        out_list = []
        for i, x in enumerate(x_list):
            x = self.final_norms[i](x)
            out_list.append(x)
            
        return out_list