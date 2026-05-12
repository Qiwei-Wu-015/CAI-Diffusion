import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------------------------------------------------------------------
# 1. 模态适配器 (Modality Adapter)
# 作用：将 Source 模态的信息映射并注入到 Target 模态
# 改进：加入了 LayerNorm (Pre-Norm) 以提高训练稳定性
# ------------------------------------------------------------------------------------------
class ModalityAdapter(nn.Module):
    def __init__(self, dim, reduction=16): 
        super().__init__()
        # 动态计算隐藏层维度，避免过小
        hidden_dim = max(8, dim // reduction)
        
        self.net = nn.Sequential(
            nn.LayerNorm(dim), # Pre-Norm：先归一化再变换，模仿 StitchFusion 的稳定性设计
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim)
        )
        
        # 初始化为接近 0，确保初始阶段不会破坏原有特征
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return self.net(x)

# ------------------------------------------------------------------------------------------
# 2. 前馈网络 (FeedForward)
# 标准的 Transformer FFN
# ------------------------------------------------------------------------------------------
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim), # Pre-Norm
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

# ------------------------------------------------------------------------------------------
# 3. 缝合块 (StitchBlock)
# 核心逻辑：MHSA -> Stitch(Attn) -> FFN -> Stitch(FFN)
# ------------------------------------------------------------------------------------------
class StitchBlock(nn.Module):
    def __init__(self, dim, num_heads, num_modals=3, mlp_ratio=4., drop=0.1):
        super().__init__()
        self.num_modals = num_modals
        
        # 3.1 Intra-Modal Self-Attention (每个模态独立的时序建模)
        self.norms_attn = nn.ModuleList([nn.LayerNorm(dim) for _ in range(num_modals)])
        self.attns = nn.ModuleList([
            nn.MultiheadAttention(dim, num_heads, dropout=drop, batch_first=True) 
            for _ in range(num_modals)
        ])
        
        # 3.2 First Stitching Layer (MHSA 之后)
        # 动态创建适配器: i(Source) -> j(Target)
        self.stitch_att = nn.ModuleDict()
        for src in range(num_modals):
            for dst in range(num_modals):
                if src != dst:
                    self.stitch_att[f'{src}_{dst}'] = ModalityAdapter(dim)

        # 3.3 Feed Forward Network (每个模态独立)
        self.ffns = nn.ModuleList([
            FeedForward(dim, int(dim * mlp_ratio), dropout=drop) 
            for _ in range(num_modals)
        ])

        # 3.4 Second Stitching Layer (FFN 之后)
        self.stitch_mlp = nn.ModuleDict()
        for src in range(num_modals):
            for dst in range(num_modals):
                if src != dst:
                    self.stitch_mlp[f'{src}_{dst}'] = ModalityAdapter(dim)

    def forward(self, x_list):
        # x_list: [Audio, 3DMM, Emo], Shape = [Batch, Seq, Dim]
        
        # --- Stage 1: Intra-Modal MHSA (时序建模) ---
        x_post_attn = []
        for i, x in enumerate(x_list):
            res = x
            x_norm = self.norms_attn[i](x)
            x_attn, _ = self.attns[i](x_norm, x_norm, x_norm)
            x_post_attn.append(res + x_attn) # Residual
        
        # --- Stage 2: Cross-Modal Stitching 1 (Attn Level) ---
        # 串行逻辑：使用 MHSA 后的特征作为 Source 进行缝合
        x_stitched_1 = [x.clone() for x in x_post_attn]
        current_state = [x.clone() for x in x_post_attn] # 冻结 Source
        
        for src in range(self.num_modals):
            for dst in range(self.num_modals):
                if src != dst:
                    # Target = Target + Adapter(Source)
                    adapter = self.stitch_att[f'{src}_{dst}']
                    x_stitched_1[dst] = x_stitched_1[dst] + adapter(current_state[src])
        
        # --- Stage 3: FFN (特征提炼) ---
        x_post_ffn = []
        for i, x in enumerate(x_stitched_1):
            res = x
            # FFN 内部包含了 Pre-Norm
            x_ffn = self.ffns[i](x)
            x_post_ffn.append(res + x_ffn)

        # --- Stage 4: Cross-Modal Stitching 2 (FFN Level) ---
        x_final = [x.clone() for x in x_post_ffn]
        current_state = [x.clone() for x in x_post_ffn]
        
        for src in range(self.num_modals):
            for dst in range(self.num_modals):
                if src != dst:
                    adapter = self.stitch_mlp[f'{src}_{dst}']
                    x_final[dst] = x_final[dst] + adapter(current_state[src])
        
        return x_final


# ------------------------------------------------------------------------------------------
# 4. 缝合编码器 (StitchEncoder) - 主入口
# ------------------------------------------------------------------------------------------
class StitchEncoder(nn.Module):
    def __init__(self, 
                 input_dims=[768, 58, 25], # 对应 [Audio, 3DMM, Emotion]
                 latent_dim=512, 
                 num_layers=2, 
                 num_heads=8):
        super().__init__()
        
        self.num_modals = len(input_dims)
        
        # 1. 投影层 (Projection)
        self.projections = nn.ModuleList([
            nn.Linear(in_dim, latent_dim) for in_dim in input_dims
        ])
        
        # 2. 位置编码 (Learnable Positional Embedding)
        # 假设最大序列长度为 1000，足够覆盖 60 帧
        self.pos_embed = nn.Parameter(torch.zeros(1, 1000, latent_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        # 3. 堆叠 StitchBlocks
        self.blocks = nn.ModuleList([
            StitchBlock(latent_dim, num_heads, num_modals=self.num_modals)
            for _ in range(num_layers)
        ])
        
        # 4. 最终 Norm
        self.final_norms = nn.ModuleList([nn.LayerNorm(latent_dim) for _ in range(self.num_modals)])

    def forward(self, inputs):
        """
        Args:
            inputs: list of tensors, e.g. [audio, 3dmm, emotion]
                    each shape is [Batch, Seq_Len, Input_Dim]
        Returns:
            out_list: list of fused tensors
                      each shape is [Seq_Len, Batch, Latent_Dim] (Ready for Transformer)
        """
        assert len(inputs) == self.num_modals, f"Input size {len(inputs)} != defined modals {self.num_modals}"
        
        # 1. 投影 & 加位置编码
        x_list = []
        for i, x in enumerate(inputs):
            # Projection: [B, L, In] -> [B, L, 512]
            x = self.projections[i](x) 
            # Pos Embed
            seq_len = x.size(1)
            x = x + self.pos_embed[:, :seq_len, :]
            x_list.append(x)
            
        # 2. 穿过缝合块
        for block in self.blocks:
            x_list = block(x_list)
            
        # 3. 最终输出处理
        out_list = []
        for i, x in enumerate(x_list):
            x = self.final_norms[i](x)
            # Permute to [L, B, D] for standard PyTorch Transformer
            out_list.append(x)
            
        return out_list