import torch
import torch.nn as nn
import math
from diffusers import UNet2DModel

# Sinusoidal timestep embedding
class SinusoidalPositionEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if self.dim % 2 == 1:
            emb = nn.functional.pad(emb, (0, 1))
        return emb

# Conditional UNet using HuggingFace Diffusers UNet2DModel
class ConditionalUNetDiffusers(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        num_classes=24,
        time_emb_dim=256,
        cond_emb_dim=256,
        device="cuda"
    ):
        super().__init__()
        self.device = device
        self.time_emb_dim = time_emb_dim
        self.cond_emb_dim = cond_emb_dim

        # Time embedding (sinusoidal + MLP)
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 2),
            nn.GELU(),
            nn.Linear(time_emb_dim * 2, time_emb_dim),
        )

        # Conditional embedding
        self.cond_mlp = nn.Sequential(
            nn.Linear(num_classes, cond_emb_dim),
            nn.GELU(),
            nn.Linear(cond_emb_dim, cond_emb_dim),
            nn.Dropout(0.1),
            nn.Linear(cond_emb_dim, time_emb_dim),
        )

        # HuggingFace UNet2DModel - 5 Layers with attention
        self.unet = UNet2DModel(
            sample_size=64,               # image resolution
            in_channels=in_channels,      # input channels (e.g. RGB)
            out_channels=out_channels,    # output channels
            layers_per_block=2,
            block_out_channels=(64, 128, 256, 512, 512),  # 5 levels
            down_block_types=(
                "DownBlock2D",           # 1
                "DownBlock2D",           # 2
                "AttnDownBlock2D",       # 3
                "AttnDownBlock2D",       # 4
                "DownBlock2D",           # 5
            ),
            up_block_types=(
                "UpBlock2D",             # 5
                "AttnUpBlock2D",         # 4
                "AttnUpBlock2D",         # 3
                "UpBlock2D",             # 2
                "UpBlock2D",             # 1
            ),
            attention_head_dim=8,
        )

    def forward(self, x, t, cond):
        # t: (B,) long tensor
        t = t.to(self.device)
        cond = cond.to(self.device)

        # Embed time and condition
        t_emb = self.time_mlp(t)
        c_emb = self.cond_mlp(cond)
        emb = t_emb + c_emb

        # HuggingFace UNet expects raw t (not embedding)
        return self.unet(x, timestep=t).sample
