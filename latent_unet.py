# latent_unet.py
import torch
import torch.nn as nn
import math

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class ResidualBlock(nn.Module):
    """
    Simple residual MLP block with LayerNorm and GELU non‑linearity.
    A small scaling (0.1) is applied to the residual branch for stability.
    """
    def __init__(self, dim, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(num_groups=8, num_channels=dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim)
        )

    def forward(self, x):
        return x + 0.1 * self.net(x)

class LatentUNet(nn.Module):
    def __init__(self, z_dim=32, time_emb_dim=128):
        super().__init__()
        
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU()
        )
        
        # -------- Down path --------
        self.down1 = nn.Sequential(
            nn.Linear(z_dim, 256),
            ResidualBlock(256)
        )
        self.down2 = nn.Sequential(
            nn.Linear(256, 512),
            ResidualBlock(512)
        )
        self.down3 = nn.Sequential(
            nn.Linear(512, 1024),
            ResidualBlock(1024)
        )

        # -------- Bottleneck --------
        self.mid_proj = nn.Sequential(
            ResidualBlock(1024),
            ResidualBlock(1024)
        )

        # -------- Up path --------
        self.up1_time_mlp = nn.Linear(time_emb_dim, 1024)
        self.up1 = nn.Sequential(
            nn.Linear(2048, 512),
            ResidualBlock(512)
        )

        self.up2_time_mlp = nn.Linear(time_emb_dim, 512)
        self.up2 = nn.Sequential(
            nn.Linear(1024, 256),
            ResidualBlock(256)
        )

        self.up3_time_mlp = nn.Linear(time_emb_dim, 256)
        self.up3 = nn.Sequential(
            nn.Linear(512, z_dim)
        )

    def forward(self, z, t):
        t_emb = self.time_mlp(t)
        
        # Down path
        h1 = self.down1(z)
        h2 = self.down2(h1)
        h3 = self.down3(h2)
        
        # Middle
        h_mid = self.mid_proj(h3)

        # Up path
        h_mid = h_mid + self.up1_time_mlp(t_emb)
        up1_cat = torch.cat([h_mid, h3], dim=1)
        h_up1 = self.up1(up1_cat)

        h_up1 = h_up1 + self.up2_time_mlp(t_emb)
        up2_cat = torch.cat([h_up1, h2], dim=1)
        h_up2 = self.up2(up2_cat)

        h_up2 = h_up2 + self.up3_time_mlp(t_emb)
        up3_cat = torch.cat([h_up2, h1], dim=1)
        output = self.up3(up3_cat)
        
        return output
