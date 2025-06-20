# conv_vae.py (高速化対応版)
import torch.nn as nn, torch

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out += residual # Skip Connection
        return self.relu(out)

class ConvVAE(nn.Module):
    # __init__にCLIPの埋め込み次元数を追加
    def __init__(self, z_dim=64, clip_emb_dim=512):
        super().__init__()
        # --- Encoder (変更なし) ---
        self.enc = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.ReLU(),          # 72→36
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),         # 36→18
            nn.Flatten(),                                  # 64*18*18
        )
        self.fc_mu     = nn.Linear(64*18*18, z_dim)
        self.fc_logvar = nn.Linear(64*18*18, z_dim)

        # ★★★★★ ここからが新しい追加部分 ★★★★★
        # VAEの潜在空間(z_dim)からCLIPの意味空間(clip_emb_dim)への翻訳機
        self.projection_head = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.ReLU(),
            nn.Linear(256, clip_emb_dim)
        )
        # ★★★★★ 追加ここまで ★★★★★

        # --- Decoder (変更なし) ---
        self.fc_up = nn.Linear(z_dim, 64*18*18)
        self.dec = nn.Sequential(
            nn.ReLU(),
            nn.Unflatten(1, (64, 18, 18)),
            ResidualBlock(64),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(), # 18→36
            # デコーダーのResidualBlockの入力チャンネル数を修正
            # ConvTranspose2dの出力が32チャンネルなので、次も32にする
            ResidualBlock(32), 
            nn.ConvTranspose2d(32, 3, 4, 2, 1), nn.Tanh()   # 36→72
        )

    def reparameterize(self, mu, logvar):
        eps = torch.randn_like(mu)
        return mu + eps * (0.5*logvar).exp()

    def encode(self, x):
        h = self.enc(x)
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        return mu, logvar

    def decode(self, z):
        return self.dec(self.fc_up(z))

    # forwardメソッドを修正
    def forward(self, x):
        mu, logvar = self.encode(x)
        z  = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        
        # ★★★★★ muを射影して、CLIP損失で使えるようにする ★★★★★
        # forwardの戻り値に、射影された潜在ベクトルを追加
        projected_mu = self.projection_head(mu)
        
        return recon, mu, logvar, projected_mu
