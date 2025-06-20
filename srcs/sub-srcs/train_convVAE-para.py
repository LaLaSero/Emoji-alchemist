# train_emoji_vae.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# 修正版のConvVAEをインポート
from conv_vae import ConvVAE

# --- データの準備 (CLIP関連削除) ---
print("Loading data...")
data = torch.load("openmoji_72x72.pt")
dataset = TensorDataset(data)
effective_bs = 256 * max(1, torch.cuda.device_count())
loader = DataLoader(dataset, batch_size=effective_bs, shuffle=True, num_workers=2, pin_memory=True)

# --- モデルとオプティマイザの準備 ---
# CLIP埋め込み次元の指定を削除
model = ConvVAE(z_dim=32)
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
    model = nn.DataParallel(model)
model = model.cuda()
opt = torch.optim.Adam(model.parameters(), 5e-4)

# --- ハイパーパラメータ ---
# Capacity Annealing parameters
C_max = 20.0        # 目標とする KL 容量 (nats)
gamma = 30.0       # KL を容量に合わせる強さ
anneal_epochs = 2500

# --- 学習ループ ---
print("Starting training...")
for epoch in range(10000):
    # Capacity Annealing: linearly increase target KL capacity C_t
    C_current = min(C_max, C_max * (epoch / anneal_epochs))
    for (x,) in loader:
        x = x.cuda()

        # projected_mu は使わないので無視
        recon, mu, logvar, _ = model(x)
        
        # 損失計算からCLIP関連を削除
        recon_loss = F.mse_loss(recon, x, reduction='mean')
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean()
        
        loss = recon_loss + gamma * torch.abs(kl - C_current)
        
        opt.zero_grad()
        loss.backward()
        opt.step()

    if epoch % 50 == 0:
        print(
            f"epoch {epoch:04d}  total={loss.item():.4f}  "
            f"recon={recon_loss.item():.4f}  KL={kl.item():.4f}  "
            f"C={C_current:.2f}"
        )

# 新しいモデルとして保存
torch.save(model.state_dict(), "emoji_vae_parallel.pth")
print("Saved capacity annealing VAE to 'emoji_vae_capacity.pth'")
