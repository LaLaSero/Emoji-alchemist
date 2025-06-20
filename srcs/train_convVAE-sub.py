# train_emoji_vae.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import lpips  # perceptual loss

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
# LPIPS perceptual loss network (frozen VGG)
lpips_fn = lpips.LPIPS(net='vgg').cuda().eval()
# --- オプティマイザと CosineAnnealingLR ---
base_lr = 5e-4
eta_min = 1e-5
total_epochs = 20000
opt = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    opt, T_max=total_epochs, eta_min=eta_min
)

# --- ハイパーパラメータ ---
# Capacity Annealing parameters
C_max = 30.0        # 目標とする KL 容量 (nats)
gamma = 50.0       # KL を容量に合わせる強さ
lambda_perc = 0.1       # LPIPS loss weight
anneal_epochs = 3000

# --- 学習ループ ---
print("Starting training...")
for epoch in range(total_epochs):
    # Capacity Annealing: linearly increase target KL capacity C_t
    C_current = min(C_max, C_max * (epoch / anneal_epochs))
    # γ を漸減させる: epoch 0→6000 で 50→30、以降は 30
    gamma_current = max(30.0, 50.0 * (1 - epoch / total_epochs))
    for (x,) in loader:
        x = x.cuda()

        # projected_mu は使わないので無視
        recon, mu, logvar, _ = model(x)
        
        # 損失計算からCLIP関連を削除
        recon_loss = F.mse_loss(recon, x, reduction='mean')
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean()
        # Perceptual (LPIPS) loss; expects inputs in [-1,1] range
        perceptual_loss = lpips_fn(recon, x).mean()
        
        loss = recon_loss + lambda_perc * perceptual_loss + \
               gamma_current * torch.abs(kl - C_current)
        
        opt.zero_grad()
        loss.backward()
        opt.step()

    # Cosine LR scheduler step (1 step per epoch)
    scheduler.step()

    if epoch % 50 == 0:
        print(
            f"epoch {epoch:04d}  total={loss.item():.4f}  "
            f"recon={recon_loss.item():.4f}  P={perceptual_loss.item():.4f}  "
            f"KL={kl.item():.4f}  C={C_current:.2f}  "
            f"γ={gamma_current:.1f}  lr={opt.param_groups[0]['lr']:.6f}"
        )

# 新しいモデルとして保存
torch.save(model.state_dict(), "emoji_vae_capacity.pth")
print("Saved capacity annealing VAE to 'emoji_vae_capacity.pth'")
