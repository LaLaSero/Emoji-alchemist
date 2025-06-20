# evaluate_convVAE.py
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from conv_vae import ConvVAE

# データのロード
data = torch.load("openmoji_72x72.pt")  # (N,3,72,72)
data = data[:64]  # 可視化用に一部だけ使う
x = data.cuda()

# モデルのロード
model = ConvVAE(z_dim=32).cuda()
# --- load checkpoint and strip "module." prefix if present ---
state_dict = torch.load("emoji_vae_capacity.pth", map_location='cpu')
if any(k.startswith("module.") for k in state_dict.keys()):
    state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
model.load_state_dict(state_dict)
model.eval()

# 推論
with torch.no_grad():
    # ConvVAE returns (recon, mu, logvar, extra); we ignore the 4th value
    recon, mu, logvar, _ = model(x)
    recon_loss = F.mse_loss(recon, x, reduction='mean').item()
    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean().item()

print(f"Reconstruction MSE: {recon_loss:.4f}")
print(f"KL Divergence: {kl:.4f}")

# 可視化（8x8 グリッド）
def show_images(orig, recon):
    orig = orig.cpu().permute(0, 2, 3, 1)
    recon = recon.cpu().permute(0, 2, 3, 1)

    # [-1, 1] → [0, 1] に変換
    orig = (orig + 1) / 2
    recon = (recon + 1) / 2

    fig, axes = plt.subplots(2, 8, figsize=(16, 4))
    for i in range(8):
        axes[0, i].imshow(orig[i])
        axes[0, i].axis('off')
        axes[1, i].imshow(recon[i])
        axes[1, i].axis('off')
    axes[0, 0].set_ylabel('Original')
    axes[1, 0].set_ylabel('Reconstructed')
    plt.tight_layout()
    plt.savefig("vae_reconstruction-cpa.png", dpi=300)

show_images(x, recon)