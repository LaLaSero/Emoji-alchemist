# test_decoder.py
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from torchvision.utils import make_grid

# --- いつものモデル読み込み -------------
from conv_vae import ConvVAE
Z_DIM = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

vae = ConvVAE(z_dim=Z_DIM).to(DEVICE)
# --- robust checkpoint loader (handles DataParallel 'module.' prefix) ---
ckpt = torch.load("emoji_vae_capacity.pth", map_location=DEVICE)
if any(k.startswith("module.") for k in ckpt.keys()):
    ckpt = {k.replace("module.", "", 1): v for k, v in ckpt.items()}
vae.load_state_dict(ckpt, strict=False)

vae.eval()

scale_factor = torch.load("vae_latent_scale.pt").to(DEVICE)
if scale_factor.ndim == 0:
    scale_factor = scale_factor.unsqueeze(0)

# --- データセット（訓練で使った72x72テンソル）
all_images = torch.load("openmoji_72x72.pt")   # (N,3,72,72) in [-1,1]
loader = DataLoader(TensorDataset(all_images), batch_size=25, shuffle=True)

# 1バッチだけ取り出す
x, = next(iter(loader))                         # 1-tuple unpack
x = x.to(DEVICE)

with torch.no_grad():
    mu, logvar = vae.encode(x)
    x_recon = vae.decode(mu)                   # 再構成
    x_recon = (x_recon + 1) / 2                # to 0-1
    x_orig  = (x + 1) / 2

# --- グリッド表示（左=original, 右=recon）
grid = torch.cat([x_orig, x_recon], dim=0)      # (2B,3,72,72)
img = make_grid(grid, nrow=5, padding=2)
plt.figure(figsize=(10,4))
plt.title("Original (top)  vs  VAE reconstruction (bottom)")
plt.axis("off")
plt.imshow(img.permute(1,2,0).cpu())
# plt.show()
plt.savefig("vae_reconstruction_test.png", bbox_inches='tight', dpi=300)

# ==============================================================
# ① Posterior‑z decode test  ------------------------------------
#    → mu + σ·ε からサンプリングされた現実的な latent を使う
# ==============================================================
mu, logvar = vae.encode(x)
std = (0.5 * logvar).exp()
eps = torch.randn_like(std)
z_posterior = mu + eps * std
imgs_prior = (vae.decode(z_posterior) + 1) / 2
grid_prior = make_grid(imgs_prior, nrow=5, padding=2)
plt.figure(figsize=(10, 4))
plt.title("Posterior‑z decode (realistic latent sample)")
plt.axis("off")
plt.imshow(grid_prior.permute(1, 2, 0).cpu())
plt.savefig("vae_prior_random_decode.png", bbox_inches="tight", dpi=300)

exit()
# ==============================================================
# ② Diffusion only  --------------------------------------------
#    → z0 ~ N(0,I) → p_sample(..., t=0) → decode
#       *p_sample* は app.py に既に定義されているものを再利用
# ==============================================================
try:
    # p_sample を動的インポート（app.py 内で定義済み）
    import importlib

    app_module = importlib.import_module("app")
    p_sample = app_module.p_sample

    zT = torch.randn(25, Z_DIM, device=DEVICE)
    for t in reversed(range(1000)):
        zT = p_sample(zT, t)
    imgs_diff = (vae.decode(zT * scale_factor) + 1) / 2
    grid_diff = make_grid(imgs_diff, nrow=5, padding=2)
    plt.figure(figsize=(10, 4))
    plt.title("Diffusion → decode")
    plt.axis("off")
    plt.imshow(grid_diff.permute(1, 2, 0).cpu())
    plt.savefig("diffusion_decode_test.png", bbox_inches="tight", dpi=300)

except (ModuleNotFoundError, AttributeError) as e:
    print("⚠  Diffusion test skipped: p_sample could not be imported.", e)