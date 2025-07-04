# train_latent_diffusion.py
import math
import torch
from torch.utils.data import DataLoader, TensorDataset
from conv_vae import ConvVAE
from latent_unet import LatentUNet
import torch.nn.functional as F

# --- ハイパーパラメータ ---
Z_DIM = 32
TIMESTEPS = 1000
DEVICE = "cuda"

# Cosine beta schedule
def cosine_beta_schedule(timesteps: int, s: float = 0.1) -> torch.Tensor:
    steps = torch.arange(timesteps + 1, dtype=torch.float32, device=DEVICE)
    alphas_cumprod = torch.cos(((steps / timesteps) + s) / (1 + s) * math.pi / 2) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return betas

betas = cosine_beta_schedule(TIMESTEPS, s=0.1)

alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)

# --- 1. 学習済みのVAEをロード（重みは固定） ---
print("Loading pre-trained VAE...")
vae = ConvVAE(z_dim=Z_DIM).to(DEVICE)
vae.load_state_dict(torch.load("emoji_vae_clip_fast.pth"))
vae.eval()
for param in vae.parameters():
    param.requires_grad = False

# --- 2. データセットから潜在ベクトルを抽出 ---
print("Extracting latents from dataset...")
data = torch.load("openmoji_72x72.pt")
loader = DataLoader(TensorDataset(data), batch_size=512, shuffle=False)

latents = []
with torch.no_grad():
    for (x,) in loader:
        x = x.to(DEVICE)
        mu, _ = vae.encode(x)
        latents.append(mu.cpu())
latents = torch.cat(latents, dim=0)

latent_mean = latents.mean(dim=0)
scale_factor = latents.std()
print(f"Original latent mean abs: {latent_mean.abs().mean():.4f} | std: {scale_factor:.4f}")

# --- 白色化: (z - mean) / std ---------------------------------
latents = (latents - latent_mean) / scale_factor

# 生成時に使うため保存
torch.save(latent_mean,   "vae_latent_mean.pt")
torch.save(scale_factor,  "vae_latent_scale.pt")
print(f"Latents normalized. New mean: {latents.mean():.4f}, std: {latents.std():.4f} | mean/std saved.")

latent_loader = DataLoader(TensorDataset(latents), batch_size=512, shuffle=True)

# --- 3. Diffusionモデルの準備 ---
model = LatentUNet(z_dim=Z_DIM).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

def get_noisy_latent(z_start, t, noise):
    sqrt_alpha_cumprod = torch.sqrt(alphas_cumprod[t])[:, None]
    sqrt_one_minus_alpha_cumprod = torch.sqrt(1. - alphas_cumprod[t])[:, None]
    return sqrt_alpha_cumprod * z_start + sqrt_one_minus_alpha_cumprod * noise

# --- 4. 学習ループ ---
print("Starting training...")
for epoch in range(5000): # 実際にはもっと多くのエポックが必要
    for (z_start,) in latent_loader:
        z_start = z_start.to(DEVICE)
        
        # t（タイムステップ）をランダムにサンプリング
        t = torch.randint(0, TIMESTEPS, (z_start.size(0),), device=DEVICE)
        
        # ノイズを生成し、noisy latentを計算
        noise = torch.randn_like(z_start)
        z_t = get_noisy_latent(z_start, t, noise)
        
        # モデルにノイズを予測させる
        predicted_noise = model(z_t, t.float())
        
        # 損失計算 (MSE)
        loss = F.mse_loss(predicted_noise, noise)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    if epoch % 25 == 0:
        print(f"Epoch {epoch:04d} | Loss: {loss.item():.6f}")

torch.save(model.state_dict(), "latent_diffusion.pth")
print("Latent diffusion model saved.")
