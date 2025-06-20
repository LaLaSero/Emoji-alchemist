# generate_by_diff.py
import torch
import torchvision.utils as vutils
from conv_vae import ConvVAE
from latent_unet import LatentUNet

# --- ハイパーパラメータ（学習時と合わせる） ---
Z_DIM = 32
TIMESTEPS = 1000
BETA_START = 0.0001
BETA_END = 0.02
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- モデルのロード ---
# 1. VAE（デコーダーとして使用）
print("Loading pre-trained VAE...")
vae = ConvVAE(z_dim=Z_DIM).to(DEVICE)
# vae.load_state_dict(torch.load("emoji_vae.pth"))
# 拡散モデルの学習時と同じ、新しいVAEの重みをロードする
vae.load_state_dict(torch.load("emoji_vae_clip_fast.pth"))
vae.eval()

# 2. Latent Diffusion Model (U-Net)
print("Loading Latent Diffusion model...")
diffusion_model = LatentUNet(z_dim=Z_DIM).to(DEVICE)
diffusion_model.load_state_dict(torch.load("latent_diffusion.pth"))
diffusion_model.eval()

# --- Diffusionのサンプリング設定 ---
betas = torch.linspace(BETA_START, BETA_END, TIMESTEPS, device=DEVICE)
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

try:
    scale_factor = torch.load("vae_latent_scale.pt").to(DEVICE)
    print(f"Loaded VAE latent scale factor: {scale_factor:.4f}")
except FileNotFoundError:
    print("Scale factor not found, using 1.0. Please re-run training script first.")
    scale_factor = 1.0

@torch.no_grad()
def p_sample(z_t, t):
    """
    1ステップ分のノイズ除去を行う関数
    """
    # U-Netを使ってノイズを予測
    pred_noise = diffusion_model(z_t, torch.tensor([t], device=DEVICE).float())
    
    # ノイズを除去してz_{t-1}を計算
    model_mean = sqrt_recip_alphas[t] * (z_t - betas[t] * pred_noise / sqrt_one_minus_alphas_cumprod[t])
    
    if t == 0:
        return model_mean
    else:
        # さらにノイズを加える
        posterior_variance = betas[t]
        noise = torch.randn_like(z_t)
        return model_mean + torch.sqrt(posterior_variance) * noise

@torch.no_grad()
def sample(num_images=16):
    """
    Tステップかけて、ノイズから潜在ベクトルを生成する関数
    """
    print("Generating latents from noise...")
    # Step 1: ランダムノイズからスタート
    z = torch.randn(num_images, Z_DIM, device=DEVICE)
    
    # Step 2: Tステップかけてノイズ除去を繰り返す
    for i in reversed(range(TIMESTEPS)):
        if i % 100 == 0:
            print(f"  Sampling step {i}/{TIMESTEPS}")
        z = p_sample(z, i)
    
    # 正規化（経験的に生成が安定することがある）
    z = z / z.std(dim=-1, keepdim=True)

    print("Decoding latents to images...")
    # Step 3: 生成された潜在ベクトルをVAEデコーダーで画像に変換
    images = vae.decode(z)
    z_unscaled = z * scale_factor
    images = vae.decode(z_unscaled)
    
    # [-1, 1] から [0, 1] に変換
    images = (images + 1) / 2
    return images

# --- メイン処理 ---
generated_images = sample(num_images=16)

# グリッド画像として保存
vutils.save_image(generated_images, "generated_emojis.png", nrow=4)
print("\nSaved generated images to 'generated_emojis.png'")

