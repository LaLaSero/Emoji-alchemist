# bo_step_clip.py
import torch
import clip # CLIPライブラリをインポート
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import UpperConfidenceBound
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
import torchvision.utils as vutils
from torchvision import transforms

# --- モデルのロード ---
from conv_vae import ConvVAE
from latent_unet import LatentUNet

Z_DIM = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading models (VAE, Diffusion, CLIP)...")
# VAE (デコーダーとして使用)
vae = ConvVAE(z_dim=Z_DIM).to(DEVICE)
vae.load_state_dict(torch.load("emoji_vae.pth"))
vae.eval()

# Latent Diffusion Model
diffusion_model = LatentUNet(z_dim=Z_DIM).to(DEVICE)
diffusion_model.load_state_dict(torch.load("latent_diffusion.pth"))
diffusion_model.eval()

# CLIPモデル
clip_model, clip_preprocess = clip.load("ViT-B/32", device=DEVICE)
clip_model.eval()

# VAEの潜在空間のスケール値をロード
scale_factor = torch.load("vae_latent_scale.pt").to(DEVICE)

# --- Diffusionサンプリング関数 (変更なし) ---
betas = torch.linspace(0.0001, 0.02, 1000, device=DEVICE)
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

@torch.no_grad()
def p_sample(z_t, t):
    pred_noise = diffusion_model(z_t, torch.tensor([t], device=DEVICE).float())
    model_mean = sqrt_recip_alphas[t] * (z_t - betas[t] * pred_noise / sqrt_one_minus_alphas_cumprod[t])
    if t == 0: return model_mean
    else:
        noise = torch.randn_like(z_t)
        return model_mean + torch.sqrt(betas[t]) * noise

@torch.no_grad()
def generate_latents(num_latents=100):
    z = torch.randn(num_latents, Z_DIM, device=DEVICE)
    for i in reversed(range(1000)):
        if i % 200 == 0: print(f"  Initial sampling step {i}")
        z = p_sample(z, i)
    return z

# ★★★★★ ここからがCLIPを使った目的関数 ★★★★★
# 探索したいテキストプロンプト
TARGET_TEXT = "cold"

# テキストをエンコード（これは一度だけ行えば良い）
text_tokens = clip.tokenize([TARGET_TEXT]).to(DEVICE)
with torch.no_grad():
    TARGET_TEXT_FEATURES = clip_model.encode_text(text_tokens).float()
    TARGET_TEXT_FEATURES /= TARGET_TEXT_FEATURES.norm(dim=-1, keepdim=True)

# CLIPが要求する画像の前処理
clip_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26588654, 0.27577711))
])

def clip_preference_function(z: torch.Tensor) -> torch.Tensor:
    """
    潜在ベクトルzから画像を生成し、ターゲットテキストとのCLIP類似度を計算する。
    """
    with torch.no_grad():
        # Step 1: zをデコードして画像にする
        z_unscaled = z * scale_factor
        images = vae.decode(z_unscaled) # [-1, 1]の範囲
        images = (images + 1) / 2      # [0, 1]に変換

        # Step 2: CLIP用に画像を前処理
        images_preprocessed = clip_transform(images)

        # Step 3: 画像の特徴量を計算
        image_features = clip_model.encode_image(images_preprocessed).float()
        image_features /= image_features.norm(dim=-1, keepdim=True)

        # Step 4: テキスト特徴量との類似度（コサイン類似度）を計算
        # Botorchは大きい値を最大化しようとするので、類似度をそのまま返す
        similarity = (100.0 * image_features @ TARGET_TEXT_FEATURES.T)
    
    return similarity.squeeze(1).detach().cpu() # (N, 1) -> (N,)

# --- ベイズ最適化の実行 ---
print(f"\nTarget text: '{TARGET_TEXT}'")
# Step 1: 高品質な初期データ(train_x)を生成
print("Generating initial high-quality latents for BO...")
train_x = generate_latents(num_latents=20) # 初期サンプルは少なめでも良い

# Step 2: 初期データのスコアをCLIPで評価
print("Evaluating initial latents with CLIP...")
# BoTorchは(N, 1)の形状を期待するのでreshape
train_y = clip_preference_function(train_x).unsqueeze(1)

# Step 3: GPモデルの学習
print("Fitting GP model...")
gp = SingleTaskGP(train_x.cpu(), train_y.cpu())
mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
fit_gpytorch_mll(mll)

# Step 4: UCBでacquisition最大化
ucb = UpperConfidenceBound(gp, beta=0.2)
bounds = torch.stack([train_x.cpu().min(dim=0).values, train_x.cpu().max(dim=0).values])

print("Optimizing acquisition function to find the next best emoji...")
new_z, _ = optimize_acqf(
    acq_function=ucb,
    bounds=bounds,
    q=1, num_restarts=10, raw_samples=128,
)
new_z = new_z.to(DEVICE) # GPUに戻す

# --- 結果の可視化 ---
with torch.no_grad():
    z_all = torch.cat([train_x[:4], new_z], dim=0) # 初期サンプル4つ + BO提案1つ
    z_unscaled = z_all * scale_factor
    decoded = vae.decode(z_unscaled)
    decoded = (decoded + 1) / 2

vutils.save_image(decoded, "bo_clip_results.png", nrow=5)
print(f"\nSaved results to 'bo_clip_results.png'")
print("左4つが初期サンプル、右端がBOによって『cowboy angel』に最も近いと提案された新しいサンプルです。")
