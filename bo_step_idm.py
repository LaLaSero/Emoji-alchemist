# bo_step_ldm.py
import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import UpperConfidenceBound
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
import torchvision.utils as vutils

# --- モデルのロード ---
from conv_vae import ConvVAE
from latent_unet import LatentUNet

Z_DIM = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading models...")
# VAE (デコーダーとして使用)
vae = ConvVAE(z_dim=Z_DIM).to(DEVICE)
vae.load_state_dict(torch.load("emoji_vae.pth"))
vae.eval()

# Latent Diffusion Model
diffusion_model = LatentUNet(z_dim=Z_DIM).to(DEVICE)
diffusion_model.load_state_dict(torch.load("latent_diffusion.pth"))
diffusion_model.eval()

# generate.pyからサンプリング関数を流用
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
    return z / z.std(dim=-1, keepdim=True)

# --- ベイズ最適化の実行 ---

# Step 1: Diffusionモデルで高品質な初期データ(train_x)を生成
print("\nGenerating initial high-quality latents for BO...")
train_x = generate_latents(num_latents=100)

# 仮の目的関数（好みのスコアを模倣）
def preference_function(z: torch.Tensor) -> torch.Tensor:
    # 例: zのL2ノルムが小さいほど「好みが高い」と仮定
    #     （より複雑な関数を定義可能）
    return -z.norm(dim=-1, keepdim=True)

train_y = preference_function(train_x)

# Step 2: GPモデルの学習
print("Fitting GP model...")
gp = SingleTaskGP(train_x, train_y.to(torch.float32))
mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
fit_gpytorch_mll(mll)

# Step 3: UCBでacquisition最大化
ucb = UpperConfidenceBound(gp, beta=0.2)

# 探索範囲はtrain_xのmin/maxを基準に設定
bounds = torch.stack([train_x.min(dim=0).values, train_x.max(dim=0).values])

print("Optimizing acquisition function...")
new_z, _ = optimize_acqf(
    acq_function=ucb,
    bounds=bounds,
    q=1, num_restarts=10, raw_samples=128,
)
print("提案された次のz:", new_z)

# --- 結果の可視化 ---
with torch.no_grad():
    # 評価の高かった初期サンプル4つと、BOが提案した新しいサンプル1つをデコード
    top_indices = torch.topk(train_y.squeeze(), 4).indices
    z_top4 = train_x[top_indices]
    
    z_all = torch.cat([z_top4, new_z], dim=0)
    decoded = vae.decode(z_all)
    decoded = (decoded + 1) / 2 # [-1, 1] -> [0, 1]

grid = vutils.make_grid(decoded, nrow=5)
from torchvision.transforms.functional import to_pil_image
to_pil_image(grid).save("bo_final_results.png")
print("\nSaved BO results to 'bo_final_results.png'")
print("左4つが評価の高かった初期サンプル、右端がBOによって提案された新しいサンプルです。")
