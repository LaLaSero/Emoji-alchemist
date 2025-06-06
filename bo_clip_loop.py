# bo_clip_loop.py
import torch
import clip
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import UpperConfidenceBound
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
import torchvision.utils as vutils
from torchvision import transforms
from tqdm import tqdm # 進捗表示用のライブラリ

# --- モデルと設定のロード (変更なし) ---
from conv_vae import ConvVAE
from latent_unet import LatentUNet

Z_DIM = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Loading models (VAE, Diffusion, CLIP)...")
vae = ConvVAE(z_dim=Z_DIM).to(DEVICE); vae.load_state_dict(torch.load("emoji_vae.pth")); vae.eval()
diffusion_model = LatentUNet(z_dim=Z_DIM).to(DEVICE); diffusion_model.load_state_dict(torch.load("latent_diffusion.pth")); diffusion_model.eval()
clip_model, _ = clip.load("ViT-B/32", device=DEVICE); clip_model.eval()
scale_factor = torch.load("vae_latent_scale.pt").to(DEVICE)

# --- Diffusionサンプリング関数 (変更なし) ---
betas = torch.linspace(0.0001, 0.02, 1000, device=DEVICE); alphas = 1. - betas; alphas_cumprod = torch.cumprod(alphas, axis=0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas); sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
@torch.no_grad()
def p_sample(z_t, t):
    pred_noise = diffusion_model(z_t, torch.tensor([t], device=DEVICE).float())
    model_mean = sqrt_recip_alphas[t] * (z_t - betas[t] * pred_noise / sqrt_one_minus_alphas_cumprod[t])
    if t == 0: return model_mean
    else: return model_mean + torch.sqrt(betas[t]) * torch.randn_like(z_t)
@torch.no_grad()
def generate_latents(num_latents=50):
    z = torch.randn(num_latents, Z_DIM, device=DEVICE)
    for i in reversed(range(1000)):
        z = p_sample(z, i)
    return z

# --- CLIP目的関数 (変更なし) ---
clip_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26588654, 0.27577711))])
def clip_preference_function(z: torch.Tensor, text_features: torch.Tensor):
    with torch.no_grad():
        z_unscaled = z * scale_factor; images = (vae.decode(z_unscaled) + 1) / 2
        images_preprocessed = clip_transform(images)
        image_features = clip_model.encode_image(images_preprocessed).float()
        image_features /= image_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ text_features.T)
    return similarity.squeeze(1).detach().cpu()

# ★★★★★ ここからが新しいBOループ ★★★★★
# --- BOの探索設定 ---
TARGET_TEXT = "freezing cold blue emoji"
NUM_INITIAL_SAMPLES = 100  # 初期サンプル数を増やす
BO_ITERATIONS = 20        # BOの反復回数

# ターゲットテキストをエンコード
text_tokens = clip.tokenize([TARGET_TEXT]).to(DEVICE)
with torch.no_grad():
    TARGET_TEXT_FEATURES = clip_model.encode_text(text_tokens).float()
    TARGET_TEXT_FEATURES /= TARGET_TEXT_FEATURES.norm(dim=-1, keepdim=True)

# --- Step 1: 初期データを生成・評価 ---
print(f"\nTarget text: '{TARGET_TEXT}'")
print(f"Generating {NUM_INITIAL_SAMPLES} initial samples...")
train_x = generate_latents(num_latents=NUM_INITIAL_SAMPLES)
print(f"Evaluating initial samples with CLIP...")
train_y = clip_preference_function(train_x, TARGET_TEXT_FEATURES).unsqueeze(1)

# --- Step 2: BOループを開始 ---
for i in range(BO_ITERATIONS):
    print(f"\n--- BO Iteration {i+1}/{BO_ITERATIONS} ---")
    
    # GPモデルの学習
    gp = SingleTaskGP(train_x.cpu(), train_y.cpu())
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll)
    
    # 獲得関数を定義 (betaを大きくして探索を重視)
    ucb = UpperConfidenceBound(gp, beta=1.5)
    
    # 探索範囲を定義
    bounds = torch.stack([train_x.cpu().min(dim=0).values, train_x.cpu().max(dim=0).values])
    
    # 獲得関数を最大化して、次の候補点を提案
    new_z, _ = optimize_acqf(
        acq_function=ucb, bounds=bounds, q=1,
        num_restarts=20,  # 探索の試行回数を増やす
        raw_samples=512, # 探索の初期候補点を増やす
    )
    new_z = new_z.to(DEVICE)
    
    # 新しい候補点を評価
    new_y = clip_preference_function(new_z, TARGET_TEXT_FEATURES).unsqueeze(1)
    
    print(f"  Candidate score: {new_y.item():.4f}")
    
    # 新しいデータをデータセットに追加
    train_x = torch.cat([train_x, new_z])
    train_y = torch.cat([train_y, new_y.cpu()])

# --- Step 3: 最終結果の可視化 ---
print("\n--- BO Finished ---")
best_idx = train_y.argmax()
best_score = train_y.max()
best_z = train_x[best_idx]

print(f"Found best emoji with score: {best_score:.4f}")

with torch.no_grad():
    # 見つかった最高の絵文字をデコード
    z_unscaled = best_z.unsqueeze(0) * scale_factor
    best_image = (vae.decode(z_unscaled) + 1) / 2
    
    vutils.save_image(best_image, "bo_final_best_result.png")
    print(f"\nSaved the best result to 'bo_final_best_result.png'")

    # 上位5つの候補を並べて表示
    print("Saving top 5 candidates...")
    top5_indices = torch.topk(train_y.squeeze(), 5).indices
    top5_z = train_x[top5_indices]
    top5_unscaled = top5_z * scale_factor
    top5_images = (vae.decode(top5_unscaled) + 1) / 2
    vutils.save_image(top5_images, "bo_top5_results.png", nrow=5)
    print("Saved top 5 candidates to 'bo_top5_results.png'")
