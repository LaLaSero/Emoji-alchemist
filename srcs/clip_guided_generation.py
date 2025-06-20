# emoji_alchemist.py (最終版：CLIP誘導 + 複合目的BO探索)
# これまでの全てのステップを統合した、完成形のスクリプトです。

import torch
import clip
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import UpperConfidenceBound
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
import torchvision.utils as vutils
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from tqdm import tqdm
from conv_vae import ConvVAE
from latent_unet import LatentUNet

# ======================================================================================
# 1. モデルと設定のロード
# ======================================================================================
Z_DIM = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Loading models (VAE, Diffusion, CLIP)...")
# 各モデルをロードします。これらは学習済みで、重みは固定です。
vae = ConvVAE(z_dim=Z_DIM).to(DEVICE); vae.load_state_dict(torch.load("emoji_vae.pth")); vae.eval()
diffusion_model = LatentUNet(z_dim=Z_DIM).to(DEVICE); diffusion_model.load_state_dict(torch.load("latent_diffusion.pth")); diffusion_model.eval()
clip_model, _ = clip.load("ViT-B/32", device=DEVICE); clip_model.eval()
scale_factor = torch.load("vae_latent_scale.pt").to(DEVICE)

# Diffusionプロセス用の設定
betas = torch.linspace(0.0001, 0.02, 1000, device=DEVICE)
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)

# ======================================================================================
# 2. 探索ハイパーパラメータ
# ここを編集することで、AIの挙動をコントロールできます。
# ======================================================================================
# --- 目的プロンプト ---
TARGET_TEXT = "A freezing cold emoji, covered in ice"
POSITIVE_PROMPT = "a high-quality, well-drawn, clear emoji face"
NEGATIVE_PROMPT = "a distorted, blurry, noisy, malformed, ugly, broken image"

# --- ステージ1：CLIP誘導の設定 ---
NUM_INITIAL_CANDIDATES = 4  # BOの初期候補として、まず32枚生成
GUIDANCE_STRENGTH = 3.0     # 崩壊しない程度の、適度なガイダンス強度
NUM_CUTS = 16                # 画像を16の断片に分けて評価し、頑健性を高める

# --- ステージ2：BOの設定 ---
BO_ITERATIONS = 20           # BOの反復（試行錯誤）回数
QUALITY_WEIGHT = 0.0       # 品質スコアの重要度
PENALTY_WEIGHT = 0.3        # 欠点スコアの重要度 (ペナルティ)

# ======================================================================================
# 3. ヘルパー関数
# ======================================================================================
# --- テキストプロンプトのエンコード ---
prompts = [TARGET_TEXT, POSITIVE_PROMPT, NEGATIVE_PROMPT]
text_tokens = clip.tokenize(prompts).to(DEVICE)
with torch.no_grad():
    all_text_features = clip_model.encode_text(text_tokens).float()
    all_text_features /= all_text_features.norm(dim=-1, keepdim=True)
    # 各特徴量を個別の変数に格納
    TARGET_FEATURES, POSITIVE_FEATURES, NEGATIVE_FEATURES = all_text_features.chunk(3)

# --- CLIP用の画像前処理 ---
# ガイダンス用（ランダム性なし、歪みなし）
guidance_transform = transforms.Compose([
    transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.Normalize(
        (0.48145466, 0.4578275, 0.40821073),
        (0.26862954, 0.26588654, 0.27577711),
    ),
])
# BO評価用（ランダム性なし、固定評価のため）
evaluation_transform = transforms.Compose([
    transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26588654, 0.27577711)),
])

def cond_fn(x, t, guidance_scale):
    """ CLIP誘導の勾配を計算するコア関数 """
    with torch.enable_grad():
        x_is_grad = x.detach().requires_grad_(True)
        # 予測される最終画像z0を計算
        z0_pred = (x_is_grad - pred_noise * torch.sqrt(1. - alphas_cumprod[t])) / torch.sqrt(alphas_cumprod[t])
        z_unscaled = z0_pred * scale_factor
        decoded_images = (vae.decode(z_unscaled) + 1) / 2
        
        # 複数に切り抜いて評価
        augmented_images = torch.cat([guidance_transform(decoded_images) for _ in range(NUM_CUTS)], dim=0)
        image_features = clip_model.encode_image(augmented_images).float()
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        # 複合的なスコアを計算（誘導時も品質とペナルティを考慮）
        target_loss = (100.0 * image_features @ TARGET_FEATURES.T).mean()
        quality_loss = (100.0 * image_features @ POSITIVE_FEATURES.T).mean()
        penalty_loss = (100.0 * image_features @ NEGATIVE_FEATURES.T).mean()
        loss = target_loss + QUALITY_WEIGHT * quality_loss - PENALTY_WEIGHT * penalty_loss

        # 勾配を計算
        grad = torch.autograd.grad(loss, x_is_grad)[0]
    return grad * guidance_scale

@torch.no_grad()
def generate_with_guidance(num_images):
    """ ステージ1: CLIP誘導を使って高品質な初期候補を生成する """
    print(f"--- Stage 1: Generating {num_images} initial candidates with CLIP guidance ---")
    z = torch.randn(num_images, Z_DIM, device=DEVICE)
    
    for i in tqdm(reversed(range(1000)), desc="Guided Initial Sampling"):
        t = torch.tensor([i], device=DEVICE).repeat(num_images)
        global pred_noise # cond_fnから参照できるようグローバル変数として扱う
        pred_noise = diffusion_model(z, t.float())
        
        # 勾配を計算し、予測されたノイズを修正（正しいDDIMガイダンスの適用）
        grad = cond_fn(z, i, GUIDANCE_STRENGTH)
        pred_noise_guided = pred_noise - torch.sqrt(1. - alphas_cumprod[i]) * grad
        
        # 誘導されたノイズを使ってzを更新
        z0_pred = (z - pred_noise_guided * torch.sqrt(1. - alphas_cumprod[i])) / torch.sqrt(alphas_cumprod[i])
        if i > 0:
            mean_next = alphas_cumprod[i-1].sqrt() * z0_pred + (1-alphas_cumprod[i-1]).sqrt() * pred_noise_guided
            z = mean_next + torch.sqrt(1 - alphas_cumprod[i-1]) * torch.randn_like(z)
        else:
            z = z0_pred
    return z

def bo_multi_objective_function(z: torch.Tensor):
    """ ステージ2: BOで使う複合目的関数 """
    with torch.no_grad():
        z_unscaled = z * scale_factor
        images = (vae.decode(z_unscaled) + 1) / 2
        images_preprocessed = evaluation_transform(images)
        image_features = clip_model.encode_image(images_preprocessed).float()
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        target_sim = (100.0 * image_features @ TARGET_FEATURES.T).squeeze(-1)
        quality_sim = (100.0 * image_features @ POSITIVE_FEATURES.T).squeeze(-1)
        penalty_sim = (100.0 * image_features @ NEGATIVE_FEATURES.T).squeeze(-1)
        
        final_score = target_sim + (QUALITY_WEIGHT * quality_sim) - (PENALTY_WEIGHT * penalty_sim)
    return final_score.detach().cpu()

# ======================================================================================
# 4. メイン実行部
# ======================================================================================
# --- ステージ1の実行 ---
initial_latents = generate_with_guidance(NUM_INITIAL_CANDIDATES)
initial_scores = bo_multi_objective_function(initial_latents).unsqueeze(1)
vutils.save_image((vae.decode(initial_latents * scale_factor) + 1) / 2, "bo_initial_candidates.png", nrow=8)
print("Saved initial candidates to 'bo_initial_candidates.png'")

# --- ステージ2の実行 ---
print(f"\n--- Stage 2: Starting BO loop with Multi-Objective score for {BO_ITERATIONS} iterations ---")
train_x = initial_latents
train_y = initial_scores

for i in range(BO_ITERATIONS):
    print(f"  BO Iteration {i+1}/{BO_ITERATIONS}...")
    gp = SingleTaskGP(train_x.cpu(), train_y.cpu())
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll)
    
    ucb = UpperConfidenceBound(gp, beta=1.5) # 未知の領域をより探索的に
    bounds = torch.stack([train_x.cpu().min(dim=0).values, train_x.cpu().max(dim=0).values])
    
    new_z, _ = optimize_acqf(acq_function=ucb, bounds=bounds, q=1, num_restarts=20, raw_samples=512)
    new_z = new_z.to(DEVICE)
    new_y = bo_multi_objective_function(new_z).unsqueeze(1)
    print(f"  Candidate found. Score: {new_y.item():.4f}")
    
    train_x = torch.cat([train_x, new_z])
    train_y = torch.cat([train_y, new_y.cpu()])

# --- 最終結果の可視化 ---
print("\n--- BO Finished ---")
best_idx = train_y.argmax()
best_score = train_y.max()
best_z = train_x[best_idx]

print(f"Found best emoji with score: {best_score:.4f}")
with torch.no_grad():
    best_image = (vae.decode(best_z.unsqueeze(0) * scale_factor) + 1) / 2
    vutils.save_image(best_image, "ultimate_best_result.png")
    print(f"\nSaved the best result to 'ultimate_best_result.png'")
    
    top5_indices = torch.topk(train_y.squeeze(), 5).indices
    top5_z = train_x[top5_indices]
    top5_images = (vae.decode(top5_z * scale_factor) + 1) / 2
    vutils.save_image(top5_images, "ultimate_top5_results.png", nrow=5)
    print("Saved top 5 candidates to 'ultimate_top5_results.png'")
