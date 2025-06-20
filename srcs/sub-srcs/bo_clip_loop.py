# bo_clip_loop_improved.py
import torch
import clip
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import UpperConfidenceBound
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
import torchvision.utils as vutils
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from pathlib import Path

# --- 設定項目 ---
# テキストプロンプト、BOの反復回数、初期サンプル数をここで設定
TARGET_TEXT = "A freezing cold emoji, covered in ice"
NUM_INITIAL_SAMPLES = 100  # 初期サンプルは少なくし、BOで探索させる
BO_ITERATIONS = 10        # BOの反復回数を増やして、より深く探索させる
UCB_BETA = 1.5            # 探索の積極性 (大きいほど未知の領域を探索)
OUTPUT_DIR = Path("optimization_progress") # 結果を保存するディレクトリ

# --- デバイス設定 ---
Z_DIM = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- 結果保存ディレクトリを作成 ---
OUTPUT_DIR.mkdir(exist_ok=True)
print(f"Results will be saved to '{OUTPUT_DIR}'")

# --- モデルと設定のロード ---
print("Loading models (VAE, Diffusion, CLIP)...")
from conv_vae import ConvVAE
from latent_unet import LatentUNet

# VAE
vae = ConvVAE(z_dim=Z_DIM).to(DEVICE)
# ★★★ 修正点1: 正しいVAEの重みファイルをロード ★★★
vae.load_state_dict(torch.load("emoji_vae_clip_fast.pth"))
vae.eval()

# Latent Diffusion Model
diffusion_model = LatentUNet(z_dim=Z_DIM).to(DEVICE)
diffusion_model.load_state_dict(torch.load("latent_diffusion.pth"))
diffusion_model.eval()

# CLIP
clip_model, _ = clip.load("ViT-B/32", device=DEVICE)
clip_model.eval()

# スケールファクター
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
    if t == 0:
        return model_mean
    else:
        noise = torch.randn_like(z_t)
        return model_mean + torch.sqrt(betas[t]) * noise

@torch.no_grad()
def generate_latents(num_latents):
    z = torch.randn(num_latents, Z_DIM, device=DEVICE)
    # TQDMで進捗を表示
    for i in tqdm(reversed(range(1000)), desc="Generating initial latents", leave=False):
        z = p_sample(z, i)
    return z

# --- CLIP目的関数 (変更なし) ---
clip_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26588654, 0.27577711))
])

def clip_preference_function(z: torch.Tensor, text_features: torch.Tensor):
    with torch.no_grad():
        z_unscaled = z * scale_factor
        images = (vae.decode(z_unscaled) + 1) / 2
        images_preprocessed = clip_transform(images)
        image_features = clip_model.encode_image(images_preprocessed).float()
        image_features /= image_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ text_features.T)
    return similarity.squeeze(1).detach().cpu()

# --- メイン処理 ---
def main():
    # --- Step 0: ターゲットテキストをエンコード & 探索範囲を決定 ---
    print(f"\nTarget text: '{TARGET_TEXT}'")
    text_tokens = clip.tokenize([TARGET_TEXT]).to(DEVICE)
    with torch.no_grad():
        target_text_features = clip_model.encode_text(text_tokens).float()
        target_text_features /= target_text_features.norm(dim=-1, keepdim=True)

    # ★★★ 修正点2: 探索範囲を全学習データから一度だけ決定 ★★★
    print("Determining search bounds from the entire dataset...")
    all_images = torch.load("openmoji_72x72.pt")
    loader = DataLoader(TensorDataset(all_images), batch_size=512)
    all_latents = []
    with torch.no_grad():
        for (x_batch,) in loader:
            mu, _ = vae.encode(x_batch.to(DEVICE))
            all_latents.append(mu / scale_factor)
    all_latents = torch.cat(all_latents, dim=0)
    bounds = torch.stack([
        all_latents.cpu().min(dim=0).values,
        all_latents.cpu().max(dim=0).values
    ])
    print("Search bounds determined.")

    # --- Step 1: 初期データを生成・評価 ---
    print(f"Generating {NUM_INITIAL_SAMPLES} initial samples...")
    train_x = generate_latents(num_latents=NUM_INITIAL_SAMPLES)
    train_y = clip_preference_function(train_x, target_text_features).unsqueeze(1)

    # --- Step 2: BOループを開始 ---
    pbar = tqdm(range(BO_ITERATIONS), desc="Bayesian Optimization")
    for i in pbar:
        # GPモデルの学習
        gp = SingleTaskGP(train_x.cpu(), train_y.cpu())
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)

        # 獲得関数を定義
        ucb = UpperConfidenceBound(gp, beta=UCB_BETA)

        # 獲得関数を最大化して、次の候補点を提案
        new_z, _ = optimize_acqf(
            acq_function=ucb, bounds=bounds, q=1,
            num_restarts=20, raw_samples=512,
        )
        new_z = new_z.to(DEVICE)

        # 新しい候補点を評価
        new_y = clip_preference_function(new_z, target_text_features).unsqueeze(1)

        # 新しいデータをデータセットに追加
        train_x = torch.cat([train_x, new_z])
        train_y = torch.cat([train_y, new_y.cpu()])

        # 現在の最高スコアと候補を表示
        best_score_so_far = train_y.max()
        pbar.set_postfix({"best_score": f"{best_score_so_far.item():.4f}"})

        # ★★★ 改良点: 各ステップでの最良の画像を保存 ★★★
        with torch.no_grad():
            best_idx_so_far = train_y.argmax()
            best_z_so_far = train_x[best_idx_so_far].unsqueeze(0)
            best_img_so_far = (vae.decode(best_z_so_far * scale_factor) + 1) / 2
            vutils.save_image(best_img_so_far, OUTPUT_DIR / f"step_{i+1:03d}_best_img.png")

    # --- Step 3: 最終結果の可視化 ---
    print("\n--- BO Finished ---")
    best_idx = train_y.argmax()
    best_score = train_y.max()
    print(f"Found best emoji with score: {best_score:.4f}")

    # 上位5つの候補を並べて最終結果として保存
    print("Saving top 5 candidates...")
    top5_indices = torch.topk(train_y.squeeze(), 5).indices
    top5_z = train_x[top5_indices]
    with torch.no_grad():
        top5_images = (vae.decode(top5_z * scale_factor) + 1) / 2
    vutils.save_image(top5_images, OUTPUT_DIR / "final_top5_results.png", nrow=5)
    print(f"Saved top 5 candidates to '{OUTPUT_DIR / 'final_top5_results.png'}'")

if __name__ == "__main__":
    main()
