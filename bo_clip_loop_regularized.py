# bo_clip_loop_final_tune.py
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
import cv2
import numpy as np

# --- ★★★ ここで芸術的な指示を調整します ★★★ ---
# 1. ポジティブプロンプト：より具体的に指示
TARGET_TEXT = "A blue emoji made of ice, freezing cold, frosted, crystalline"

# 2. ネガティブプロンプト：不要な要素を明確に排除
NEGATIVE_TARGET_TEXT = "yellow, orange, brown, warm, suntan, human skin, happy"

# 3. パラメータ調整
NUM_INITIAL_SAMPLES = 400 # 初期候補は多めに生成
BO_ITERATIONS = 50        # BOの反復回数
UCB_BETA = 1.5            # 探索の積極性
# 潜在空間正則化はゼロに設定し、色の制約をなくす
LATENT_REG_STRENGTH = 0.0
# 顔検出ペナルティは構造維持のため、強めに設定
FACE_PENALTY_STRENGTH = 7.0
# ネガティブプロンプトの影響力
NEGATIVE_WEIGHT = 0.8

# --- その他設定 ---
OUTPUT_DIR = Path("optimization_final_results")
Z_DIM = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- 初期化処理 ---
OUTPUT_DIR.mkdir(exist_ok=True)
print(f"Results will be saved to '{OUTPUT_DIR}'")
print("Loading models (VAE, Diffusion, CLIP)...")
from conv_vae import ConvVAE
from latent_unet import LatentUNet
vae = ConvVAE(z_dim=Z_DIM).to(DEVICE); vae.load_state_dict(torch.load("emoji_vae_clip_fast.pth")); vae.eval()
diffusion_model = LatentUNet(z_dim=Z_DIM).to(DEVICE); diffusion_model.load_state_dict(torch.load("latent_diffusion.pth")); diffusion_model.eval()
clip_model, _ = clip.load("ViT-B/32", device=DEVICE); clip_model.eval()
scale_factor = torch.load("vae_latent_scale.pt").to(DEVICE)
print("Loading face detector (OpenCV Haar Cascade)...")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

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
def generate_latents(num_latents):
    z = torch.randn(num_latents, Z_DIM, device=DEVICE)
    for i in tqdm(reversed(range(1000)), desc="Generating initial latents", leave=False):
        z = p_sample(z, i)
    return z

# --- 評価関数 ---
clip_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26588654, 0.27577711))])

def get_face_detection_penalty(images_tensor: torch.Tensor, penalty_strength: float) -> torch.Tensor:
    penalties = []
    for img_tensor in images_tensor:
        img_np = (img_tensor.permute(1, 2, 0).mul(255).clamp_(0, 255).to(torch.uint8).cpu().numpy())
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, flags=cv2.CASCADE_SCALE_IMAGE)
        penalties.append(penalty_strength if len(faces) == 0 else 0.0)
    return torch.tensor(penalties, device=images_tensor.device, dtype=torch.float32)

def objective_function(z: torch.Tensor, pos_feat: torch.Tensor, neg_feat: torch.Tensor, dist_center: torch.Tensor, p: dict):
    with torch.no_grad():
        z_unscaled = z * scale_factor
        images = (vae.decode(z_unscaled) + 1) / 2
        images_clip_preprocessed = clip_transform(images)
        image_features = clip_model.encode_image(images_clip_preprocessed).float()
        image_features /= image_features.norm(dim=-1, keepdim=True)
        pos_similarity = (100.0 * image_features @ pos_feat.T).squeeze(1)
        neg_similarity = (100.0 * image_features @ neg_feat.T).squeeze(1)
        latent_reg_penalty = p['lambda_reg'] * torch.sum((z - dist_center) ** 2, dim=-1)
        face_det_penalty = get_face_detection_penalty(images, p['lambda_face'])
        final_score = (pos_similarity - p['lambda_neg'] * neg_similarity) - latent_reg_penalty.to(DEVICE) - face_det_penalty.to(DEVICE)
    return final_score.detach().cpu()

# --- メイン処理 ---
def main():
    params = {
        'lambda_reg': LATENT_REG_STRENGTH,
        'lambda_face': FACE_PENALTY_STRENGTH,
        'lambda_neg': NEGATIVE_WEIGHT
    }

    # Step 0: プロンプトのエンコード & 探索範囲の決定
    print(f"\nPositive Prompt: '{TARGET_TEXT}'")
    print(f"Negative Prompt: '{NEGATIVE_TARGET_TEXT}'")
    with torch.no_grad():
        pos_tokens = clip.tokenize([TARGET_TEXT]).to(DEVICE)
        pos_feat = clip_model.encode_text(pos_tokens).float()
        pos_feat /= pos_feat.norm(dim=-1, keepdim=True)
        neg_tokens = clip.tokenize([NEGATIVE_TARGET_TEXT]).to(DEVICE)
        neg_feat = clip_model.encode_text(neg_tokens).float()
        neg_feat /= neg_feat.norm(dim=-1, keepdim=True)

    print("Determining search bounds...")
    all_images = torch.load("openmoji_72x72.pt")
    loader = DataLoader(TensorDataset(all_images), batch_size=512)
    all_latents = []
    with torch.no_grad():
        for (x_batch,) in loader:
            mu, _ = vae.encode(x_batch.to(DEVICE))
            all_latents.append(mu / scale_factor)
    all_latents = torch.cat(all_latents, dim=0)
    bounds = torch.stack([all_latents.cpu().min(dim=0).values, all_latents.cpu().max(dim=0).values])
    print("Search bounds determined.")

    # Step 1: 初期データ生成 & 評価
    print(f"Generating {NUM_INITIAL_SAMPLES} initial samples...")
    train_x = generate_latents(num_latents=NUM_INITIAL_SAMPLES)
    distribution_center = train_x.mean(dim=0)
    print("Evaluating initial samples...")
    train_y = objective_function(train_x, pos_feat, neg_feat, distribution_center, params).unsqueeze(1)
    
    # Step 2: BOループ
    pbar = tqdm(range(BO_ITERATIONS), desc="Bayesian Optimization")
    for i in pbar:
        gp = SingleTaskGP(train_x.cpu(), train_y.cpu())
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)
        ucb = UpperConfidenceBound(gp, beta=UCB_BETA)
        new_z, _ = optimize_acqf(acq_function=ucb, bounds=bounds, q=1, num_restarts=20, raw_samples=512)
        new_z = new_z.to(DEVICE)
        new_y = objective_function(new_z, pos_feat, neg_feat, distribution_center, params).unsqueeze(1)
        print(f"Iteration {i+1}/{BO_ITERATIONS}, New sample score: {new_y.max().item():.4f}")
        train_x = torch.cat([train_x, new_z])
        train_y = torch.cat([train_y, new_y.cpu()])
        pbar.set_postfix({"best_score": f"{train_y.max().item():.4f}"})
        with torch.no_grad():
            best_idx = train_y.argmax()
            best_img = (vae.decode(train_x[best_idx].unsqueeze(0) * scale_factor) + 1) / 2
            vutils.save_image(best_img, OUTPUT_DIR / f"step_{i+1:03d}_best_img.png")

    # Step 3: 最終結果の可視化
    print("\n--- BO Finished ---")
    best_idx = train_y.argmax()
    print(f"Found best emoji with score: {train_y.max().item():.4f}")
    print("Saving top 5 candidates...")
    top5_indices = torch.topk(train_y.squeeze(), 5).indices
    with torch.no_grad():
        top5_images = (vae.decode(train_x[top5_indices] * scale_factor) + 1) / 2
    vutils.save_image(top5_images, OUTPUT_DIR / "final_top5_results.png", nrow=5)
    print(f"Saved top 5 candidates to '{OUTPUT_DIR / 'final_top5_results.png'}'")

if __name__ == "__main__":
    main()
