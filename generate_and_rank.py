# generate_and_rank.py
import torch
import clip
import torchvision.utils as vutils
from torchvision import transforms
from tqdm import tqdm
from pathlib import Path
import cv2
import numpy as np

# --- ★★★ ここで芸術的な指示を調整します ★★★ ---
# 1. ポジティブプロンプト：より具体的に指示
TARGET_TEXT = "A blue emoji made of ice, freezing cold, frosted, crystalline"

# 2. ネガティブプロンプト：不要な要素を明確に排除
NEGATIVE_TARGET_TEXT = "yellow, orange, brown, warm, suntan, human skin, happy"

# 3. 生成と選出の数
NUM_CANDIDATES_TO_GENERATE = 1000  # 生成する候補の総数 (多いほど多様な結果が出る)
NUM_TOP_CANDIDATES_TO_SHOW = 10    # 最終的に表示する上位候補の数

# 4. 評価パラメータ
# 顔検出ペナルティは構造維持のため、強めに設定
FACE_PENALTY_STRENGTH = 10.0
# ネガティブプロンプトの影響力
NEGATIVE_WEIGHT = 0.8

# --- その他設定 ---
OUTPUT_FILE = Path("ranked_results.png")
Z_DIM = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 100 # 一度に評価するバッチサイズ (GPUメモリに応じて調整)

# --- モデルと設定のロード ---
print("--- Emoji Alchemist: Generate & Rank Mode ---")
print("Loading models (VAE, Diffusion, CLIP)...")
from conv_vae import ConvVAE
from latent_unet import LatentUNet
vae = ConvVAE(z_dim=Z_DIM).to(DEVICE); vae.load_state_dict(torch.load("emoji_vae_clip_fast.pth")); vae.eval()
diffusion_model = LatentUNet(z_dim=Z_DIM).to(DEVICE); diffusion_model.load_state_dict(torch.load("latent_diffusion.pth")); diffusion_model.eval()
clip_model, _ = clip.load("ViT-B/32", device=DEVICE); clip_model.eval()
scale_factor = torch.load("vae_latent_scale.pt").to(DEVICE)
latent_mean = torch.load("vae_latent_mean.pt").to(DEVICE)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
print("All models loaded.")

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
def generate_latents_batch(num_latents):
    latents = []
    for _ in tqdm(range(int(np.ceil(num_latents / BATCH_SIZE))), desc="Phase 1: Generating Candidates"):
        z = torch.randn(BATCH_SIZE, Z_DIM, device=DEVICE)
        for i in reversed(range(1000)):
            z = p_sample(z, i)
        latents.append(z)
    return torch.cat(latents)[:num_latents]

# --- 評価関数 (変更なし) ---
clip_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26588654, 0.27577711))])

def get_face_detection_penalty(images_tensor: torch.Tensor, penalty_strength: float) -> torch.Tensor:
    penalties = []
    for img_tensor in images_tensor:
        img_np = (img_tensor.permute(1, 2, 0).mul(255).clamp_(0, 255).to(torch.uint8).cpu().numpy())
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, flags=cv2.CASCADE_SCALE_IMAGE)
        penalties.append(penalty_strength if len(faces) == 0 else 0.0)
    return torch.tensor(penalties, device=images_tensor.device, dtype=torch.float32)

def evaluate_latents_batch(z: torch.Tensor, pos_feat: torch.Tensor, neg_feat: torch.Tensor, p: dict):
    with torch.no_grad():
        z_denormalized = z * scale_factor + latent_mean
        images = (vae.decode(z_denormalized) + 1) / 2
        images_clip = clip_transform(images)
        image_features = clip_model.encode_image(images_clip).float()
        image_features /= image_features.norm(dim=-1, keepdim=True)
        pos_similarity = (100.0 * image_features @ pos_feat.T).squeeze()
        neg_similarity = (100.0 * image_features @ neg_feat.T).squeeze() if neg_feat is not None else 0
        face_penalty = get_face_detection_penalty(images, p['lambda_face'])
        score = (pos_similarity - p['lambda_neg'] * neg_similarity) - face_penalty
    return score

# --- メイン処理 ---
def main():
    # 0. プロンプトのエンコード
    print(f"\nPositive Prompt: '{TARGET_TEXT}'")
    print(f"Negative Prompt: '{NEGATIVE_TARGET_TEXT}'")
    with torch.no_grad():
        pos_tokens = clip.tokenize([TARGET_TEXT]).to(DEVICE)
        pos_feat = clip_model.encode_text(pos_tokens).float()
        pos_feat /= pos_feat.norm(dim=-1, keepdim=True)
        neg_feat = None
        if NEGATIVE_TARGET_TEXT:
            neg_tokens = clip.tokenize([NEGATIVE_TARGET_TEXT]).to(DEVICE)
            neg_feat = clip_model.encode_text(neg_tokens).float()
            neg_feat /= neg_feat.norm(dim=-1, keepdim=True)

    # 1. 候補の大量生産
    candidate_latents = generate_latents_batch(NUM_CANDIDATES_TO_GENERATE)

    # 2. 全候補を評価
    all_scores = []
    params = {'lambda_face': FACE_PENALTY_STRENGTH, 'lambda_neg': NEGATIVE_WEIGHT}
    
    # メモリを節約するため、バッチ処理で評価
    for i in tqdm(range(0, len(candidate_latents), BATCH_SIZE), desc="Phase 2: Evaluating All Candidates"):
        batch_z = candidate_latents[i:i+BATCH_SIZE]
        scores = evaluate_latents_batch(batch_z, pos_feat, neg_feat, params)
        all_scores.append(scores)
    all_scores = torch.cat(all_scores)
    
    # 3. スコア順に並べ替え、上位を選出
    print("\nPhase 3: Ranking and Selecting Top Candidates...")
    top_scores, top_indices = torch.topk(all_scores, NUM_TOP_CANDIDATES_TO_SHOW)
    top_latents = candidate_latents[top_indices]

    # 4. 最終結果を画像として保存
    with torch.no_grad():
        z_denormalized = top_latents * scale_factor + latent_mean
        top_images = (vae.decode(z_denormalized) + 1) / 2
    
    vutils.save_image(top_images, OUTPUT_FILE, nrow=5, padding=4)
    
    print("\n--- 創造の時代、終了 ---")
    print(f"Top {NUM_TOP_CANDIDATES_TO_SHOW} candidates saved to '{OUTPUT_FILE}'")
    print("Top scores:", [f"{s.item():.2f}" for s in top_scores])
    print("プロンプトやパラメータを調整して、再度実行してみてください。")


if __name__ == "__main__":
    main()
