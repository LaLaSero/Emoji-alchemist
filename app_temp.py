# app.py
import torch
import clip
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import qUpperConfidenceBound
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import cv2
import numpy as np
from flask import Flask, request, jsonify, render_template_string
from PIL import Image
import base64
import io

app = Flask(__name__)

# --- グローバル変数とモデルのロード ---
print("Loading models and configuration...")
Z_DIM = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# モデル
from conv_vae import ConvVAE
from latent_unet import LatentUNet
vae = ConvVAE(z_dim=Z_DIM).to(DEVICE); vae.load_state_dict(torch.load("emoji_vae_clip_fast.pth")); vae.eval()
diffusion_model = LatentUNet(z_dim=Z_DIM).to(DEVICE); diffusion_model.load_state_dict(torch.load("latent_diffusion.pth")); diffusion_model.eval()
clip_model, _ = clip.load("ViT-B/32", device=DEVICE); clip_model.eval()
scale_factor = torch.load("vae_latent_scale.pt").to(DEVICE)
latent_mean = torch.load("vae_latent_mean.pt").to(DEVICE)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
print("All models loaded.")
# Diffusion関連
# -- Cosine β schedule (s = 0.1) to match training --
def cosine_beta_schedule(timesteps: int, s: float = 0.1):
    steps = torch.arange(timesteps + 1, dtype=torch.float32, device=DEVICE)
    alphas_cumprod = torch.cos(((steps / timesteps) + s) / (1 + s) * torch.pi / 2) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas_ = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return betas_

betas = cosine_beta_schedule(1000, s=0.1)
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
# 探索範囲
print("Determining search bounds...")
all_images_data = torch.load("openmoji_72x72.pt")
loader = DataLoader(TensorDataset(all_images_data), batch_size=512)
all_latents = []
with torch.no_grad():
    for (x_batch,) in loader:
        mu, _ = vae.encode(x_batch.to(DEVICE))
        all_latents.append((mu - latent_mean) / scale_factor)
all_latents = torch.cat(all_latents, dim=0)
SEARCH_BOUNDS = torch.stack([all_latents.cpu().min(dim=0).values, all_latents.cpu().max(dim=0).values])
SEARCH_BOUNDS = SEARCH_BOUNDS.to(DEVICE)
print("Search bounds determined.")

# セッションデータ
session_data = {}

# --- 評価関数とヘルパー関数 ---
clip_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26588654, 0.27577711))])

def get_face_detection_penalty(images_tensor: torch.Tensor, penalty_strength: float) -> torch.Tensor:
    penalties = []
    for img_tensor in images_tensor:
        img_np = (img_tensor.permute(1, 2, 0).mul(255).clamp_(0, 255).to(torch.uint8).cpu().numpy())
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, flags=cv2.CASCADE_SCALE_IMAGE)
        penalties.append(penalty_strength if len(faces) == 0 else 0.0)
    return torch.tensor(penalties, device=images_tensor.device, dtype=torch.float32)

def objective_function(z: torch.Tensor, p_feat: torch.Tensor, n_feat: torch.Tensor, p: dict):
    with torch.no_grad():
        images = (vae.decode(z * scale_factor + latent_mean) + 1) / 2
        images = torch.nan_to_num(images).clamp_(0.0, 1.0)   # 🔧 NaN除去 & 範囲固定
        images_clip = clip_transform(images)
        image_features = clip_model.encode_image(images_clip).float()
        image_features /= image_features.norm(dim=-1, keepdim=True)
        pos_sim = (100.0 * image_features @ p_feat.T).squeeze(1)
        neg_sim = (100.0 * image_features @ n_feat.T).squeeze(1) if n_feat is not None else 0
        face_penalty = get_face_detection_penalty(images, p['lambda_face'])
        score = (pos_sim - p['lambda_neg'] * neg_sim) - 3.0* face_penalty
    # ★★★ 修正点: .cpu()を削除し、GPU上にデータを保持する ★★★
    return score

@torch.no_grad()
def p_sample(z_t, t):
    pred_noise = diffusion_model(z_t, torch.tensor([t], device=DEVICE).float())
    model_mean = sqrt_recip_alphas[t] * (z_t - betas[t] * pred_noise / sqrt_one_minus_alphas_cumprod[t])
    if t == 0: return model_mean
    else: return model_mean + torch.sqrt(betas[t]) * torch.randn_like(z_t)

@torch.no_grad()
def generate_latents(num_latents):
    """Sample posterior latents and clamp to a reasonable range."""
    idx = torch.randint(0, all_images_data.shape[0], (num_latents,))
    imgs = all_images_data[idx].to(DEVICE)
    mu, logvar = vae.encode(imgs)
    std = (0.5 * logvar).exp()
    eps = torch.randn_like(std)
    z_post = mu + eps * std                       # (num_latents, Z_DIM)
    z_white = (z_post - latent_mean) / scale_factor
    # Clamp to [-3, 3] in whitened space to avoid extreme latents that decode to black
    z_white = z_white.clamp(-3.0, 3.0)
    return z_white

def tensor_to_base64(z_tensor):
    with torch.no_grad():
        img_tensor = (vae.decode(z_tensor.unsqueeze(0) * scale_factor + latent_mean) + 1) / 2
        img_tensor = torch.nan_to_num(img_tensor).clamp_(0.0, 1.0)
    img = transforms.ToPILImage()(img_tensor.squeeze(0).cpu())
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

# --- APIエンドポイント ---
@app.route('/')
def index():
    with open('interactive_ui.html', 'r', encoding='utf-8') as f:
        return render_template_string(f.read())

@app.route('/start_session', methods=['POST'])
def start_session_route():
    global session_data
    data = request.json
    print(f"Starting new session. P: '{data['prompt']}', N: '{data['neg_prompt']}'")
    
    with torch.no_grad():
        pos_tokens = clip.tokenize([data['prompt']]).to(DEVICE)
        pos_feat = clip_model.encode_text(pos_tokens).float()
        pos_feat /= pos_feat.norm(dim=-1, keepdim=True)

        neg_feat = None
        if data['neg_prompt']:
            neg_tokens = clip.tokenize([data['neg_prompt']]).to(DEVICE)
            neg_feat = clip_model.encode_text(neg_tokens).float()
            neg_feat /= neg_feat.norm(dim=-1, keepdim=True)
            
    initial_latents = generate_latents(5)
    params = {'lambda_face': 7.0, 'lambda_neg': 0.8}
    initial_scores = objective_function(initial_latents, pos_feat, neg_feat, params)

    session_data = {
        'pos_feat': pos_feat,
        'neg_feat': neg_feat,
        'params': params,
        'train_x': initial_latents,
        'train_y': initial_scores.unsqueeze(1),
        'iteration': 0
    }
    
    images_b64 = [tensor_to_base64(z) for z in initial_latents]
    return jsonify({'images': images_b64, 'iteration': 0})

@app.route('/select_candidate', methods=['POST'])
def select_candidate_route():
    global session_data
    data = request.json
    selected_index = data['selected_index']
    
    last_scores = session_data['train_y'][-5:]
    bonus = torch.zeros_like(last_scores)
    bonus[selected_index] = 5.0
    adjusted_scores = last_scores + bonus
    session_data['train_y'][-5:] = adjusted_scores
    
    session_data['iteration'] += 1
    print(f"Iter {session_data['iteration']}: User selected {selected_index}. Scores adjusted.")

    print("Fitting GP and optimizing acquisition function...")
    gp = SingleTaskGP(session_data['train_x'], session_data['train_y'])
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll)
    
    ucb = qUpperConfidenceBound(gp, beta=1.5)
    new_candidates_z, _ = optimize_acqf(
        ucb, SEARCH_BOUNDS, q=5, num_restarts=10, raw_samples=256
    )
    
    new_scores = objective_function(
        new_candidates_z, session_data['pos_feat'], session_data['neg_feat'], session_data['params']
    ).unsqueeze(1)
    
    session_data['train_x'] = torch.cat([session_data['train_x'], new_candidates_z], dim=0)
    session_data['train_y'] = torch.cat([session_data['train_y'], new_scores], dim=0)
    
    images_b64 = [tensor_to_base64(z) for z in new_candidates_z]
    
    best_idx = session_data['train_y'].argmax()
    best_image_b64 = tensor_to_base64(session_data['train_x'][best_idx])
    
    return jsonify({
        'images': images_b64,
        'iteration': session_data['iteration'],
        'best_image': best_image_b64
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
