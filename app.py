# app.py
import torch
import clip
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import qUpperConfidenceBound
from botorch.optim import optimize_acqf
from typing import Tuple
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
Z_DIM = 32   # must match the trained ConvVAE checkpoint
# Automatically run a few BO iterations before asking the user for feedback
PRE_WARMUP_STEPS = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# モデル
from conv_vae import ConvVAE
# ----- load ConvVAE (handles DataParallel checkpoints) -----
state_dict = torch.load("emoji_vae_capacity.pth", map_location=DEVICE)
if any(k.startswith("module.") for k in state_dict.keys()):
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

vae = ConvVAE(z_dim=Z_DIM, clip_emb_dim=512).to(DEVICE)
missing, unexpected = vae.load_state_dict(state_dict, strict=False)
if missing:
    print(f"[VAE] Missing keys: {missing}")
if unexpected:
    print(f"[VAE] Unexpected keys: {unexpected}")
vae.eval()
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

# 外れ値を除外した探索範囲（5〜95パーセンタイル）
low = torch.quantile(all_latents, 0.05, dim=0)
high = torch.quantile(all_latents, 0.95, dim=0)
SEARCH_BOUNDS = torch.stack([low, high]).to(DEVICE)

print("Search bounds determined.")

# --- Mahalanobis distance pre‑compute ---
cov_eps = 1e-4
latent_cov = torch.cov(all_latents.T) + cov_eps * torch.eye(Z_DIM, device=DEVICE)
C_INV = torch.inverse(latent_cov)
print("Mahalanobis inverse covariance prepared.")

# セッションデータ
session_data = {}

# --- Trust‑region helpers ----------------------------------------------------
MAX_TR_RADIUS = 4.0   # allow exploration a bit farther
INIT_TR_RADIUS = 2.0  # begin with a wider neighbourhood

def tr_bounds(center: torch.Tensor, radius: float) -> torch.Tensor:
    """
    Return 2×Z_DIM tensor of [low, high] that satisfies low ≤ high element‑wise.
    If clipping causes an inversion (rare but possible with tight global bounds),
    swap the offending dimensions on the fly.
    """
    low = torch.max(center - radius, SEARCH_BOUNDS[0])
    high = torch.min(center + radius, SEARCH_BOUNDS[1])

    # --- safety: ensure low <= high for every dimension ---
    mask = low > high
    if mask.any():
        swapped_low = torch.where(mask, high, low)
        swapped_high = torch.where(mask, low, high)
        low, high = swapped_low, swapped_high

    return torch.stack([low, high]).to(DEVICE)

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
        distance_penalty = ( (z @ C_INV) * z ).sum(dim=-1)   # squared Mahalanobis
        score = (pos_sim - p['lambda_neg'] * neg_sim) \
                - 10.0 * face_penalty \
                - p['lambda_dist'] * distance_penalty
    # ★★★ 修正点: .cpu()を削除し、GPU上にデータを保持する ★★★
    return score

@torch.no_grad()
def generate_latents(num_latents):
    """Sample posterior latents from the VAE."""
    idx = torch.randint(0, all_images_data.shape[0], (num_latents,))
    imgs = all_images_data[idx].to(DEVICE)
    mu, logvar = vae.encode(imgs)
    std = (0.5 * logvar).exp()
    eps = torch.randn_like(std)
    z_post = mu + eps * std
    z_white = (z_post - latent_mean) / scale_factor
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
    params = {'lambda_face': 7.0, 'lambda_neg': 0.2, 'lambda_dist': 0.01}
    initial_scores = objective_function(initial_latents, pos_feat, neg_feat, params)

    best_idx0 = initial_scores.argmax()
    session_data = {
        'pos_feat': pos_feat,
        'neg_feat': neg_feat,
        'params': params,
        'train_x': initial_latents,
        'train_y': initial_scores.unsqueeze(1),
        'iteration': 0,
        'best_score': initial_scores[best_idx0].item(),
        'tr_center': initial_latents[best_idx0],   # starting center
        'tr_radius': INIT_TR_RADIUS
    }

    # --- automatic warm‑up BO steps (no user feedback) ---
    for _ in range(PRE_WARMUP_STEPS):
        session_data['iteration'] += 1

        # Fit GP
        gp = SingleTaskGP(session_data['train_x'], session_data['train_y'])
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)

        # ---- Trust‑region bounds around current center ----
        center = session_data['tr_center']
        radius = session_data['tr_radius']
        local_bounds = tr_bounds(center, radius)

        ucb = qUpperConfidenceBound(gp, beta=2.5)
        new_z, _ = optimize_acqf(
            ucb, local_bounds, q=5, num_restarts=10, raw_samples=512
        )

        # Evaluate objective and update training data
        new_scores = objective_function(
            new_z, session_data['pos_feat'], session_data['neg_feat'], session_data['params']
        ).unsqueeze(1)

        session_data['train_x'] = torch.cat([session_data['train_x'], new_z], dim=0)
        session_data['train_y'] = torch.cat([session_data['train_y'], new_scores], dim=0)

        # ---- Update TR state ----
        best_idx = session_data['train_y'].argmax()
        best_score = session_data['train_y'][best_idx].item()
        if best_score > session_data['best_score']:
            # improvement ⇒ move center & shrink radius
            session_data['best_score'] = best_score
            session_data['tr_center'] = session_data['train_x'][best_idx]
            session_data['tr_radius'] = max(1.0, session_data['tr_radius'] * 0.9)
        else:
            # no improvement ⇒ expand radius (capped)
            session_data['tr_radius'] = min(MAX_TR_RADIUS, session_data['tr_radius'] * 1.5)

    latest_latents = session_data['train_x'][-5:]
    images_b64 = [tensor_to_base64(z) for z in latest_latents]
    return jsonify({'images': images_b64, 'iteration': session_data['iteration']})

@app.route('/select_candidate', methods=['POST'])
def select_candidate_route():
    global session_data
    data = request.json
    selected_index = data['selected_index']
    multiplier = data.get('multiplier', 1)
    
    last_scores = session_data['train_y'][-5:]
    bonus = torch.zeros_like(last_scores)
    bonus_value = 12.0 * multiplier
    bonus[selected_index] = bonus_value
    print(f"Applying bonus {bonus_value} (multiplier={multiplier}) to index {selected_index}")
    
    session_data['train_y'][-5:] = adjusted_scores = last_scores + bonus
    
    # 追加: 選択されたzをselected_z_listに追記
    selected_z = session_data['train_x'][-5 + selected_index].unsqueeze(0)
    if 'selected_z_list' not in session_data:
        session_data['selected_z_list'] = selected_z
    else:
        session_data['selected_z_list'] = torch.cat([session_data['selected_z_list'], selected_z], dim=0)
    
    session_data['iteration'] += 1
    print(f"Iter {session_data['iteration']}: User selected {selected_index}. Scores adjusted.")

    print("Fitting GP and optimizing acquisition function...")
    gp = SingleTaskGP(session_data['train_x'], session_data['train_y'])
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll)
    
    # ---- Trust‑region bounds ----
    center = session_data['tr_center']
    radius = session_data['tr_radius']
    local_bounds = tr_bounds(center, radius)

    # Slightly larger beta encourages exploration under UCB
    ucb = qUpperConfidenceBound(gp, beta=2.5)
    new_candidates_z, _ = optimize_acqf(
        ucb, local_bounds, q=5, num_restarts=10, raw_samples=512
    )
    
    new_scores = objective_function(
        new_candidates_z, session_data['pos_feat'], session_data['neg_feat'], session_data['params']
    ).unsqueeze(1)
    
    session_data['train_x'] = torch.cat([session_data['train_x'], new_candidates_z], dim=0)
    session_data['train_y'] = torch.cat([session_data['train_y'], new_scores], dim=0)

    # ---- Update TR state ----
    best_idx = session_data['train_y'].argmax()
    best_score = session_data['train_y'][best_idx].item()
    if best_score > session_data['best_score']:
        session_data['best_score'] = best_score
        session_data['tr_center'] = session_data['train_x'][best_idx]
        session_data['tr_radius'] = max(1.0, session_data['tr_radius'] * 0.9)
    else:
        session_data['tr_radius'] = min(MAX_TR_RADIUS, session_data['tr_radius'] * 1.5)
    
    images_b64 = [tensor_to_base64(z) for z in new_candidates_z]
    
    best_idx = session_data['train_y'].argmax()
    best_image_b64 = tensor_to_base64(session_data['train_x'][best_idx])
    
    return jsonify({
        'images': images_b64,
        'iteration': session_data['iteration'],
        'best_image': best_image_b64
    })


# --- 可視化用エンドポイント ---
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

@app.route('/visualize_latents', methods=['GET'])
def visualize_latents():
    try:
        z_all = all_latents.detach().cpu().numpy()
        z_bo = session_data['train_x'].detach().cpu().numpy() if 'train_x' in session_data else None

        pca = PCA(n_components=2)
        z_all_2d = pca.fit_transform(z_all)
        z_bo_2d = pca.transform(z_bo) if z_bo is not None else None

        plt.figure(figsize=(8, 6))
        plt.scatter(z_all_2d[:, 0], z_all_2d[:, 1], s=5, alpha=0.3, label='Training Latents')
        # --- Auto‑crop view to central 98 % of training latents for readability ---
        x_lo, x_hi = np.percentile(z_all_2d[:, 0], [1, 99])
        y_lo, y_hi = np.percentile(z_all_2d[:, 1], [1, 99])
        x_margin = 0.05 * (x_hi - x_lo)
        y_margin = 0.05 * (y_hi - y_lo)
        plt.xlim(x_lo - x_margin, x_hi + x_margin)
        plt.ylim(y_lo - y_margin, y_hi + y_margin)
        if z_bo_2d is not None:
            plt.scatter(z_bo_2d[:, 0], z_bo_2d[:, 1], s=30, c='red', label='BO Samples')
        # 追加: selected_z_listがあれば軌道として描画
        if 'selected_z_list' in session_data:
            z_path = session_data['selected_z_list'].detach().cpu().numpy()
            z_path_2d = pca.transform(z_path)
            plt.plot(z_path_2d[:, 0], z_path_2d[:, 1], c='red', linewidth=2, label='BO Path')
            plt.scatter(z_path_2d[:, 0], z_path_2d[:, 1], s=30, c='red')
        plt.legend()
        plt.title("Latent Space (PCA projection)")
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        return jsonify({'pca_image': img_base64})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
