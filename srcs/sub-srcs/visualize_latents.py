# visualize_latents.py
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from tqdm import tqdm

# --- モデルと設定のロード ---
print("--- Latent Space Visualization ---")
print("Loading models and configuration...")
from conv_vae import ConvVAE
from latent_unet import LatentUNet

Z_DIM = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 256

# モデルのロード
vae = ConvVAE(z_dim=Z_DIM).to(DEVICE); vae.load_state_dict(torch.load("emoji_vae_capacity.pth")); vae.eval()
diffusion_model = LatentUNet(z_dim=Z_DIM).to(DEVICE); diffusion_model.load_state_dict(torch.load("latent_diffusion.pth")); diffusion_model.eval()

# 正規化のための平均とスケールをロード
latent_mean = torch.load("vae_latent_mean.pt").to(DEVICE)
scale_factor = torch.load("vae_latent_scale.pt").to(DEVICE)

# --- Diffusionサンプリング関数 ---
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
    else: return model_mean + torch.sqrt(betas[t]) * torch.randn_like(z_t)

@torch.no_grad()
def generate_diffusion_latents(num_latents):
    latents = []
    for _ in tqdm(range(int(np.ceil(num_latents / BATCH_SIZE))), desc="Generating Diffusion Latents"):
        z = torch.randn(BATCH_SIZE, Z_DIM, device=DEVICE)
        for i in reversed(range(1000)):
            z = p_sample(z, i)
        latents.append(z)
    return torch.cat(latents)[:num_latents]

# --- メイン処理 ---
def main():
    # 1. VAE潜在ベクトル群を生成 (教師データ)
    print("\nStep 1: Generating latents from VAE Encoder...")
    all_images_data = torch.load("openmoji_72x72.pt")
    loader = DataLoader(TensorDataset(all_images_data), batch_size=BATCH_SIZE)
    vae_latents = []
    with torch.no_grad():
        for (x_batch,) in tqdm(loader, desc="Encoding dataset with VAE"):
            mu, _ = vae.encode(x_batch.to(DEVICE))
            normalized_mu = (mu - latent_mean) / scale_factor
            vae_latents.append(normalized_mu)
    vae_latents = torch.cat(vae_latents)
    num_samples = len(vae_latents)

    # 2. 拡散モデル潜在ベクトル群を生成
    print(f"\nStep 2: Generating {num_samples} latents from Diffusion Model...")
    diffusion_latents = generate_diffusion_latents(num_samples)

    # 3. データを結合し、t-SNEを実行
    print("\nStep 3: Combining data and running t-SNE (this may take a while)...")
    all_latents = torch.cat([vae_latents, diffusion_latents], dim=0).cpu().numpy()
    
    # ラベルを作成 (0: VAE, 1: Diffusion)
    labels = np.array([0] * num_samples + [1] * num_samples)
    
    tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42, verbose=1)
    tsne_results = tsne.fit_transform(all_latents)

    # 4. 可視化
    print("\nStep 4: Visualizing results...")
    plt.figure(figsize=(14, 10))
    palette = sns.color_palette("bright", 2)
    sns.scatterplot(
        x=tsne_results[:, 0], y=tsne_results[:, 1],
        hue=labels,
        palette=palette,
        alpha=0.6,
        s=50,
        legend='full'
    )
    
    # 凡例のテキストを修正
    legend = plt.legend()
    legend.texts[0].set_text('VAE Encoded Latents (教師データ)')
    legend.texts[1].set_text('Diffusion Generated Latents (生成データ)')
    
    plt.title('t-SNE Visualization of VAE vs. Diffusion Latent Spaces', fontsize=18)
    plt.xlabel('t-SNE Dimension 1', fontsize=14)
    plt.ylabel('t-SNE Dimension 2', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    output_path = "latent_space_visualization.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to '{output_path}'")

if __name__ == "__main__":
    main()
