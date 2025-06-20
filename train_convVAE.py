# train_emoji_vae.py
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import open_clip

# 修正版のConvVAEをインポート
from conv_vae import ConvVAE

# --- データとCLIPの準備 (変更なし) ---
print("Loading data and CLIP model...")
data = torch.load("openmoji_72x72.pt")
texts = [line.strip() for line in open("openmoji_labels.txt")]

clip_model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
clip_model = clip_model.cuda().eval()
tokenizer = open_clip.get_tokenizer("ViT-B-32")

with torch.no_grad():
    text_tokens = tokenizer(texts).cuda()
    text_emb_all = clip_model.encode_text(text_tokens).float()
    # テキスト埋め込みを事前に正規化しておく
    text_emb_all /= text_emb_all.norm(dim=-1, keepdim=True)

dataset = TensorDataset(data, text_emb_all)
loader = DataLoader(dataset, batch_size=256, shuffle=True)

# --- モデルとオプティマイザの準備 ---
# CLIPの埋め込み次元(512)を指定してVAEをインスタンス化
model = ConvVAE(z_dim=32, clip_emb_dim=512).cuda()
opt = torch.optim.Adam(model.parameters(), 1e-3)

# --- ハイパーパラメータ ---
beta_max = 0.05
anneal_epochs = 2500
lambda_clip = 1.5  # 潜在空間での損失は値が小さめに出るので、重みを少し上げる

# --- 高速な学習ループ ---
print("Starting FAST training with latent-space CLIP loss...")
for epoch in range(3000):
    current_beta = min(beta_max, beta_max * (epoch / anneal_epochs))
    for (x, t_emb) in loader:
        x = x.cuda()
        t_emb = t_emb.cuda()

        # 修正されたモデルは4つの値を返す
        recon, mu, logvar, projected_mu = model(x)
        
        # ★★★★★ ここからが新しい高速なCLIP損失計算 ★★★★★
        # VAEが生成した潜在ベクトル(mu)の射影を正規化
        projected_mu = projected_mu / projected_mu.norm(dim=-1, keepdim=True)
        # テキスト埋め込みとのコサイン距離を計算
        clip_loss = (1 - F.cosine_similarity(projected_mu, t_emb, dim=1)).mean()
        
        # --- 全ての損失を結合 ---
        recon_loss = F.mse_loss(recon, x, reduction='mean')
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean()
        
        loss = recon_loss + current_beta * kl + lambda_clip * clip_loss
        
        opt.zero_grad()
        loss.backward()
        opt.step()

    if epoch % 25 == 0:
        print(
            f"epoch {epoch:04d}  total={loss.item():.4f}  "
            f"recon={recon_loss.item():.4f}  KL={kl.item():.4f}  "
            f"clip={clip_loss.item():.4f}  beta={current_beta:.4f}"
        )

# 新しいモデルとして保存
torch.save(model.state_dict(), "emoji_vae_clip_fast.pth")
print("Saved FAST CLIP-guided VAE to 'emoji_vae_clip_fast.pth'")
