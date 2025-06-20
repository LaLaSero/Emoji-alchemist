# bo_step.py
import torch
from botorch.models import SingleTaskGP
# BoTorch version compatibility shim
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import UpperConfidenceBound
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood

# 仮の目的関数（好みのスコアを模倣）
def preference_function(z: torch.Tensor) -> torch.Tensor:
    # 例: zのL2ノルムが小さいほど「好みが高い」と仮定
    return -z.norm(dim=-1, keepdim=True)

# Step 1: 既知のzとスコア
# VAEロード前なので、後でtrain_x, train_yを定義

# Step 2以降はVAEロード後に実行

# Load trained VAE for decoding
from conv_vae import ConvVAE
import matplotlib.pyplot as plt
import torchvision.utils as vutils

vae = ConvVAE(z_dim=32)
vae.load_state_dict(torch.load("emoji_vae.pth", map_location="cpu"))
vae.eval()

with torch.no_grad():
    # 100個の絵文字画像からz(mu)ベクトルを抽出
    imgs = torch.load("openmoji_72x72.pt")[:100]
    mu_list = []
    for i in range(0, len(imgs), 32):
        mu, _ = vae.encode(imgs[i:i+32])
        mu_list.append(mu)
    train_x = torch.cat(mu_list)
    train_y = preference_function(train_x)

# Step 2: GPモデルの学習
gp = SingleTaskGP(train_x, train_y)
mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
fit_gpytorch_mll(mll)

# Step 3: UCBでacquisition最大化
ucb = UpperConfidenceBound(gp, beta=0.2)
bounds = torch.tensor([[ -2.5]*32, [ 2.5]*32])  # z空間の制限範囲（拡張して崩れを防ぐ）

new_z, _ = optimize_acqf(
    acq_function=ucb,
    bounds=bounds,
    q=1, num_restarts=5, raw_samples=64,
)
print("提案された次のz:", new_z)

with torch.no_grad():
    z_all = torch.cat([train_x[:4], new_z], dim=0)  # 4 init + 1 BO result
    decoded = vae.decode(z_all).clamp(0, 1)         # (5,3,72,72)

grid = vutils.make_grid(decoded, nrow=5)
from torchvision.transforms.functional import to_pil_image
to_pil_image(grid).save("bo_results.png")
print("保存しました: bo_results.png")