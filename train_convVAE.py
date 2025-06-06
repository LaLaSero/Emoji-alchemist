# train_emoji_vae.py
import torch, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from conv_vae import ConvVAE

data = torch.load("openmoji_72x72.pt")        # (N,3,72,72)
loader = DataLoader(TensorDataset(data), batch_size=512, shuffle=True)

beta_max = 0.03       # betaの最大値を0.1に設定
anneal_epochs = 2000

model = ConvVAE(z_dim=32).cuda()
opt = torch.optim.Adam(model.parameters(), 1e-3)

for epoch in range(6000):
    current_beta = min(beta_max, beta_max * (epoch / anneal_epochs))
    for (x,) in loader:
        x = x.cuda()
        recon, mu, logvar = model(x)
        recon_loss = F.mse_loss(recon, x, reduction='mean')
        kl = -0.5*(1 + logvar - mu.pow(2) - logvar.exp()).mean()
        loss = recon_loss + current_beta*kl
        opt.zero_grad(); loss.backward(); opt.step()
    if epoch % 100 == 0:
        print(f"epoch {epoch:02d}  loss={loss.item():.4f}")
torch.save(model.state_dict(), "emoji_vae.pth")