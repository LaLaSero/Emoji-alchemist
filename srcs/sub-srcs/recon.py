import torch, matplotlib.pyplot as plt
from conv_vae import ConvVAE

model = ConvVAE(z_dim=32)
model.load_state_dict(torch.load("emoji_vae.pth"))
model.eval()

data = torch.load("openmoji_72x72.pt")
x = data[0].unsqueeze(0)
with torch.no_grad():
    recon, _, _ = model(x)

to_img = lambda t: (t.squeeze().permute(1,2,0) * 0.5 + 0.5).clamp(0,1)

fig, axs = plt.subplots(1, 2)
axs[0].imshow(to_img(x)); axs[0].set_title("Original")
axs[1].imshow(to_img(recon)); axs[1].set_title("Reconstruction")
for ax in axs: ax.axis("off")
plt.tight_layout()
plt.savefig("reconstruction_compare.png", dpi=150)