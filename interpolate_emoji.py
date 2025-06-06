

import torch
import torchvision.utils as vutils
from conv_vae import ConvVAE

# 1. Load model and data
z_dim = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ConvVAE(z_dim=z_dim).to(device)
model.load_state_dict(torch.load("emoji_vae.pth", map_location=device))
model.eval()

data = torch.load("openmoji_72x72.pt")  # (N,3,72,72)


# 2. Extract two random emoji images
import random
idx1, idx2 = random.sample(range(len(data)), 2)
img1 = data[idx1].unsqueeze(0).to(device)
img2 = data[idx2].unsqueeze(0).to(device)

# 3. Encode to latent vectors
with torch.no_grad():
    # model.encode returns (mu, logvar)
    mu1, _ = model.encode(img1)
    mu2, _ = model.encode(img2)

# 4. Interpolate between two latent vectors over 8 steps
steps = 8
alphas = torch.linspace(0, 1, steps, device=device).unsqueeze(1)  # (8,1)
z_interp = (1 - alphas) * mu1 + alphas * mu2  # (8, z_dim)

# 5. Decode each intermediate vector
with torch.no_grad():
    recon_imgs = model.decode(z_interp)  # (8,3,72,72)

# 6. Normalize output to [0,1] for visualization
recon_imgs = recon_imgs.clamp(0, 1)

# 7. Save the interpolation grid
grid = vutils.make_grid(recon_imgs.cpu(), nrow=8, padding=2)
import torchvision.transforms.functional as TF
from PIL import Image
img = TF.to_pil_image(grid)
img.save("interpolation.png")