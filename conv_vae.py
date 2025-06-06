# conv_vae.py
import torch.nn as nn, torch
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out += residual # Skip Connection
        return self.relu(out)

class ConvVAE(nn.Module):
    def __init__(self, z_dim=64):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.ReLU(),          # 72→36
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),         # 36→18
            nn.Flatten(),                                  # 64*18*18
        )
        self.fc_mu     = nn.Linear(64*18*18, z_dim)
        self.fc_logvar = nn.Linear(64*18*18, z_dim)

        self.fc_up = nn.Linear(z_dim, 64*18*18)
        self.dec = nn.Sequential(
            nn.ReLU(),
            nn.Unflatten(1, (64, 18, 18)),
            ResidualBlock(64),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(), # 18→36
            ResidualBlock(32),
            nn.ConvTranspose2d(32, 3, 4, 2, 1), nn.Tanh()   # 36→72
        )

    def reparameterize(self, mu, logvar):
        eps = torch.randn_like(mu)
        return mu + eps * (0.5*logvar).exp()

    def encode(self, x):
        h = self.enc(x)
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        return mu, logvar

    def decode(self, z):
        return self.dec(self.fc_up(z))


    def forward(self, x):
        h = self.enc(x)
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        z  = self.reparameterize(mu, logvar)
        recon = self.dec(self.fc_up(z))
        return recon, mu, logvar