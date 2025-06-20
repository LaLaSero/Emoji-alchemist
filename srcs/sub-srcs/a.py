# train_emoji_vae.py
import torch, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import open_clip
import torchvision.transforms as T
from conv_vae import ConvVAE

data = torch.load("openmoji_72x72.pt")        # (N,3,72,72)
texts = [line.strip() for line in open("openmoji_labels.txt")]   # 1-to-1 with images
print(f"text labels: {len(texts)}")
print("First 5 labels:", texts[:5])