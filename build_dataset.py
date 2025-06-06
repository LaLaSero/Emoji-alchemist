# build_dataset.py
import glob, os, json, torch
from torchvision import transforms
from PIL import Image


META_JSON = "./openmoji/data/openmoji.json"   # metadata file
TARGET_GROUP = "smileys-emotion"            # change here if you want a different group

ROOT = "./openmoji/color/72x72"         # openmoji/color/72x72
files = glob.glob(os.path.join(ROOT, "*.png"))

# --- filter to emotion group ------------------------------------------------
with open(META_JSON, "r", encoding="utf-8") as fp:
    meta = json.load(fp)

hex_set = {
    entry["hexcode"].lower()
    for entry in meta
    if entry.get("group") == TARGET_GROUP and "face" in entry.get("annotation", "").lower()
}


files = [f for f in files
         if os.path.splitext(os.path.basename(f))[0].lower() in hex_set]

print(f"Selected {len(files)} PNGs in group: {TARGET_GROUP}")
# ---------------------------------------------------------------------------

transform = transforms.Compose([
    transforms.PILToTensor(),                # [0,255] uint8 → [C,H,W] uint8
    transforms.ConvertImageDtype(torch.float32),
    transforms.Normalize(0.5, 0.5)           # → [-1,1]
])

tensor_list = [transform(Image.open(f).convert("RGBA")\
                          .convert("RGB"))     # RGBA→RGB
               for f in files]

data = torch.stack(tensor_list)               # [N,3,72,72]
torch.save(data, "openmoji_72x72.pt")
print("saved", data.shape)