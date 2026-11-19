from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torch import nn
from torchvision import models, transforms

CLASSES = ["sehat", "sakit"]
IMG_SIZE = (224, 448)  # tinggi x lebar
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
MODEL_PATH = "models/baseline_efficientnet_b0.pt"
N = 4  # jumlah contoh per baris

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.efficientnet_b0(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(CLASSES))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device).eval()

tf = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

df = pd.read_csv("data/processed/metadata.csv")
test = df[df["split"] == "test"].reset_index(drop=True)

# Prediksi semua gambar di data test
preds, confs = [], []
with torch.no_grad():
    for path in test["filepath"]:
        x = tf(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
        p = torch.softmax(model(x), dim=1)[0]
        preds.append(CLASSES[int(p.argmax())])
        confs.append(float(p.max()))
test["pred"] = preds
test["conf"] = confs

groups = {
    "SALAH (tebakan keliru)": test[test["label"] != test["pred"]],
    "BENAR: daun sakit": test[(test["label"] == "sakit") & (test["pred"] == "sakit")],
    "BENAR: daun sehat": test[(test["label"] == "sehat") & (test["pred"] == "sehat")],
}
print({name: len(g) for name, g in groups.items()})

cam = GradCAM(model=model, target_layers=[model.features[-1]])

fig, axes = plt.subplots(3, N, figsize=(4.5 * N, 9))
for r, (title, grp) in enumerate(groups.items()):
    grp = grp.sample(n=min(N, len(grp)), random_state=42)
    for c in range(N):
        ax = axes[r, c]
        ax.set_xticks([])
        ax.set_yticks([])
        if c == 0:
            ax.set_ylabel(title, fontsize=9)
        if c >= len(grp):
            continue
        row = grp.iloc[c]
        orig = Image.open(row["filepath"]).convert("RGB")
        x = tf(orig).unsqueeze(0).to(device)
        rgb = np.asarray(orig.resize((IMG_SIZE[1], IMG_SIZE[0])), dtype=np.float32) / 255.0
        target = [ClassifierOutputTarget(CLASSES.index(row["pred"]))]
        heat = cam(input_tensor=x, targets=target)[0]
        ax.imshow(show_cam_on_image(rgb, heat, use_rgb=True))
        ax.set_title(
            f"asli: {row['label']} | tebakan: {row['pred']} ({row['conf']:.2f})\nkarton {row['carton']}",
            fontsize=8,
        )

plt.tight_layout()
Path("reports/figures").mkdir(parents=True, exist_ok=True)
plt.savefig("reports/figures/gradcam_baseline.png", dpi=120)
print("Tersimpan di reports/figures/gradcam_baseline.png")