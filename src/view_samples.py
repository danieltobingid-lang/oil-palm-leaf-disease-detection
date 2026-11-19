import random
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

root = Path("data/raw/Image Dataset of Oil Palm Leaf Diseases")
classes = {"Sehat": root / "Daun sehat (684)", "Sakit": root / "daun sakit (899)"}

random.seed(42)
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
for row, (name, folder) in enumerate(classes.items()):
    files = list(folder.rglob("*.jpg"))
    for col, path in enumerate(random.sample(files, 5)):
        img = Image.open(path)
        axes[row, col].imshow(img)
        axes[row, col].set_title(f"{name}\n{path.parent.name[:14]} | {img.size[0]}x{img.size[1]}", fontsize=8)
        axes[row, col].axis("off")
plt.tight_layout()
Path("reports/figures").mkdir(parents=True, exist_ok=True)
plt.savefig("reports/figures/sample_images.png", dpi=120)

sizes = Counter()
for folder in classes.values():
    for p in folder.rglob("*.jpg"):
        with Image.open(p) as im:
            sizes[im.size] += 1
print("Ukuran gambar terbanyak:", sizes.most_common(5))
print("Tersimpan di reports/figures/sample_images.png")