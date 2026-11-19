import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

SEED = 42
EPOCHS = 8
BATCH_SIZE = 32
IMG_SIZE = (224, 448)  # tinggi x lebar, mengikuti rasio gambar asli (sekitar 1:2)
CLASSES = ["sehat", "sakit"]
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
MODEL_PATH = "models/grayscale_efficientnet_b0.pt"


class LeafDataset(Dataset):
    def __init__(self, df, transform):
        self.paths = df["filepath"].tolist()
        self.labels = [CLASSES.index(label) for label in df["label"]]
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return self.transform(img), self.labels[i]


def run_epoch(model, loader, loss_fn, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                out = model(x)
                loss = loss_fn(out, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(y)
            correct += (out.argmax(1) == y).sum().item()
            n += len(y)
    return total_loss / n, correct / n


def main():
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print("MODE: HITAM-PUTIH (warna dihilangkan)")

    df = pd.read_csv("data/processed/metadata.csv")

    train_tf = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    loaders = {}
    for split, tf, shuffle in [("train", train_tf, True), ("val", eval_tf, False), ("test", eval_tf, False)]:
        ds = LeafDataset(df[df["split"] == split], tf)
        loaders[split] = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=4, pin_memory=True)
        print(f"{split}: {len(ds)} gambar")

    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(CLASSES))
    model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

    Path("models").mkdir(exist_ok=True)
    best_val_acc = 0.0
    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, loaders["train"], loss_fn, device, optimizer)
        va_loss, va_acc = run_epoch(model, loaders["val"], loss_fn, device)
        print(
            f"Epoch {epoch}/{EPOCHS} | train loss {tr_loss:.3f} acc {tr_acc:.3f} "
            f"| val loss {va_loss:.3f} acc {va_acc:.3f} | {time.time() - t0:.0f}s"
        )
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save(model.state_dict(), MODEL_PATH)

    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in loaders["test"]:
            out = model(x.to(device))
            y_pred += out.argmax(1).cpu().tolist()
            y_true += y.tolist()

    print("\nHasil di data TEST (hitam-putih, karton yang belum pernah dilihat model):")
    print(classification_report(y_true, y_pred, target_names=CLASSES, digits=3))

    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=CLASSES, cmap="Blues")
    plt.title("Confusion Matrix - Grayscale (Test)")
    Path("reports/figures").mkdir(parents=True, exist_ok=True)
    plt.savefig("reports/figures/confusion_matrix_grayscale.png", dpi=120, bbox_inches="tight")
    print("Tersimpan di reports/figures/confusion_matrix_grayscale.png")


if __name__ == "__main__":
    main()