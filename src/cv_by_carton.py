from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import classification_report
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models, transforms

from train_baseline import BATCH_SIZE, CLASSES, IMG_SIZE, MEAN, STD, LeafDataset, run_epoch

EPOCHS = 6
SEED = 42

# Setiap fold menyisihkan beberapa karton (sakit: 1-10, sehat: 11-18) untuk diuji
FOLDS = {
    "fold1": {1, 5, 9, 11, 15},
    "fold2": {2, 6, 10, 12, 16},
    "fold3": {3, 7, 13, 17},
    "fold4": {4, 8, 14, 18},
}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    train_tf = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.03),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    df = pd.read_csv("data/processed/metadata.csv")
    results = []

    for name, test_cartons in FOLDS.items():
        torch.manual_seed(SEED)
        train_df = df[~df["carton"].isin(test_cartons)]
        test_df = df[df["carton"].isin(test_cartons)].copy()

        train_loader = DataLoader(LeafDataset(train_df, train_tf), batch_size=BATCH_SIZE,
                                  shuffle=True, num_workers=4, pin_memory=True)
        test_loader = DataLoader(LeafDataset(test_df, eval_tf), batch_size=BATCH_SIZE,
                                 shuffle=False, num_workers=4, pin_memory=True)

        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(CLASSES))
        model.to(device)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

        for _ in range(EPOCHS):
            run_epoch(model, train_loader, loss_fn, device, optimizer)

        model.eval()
        preds = []
        with torch.no_grad():
            for x, _ in test_loader:
                preds += model(x.to(device)).argmax(1).cpu().tolist()

        test_df["pred"] = [CLASSES[p] for p in preds]
        test_df["fold"] = name
        results.append(test_df)
        acc = (test_df["pred"] == test_df["label"]).mean()
        print(f"{name}: karton uji {sorted(test_cartons)} | latih {len(train_df)} | uji {len(test_df)} | akurasi {acc:.3f}")

    res = pd.concat(results)
    res["benar"] = res["pred"] == res["label"]

    print("\nAkurasi per karton:")
    table = res.groupby(["label", "carton"])["benar"].agg(["mean", "count"]).round(3)
    print(table)

    print("\nGabungan semua fold (setiap foto diuji tepat sekali, saat kartonnya disisihkan):")
    print(classification_report(res["label"], res["pred"], labels=CLASSES, target_names=CLASSES, digits=3))

    Path("reports").mkdir(exist_ok=True)
    table.to_csv("reports/cv_by_carton.csv")
    print("Tersimpan di reports/cv_by_carton.csv")


if __name__ == "__main__":
    main()