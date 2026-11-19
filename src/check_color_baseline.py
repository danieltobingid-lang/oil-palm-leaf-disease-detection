import numpy as np
import pandas as pd
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler


def color_features(path):
    img = np.asarray(Image.open(path).convert("RGB").resize((224, 112)), dtype=np.float32) / 255.0
    return np.concatenate([img.mean(axis=(0, 1)), img.std(axis=(0, 1))])  # 6 angka: mean & std RGB


df = pd.read_csv("data/processed/metadata.csv")
X = {s: np.stack([color_features(p) for p in df[df["split"] == s]["filepath"]]) for s in ["train", "test"]}
y = {s: (df[df["split"] == s]["label"] == "sakit").astype(int).values for s in ["train", "test"]}

scaler = StandardScaler().fit(X["train"])
clf = LogisticRegression(max_iter=1000).fit(scaler.transform(X["train"]), y["train"])
pred = clf.predict(scaler.transform(X["test"]))

print("Baseline warna sederhana (6 angka per foto), diuji di data TEST:")
print(classification_report(y["test"], pred, target_names=["sehat", "sakit"], digits=3))
