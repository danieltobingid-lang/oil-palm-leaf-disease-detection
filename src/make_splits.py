import re
from pathlib import Path

import pandas as pd

ROOT = Path("data/raw/Image Dataset of Oil Palm Leaf Diseases")
LABELS = {"Daun sehat (684)": "sehat", "daun sakit (899)": "sakit"}

# Dibagi per karton (bukan per gambar), supaya model diuji pada karton yang belum pernah dilihat
TEST_CARTONS = {9, 10, 17, 18}
VAL_CARTONS = {4, 8, 13, 16}

rows = []
for folder_name, label in LABELS.items():
    for path in (ROOT / folder_name).rglob("*.jpg"):
        carton = int(re.search(r"karton\s*(\d+)", path.parent.name, re.IGNORECASE).group(1))
        if carton in TEST_CARTONS:
            split = "test"
        elif carton in VAL_CARTONS:
            split = "val"
        else:
            split = "train"
        rows.append({"filepath": path.as_posix(), "label": label, "carton": carton, "split": split})

df = pd.DataFrame(rows)
Path("data/processed").mkdir(parents=True, exist_ok=True)
df.to_csv("data/processed/metadata.csv", index=False)

print(pd.crosstab(df["split"], df["label"], margins=True))
print("Tersimpan di data/processed/metadata.csv")