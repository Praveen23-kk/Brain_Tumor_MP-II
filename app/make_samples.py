"""One-off script that fills app/model/ and app/samples/ (run it from the project root: python app/make_samples.py).

model/   : best.pt, config.json and metrics.json of the best lightweight model on the CLEAN split.
samples/ : one MRI per class, copied from data/ (data/ itself is only read, never changed).
The samples are the FIRST row of each class among the clean split's test rows, so the model never saw them
in training or validation. Nothing is hand-picked.
"""
import shutil
from pathlib import Path

import pandas as pd

RUN_DIR = Path("outputs/runs/v2_clean/mobilenetv3_small_100")   # best light model on the clean split (Week 8 table)
SPLIT_CSV = Path("outputs/splits/v2_split_clean.csv")
APP_DIR = Path("app")
CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]

# 1) model files
(APP_DIR / "model").mkdir(parents=True, exist_ok=True)
for name in ["best.pt", "config.json", "metrics.json"]:
    shutil.copy2(RUN_DIR / name, APP_DIR / "model" / name)
    print("copied", RUN_DIR / name)

# 2) one held-out sample per class
(APP_DIR / "samples").mkdir(parents=True, exist_ok=True)
table = pd.read_csv(SPLIT_CSV)
test = table[table["split"] == "test"].reset_index(drop=True)   # same row order as test_outputs.npz
for cls in CLASSES:
    row = test[test["label"] == cls].iloc[0]                     # first test row of this class
    source = Path(str(row["source"]).replace("\\", "/"))        # original JPG in data/
    shutil.copy2(source, APP_DIR / "samples" / f"{cls}.jpg")
    print(f"{cls:11s} test row {test[test['label'] == cls].index[0]:3d}  <-  {source}")
