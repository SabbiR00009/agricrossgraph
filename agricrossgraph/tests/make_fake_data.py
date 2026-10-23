"""Create a tiny fake Agri-Vision4-like folder tree (random images) to test the pipeline without real data.

    python tests/make_fake_data.py ./data/agrivision4 --per-class 6
"""
import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image

CROP_DIR = {"tomato": "Tomato", "papaya": "Papaya", "zucchini": "Zucchini", "bottle_gourd": "Bottle Gourd"}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--per-class", type=int, default=6)
    a = ap.parse_args()
    meta = Path(__file__).resolve().parents[1] / "metadata" / "agrivision4_classes.csv"
    rng = np.random.default_rng(0)
    for r in csv.DictReader(open(meta)):
        crop = CROP_DIR[r["crop"]]
        cls = f"{crop}_{r['class_name']}".replace(" ", "_")
        for src, n in (("Original_Images", a.per_class), ("Augmented_Images", 2)):
            d = Path(a.root) / src / crop / cls
            d.mkdir(parents=True, exist_ok=True)
            for i in range(n):
                arr = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
                Image.fromarray(arr).save(d / f"{cls}_{i:03d}.jpg")
    print("fake data at", a.root)
