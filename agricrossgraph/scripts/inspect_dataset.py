"""Print a folder tree with image counts, to see how an uploaded dataset is laid out.

    python scripts/inspect_dataset.py /kaggle/input            # find your dataset folders
    python scripts/inspect_dataset.py /kaggle/input/agri-vision4 --depth 4
"""
import argparse
from pathlib import Path

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def count_images(d: Path) -> int:
    return sum(1 for f in d.rglob("*") if f.suffix.lower() in IMG_EXT)


def walk(d: Path, depth: int, level: int = 0):
    subdirs = sorted(p for p in d.iterdir() if p.is_dir())
    direct = sum(1 for f in d.iterdir() if f.suffix.lower() in IMG_EXT)
    print(f"{'    ' * level}{d.name}/  [{count_images(d)} images{f', {direct} directly here' if direct else ''}]")
    if direct:
        ex = next(f.name for f in d.iterdir() if f.suffix.lower() in IMG_EXT)
        print(f"{'    ' * (level + 1)}e.g. {ex}")
    if level + 1 < depth:
        for s in subdirs:
            walk(s, depth, level + 1)
    elif subdirs:
        print(f"{'    ' * (level + 1)}... {len(subdirs)} more folders")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--depth", type=int, default=3)
    a = ap.parse_args()
    walk(Path(a.root), a.depth)
