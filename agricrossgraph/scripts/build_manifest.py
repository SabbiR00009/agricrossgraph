"""Index a dataset into {paths.work_root}/manifests/<name>.csv and print a sanity report.

    python scripts/build_manifest.py --config configs/data/agrivision4.yaml
"""
import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
from agricrossgraph.config import add_common_args, load_config
from agricrossgraph.data.manifest import build_manifest, summarize

if __name__ == "__main__":
    a = add_common_args(argparse.ArgumentParser()).parse_args()
    cfg = load_config(a.config, a.overrides, a.paths)
    df, problems = build_manifest(cfg.dataset)
    out = Path(cfg.paths.work_root) / "manifests" / f"{cfg.dataset.name}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    report = summarize(df, cfg.dataset)
    if problems:
        report += "\n\nPROBLEMS TO FIX:\n" + "\n".join(f"  - {p}" for p in problems)
    else:
        report += "\n\nNo problems: every image matched a crop and class."
    out.with_suffix(".report.txt").write_text(report)
    print(report)
    print(f"\nManifest: {out}")
