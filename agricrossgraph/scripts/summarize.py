"""Collect metrics.json files into one results table (mean ± std over seeds).

    python scripts/summarize.py /kaggle/working/outputs            # all experiments
    python scripts/summarize.py /kaggle/working/outputs --exp baseline_cnn_loco
"""
import argparse
import json
from pathlib import Path

import pandas as pd

KEYS = ["acc_overall", "acc_seen", "macro_f1_seen", "healthy_auroc", "unseen_fraction"]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("output_root")
    ap.add_argument("--exp", default=None)
    a = ap.parse_args()
    files = Path(a.output_root).glob(f"{a.exp or '*'}/*/seed*/metrics.json")
    rows = [json.loads(f.read_text()) for f in files]
    if not rows:
        raise SystemExit("No metrics.json found.")
    df = pd.DataFrame(rows)
    keys = [k for k in KEYS if k in df]
    g = df.groupby(["exp_name", "fold"])[keys]
    table = g.mean().round(4).astype(str) + " ± " + g.std().fillna(0).round(4).astype(str)
    table["n_seeds"] = g.size()
    print(table.to_string())
    out = Path(a.output_root) / "results_summary.csv"
    table.to_csv(out)
    df.to_csv(Path(a.output_root) / "results_all_runs.csv", index=False)
    print(f"\nSaved {out}")
