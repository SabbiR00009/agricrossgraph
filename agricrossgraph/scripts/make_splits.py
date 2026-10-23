"""Create (once) and freeze the split CSVs for every fold of an experiment's protocol.

    python scripts/make_splits.py --config configs/experiments/baseline_cnn_loco.yaml

Copy the resulting splits/ folder into the repo and commit it, so every experiment and every
teammate uses identical splits.
"""
import argparse

import _bootstrap  # noqa: F401
from agricrossgraph.config import add_common_args, load_config
from agricrossgraph.data.splits import load_or_make_fold, split_path

if __name__ == "__main__":
    a = add_common_args(argparse.ArgumentParser()).parse_args()
    cfg = load_config(a.config, a.overrides, a.paths)
    for fold in cfg.run.folds:
        df = load_or_make_fold(cfg, fold)
        print(f"\n== fold {fold}  ({split_path(cfg, fold)})")
        print(df.groupby(["split", "crop"]).size().unstack(fill_value=0).to_string())
        if cfg.split.protocol == "loco":
            train_concepts = set(df[df.split == "train"].concept)
            test = df[df.split == "test"]
            seen = test.concept.isin(train_concepts)
            print(f"   test concepts seen in training: {sorted(set(test.concept[seen]))}")
            print(f"   test concepts NEVER seen:       {sorted(set(test.concept[~seen]))}")
            print(f"   -> {seen.mean():.1%} of test images have a seen concept")
