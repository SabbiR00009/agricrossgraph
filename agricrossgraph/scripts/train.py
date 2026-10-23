"""Train + test one experiment over all folds x seeds listed in its config.

    python scripts/train.py --config configs/experiments/smoke_test.yaml
    python scripts/train.py --config configs/experiments/baseline_cnn_loco.yaml run.folds=[zucchini] run.seeds=[0]
"""
import argparse

import _bootstrap  # noqa: F401
from agricrossgraph.config import add_common_args, load_config
from agricrossgraph.engine.trainer import run_one

if __name__ == "__main__":
    a = add_common_args(argparse.ArgumentParser()).parse_args()
    cfg = load_config(a.config, a.overrides, a.paths)
    for fold in cfg.run.folds:
        for seed in cfg.run.seeds:
            run_one(cfg, fold, seed)
