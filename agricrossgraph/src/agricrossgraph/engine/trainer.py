"""Supervised training loop used by the baselines. One call = one (fold, seed) run.

Output folder: {paths.output_root}/{exp_name}/{fold}/seed{seed}/
    config.yaml            fully resolved config (exactly what ran)
    best.pt                best checkpoint by val macro-F1
    history.csv            per-epoch train loss + val metrics
    metrics.json           final test metrics
    test_predictions.csv   per-image predictions (for confusion matrices / error analysis)
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from ..data.dataset import build_loaders
from ..data.splits import load_or_make_fold
from ..models import build_model
from ..utils.common import get_device, save_json, seed_everything
from .metrics import compute_metrics


@torch.no_grad()
def predict(model, loader, device, amp):
    model.eval()
    ys, ps, idx = [], [], []
    for x, y, i in loader:
        with torch.autocast(device.type, enabled=amp and device.type == "cuda"):
            logits = model(x.to(device, non_blocking=True))
        ps.append(logits.float().softmax(1).cpu())
        ys.append(y)
        idx.append(i)
    probs = torch.cat(ps).numpy() if ps else np.zeros((0, 1))
    return torch.cat(ys).numpy(), probs, torch.cat(idx).numpy()


def run_one(cfg, fold: str, seed: int) -> dict:
    seed_everything(seed)
    device = get_device()
    out_dir = Path(cfg.paths.output_root) / cfg.exp_name / fold / f"seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, out_dir / "config.yaml")

    fold_df = load_or_make_fold(cfg, fold)
    loaders, classes, label_col = build_loaders(cfg, fold_df, seed)
    print(f"[{cfg.exp_name} | fold={fold} | seed={seed}] classes={len(classes)} "
          f"train={len(loaders['train'].dataset)} val={len(loaders['val'].dataset)} "
          f"test={len(loaders['test'].dataset)} device={device}")

    model = build_model(cfg, len(classes)).to(device)
    t = cfg.train
    opt = torch.optim.AdamW(model.parameters(), lr=t.lr, weight_decay=t.weight_decay)
    steps_per_epoch = max(1, len(loaders["train"]))
    total, warm = t.epochs * steps_per_epoch, t.warmup_epochs * steps_per_epoch

    def lr_lambda(step):
        if step < warm:
            return (step + 1) / warm
        return 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    scaler = torch.amp.GradScaler(enabled=t.amp and device.type == "cuda")
    crit = nn.CrossEntropyLoss(label_smoothing=t.label_smoothing)

    best, bad, history = -1.0, 0, []
    for epoch in range(t.epochs):
        model.train()
        t0, losses = time.time(), []
        for step, (x, y, _) in enumerate(loaders["train"]):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device.type, enabled=t.amp and device.type == "cuda"):
                loss = crit(model(x), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            losses.append(loss.item())
            if cfg.log.print_every and step % cfg.log.print_every == 0:
                print(f"  ep{epoch} step{step}/{steps_per_epoch} loss={loss.item():.4f}")

        yv, pv, _ = predict(model, loaders["val"], device, t.amp)
        vm = compute_metrics(yv, pv.argmax(1), pv, classes)
        score = vm.get("macro_f1_seen", 0.0)
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), **{f"val_{k}": v for k, v in vm.items()},
                        "time_s": time.time() - t0})
        print(f"  epoch {epoch}: loss={np.mean(losses):.4f} val_acc={vm.get('acc_seen', 0):.4f} "
              f"val_macroF1={score:.4f} ({time.time() - t0:.0f}s)")
        if score > best:
            best, bad = score, 0
            torch.save({"model": model.state_dict(), "classes": classes, "epoch": epoch}, out_dir / "best.pt")
        else:
            bad += 1
            if bad >= t.early_stop_patience:
                print(f"  early stop at epoch {epoch}")
                break
    pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)

    ckpt = torch.load(out_dir / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model"])
    yt, pt, it = predict(model, loaders["test"], device, t.amp)
    pred = pt.argmax(1)
    metrics = compute_metrics(yt, pred, pt, classes)
    metrics.update(exp_name=cfg.exp_name, fold=fold, seed=seed, best_epoch=int(ckpt["epoch"]), best_val_macro_f1=best)
    save_json(metrics, out_dir / "metrics.json")

    test_df = loaders["test"].dataset.df.iloc[it][["rel_path", "crop", "class_id", "class_name", "concept"]].copy()
    test_df["true_label"] = test_df[label_col]
    test_df["seen"] = yt >= 0
    test_df["pred_label"] = [classes[p] for p in pred]
    test_df["confidence"] = pt.max(1)
    test_df.to_csv(out_dir / "test_predictions.csv", index=False)
    print(f"  TEST: {metrics}")
    return metrics
