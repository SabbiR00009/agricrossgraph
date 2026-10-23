"""Config loading: YAML files with `_base_` inheritance + a paths profile + command-line overrides.

Order of precedence (later wins):
    paths/<profile>.yaml  <  _base_ files (in order)  <  the file itself  <  CLI overrides (key.sub=value)
"""
from __future__ import annotations

import os
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"


def _load_with_bases(path: Path) -> DictConfig:
    path = Path(path).resolve()
    cfg = OmegaConf.load(path)
    bases = cfg.pop("_base_", []) or []
    if isinstance(bases, str):
        bases = [bases]
    merged = OmegaConf.create()
    for b in bases:
        merged = OmegaConf.merge(merged, _load_with_bases(path.parent / b))
    return OmegaConf.merge(merged, cfg)


def default_paths_profile() -> str:
    return "kaggle" if os.path.isdir("/kaggle/input") else "local"


def load_config(path: str | Path, overrides: list[str] | None = None, paths_profile: str | None = None) -> DictConfig:
    profile = paths_profile or default_paths_profile()
    paths_file = CONFIG_DIR / "paths" / f"{profile}.yaml"
    if not paths_file.exists():
        raise FileNotFoundError(f"No paths profile '{profile}' at {paths_file}")
    cfg = OmegaConf.merge(OmegaConf.load(paths_file), _load_with_bases(Path(path)))
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
    cfg.paths_profile = profile
    OmegaConf.resolve(cfg)
    return cfg


def repo_path(p: str | Path) -> Path:
    """Resolve a path relative to the repo root unless it is already absolute."""
    p = Path(p)
    return p if p.is_absolute() else REPO_ROOT / p


def add_common_args(parser):
    parser.add_argument("--config", required=True, help="YAML config file")
    parser.add_argument("--paths", default=None, help="paths profile: kaggle | local (default: auto-detect)")
    parser.add_argument("overrides", nargs="*", help="key=value overrides, e.g. train.epochs=5")
    return parser
