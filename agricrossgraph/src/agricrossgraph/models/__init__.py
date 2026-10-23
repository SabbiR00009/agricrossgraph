"""Model registry. Add a new model by writing a builder and registering it here;
then select it from YAML with `model.name: <key>`.

Planned keys (later phases): simclr_cnn, clip_zeroshot, gnn_supervised, agricrossgraph.
"""
from .cnn import build_cnn

REGISTRY = {
    "cnn": build_cnn,
}


def build_model(cfg, num_classes: int):
    name = cfg.model.name
    if name not in REGISTRY:
        raise KeyError(f"Unknown model.name '{name}'. Available: {list(REGISTRY)}")
    return REGISTRY[name](cfg, num_classes)
