# AgriCrossGraph

Cross-crop contrastive graph pretraining for zero-shot leaf disease transfer across tomato, papaya,
zucchini and bottle gourd (Agri-Vision4), plus extra Bangladeshi leaf datasets.

## Repo layout

```
configs/
  base.yaml                     shared defaults (every experiment inherits these)
  paths/kaggle.yaml             dataset + output locations on Kaggle (auto-selected on Kaggle)
  paths/local.yaml              same for your laptop
  data/agrivision4.yaml         how to read the Agri-Vision4 folders
  data/bd_freshness.yaml        template for the extra Bangladesh dataset
  experiments/*.yaml            one file per experiment = one row in your results table
metadata/
  agrivision4_classes.csv       28 classes -> shared concept, pathogen type, symptom text
splits/                         frozen split CSVs (commit these!)
src/agricrossgraph/
  config.py                     YAML loader (_base_ inheritance, paths profile, CLI overrides)
  data/manifest.py              folder -> one CSV row per image
  data/splits.py                leave-one-crop-out and within-crop splits
  data/dataset.py               PyTorch Dataset, transforms, loaders
  models/__init__.py            model registry (model.name in YAML -> builder)
  models/cnn.py                 timm CNN baseline
  engine/trainer.py             train / early-stop / test loop, writes all outputs
  engine/metrics.py             seen/unseen-aware metrics
scripts/                        command-line entry points (thin wrappers around src/)
notebooks/kaggle_runner.ipynb   the notebook you run on Kaggle
tests/make_fake_data.py         fake dataset to test the pipeline without real data
```

## How the configs work

An experiment file lists its parents in `_base_` and only states what is different:

```yaml
_base_: [../base.yaml, ../data/agrivision4.yaml]
exp_name: baseline_cnn_loco
model:
  backbone: resnet50
```

Final config = `paths/<profile>.yaml` + base files + this file + command-line overrides (later wins):

```bash
python scripts/train.py --config configs/experiments/baseline_cnn_loco.yaml train.epochs=10 model.backbone=convnext_tiny
```

Every run saves the fully resolved `config.yaml` next to its results, so any number in your report
can be traced to exactly what produced it.

**Rule:** never change code to run a new experiment. Add a new YAML file in `configs/experiments/`.

## Workflow

```bash
pip install -r requirements.txt
python scripts/inspect_dataset.py /kaggle/input --depth 4                     # 1. see folder layout
python scripts/build_manifest.py --config configs/data/agrivision4.yaml        # 2. index + sanity report
python scripts/make_splits.py   --config configs/experiments/baseline_cnn_loco.yaml   # 3. freeze splits
python scripts/train.py         --config configs/experiments/smoke_test.yaml          # 4. quick check
python scripts/train.py         --config configs/experiments/baseline_cnn_loco.yaml   # 5. real run
python scripts/summarize.py /kaggle/working/outputs                            # 6. results table
```

Outputs: `{output_root}/{exp_name}/{fold}/seed{seed}/` with `config.yaml`, `best.pt`, `history.csv`,
`metrics.json`, `test_predictions.csv`.

## Evaluation protocol

- **LOCO (leave-one-crop-out):** train on 3 crops, test on all *original* images of the 4th.
- **Never test on augmented images.** Augmented copies are rotations/flips of originals.
- `split.label_level: concept` trains on shared disease concepts (healthy, mosaic_virus, downy_mildew,
  anthracnose, insect_damage, leaf_spot + crop-specific ones), so cross-crop transfer is measurable.
- Metrics: `acc_overall` (unseen concepts count as wrong), `acc_seen`, `macro_f1_seen`, `healthy_auroc`.

## Test without real data

```bash
python tests/make_fake_data.py ./data/agrivision4
python scripts/build_manifest.py --config configs/data/agrivision4.yaml --paths local
python scripts/train.py --config configs/experiments/smoke_test.yaml --paths local model.pretrained=false data.num_workers=0
```
Delete `splits/agrivision4/` afterwards, since those splits come from the fake data.

## Roadmap (where new code goes)

| Phase | Add | Where |
|---|---|---|
| Baselines | CLIP / BioCLIP zero-shot, SimCLR CNN | `models/clip_zs.py`, `models/simclr.py` + experiment YAMLs |
| Core | superpixel graphs, cached backbone features | `graphs/build.py`, `scripts/extract_features.py` |
| Core | GNN encoder + GraphCL pretraining | `models/gnn.py`, `engine/pretrain.py` |
| Core | text-aligned zero-shot head | `models/zeroshot_head.py` (uses `description` column) |
| Extra data | Bangladesh freshness dataset as unseen crops | fill `metadata/bd_freshness_classes.csv` |
| Analysis | confusion matrices, t-SNE, GNNExplainer | `analysis/` |
