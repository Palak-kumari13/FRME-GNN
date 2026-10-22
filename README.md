# FRME-GNN: Stage 1

Tests whether a learned edge mask is useful before adding fairness and robustness losses.

This repository contains **A0–A2 only**. Fairness-aware mask regularization, graph attacks, GeoXGNN directions and knowledge distillation of geometry are future experiments, not implemented results.

## Experiments

| Run | Model | Loss |
|---|---|---|
| A0 | 2-layer GCN | Cross entropy |
| A1 | 2-layer GCN with learned edge weights | Cross entropy |
| A2 | A1 + frozen A0 prediction reference | Cross entropy + KL fidelity + mean mask penalty |

Edges have one shared score in both directions. Self loops remain unmasked. A2's KL term uses the **A0 teacher's soft predictions on training nodes**. This is prediction fidelity, not a guarantee that the explanation mask identifies ground-truth causal edges.

## Datasets and comparison protocol

Use the **Bail, Credit and German** graphs prepared for the NIFTY, FairGNN and EDITS baselines. The graph edges and node features must be the same as those previous runs. This repo does not download graph data or recreate graph construction, because doing so would produce a different benchmark.

Save `data/bail.npz`, `data/credit.npz`, `data/german.npz` (or `<dataset>_seed0.npz` through `<dataset>_seed4.npz` if you previously used distinct splits for each seed).

Each NPZ must contain:

| Key | Shape | Description |
|---|---|---|
| `x` | N × F | Finite numeric features, identically preprocessed to baselines |
| `edge_index` | 2 × E (or E × 2) | Zero-based node edges |
| `y` | N | Binary classification target, encoded 0/1 |
| `sensitive` | N | Binary sensitive attribute, encoded 0/1 |
| `train_idx` | 1D | Training node IDs |
| `val_idx` | 1D | Validation node IDs |
| `test_idx` | 1D | Test node IDs |

Example, once you have extracted the arrays from your **existing baseline loader**:

```python
import numpy as np
np.savez_compressed('data/bail.npz', x=x, edge_index=edge_index,
                    y=y, sensitive=sensitive, train_idx=train_idx,
                    val_idx=val_idx, test_idx=test_idx)
```

### Balanced, reproducible split (only if no prior split exists)

There are two separate experimental protocols:

- **Direct baseline comparison (recommended):** reuse the *exact* train/val/test indices of NIFTY, FairGNN and EDITS. Do not generate new splits. Training seeds remain **0–4**.
- **New controlled split study:** prepare `data/bail_raw.npz` with `x`, `edge_index`, `y`, `sensitive` and run `python prepare_splits.py --dataset bail --split-seed 42`. Repeat for Credit and German. The script makes a fixed 60/20/20 stratified split, prioritizing joint label–sensitive-group balance and falling back to label stratification if groups are small. It prints split counts and never overwrites an existing dataset. Every training seed reuses the same fixed split.

Stratification only aims for similar distributions; it does **not** make equal counts across classes, and it must not rebalance test labels. Reusing the same fixed graph across train/test is the usual transductive node-classification protocol; if your original baselines used inductive evaluation, that requires a matching loader instead. Feature normalization must be fit using training nodes only if it is not already in your baseline preprocessing.

**Do not compare numbers from the new split directly against previously recorded baseline results.** Rerun all baselines on it first.

## Setup

Python 3.10+; install PyTorch suited to your CPU or GPU, then:

```bash
pip install -r requirements.txt
```

No PyTorch Geometric dependency is needed. The two-layer GCN uses standard PyTorch operations.

## Train

```bash
python train.py --data-dir data --dataset bail --output results/bail
python train.py --data-dir data --dataset credit --output results/credit
python train.py --data-dir data --dataset german --output results/german
```

Each command trains A0, A1 and A2 with seeds `0 1 2 3 4`, saving weights and a `stage1_results.csv`. A2 uses the same seed's trained A0 checkpoint as a frozen teacher. Save each experiment to a **new output folder**: the script refuses to append to an existing CSV.

```bash
python summarize.py results/bail/stage1_results.csv --output results/bail/summary.csv
python summarize.py results/credit/stage1_results.csv --output results/credit/summary.csv
python summarize.py results/german/stage1_results.csv --output results/german/summary.csv
```

Custom hyperparameters:

```bash
python train.py --data-dir data --dataset bail --output results/bail_lrtest \
  --epochs 300 --patience 40 --hidden 64 --lr 0.005 \
  --lambda-fid 1.0 --lambda-sparse 0.005
```

Treat those as initial hyperparameters, not dataset-optimized values. If you tune them, do so using validation data only and report the tuning procedure.

## Metrics

The run file contains **test** F1, ROC-AUC, demographic parity gap, equal opportunity gap, positive prediction rate, mean mask weight, fraction of mask scores above 0.5, validation score and epochs.

- Larger F1/AUC indicates better predictive performance.
- Smaller parity/equality gaps means smaller measured differences between the two sensitive groups, **not proof of fairness**.
- Sparse masks and KL to a teacher do not establish explanation correctness. Evaluate motif recovery or ground-truth explanation accuracy in a later stage.
- The original baseline `robustness` and `counterfactual` metrics are **not reproduced here**. Do not compare them until their exact attacks and formulas have been implemented identically.
- AUC can be undefined if a test split contains only one class. Equality can be undefined if either sensitive group has no positive ground-truth examples.

For a fair comparison with the original five configurations (NIFTY Vanilla, NIFTY SSF, FairGNN from both run sets, EDITS), match preprocessing, splits, thresholding, fairness definitions, and seed handling. Results are mean ± **sample SD** (n=5) from `summarize.py`.

## Files

```
frme-gnn-stage1/
  train.py              # data validation, A0/A1/A2, training and test metrics
  prepare_splits.py     # optional fixed stratified split generator
  summarize.py          # average and standard deviation by model
  requirements.txt
  README.md
  .gitignore
  data/.gitkeep
  results/.gitkeep
```

## Next stage

First compare A0/A1/A2 using the same graph data and seeds. If the learned mask retains utility, develop a selective fairness mask loss, then robustness attacks, then predictive/geometric distillation. If the mask fails at Stage 1, investigate its design before adding further losses.
