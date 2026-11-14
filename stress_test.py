"""
Stress tests for a trained run: corrupted inputs, partial feature loss, OOD inputs and class rarity.

Every corruption (corruptions.py) is applied to the 224x224 test crops (same split and preprocessing
as training) before ImageNet normalisation. OOD images come from CIFAR-100 (downloaded to ood_data/):
unseen produce (apple, pear, sweet pepper, mushroom) and non-produce classes.

Writes to runs/<run>/stress/: corruption_results.csv, ood_results.csv, rarity_results.csv

Usage:  python stress_test.py --run s4x40_reg [--rare-run s4x40_reg_rare_semi]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, roc_auc_score
from torchvision.datasets import CIFAR100

from calibration import ece, load_model, predict_logits
from corruptions import CORRUPTIONS, clean, low_res
from eda import FRESHNESS_ORDER
from train import load_images

REJECT_BELOW = 0.6  # post-prediction confidence threshold
CIFAR_PRODUCE = ["apple", "pear", "sweet_pepper", "mushroom"]  # unseen produce (orange is one of ours)
CIFAR_EXCLUDE = CIFAR_PRODUCE + ["orange"]


def predict(model, X_uint8, corrupt, device, seed):
    return predict_logits(model, X_uint8, device, corrupt, seed).softmax(1).numpy()


def summarise(probs, y):
    pred = probs.argmax(1)
    p, r, _, _ = precision_recall_fscore_support(y, pred, labels=[0, 1, 2], zero_division=0)
    conf = probs.max(1)
    return {"accuracy": accuracy_score(y, pred), "f1_macro": f1_score(y, pred, average="macro"),
            **{f"precision_{c}": p[i] for i, c in enumerate(FRESHNESS_ORDER)},
            **{f"recall_{c}": r[i] for i, c in enumerate(FRESHNESS_ORDER)},
            "rotten_to_fresh": int(((y == 2) & (pred == 0)).sum()),
            "mean_confidence": conf.mean(), "share_conf_over_0.9": (conf > 0.9).mean(),
            "rejected_share": (conf < REJECT_BELOW).mean(), "ece": ece(probs, y)}


def load_cifar_ood(root, n_other, seed):
    """uint8 224x224 tensors (CIFAR's 32x32 upsampled) for the two OOD sets."""
    ds = CIFAR100(root, train=False, download=True)
    names = np.array(ds.classes)[ds.targets]
    produce = np.flatnonzero(np.isin(names, CIFAR_PRODUCE))
    other = np.flatnonzero(~np.isin(names, CIFAR_EXCLUDE))
    other = np.random.default_rng(seed).choice(other, n_other, replace=False)
    to_tensor = lambda idx: torch.from_numpy(ds.data[idx]).permute(0, 3, 1, 2)  # uint8, 32x32
    up = lambda x: F.interpolate(x.float(), size=(224, 224), mode="bilinear", align_corners=False).round().byte()
    return {"Unseen produce (CIFAR-100 apple/pear/pepper/mushroom)": (up(to_tensor(produce)), names[produce]),
            "Non-fruit objects (CIFAR-100, 1000 random)": (up(to_tensor(other)), names[other])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="s4x40_reg")
    ap.add_argument("--rare-run", default=None, help="run trained with a rare class (train.py --rare-class)")
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--data", default="Processed Data")
    ap.add_argument("--ood-dir", default="ood_data")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--resamples", type=int, default=20, help="bootstrap draws for the test-prior shift")
    args = ap.parse_args()
    runs_dir = Path(args.runs_dir)
    out = runs_dir / args.run / "stress"
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    split = pd.read_csv(runs_dir / f"split_seed{args.seed}.csv")
    test = split[split["split"] == "test"].reset_index(drop=True)
    X = load_images(Path(args.data), test["path"], args.workers)
    y = test["freshness"].map(FRESHNESS_ORDER.index).to_numpy()
    model = load_model(runs_dir / args.run / "best.pt", device)

    # 1. Corruptions and feature loss
    rows = []
    for name, group, _, fn in CORRUPTIONS:
        rows.append({"test": name, "group": group, **summarise(predict(model, X, fn, device, args.seed), y)})
        print(f"{name:28s} acc {rows[-1]['accuracy']:.3f}  macro-F1 {rows[-1]['f1_macro']:.3f}")
    corr = pd.DataFrame(rows)
    corr.insert(3, "delta_accuracy", corr["accuracy"] - corr["accuracy"].iloc[0])
    corr.to_csv(out / "corruption_results.csv", index=False)

    # 2. OOD: confidence, rejection rate, and AUROC of max-softmax for ID (positive) vs OOD
    clean_p = predict(model, X, clean(), device, args.seed)
    clean_lr = predict(model, X, low_res(32), device, args.seed)
    ood_rows = [{"set": "In-distribution test (clean)", "n": len(X), "mean_confidence": clean_p.max(1).mean(),
                 "rejected_share": (clean_p.max(1) < REJECT_BELOW).mean(),
                 "share_conf_over_0.9": (clean_p.max(1) > 0.9).mean()},
                {"set": "In-distribution test at 32px (resolution-matched)", "n": len(X),
                 "mean_confidence": clean_lr.max(1).mean(), "rejected_share": (clean_lr.max(1) < REJECT_BELOW).mean(),
                 "share_conf_over_0.9": (clean_lr.max(1) > 0.9).mean()}]
    for name, (Xo, _) in load_cifar_ood(args.ood_dir, 1000, args.seed).items():
        p = predict(model, Xo, clean(), device, args.seed)
        conf = p.max(1)
        auroc = lambda ref: roc_auc_score(np.r_[np.ones(len(ref)), np.zeros(len(conf))], np.r_[ref.max(1), conf])
        ood_rows.append({"set": name, "n": len(Xo), "mean_confidence": conf.mean(),
                         "rejected_share": (conf < REJECT_BELOW).mean(),
                         "share_conf_over_0.9": (conf > 0.9).mean(),
                         "auroc_vs_clean_id": auroc(clean_p), "auroc_vs_32px_id": auroc(clean_lr),
                         **{f"predicted_{c}": (p.argmax(1) == i).mean() for i, c in enumerate(FRESHNESS_ORDER)}})
    ood = pd.DataFrame(ood_rows)
    ood.to_csv(out / "ood_results.csv", index=False)

    # 3. Class rarity: test prior shifted to 60/10/30 (bootstrap mean), and a model trained with rare semi-fresh
    rng = np.random.default_rng(args.seed)
    idx_by = [np.flatnonzero(y == k) for k in range(3)]
    n_total = round(len(idx_by[0]) / 0.6)
    draws = []
    for _ in range(args.resamples):
        idx = np.concatenate([rng.choice(idx_by[k], round(n_total * s), replace=False)
                              for k, s in enumerate((0.6, 0.1, 0.3))])
        draws.append(summarise(clean_p[idx], y[idx]))
    rarity = [{"scenario": "Balanced test (baseline)", **summarise(clean_p, y)},
              {"scenario": "Test prior 60/10/30 (fewer semi-fresh)", "n": len(idx),
               **pd.DataFrame(draws).mean().to_dict()}]
    if args.rare_run:
        rare_probs = predict(load_model(runs_dir / args.rare_run / "best.pt", device), X, clean(), device, args.seed)
        share = json.loads((runs_dir / args.rare_run / "selection_precision.json").read_text())["true_freshness_share"]
        rarity.append({"scenario": f"Trained with rare semi-fresh ({share.get('Semi Fresh', 0):.0%} of training set)",
                       **summarise(rare_probs, y)})
    pd.DataFrame(rarity).to_csv(out / "rarity_results.csv", index=False)

    pd.set_option("display.width", 250, "display.max_columns", 30)
    for name, df in [("Corruptions", corr), ("OOD", ood), ("Rarity", pd.DataFrame(rarity))]:
        print(f"\n{name}\n" + df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nWritten to {out}/")


if __name__ == "__main__":
    main()
