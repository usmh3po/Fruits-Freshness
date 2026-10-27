"""
Compare training runs (the sentence-count ablation) and add a zero-shot CLIP baseline.

Reads runs/<name>/metrics.json, history.csv, selection_precision.json and writes to runs/:
  summary.csv                  : one row per run (selection precision, val/test metrics, efficiency)
  ablation_test.png            : test accuracy and test F1 (macro + per class) for every run
  zero_shot_clip.json          : CLIP zero-shot freshness metrics on the same test split

Usage:  python summarize_runs.py --runs s4x40_reg s2x80_reg s1x160_reg
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from eda import FRESHNESS_ORDER
from train import freshness_metrics


def zero_shot_clip(runs_dir, clip_dir, seed):
    """Freshness predicted by CLIP alone: softmax over the 96 sentences, summed per freshness level."""
    split = pd.read_csv(runs_dir / f"split_seed{seed}.csv")
    sims = pd.read_csv(clip_dir / "similarities.csv")
    sentences = pd.read_csv(clip_dir / "sentences.csv")
    sent_fresh = sentences["label"].str.rsplit(" ", n=1).str[0].map(FRESHNESS_ORDER.index).to_numpy()
    res = {}
    for name in ["val", "test"]:
        d = split[split["split"] == name].merge(sims, on="path")
        logits = d[[f"s{i}" for i in sentences["sentence_id"]]].to_numpy() * 100  # CLIP logit scale
        p = np.exp(logits - logits.max(1, keepdims=True))
        p /= p.sum(1, keepdims=True)
        probs = np.stack([p[:, sent_fresh == k].sum(1) for k in range(len(FRESHNESS_ORDER))], 1)
        y = d["freshness"].map(FRESHNESS_ORDER.index).to_numpy()
        res[name] = freshness_metrics(y, probs, d["fruit"].to_numpy())
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=["s4x40_reg", "s2x80_reg", "s1x160_reg"])
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--clip-dir", default="clip_output")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    runs_dir = Path(args.runs_dir)

    rows, hists = [], {}
    for r in args.runs:
        m = json.loads((runs_dir / r / "metrics.json").read_text())
        cfg = json.loads((runs_dir / r / "config.json").read_text())
        sp = json.loads((runs_dir / r / "selection_precision.json").read_text())
        hists[r] = pd.read_csv(runs_dir / r / "history.csv")
        row = {"run": r, "sentences_per_class": cfg["selection"]["sentences_per_class"],
               "images_per_sentence": cfg["selection"]["per_sentence"], "n_selected": sp["n_selected"],
               "sel_precision_class": sp["precision_class"], "sel_precision_freshness": sp["precision_freshness"],
               **{f"train_share_{k}": sp["true_freshness_share"].get(k, 0) for k in FRESHNESS_ORDER}}
        for split in ["val", "test"]:
            s = m[split]
            row.update({f"{split}_accuracy": s["accuracy"], f"{split}_f1_macro": s["f1_macro"],
                        f"{split}_f1_micro": s["f1_micro"], f"{split}_auroc": s["auroc_macro_ovr"],
                        f"{split}_pr_auc": s["pr_auc_macro"]})
            row.update({f"{split}_f1_{c}": v["f1"] for c, v in s["per_class"].items()})
        row["test_accuracy_originals_only"] = m["test"]["accuracy_originals_only"]
        e = m["efficiency"]
        row.update({"epochs_run": e["epochs_run"], "best_epoch": e["best_epoch"], "train_time_s": e["train_time_s"]})
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(runs_dir / "summary.csv", index=False)
    print(summary.T.to_string(header=False, float_format=lambda v: f"{v:.4f}"))

    zs = zero_shot_clip(runs_dir, Path(args.clip_dir), args.seed)
    (runs_dir / "zero_shot_clip.json").write_text(json.dumps(zs, indent=2))
    t = zs["test"]
    print(f"\nZero-shot CLIP (test): accuracy {t['accuracy']:.4f}, macro-F1 {t['f1_macro']:.4f}, "
          + ", ".join(f"F1 {c} {v['f1']:.3f}" for c, v in t["per_class"].items()))

    # Test accuracy and test F1 per run (bar charts)
    names = [f"{r.sentences_per_class} sentence{'s' if r.sentences_per_class > 1 else ''} x "
             f"{r.images_per_sentence}" for r in summary.itertuples()]
    x = np.arange(len(summary))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), gridspec_kw={"width_ratios": [1, 2]})
    bars = axes[0].bar(x, summary["test_accuracy"], color="#4c72b0", width=0.6)
    axes[0].bar_label(bars, fmt="%.3f", padding=2)
    axes[0].set_title("Test accuracy (freshness)")
    f1_cols = [("test_f1_macro", "Macro"), *[(f"test_f1_{c}", c) for c in FRESHNESS_ORDER]]
    w = 0.8 / len(f1_cols)
    for i, (col, label) in enumerate(f1_cols):
        b = axes[1].bar(x + (i - (len(f1_cols) - 1) / 2) * w, summary[col], w, label=label)
        axes[1].bar_label(b, fmt="%.2f", padding=2, fontsize=8)
    axes[1].set_title("Test F1 (freshness)")
    axes[1].legend(ncol=4, loc="upper center", fontsize=9)
    for ax in axes:
        ax.set_xticks(x, names)
        ax.set_ylim(0.6, 1.02)
        ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(runs_dir / "ablation_test.png", dpi=110)
    plt.close()
    print(f"\nWritten to {runs_dir}/")


if __name__ == "__main__":
    main()
