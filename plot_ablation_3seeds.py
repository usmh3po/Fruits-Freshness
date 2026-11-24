"""
The ablation_test.png figure (test accuracy + per-class test F1) redrawn for the four Milestone 2 setups
with three seeds each: ResNet-18 1 x 160, 2 x 80, 4 x 40, and Swin-T 4 x 40. Bars are means over seeds,
error bars the standard deviation. Nothing is retrained: the numbers come from runs/<run>/metrics.json.

Writes to runs/backbones/:
  ablation_3seeds.csv          one row per setup and seed (edit this to change the figure)
  ablation_3seeds_summary.csv  mean / std per setup
  ablation_test_3seeds.png

Usage:  python plot_ablation_3seeds.py              # build the CSV from the runs, then plot
        python plot_ablation_3seeds.py --from-csv   # redraw from an edited ablation_3seeds.csv
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

SETUPS = {"ResNet-18\n1 sentence x 160": ["s1x160_reg", "s1x160_reg_seed43", "s1x160_reg_seed44"],
          "ResNet-18\n2 sentences x 80": ["s2x80_reg", "s2x80_reg_seed43", "s2x80_reg_seed44"],
          "ResNet-18\n4 sentences x 40": ["s4x40_reg", "s4x40_reg_seed43", "s4x40_reg_seed44"],
          "Swin-T\n4 sentences x 40": ["swin_t_seed42", "swin_t_seed43", "swin_t_seed44"]}
SEEDS = [42, 43, 44]
F1_COLS = [("f1_macro", "Macro"), *[(f"f1_{c}", c) for c in FRESHNESS_ORDER]]


def build_csv(runs_dir):
    rows = []
    for setup, runs in SETUPS.items():
        for seed, run in zip(SEEDS, runs):
            t = json.loads((runs_dir / run / "metrics.json").read_text())["test"]
            rows.append({"setup": setup.replace("\n", " | "), "seed": seed, "run": run, "accuracy": t["accuracy"],
                         "f1_macro": t["f1_macro"], **{f"f1_{c}": v["f1"] for c, v in t["per_class"].items()}})
    return pd.DataFrame(rows)


def plot(df, path):
    setups = list(dict.fromkeys(df["setup"]))
    mean = df.groupby("setup", sort=False).mean(numeric_only=True).loc[setups]
    std = df.groupby("setup", sort=False).std(numeric_only=True).loc[setups]
    x = np.arange(len(setups))
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), gridspec_kw={"width_ratios": [1, 2]})
    bars = axes[0].bar(x, mean["accuracy"], yerr=std["accuracy"], capsize=4, color="#4c72b0", width=0.6)
    axes[0].bar_label(bars, fmt="%.3f", padding=3)
    axes[0].set_title("Test accuracy (freshness), mean ± std over 3 seeds")
    w = 0.8 / len(F1_COLS)
    for i, (col, label) in enumerate(F1_COLS):
        b = axes[1].bar(x + (i - (len(F1_COLS) - 1) / 2) * w, mean[col], w, yerr=std[col], capsize=2, label=label)
        axes[1].bar_label(b, fmt="%.2f", padding=3, fontsize=8)
    axes[1].set_title("Test F1 (freshness), mean ± std over 3 seeds")
    axes[1].legend(ncol=4, loc="upper center", fontsize=9)
    for ax in axes:
        ax.set_xticks(x, [s.replace(" | ", "\n") for s in setups], fontsize=9)
        ax.set_ylim(0.6, 1.02)
        ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=110)
    plt.close()
    return mean, std


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-csv", action="store_true", help="redraw from runs/backbones/ablation_3seeds.csv")
    args = ap.parse_args()
    runs_dir = Path("runs")
    out = runs_dir / "backbones"
    out.mkdir(exist_ok=True)
    csv = out / "ablation_3seeds.csv"
    if args.from_csv:
        df = pd.read_csv(csv)
    else:
        df = build_csv(runs_dir)
        df.to_csv(csv, index=False)
    mean, std = plot(df, out / "ablation_test_3seeds.png")
    summary = mean.add_suffix("_mean").join(std.add_suffix("_std")).drop(columns=["seed_mean", "seed_std"])
    summary.to_csv(out / "ablation_3seeds_summary.csv")
    print(summary.round(4).T.to_string())


if __name__ == "__main__":
    main()
