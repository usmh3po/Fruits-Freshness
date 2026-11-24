"""
Backbone comparison on the Milestone 2 setup: ResNet-18 vs Swin-T, same 3,840 selected training images,
(plus the ResNet-18 sentence-count ablations 1 x 160 and 2 x 80),
same split, preprocessing and v1.1 training recipe, three training seeds each.

Reads runs/<run>/metrics.json and writes to runs/backbones/:
  comparison.csv           one row per run (test accuracy, macro-F1, AUROC, efficiency)
  summary.csv              mean / std per backbone
  test_accuracy.png, test_f1.png, backbone_comparison.png (both panels)
  plot_data.csv            editable numbers behind the accuracy / F1 graphs (setup, label, colour, seed, values)
  latency.csv, latency.png  back-to-back latency / throughput benchmark of both models (seed 42)

Redraw every graph from the CSVs (after editing plot_data.csv or latency.csv):
  python compare_backbones.py --plot-only

Usage:  python compare_backbones.py
"""

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.flop_counter import FlopCounterMode

from train import FreshnessNet

# Setups compared in the accuracy / F1 graphs: the sentence-count ablations and both backbones, 3 seeds each
RUNS = {"ResNet-18, 1 x 160": ["s1x160_reg", "s1x160_reg_seed43", "s1x160_reg_seed44"],
        "ResNet-18, 2 x 80": ["s2x80_reg", "s2x80_reg_seed43", "s2x80_reg_seed44"],
        "ResNet-18 (CNN)": ["s4x40_reg", "s4x40_reg_seed43", "s4x40_reg_seed44"],
        "Swin-T (transformer)": ["swin_t_seed42", "swin_t_seed43", "swin_t_seed44"]}
LABELS = {"ResNet-18, 1 x 160": "ResNet-18\n1 sentence x 160\n(3 seeds)",
          "ResNet-18, 2 x 80": "ResNet-18\n2 sentences x 80\n(3 seeds)",
          "ResNet-18 (CNN)": "ResNet-18\n4 sentences x 40\n(3 seeds)",
          "Swin-T (transformer)": "Swin-T\n4 sentences x 40\n(3 seeds)"}
SEEDS = [42, 43, 44]
COLOURS = {"ResNet-18, 1 x 160": "#a6bddb", "ResNet-18, 2 x 80": "#74a9cf",
           "ResNet-18 (CNN)": "#4c72b0", "Swin-T (transformer)": "#dd8452"}
METRICS = {"accuracy": "Test accuracy", "f1_macro": "Test macro-F1"}
BENCH = {"ResNet-18 (CNN)": ("resnet18", "s4x40_reg"), "Swin-T (transformer)": ("swin_t", "swin_t_seed42")}
SLO = {"GPU": 10.0, "CPU": 50.0}  # p95 latency targets from Milestone 2


@torch.no_grad()
def timings(model, x, reps, cuda):
    with torch.autocast("cuda", enabled=cuda):
        for _ in range(5):
            model(x)
        out = []
        for _ in range(reps):
            if cuda:
                torch.cuda.synchronize()
            t = time.perf_counter()
            model(x)
            if cuda:
                torch.cuda.synchronize()
            out.append((time.perf_counter() - t) * 1000)
    return out


def benchmark(runs_dir, rounds=3):
    """Latency / throughput of both trained models, measured back to back in alternating rounds so both
    see the same machine state (this laptop throttles); FLOPs and parameters per image."""
    models = {}
    for name, (backbone, run) in BENCH.items():
        m = FreshnessNet(0.5, backbone)
        m.load_state_dict(torch.load(runs_dir / run / "best.pt", map_location="cpu"))
        models[name] = m.eval()
    res = {n: {"gpu_bs1": [], "gpu_bs64": [], "cpu_bs1": []} for n in models}
    configs = [("gpu_bs1", "cuda", 1, 100), ("gpu_bs64", "cuda", 64, 20), ("cpu_bs1", "cpu", 1, 30)]
    for _ in range(rounds):
        for key, dev, bs, reps in configs:
            if dev == "cuda" and not torch.cuda.is_available():
                continue
            x = torch.randn(bs, 3, 224, 224, device=dev)
            for name, m in models.items():
                res[name][key] += timings(m.to(dev), x, reps, dev == "cuda")
    rows = []
    for name, m in models.items():
        m.cpu()
        with FlopCounterMode(display=False) as fc:
            with torch.no_grad():
                m(torch.randn(1, 3, 224, 224))
        r = res[name]
        rows.append({"backbone": name, "params_m": sum(p.numel() for p in m.parameters()) / 1e6,
                     "gflops_per_image": fc.get_total_flops() / 1e9,
                     "gpu_bs1_p50_ms": np.percentile(r["gpu_bs1"], 50), "gpu_bs1_p90_ms": np.percentile(r["gpu_bs1"], 90),
                     "gpu_bs64_throughput_img_s": 64 / np.percentile(r["gpu_bs64"], 50) * 1000,
                     "cpu_bs1_p50_ms": np.percentile(r["cpu_bs1"], 50), "cpu_bs1_p90_ms": np.percentile(r["cpu_bs1"], 90)})
    return pd.DataFrame(rows)


def plot_latency(bench, path):
    names = bench["backbone"].tolist()
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    for ax, dev in zip(axes[:2], ["GPU", "CPU"]):
        key = dev.lower()
        for i, (_, r) in enumerate(bench.iterrows()):
            c = COLOURS.get(r["backbone"], "#777777")
            ax.bar(i - 0.18, r[f"{key}_bs1_p50_ms"], 0.36, color=c, alpha=0.85, label="p50" if i == 0 else None)
            ax.bar(i + 0.18, r[f"{key}_bs1_p90_ms"], 0.36, color=c, alpha=0.45, hatch="//", label="p90" if i == 0 else None)
            for dx, v in [(-0.18, r[f"{key}_bs1_p50_ms"]), (0.18, r[f"{key}_bs1_p90_ms"])]:
                ax.text(i + dx, v, f"{v:.1f}", ha="center", va="bottom", fontsize=9)
        ax.axhline(SLO[dev], color="red", ls="--", lw=1.4, label=f"SLO (p95 ≤ {SLO[dev]:.0f} ms)")
        ax.set_xticks(range(len(names)), names)
        ax.set_ylabel("Latency per image (ms)")
        ax.set_title(f"{dev} latency, batch 1", fontsize=10)
        ax.set_ylim(0, max(SLO[dev], bench[f"{key}_bs1_p90_ms"].max()) * 1.2)
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(axis="y", alpha=0.3)
    ax = axes[2]
    for i, (_, r) in enumerate(bench.iterrows()):
        ax.bar(i, r["gpu_bs64_throughput_img_s"], 0.55, color=COLOURS.get(r["backbone"], "#777777"), alpha=0.85)
        ax.text(i, r["gpu_bs64_throughput_img_s"], f"{r['gpu_bs64_throughput_img_s']:,.0f} img/s\n"
                f"{r['gflops_per_image']:.1f} GFLOPs, {r['params_m']:.1f}M params", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(range(len(names)), names)
    ax.set_ylabel("Images per second")
    ax.set_title("GPU throughput, batch 64", fontsize=10)
    ax.set_ylim(0, bench["gpu_bs64_throughput_img_s"].max() * 1.3)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def plot_accuracy(plot_data, out):
    """Test accuracy / macro-F1 bars (mean, std error bar and one dot per seed) from plot_data.csv columns:
    setup, label (' | ' = line break), colour, seed, accuracy, f1_macro. Setups appear in file order."""
    setups = list(dict.fromkeys(plot_data["setup"]))

    def panel(ax, metric):
        for i, setup in enumerate(setups):
            d = plot_data[plot_data["setup"] == setup]
            mean = d[metric].mean()
            std = d[metric].std() if len(d) > 1 else None
            ax.bar(i, mean, 0.6, yerr=std, capsize=6, color=d["colour"].iloc[0], alpha=0.9)
            if len(d) > 1:
                xs = np.linspace(i - 0.14, i + 0.14, len(d))
                ax.scatter(xs, d[metric], color="black", zorder=3, s=20)
                for x, (s, v) in zip(xs, zip(d["seed"], d[metric])):
                    ax.annotate(f"s{s}", (x, v), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=7)
        lo = max(0, plot_data[metric].min() - 0.06)
        for i, setup in enumerate(setups):  # values at the foot of each bar, clear of the seed dots
            d = plot_data[plot_data["setup"] == setup]
            std = d[metric].std() if len(d) > 1 else None
            ax.text(i, lo + 0.004, f"{d[metric].mean():.3f}" + (f"\n± {std:.3f}" if std is not None else ""),
                    ha="center", va="bottom", fontsize=9.5, color="white", weight="bold")
        labels = [plot_data[plot_data["setup"] == s]["label"].iloc[0].replace(" | ", "\n") for s in setups]
        ax.set_xticks(range(len(setups)), labels, fontsize=8.5)
        ax.set_ylim(lo, min(1, plot_data[metric].max() + 0.025))
        ax.set_title(f"{METRICS[metric]} (mean ± std over seeds)", fontsize=10)
        ax.set_ylabel(METRICS[metric])
        ax.grid(axis="y", alpha=0.3)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    for ax, metric in zip(axes, METRICS):
        panel(ax, metric)
    plt.tight_layout()
    plt.savefig(out / "backbone_comparison.png", dpi=120)
    plt.close()
    for metric, name in [("accuracy", "test_accuracy.png"), ("f1_macro", "test_f1.png")]:
        fig, ax = plt.subplots(figsize=(8, 5))
        panel(ax, metric)
        plt.tight_layout()
        plt.savefig(out / name, dpi=120)
        plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot-only", action="store_true",
                    help="redraw all graphs from runs/backbones/plot_data.csv and latency.csv (e.g. after editing them)")
    args = ap.parse_args()
    runs_dir = Path("runs")
    out = runs_dir / "backbones"
    out.mkdir(exist_ok=True)
    if args.plot_only:
        plot_accuracy(pd.read_csv(out / "plot_data.csv"), out)
        if (out / "latency.csv").exists():
            plot_latency(pd.read_csv(out / "latency.csv"), out / "latency.png")
        return
    rows = []
    for backbone, runs in RUNS.items():
        for seed, run in zip(SEEDS, runs):
            m = json.loads((runs_dir / run / "metrics.json").read_text())
            t, e = m["test"], m["efficiency"]
            rows.append({"backbone": backbone, "seed": seed, "run": run, "accuracy": t["accuracy"],
                         "f1_macro": t["f1_macro"], "auroc": t["auroc_macro_ovr"],
                         **{f"f1_{c}": v["f1"] for c, v in t["per_class"].items()},
                         "params_m": e["params_total"] / 1e6, "train_time_s": e["train_time_s"],
                         "epochs_run": e["epochs_run"], "peak_train_vram_mb": e["peak_gpu_memory_mb"],
                         "gpu_bs1_latency_ms": e["cuda_batch1_latency_ms"],
                         "gpu_bs64_throughput_img_s": e["cuda_batch64_throughput_img_s"]})
    df = pd.DataFrame(rows)
    df.to_csv(out / "comparison.csv", index=False)
    summary = df.drop(columns=["seed", "run"]).groupby("backbone", sort=False).agg(["mean", "std"])
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary.to_csv(out / "summary.csv")

    # Editable plot data: one row per setup and seed; --plot-only redraws the graphs from this file
    plot_data = pd.DataFrame({"setup": df["backbone"], "label": df["backbone"].map(LABELS).str.replace("\n", " | "),
                              "colour": df["backbone"].map(COLOURS), "seed": df["seed"],
                              "accuracy": df["accuracy"], "f1_macro": df["f1_macro"]})
    plot_data.to_csv(out / "plot_data.csv", index=False)
    plot_accuracy(plot_data, out)

    bench = benchmark(runs_dir)
    bench.to_csv(out / "latency.csv", index=False)
    plot_latency(bench, out / "latency.png")

    pd.set_option("display.width", 200, "display.max_columns", 30)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\n" + summary.T.to_string(float_format=lambda v: f"{v:.4f}"))
    print("\n" + bench.to_string(index=False, float_format=lambda v: f"{v:.2f}"))


if __name__ == "__main__":
    main()
