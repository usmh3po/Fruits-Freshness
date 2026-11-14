"""
Robustness evaluation: original (v1.1) vs. robust (v1.2) model, three training seeds each.

For every model: temperature fitted on the validation set, then the Milestone-2 metric families
(accuracy, per-class P/R/F1, macro/micro F1, AUROC, PR-AUC) plus ECE on the clean test set and under
every corruption in corruptions.CORRUPTIONS. 95% bootstrap CIs for the main seed.

Writes to runs/robustness/:
  corruption_seeds.csv      one row per model, seed and condition
  corruption_summary.csv    mean / std over seeds, bootstrap CI for seed 42
  temperatures.json         fitted temperature and val/test ECE per model
  robustness_curves.png     accuracy vs. severity per corruption family (mean +- std over seeds)
  adversarial_curves.png    robust accuracy vs. epsilon (needs adversarial.py first)
  reliability.png           reliability diagrams before / after temperature scaling
  confidence_histograms.png clean correct / wrong, noisy, OOD and adversarial confidence

Usage:  python robustness_eval.py
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from adversarial import adversarial_probs, stratified_subset
from calibration import ece, fit_temperature, load_model, predict_logits, reliability_diagram
from corruptions import CORRUPTIONS, noise
from eda import FRESHNESS_ORDER
from stress_test import load_cifar_ood
from train import freshness_metrics, load_images

FAMILIES = {"Original (v1.1)": ["s4x40_reg", "s4x40_reg_seed43", "s4x40_reg_seed44"],
            "Robust (v1.2)": ["s4x40_robust", "s4x40_robust_seed43", "s4x40_robust_seed44"]}
SEEDS = [42, 43, 44]
COLOURS = {"Original (v1.1)": "#4c72b0", "Robust (v1.2)": "#dd8452"}


def fast_acc_f1(y, pred, k=3):
    cm = np.bincount(y * k + pred, minlength=k * k).reshape(k, k)
    tp = np.diag(cm)
    f1 = np.divide(2 * tp, cm.sum(0) + cm.sum(1), out=np.zeros(k), where=(cm.sum(0) + cm.sum(1)) > 0)
    return tp.sum() / len(y), f1.mean()


def bootstrap_ci(y, pred, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    stats = np.array([fast_acc_f1(y[i], pred[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(n))])
    lo, hi = np.percentile(stats, [2.5, 97.5], axis=0)
    return {"acc_ci_low": lo[0], "acc_ci_high": hi[0], "f1_ci_low": lo[1], "f1_ci_high": hi[1]}


def flat_metrics(probs, y):
    m = freshness_metrics(y, probs)
    row = {k: m[k] for k in ["accuracy", "f1_macro", "f1_micro", "auroc_macro_ovr", "pr_auc_macro"]}
    for c, v in m["per_class"].items():
        row.update({f"precision_{c}": v["precision"], f"recall_{c}": v["recall"], f"f1_{c}": v["f1"]})
    pred = probs.argmax(1)
    row["rotten_to_fresh"] = int(((y == 2) & (pred == 0)).sum())
    return row


def plot_robustness_curves(summary, path):
    groups = [g for g in dict.fromkeys(c[1] for c in CORRUPTIONS) if g not in ("none", "Feature loss")]
    fig, axes = plt.subplots(2, 5, figsize=(20, 7.5))
    axes = axes.ravel()
    clean = summary[summary["condition"] == "Clean (baseline)"]
    for ax, group in zip(axes, groups):
        for fam, d in summary[summary["group"] == group].groupby("family", sort=False):
            c0 = clean[clean["family"] == fam].iloc[0]
            clean_x = 100 if group == "JPEG quality" else 0
            pts = pd.concat([d, c0.to_frame().T.assign(severity=clean_x)]).astype({"severity": float}).sort_values("severity")
            if group == "JPEG quality":
                pts = pts.iloc[::-1]
            ax.errorbar(pts["severity"], pts["accuracy_mean"].astype(float), yerr=pts["accuracy_std"].astype(float),
                        marker="o", capsize=3, label=fam, color=COLOURS[fam])
        ax.set_title(group)
        ax.set_ylim(0.3, 0.95)
        ax.grid(alpha=0.3)
        ax.set_xlabel({"Brightness": "change (%)", "Contrast": "change (%)", "Gaussian noise": "sigma",
                       "Gaussian blur": "sigma (px)", "JPEG quality": "quality (100 = clean)",
                       "Mask (centre)": "area masked", "Mask (random)": "area masked",
                       "Patch dropout": "patches dropped"}[group])
        if group == "JPEG quality":
            ax.invert_xaxis()
    ax = axes[len(groups)]
    feat = summary[summary["condition"].isin(["Clean (baseline)", "Grayscale", "Low resolution 64px"])]
    names = ["Clean (baseline)", "Grayscale", "Low resolution 64px"]
    for i, (fam, d) in enumerate(feat.groupby("family", sort=False)):
        d = d.set_index("condition").loc[names]
        ax.bar(np.arange(3) + (i - 0.5) * 0.38, d["accuracy_mean"].astype(float), 0.38,
               yerr=d["accuracy_std"].astype(float), capsize=3, color=COLOURS[fam], label=fam)
    ax.set_xticks(range(3), ["Clean", "Grayscale", "Low-res 64px"])
    ax.set_ylim(0.3, 0.95)
    ax.set_title("Feature loss")
    ax.grid(axis="y", alpha=0.3)
    for ax in axes[len(groups) + 1:]:
        ax.axis("off")
    axes[0].set_ylabel("Test accuracy")
    axes[5].set_ylabel("Test accuracy")
    axes[0].legend(loc="lower left", fontsize=9)
    fig.suptitle("Test accuracy under corruption (mean +- std over 3 training seeds)")
    plt.tight_layout()
    plt.savefig(path, dpi=110)
    plt.close()


def plot_adversarial_curves(adv, path):
    attacks = [a for a in adv["attack"].unique() if "patch" not in a.lower()]
    fig, axes = plt.subplots(1, len(attacks), figsize=(5 * len(attacks), 4))
    for ax, a in zip(axes, attacks):
        for (model, fam) in [("s4x40_reg", "Original (v1.1)"), ("s4x40_robust", "Robust (v1.2)")]:
            d = adv[(adv["attack"] == a) & (adv["model"] == model)].sort_values("epsilon_255")
            col = "non_fresh_predicted_fresh" if "targeted" in a else "accuracy"
            ax.plot(d["epsilon_255"].astype(str), d[col], marker="o", label=fam, color=COLOURS[fam])
        ax.set_title(a, fontsize=10)
        ax.set_xlabel("epsilon (/255)")
        ax.set_ylabel("Non-fresh predicted Fresh" if "targeted" in a else "Robust accuracy")
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(path, dpi=110)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--data", default="Processed Data")
    ap.add_argument("--ood-dir", default="ood_data")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    runs_dir = Path(args.runs_dir)
    out = runs_dir / "robustness"
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    split = pd.read_csv(runs_dir / f"split_seed{args.seed}.csv")
    data = {}
    for name in ["val", "test"]:
        d = split[split["split"] == name]
        data[name] = (load_images(Path(args.data), d["path"], args.workers), d["freshness"].map(FRESHNESS_ORDER.index).to_numpy())
    (Xv, yv), (Xt, yt) = data["val"], data["test"]

    rows, temps, main_probs = [], {}, {}
    for fam, runs in FAMILIES.items():
        for seed, run in zip(SEEDS, runs):
            model = load_model(runs_dir / run / "best.pt", device)
            lv = predict_logits(model, Xv, device)
            T = fit_temperature(lv, yv)
            temps[run] = {"family": fam, "seed": seed, "temperature": T,
                          "val_ece_raw": ece(lv.softmax(1).numpy(), yv), "val_ece_calibrated": ece((lv / T).softmax(1).numpy(), yv)}
            for name, group, sev, fn in CORRUPTIONS:
                lt = predict_logits(model, Xt, device, fn, args.seed)
                p_raw, p_cal = lt.softmax(1).numpy(), (lt / T).softmax(1).numpy()
                rows.append({"family": fam, "run": run, "seed": seed, "condition": name, "group": group, "severity": sev,
                             **flat_metrics(p_raw, yt), "ece_raw": ece(p_raw, yt), "ece_calibrated": ece(p_cal, yt),
                             "mean_conf_raw": p_raw.max(1).mean(), "mean_conf_calibrated": p_cal.max(1).mean()})
                if seed == args.seed:
                    main_probs[(fam, name)] = (p_raw, p_cal)
            print(f"{run:22s} T={T:.3f}  clean acc {rows[-len(CORRUPTIONS)]['accuracy']:.4f}")
            del model
    per_seed = pd.DataFrame(rows)
    per_seed.to_csv(out / "corruption_seeds.csv", index=False)
    (out / "temperatures.json").write_text(json.dumps(temps, indent=2))

    # Mean / std over seeds + bootstrap CI of the main seed
    metric_cols = [c for c in per_seed.columns if c not in ("family", "run", "seed", "condition", "group", "severity")]
    agg = per_seed.groupby(["family", "condition"], sort=False)[metric_cols].agg(["mean", "std"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    agg = agg.reset_index()
    meta = per_seed.drop_duplicates(["family", "condition"])[["family", "condition", "group", "severity"]]
    summary = meta.merge(agg, on=["family", "condition"])
    cis = [{"family": fam, "condition": name, **bootstrap_ci(yt, p[0].argmax(1))} for (fam, name), p in main_probs.items()]
    summary = summary.merge(pd.DataFrame(cis), on=["family", "condition"])
    summary.to_csv(out / "corruption_summary.csv", index=False)

    plot_robustness_curves(summary, out / "robustness_curves.png")
    if (out / "adversarial.csv").exists():
        plot_adversarial_curves(pd.read_csv(out / "adversarial.csv"), out / "adversarial_curves.png")

    # Reliability diagrams (clean test)
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.4))
    for ax, (fam, i, label) in zip(axes, [("Original (v1.1)", 0, "raw"), ("Original (v1.1)", 1, "temperature-scaled"),
                                         ("Robust (v1.2)", 0, "raw"), ("Robust (v1.2)", 1, "temperature-scaled")]):
        reliability_diagram(ax, main_probs[(fam, "Clean (baseline)")][i], yt, f"{fam}, {label}")
    plt.tight_layout()
    plt.savefig(out / "reliability.png", dpi=110)
    plt.close()

    # Confidence histograms (temperature-scaled): clean correct / wrong, noise, OOD, adversarial
    ood_X = torch.cat([x for x, _ in load_cifar_ood(args.ood_dir, 1000, args.seed).values()])
    adv_sub = stratified_subset(split[split["split"] == "test"], 167, args.seed)
    X_adv = load_images(Path(args.data), adv_sub["path"], args.workers)
    y_adv = adv_sub["freshness"].map(FRESHNESS_ORDER.index).to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), sharey=True)
    bins = np.linspace(1 / 3, 1, 26)
    for ax, (fam, run) in zip(axes, [("Original (v1.1)", "s4x40_reg"), ("Robust (v1.2)", "s4x40_robust")]):
        model = load_model(runs_dir / run / "best.pt", device)
        for p in model.parameters():
            p.requires_grad_(False)
        T = temps[run]["temperature"]
        p_clean = main_probs[(fam, "Clean (baseline)")][1]
        ok = p_clean.argmax(1) == yt
        cal = lambda logits: (logits / T).softmax(1).numpy().max(1)
        p_adv = adversarial_probs(model, model, X_adv, y_adv, 2 / 255, 10, False, device, args.seed)
        p_adv_conf = (torch.log(torch.from_numpy(p_adv)) / T).softmax(1).numpy().max(1)
        groups = {"Clean, correct": p_clean.max(1)[ok], "Clean, wrong": p_clean.max(1)[~ok],
                  "Gaussian noise sigma=0.1": cal(predict_logits(model, Xt, device, noise(0.1), args.seed)),
                  "OOD (CIFAR-100)": cal(predict_logits(model, ood_X, device)),
                  "PGD-10 eps=2/255": p_adv_conf}
        for label, conf in groups.items():
            ax.hist(conf, bins=bins, histtype="step", lw=1.8, density=True, label=f"{label} (n={len(conf)})")
        ax.set_title(f"{fam}, temperature-scaled (T={T:.2f})")
        ax.set_xlabel("Max softmax confidence")
        ax.grid(alpha=0.3)
        del model
    axes[0].set_ylabel("Density")
    axes[1].legend(fontsize=8, loc="upper left")
    plt.tight_layout()
    plt.savefig(out / "confidence_histograms.png", dpi=110)
    plt.close()

    show = summary[summary["condition"].isin(["Clean (baseline)", "Gaussian noise sigma=0.1", "Mask 20% centre", "Grayscale"])]
    print(show[["family", "condition", "accuracy_mean", "accuracy_std", "f1_macro_mean", "acc_ci_low", "acc_ci_high",
                "ece_raw_mean", "ece_calibrated_mean"]].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nWritten to {out}/")


if __name__ == "__main__":
    main()
