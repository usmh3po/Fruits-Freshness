"""
End-to-end demo of the deployed system (about 1-2 minutes on a GPU):

  1. Load    the robust ResNet-18 (v1.2), fit its temperature and the drift monitor
  2. Infer   on a clean incoming batch: grade, or route low-confidence images to manual inspection
  3. Drift   a new batch arrives from a cheaper camera (blur + noise + JPEG); the monitor compares its
             OOD-gate rejection rate and predicted class shares with the clean baseline
  4. Query   if drift is detected, half of the labelling budget goes to the images the model is least
             sure about (flagged first), half to a random sample (keeps the labelled set representative);
             labels come from a "human" (simulated with the dataset labels; --interactive asks you instead)
  5. Update  class-balanced replay fine-tune (layer4 + heads), release gate, before/after evaluation

Needs: Processed Data/, runs/split_seed42.csv and runs/s4x40_robust/ (best.pt, selection.csv).
Writes to demo_output/: demo_report.json, queried_images.csv, demo_summary.png, model_updated.pt

Usage:  python demo.py                 # simulated annotator
        python demo.py --interactive   # you label the queried images (f / s / r)
"""

import argparse
import copy
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from adapt import camera_drift
from calibration import ece, fit_temperature, load_model, predict_logits
from corruptions import augment
from eda import FRESHNESS_ORDER
from feature_ood import Mahalanobis, features_and_logits
from robust_inference import corrupt_set
from train import FRUITS, GPUNormalize, freshness_metrics, load_images, run_epoch, set_stage, train_mode

CONFIDENCE_MIN = 0.54  # abstention threshold of the deployed pipeline (Milestone 3)
CLEAN_TOLERANCE = 0.02  # release gate: max clean macro-F1 drop
KEYS = {"f": "Fresh", "s": "Semi Fresh", "r": "Rotten"}


def step(n, title):
    print(f"\n[{n}] {title}\n" + "-" * (len(title) + 4))


def metrics(model, X, y, device, T):
    logits = predict_logits(model, X, device)
    m = freshness_metrics(y, logits.softmax(1).numpy())
    return {"accuracy": m["accuracy"], "f1_macro": m["f1_macro"], "recall_semi": m["per_class"]["Semi Fresh"]["recall"],
            "rotten_to_fresh": m["confusion_matrix"][2][0], "ece": ece((logits / T).softmax(1).numpy(), y)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="s4x40_robust", help="deployed model (runs/<run>/best.pt)")
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--data", default="Processed Data")
    ap.add_argument("--out", default="demo_output")
    ap.add_argument("--batch", type=int, default=600, help="images per incoming batch")
    ap.add_argument("--budget", type=int, default=300, help="images sent to the human per update")
    ap.add_argument("--eval-per-class", type=int, default=0, help="held-out test images per class (0 = whole test set)")
    ap.add_argument("--replay", type=int, default=600, help="replayed earlier training images per update")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--interactive", action="store_true", help="label the queried images yourself")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    runs_dir, data_dir, out = Path(args.runs_dir), Path(args.data), Path(args.out)
    out.mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    report = {"device": str(device), "model": args.run}
    t_start = time.perf_counter()

    split = pd.read_csv(runs_dir / f"split_seed{args.seed}.csv")
    selected = pd.read_csv(runs_dir / args.run / "selection.csv")
    enc = lambda d: d["freshness"].map(FRESHNESS_ORDER.index).to_numpy()
    load = lambda d: load_images(data_dir, d["path"], args.workers)

    # 1. Load the deployed model, its temperature and the drift monitor -----------------------------
    step(1, "Load model and monitor")
    model = load_model(runs_dir / args.run / "best.pt", device)
    val = split[split["split"] == "val"].groupby("freshness").sample(200, random_state=args.seed)
    Xv, yv = load(val), enc(val)
    T = fit_temperature(predict_logits(model, Xv, device), yv)
    reference = selected.groupby("freshness").sample(300, random_state=args.seed)
    X_ref = load(reference)
    maha = Mahalanobis().fit(features_and_logits(model, X_ref, device)[0], enc(reference)).to(device)
    fv, lv = features_and_logits(model, Xv, device)
    maha_max = float(np.percentile(maha.score(fv), 98))
    base_share = np.bincount(lv.argmax(1).numpy(), minlength=3) / len(lv)
    print(f"Loaded runs/{args.run}/best.pt on {device}; temperature {T:.2f}; OOD threshold = 98th pct of clean val")

    def monitor(X):
        f, logits = features_and_logits(model, X, device)
        probs = (logits / T).softmax(1).numpy()
        share = np.bincount(probs.argmax(1), minlength=3) / len(probs)
        ood = maha.score(f) > maha_max
        return probs, ood, {"ood_rate": float(ood.mean()), "class_share_shift": float(np.abs(share - base_share).sum() / 2),
                            "mean_confidence": float(probs.max(1).mean()),
                            "flagged_for_manual_inspection": float(((probs.max(1) < CONFIDENCE_MIN) | ood).mean())}

    # 2. Inference on a clean incoming batch -------------------------------------------------------
    step(2, "Inference on a clean batch")
    unused = split[(split["split"] == "train") & ~split["path"].isin(selected["path"])]
    clean_batch = unused.sample(args.batch, random_state=args.seed)
    probs, ood, clean_stats = monitor(load(clean_batch))
    for p, pr, o in list(zip(clean_batch["path"], probs, ood))[:5]:
        decision = "MANUAL INSPECTION" if (pr.max() < CONFIDENCE_MIN or o) else f"{FRESHNESS_ORDER[pr.argmax()]} ({pr.max():.2f})"
        print(f"  {p:60s} -> {decision}")
    print(f"  ... {args.batch} images: {clean_stats['flagged_for_manual_inspection']:.1%} routed to manual inspection, "
          f"OOD rate {clean_stats['ood_rate']:.1%}")
    report["clean_batch"] = clean_stats

    # 3. A drifted batch arrives (cheaper camera); the monitor decides whether to update -----------
    step(3, "Drift detection on a new batch (cheaper camera)")
    drift_batch = unused.drop(clean_batch.index).groupby("freshness").sample(args.batch // 3, random_state=args.seed)
    X_drift = corrupt_set(load(drift_batch), camera_drift, args.seed)
    probs_d, ood_d, drift_stats = monitor(X_drift)
    triggered = drift_stats["ood_rate"] > 2 * max(clean_stats["ood_rate"], 0.02) or drift_stats["class_share_shift"] > 0.10
    print(f"  OOD rate {clean_stats['ood_rate']:.1%} (clean) -> {drift_stats['ood_rate']:.1%} (new batch); "
          f"class-share shift {drift_stats['class_share_shift']:.3f}; "
          f"manual inspection {drift_stats['flagged_for_manual_inspection']:.1%}")
    print(f"  Drift detected: {'YES -> query labels and update' if triggered else 'no -> keep collecting'}")
    report["drift_batch"] = {**drift_stats, "update_triggered": bool(triggered)}

    test = split[split["split"] == "test"]
    if args.eval_per_class:
        test = test.groupby("freshness").sample(args.eval_per_class, random_state=args.seed)
    X_test, y_test = load(test), enc(test)
    X_test_drift = corrupt_set(X_test, camera_drift, args.seed + 1)
    before = {"clean": metrics(model, X_test, y_test, device, T), "drift": metrics(model, X_test_drift, y_test, device, T)}
    report["before_update"] = before
    final_model, after = model, before

    if triggered:
        # 4. Query the human: flagged images first, then the least confident ones -------------------
        step(4, f"Query the human for {args.budget} labels")
        conf = probs_d.max(1)
        flagged = (conf < CONFIDENCE_MIN) | ood_d
        ranked = np.lexsort((conf, ~flagged))  # flagged first, then lowest confidence
        uncertain = ranked[:args.budget // 2]
        rest = np.setdiff1d(np.arange(len(conf)), uncertain)
        order = np.r_[uncertain, rng.choice(rest, args.budget - len(uncertain), replace=False)]
        query = drift_batch.iloc[order].copy()
        if args.interactive:
            labels = []
            for p in query["path"]:
                ans = ""
                while ans not in KEYS:
                    ans = input(f"  {data_dir / p}  -> fresh / semi / rotten [f/s/r]: ").strip().lower()[:1]
                labels.append(KEYS[ans])
            query["human_label"] = labels
        else:
            query["human_label"] = query["freshness"]  # simulated annotator: the dataset label
        query["model_confidence"] = conf[order]
        query["was_flagged"] = flagged[order]
        query["selected_as"] = ["least confident"] * len(uncertain) + ["random"] * (len(order) - len(uncertain))
        query[["path", "human_label", "selected_as", "model_confidence", "was_flagged"]].to_csv(out / "queried_images.csv", index=False)
        print(f"  {len(uncertain)} least-confident ({int(flagged[uncertain].sum())} of them flagged) + "
              f"{len(order) - len(uncertain)} random images labelled ({'by you' if args.interactive else 'simulated annotator'}); "
              f"labelled class mix: {query['human_label'].value_counts().to_dict()}")

        # 5. Update: replay fine-tune, release gate, before / after ---------------------------------
        step(5, "Replay update and release gate")
        X_new = X_drift[order]
        replay = selected.groupby("freshness").sample(args.replay // 3, random_state=args.seed + 1)
        X = torch.cat([X_new, load(replay)])
        yf = torch.as_tensor(np.r_[query["human_label"].map(FRESHNESS_ORDER.index).to_numpy(), enc(replay)])
        yr = torch.as_tensor(np.r_[query["fruit"].map(FRUITS.index).to_numpy(), replay["fruit"].map(FRUITS.index).to_numpy()])
        candidate = copy.deepcopy(model)
        opt = torch.optim.AdamW(set_stage(candidate, 2), lr=1e-4, weight_decay=0.05)
        scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
        cfg = {"batch_size": 32, "label_smoothing": 0.2, "fruit_loss_weight": 0.5, "amp": device.type == "cuda",
               "augment": augment}
        gen = torch.Generator().manual_seed(args.seed)
        t0 = time.perf_counter()
        for _ in range(args.epochs):
            train_mode(candidate, 2)
            run_epoch(candidate, X, yf, yr, cfg, GPUNormalize(device), device, opt, scaler, gen)
        candidate.eval()
        update_time = time.perf_counter() - t0
        T_new = fit_temperature(predict_logits(candidate, Xv, device), yv)
        after = {"clean": metrics(candidate, X_test, y_test, device, T_new),
                 "drift": metrics(candidate, X_test_drift, y_test, device, T_new)}
        gate = bool(after["drift"]["f1_macro"] >= before["drift"]["f1_macro"]
                    and after["clean"]["f1_macro"] >= before["clean"]["f1_macro"] - CLEAN_TOLERANCE)
        print(f"  Update: {len(X)} images ({args.budget} new + {len(replay)} replay), {update_time:.1f} s")
        for name in ["drift", "clean"]:
            b, a = before[name], after[name]
            print(f"  {name:5s} test: accuracy {b['accuracy']:.1%} -> {a['accuracy']:.1%}, "
                  f"macro-F1 {b['f1_macro']:.1%} -> {a['f1_macro']:.1%}")
        print(f"  Release gate: {'PASSED -> new version deployed' if gate else 'FAILED -> keep current model'}")
        report.update({"labels_queried": args.budget, "update_time_s": update_time, "after_update": after,
                       "release_gate_passed": gate})
        if gate:
            final_model = candidate
            torch.save(candidate.state_dict(), out / "model_updated.pt")

    # Summary figure and report --------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, metric, title in [(axes[0], "accuracy", "Test accuracy"), (axes[1], "f1_macro", "Test macro-F1")]:
        x = np.arange(2)
        ax.bar(x - 0.18, [before[s][metric] for s in ["clean", "drift"]], 0.36, label="before update", color="#8c8c8c")
        ax.bar(x + 0.18, [after[s][metric] for s in ["clean", "drift"]], 0.36, label="after update", color="#4c72b0")
        for i, s in enumerate(["clean", "drift"]):
            for dx, v in [(-0.18, before[s][metric]), (0.18, after[s][metric])]:
                ax.text(i + dx, v, f"{v:.1%}", ha="center", va="bottom", fontsize=9)
        ax.set_xticks(x, ["Clean images", "New camera (drift)"])
        ax.set_ylim(0.5, 1.0)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(fontsize=8, loc="lower left")
    plt.suptitle(f"Demo: drift -> {args.budget} human labels -> replay update "
                 f"({'deployed' if final_model is not model else 'not deployed'})", fontsize=10)
    plt.tight_layout()
    plt.savefig(out / "demo_summary.png", dpi=110)
    plt.close()
    report["total_time_s"] = time.perf_counter() - t_start
    (out / "demo_report.json").write_text(json.dumps(report, indent=2, default=float))
    print(f"\nDone in {report['total_time_s']:.0f} s. Outputs in {out}/: demo_report.json, demo_summary.png"
          + (", queried_images.csv" if triggered else "") + (", model_updated.pt" if final_model is not model else ""))


if __name__ == "__main__":
    main()
