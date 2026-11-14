"""
Adaptation experiment: simulated camera drift -> small fine-tune -> before / after metrics.

Drift: the deployment camera changes to a low-cost one (blur sigma=1, noise sigma=0.08, JPEG q30),
applied to the test set. Update: fine-tune layer4 + heads for 5 epochs on 600 newly labelled drifted
images from the unused part of the train split, mixed with 600 replayed images from the original
training selection (to limit forgetting). Done for the original and the robust model.

Writes runs/robustness/adaptation.csv

Usage:  python adapt.py
"""

import argparse
import copy
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from calibration import load_model, predict_logits
from corruptions import blur, jpeg, noise
from eda import FRESHNESS_ORDER
from robust_inference import corrupt_set, time_fn
from train import FRUITS, GPUNormalize, freshness_metrics, load_images, run_epoch, set_stage, train_mode


def camera_drift(x, g):
    return jpeg(30)(noise(0.08)(blur(1)(x, g), g), g)


def labels(df):
    return (torch.tensor(df["freshness"].map(FRESHNESS_ORDER.index).to_numpy()),
            torch.tensor(df["fruit"].map(FRUITS.index).to_numpy()))


def evaluate(model, X, y, device):
    m = freshness_metrics(y, predict_logits(model, X, device).softmax(1).numpy())
    return {"accuracy": m["accuracy"], "f1_macro": m["f1_macro"], "auroc": m["auroc_macro_ovr"],
            "recall_semi": m["per_class"]["Semi Fresh"]["recall"],
            "rotten_to_fresh": m["confusion_matrix"][2][0]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=["s4x40_reg", "s4x40_robust"])
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--data", default="Processed Data")
    ap.add_argument("--n-new", type=int, default=600, help="newly labelled drifted images")
    ap.add_argument("--n-replay", type=int, default=600, help="replayed original training images")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    runs_dir, data_dir = Path(args.runs_dir), Path(args.data)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    split = pd.read_csv(runs_dir / f"split_seed{args.seed}.csv")
    test = split[split["split"] == "test"]
    Xt = load_images(data_dir, test["path"], args.workers)
    yt = test["freshness"].map(FRESHNESS_ORDER.index).to_numpy()
    Xt_drift = corrupt_set(Xt, camera_drift, args.seed)

    rows = []
    for run in args.runs:
        selected = pd.read_csv(runs_dir / run / "selection.csv")
        unused = split[(split["split"] == "train") & ~split["path"].isin(selected["path"])]
        new = unused.groupby("freshness").sample(args.n_new // 3, random_state=args.seed)
        replay = selected.sample(args.n_replay, random_state=args.seed)
        X_new = corrupt_set(load_images(data_dir, new["path"], args.workers), camera_drift, args.seed + 1)
        X = torch.cat([X_new, load_images(data_dir, replay["path"], args.workers)])
        yf, yr = [torch.cat(t) for t in zip(labels(new), labels(replay))]

        model = load_model(runs_dir / run / "best.pt", device)
        before = {"clean": evaluate(model, Xt, yt, device), "drift": evaluate(model, Xt_drift, yt, device)}

        adapted = copy.deepcopy(model)
        params = set_stage(adapted, 2)
        opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.05)
        scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
        cfg = {"batch_size": 32, "label_smoothing": 0.2, "fruit_loss_weight": 0.5, "amp": device.type == "cuda"}
        gen = torch.Generator().manual_seed(args.seed)
        t0 = time.perf_counter()
        for _ in range(args.epochs):
            train_mode(adapted, 2)
            run_epoch(adapted, X, yf, yr, cfg, GPUNormalize(device), device, opt, scaler, gen)
        adapt_time = time.perf_counter() - t0
        adapted.eval()
        after = {"clean": evaluate(adapted, Xt, yt, device), "drift": evaluate(adapted, Xt_drift, yt, device)}

        lat = {}
        for name, m in [("before", model), ("after", adapted)]:
            t = time_fn(lambda x: predict_logits(m, x, device, bs=1), Xt[:1], 100, device.type == "cuda")
            lat[name] = (np.percentile(t, 50), np.percentile(t, 90))
        for stage, res in [("before", before), ("after", after)]:
            for split_name, m in res.items():
                rows.append({"model": run, "stage": stage, "test_set": split_name, **m,
                             "latency_p50_ms": lat[stage][0], "latency_p90_ms": lat[stage][1],
                             "params": sum(p.numel() for p in model.parameters()),
                             "adapt_time_s": adapt_time if stage == "after" else 0.0})
        print(f"{run}: drift acc {before['drift']['accuracy']:.3f} -> {after['drift']['accuracy']:.3f}, "
              f"clean acc {before['clean']['accuracy']:.3f} -> {after['clean']['accuracy']:.3f} ({adapt_time:.0f}s)")
    out = pd.DataFrame(rows)
    out.to_csv(runs_dir / "robustness" / "adaptation.csv", index=False)
    print(out.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
