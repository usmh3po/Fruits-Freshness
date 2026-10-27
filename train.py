"""
Train a freshness classifier on images chosen by the VLM (CLIP) selection head.

Pipeline:
1. Split: 70/15/15 split of the ORIGINAL photos per class folder (seeded). Every augmented copy
   (aug_<n>_<original>) goes to the same split as its original, so no photo leaks across splits.
   Byte-identical originals are kept together as one group.
2. Selection: on the train split only, CLIP round-robin selection with K sentences per class and
   N images per sentence (uses the similarities from clip_selection.py).
3. Oracle labelling: selected images get their true folder labels (like a human annotator would).
   The selection precision (how often CLIP's sentence matched the true label) is stored as-is.
4. Training: ImageNet-pretrained ResNet-18 with two heads (freshness: 3 classes, fruit: 8 classes),
   two stages: (1) heads only, backbone frozen; (2) layer4 + heads, early stopping on val macro-F1.
5. Evaluation on val and test (freshness metrics) + training/inference efficiency.

Outputs in runs/<run-name>/: config.json, selection.csv, selection_precision.json, history.csv,
metrics.json, learning_curves.png, confusion_test.png, best.pt

Usage:
  python clip_selection.py                                  # once, to compute similarities
  python train.py --sentences-per-class 4 --per-sentence 40 --run-name s4x40
"""

import argparse
import hashlib
import json
import platform
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support, roc_auc_score)
from torchvision import models, transforms

from clip_selection import select_round_robin
from eda import FRESHNESS_ORDER, IMAGE_EXTS, parse_folder

CONFIG_VERSION = "v1.0"
FRUITS = ["Banana", "Bittermelon", "Cucumber", "Eggplant", "Orange", "Papaya", "Pineapple", "Tomato"]
IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


# ---------------------------------------------------------------- data split

def build_split(data_dir, seed, ratios=(0.70, 0.15, 0.15)):
    """Per class folder: shuffle the original photos (grouped by MD5) and split 70/15/15;
    augmented copies follow their original."""
    rows = []
    for folder in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        state, fruit, _, _ = parse_folder(folder.name)
        for f in sorted(folder.iterdir()):
            if f.suffix.lower() not in IMAGE_EXTS:
                continue
            m = re.match(r"^aug_\d+_(.+)$", f.name)
            rows.append({"path": f"{folder.name}/{f.name}", "folder": folder.name,
                         "freshness": state, "fruit": fruit, "is_augmented": bool(m),
                         "source": f"{folder.name}/{m.group(1) if m else f.name}"})
    df = pd.DataFrame(rows)

    # Identical original files form one group, so a duplicated photo can't be in two splits
    orig = df[~df["is_augmented"]].copy()
    orig["md5"] = [hashlib.md5((data_dir / p).read_bytes()).hexdigest() for p in orig["path"]]
    group_of_source = dict(zip(orig["path"], orig["folder"] + "/" + orig["md5"]))
    df["group"] = df["source"].map(group_of_source)

    rng = np.random.default_rng(seed)
    split_of_group = {}
    for _, g in df[~df["is_augmented"]].groupby("folder"):
        groups = g["group"].unique()
        rng.shuffle(groups)
        n_train, n_val = round(len(groups) * ratios[0]), round(len(groups) * ratios[1])
        for i, grp in enumerate(groups):
            split_of_group[grp] = "train" if i < n_train else "val" if i < n_train + n_val else "test"
    df["split"] = df["group"].map(split_of_group)
    return df


# ---------------------------------------------------------------- selection + oracle labelling

def select_training_images(split_df, sim_csv, sentences_csv, k_sentences, per_sentence):
    """CLIP round-robin selection over the train split using the first k sentences of each class."""
    sims = pd.read_csv(sim_csv)
    sentences = pd.read_csv(sentences_csv)
    sentences["rank_in_class"] = sentences.groupby("class_id").cumcount()
    used = sentences[sentences["rank_in_class"] < k_sentences].reset_index(drop=True)

    pool = split_df[split_df["split"] == "train"].merge(sims, on="path", how="left")
    if pool[f"s{used['sentence_id'].iloc[0]}"].isna().any():
        raise SystemExit("Similarities missing for some images; rerun clip_selection.py first.")
    sim = pool[[f"s{i}" for i in used["sentence_id"]]].to_numpy()

    picks = select_round_robin(sim, per_sentence)
    sel = pool.iloc[[p[1] for p in picks]][["path", "folder", "freshness", "fruit", "is_augmented"]].copy()
    sel["sentence_id"] = used["sentence_id"].to_numpy()[[p[0] for p in picks]]
    sel["sentence"] = used["sentence"].to_numpy()[[p[0] for p in picks]]
    sel["sentence_label"] = used["label"].to_numpy()[[p[0] for p in picks]]
    sel["similarity"] = sim[[p[1] for p in picks], [p[0] for p in picks]]
    # Oracle labelling: the true label is the folder label; CLIP's guess is kept for precision
    sel["true_label"] = sel["freshness"] + " " + sel["fruit"]
    sel["correct_class"] = sel["sentence_label"] == sel["true_label"]
    sel["correct_freshness"] = sel["sentence_label"].str.rsplit(" ", n=1).str[0] == sel["freshness"]
    sel["correct_fruit"] = sel["sentence_label"].str.rsplit(" ", n=1).str[-1] == sel["fruit"]
    return sel.reset_index(drop=True), len(used)


def selection_precision(sel):
    per_class = sel.groupby("sentence_label")["correct_class"].agg(["size", "sum", "mean"])
    return {
        "n_selected": int(len(sel)),
        "precision_class": float(sel["correct_class"].mean()),
        "precision_freshness": float(sel["correct_freshness"].mean()),
        "precision_fruit": float(sel["correct_fruit"].mean()),
        "per_sentence_label": {k: {"selected": int(r["size"]), "correct": int(r["sum"]),
                                   "precision": round(float(r["mean"]), 4)} for k, r in per_class.iterrows()},
        "true_freshness_share": sel["freshness"].value_counts(normalize=True).round(4).to_dict(),
        "true_fruit_share": sel["fruit"].value_counts(normalize=True).round(4).to_dict(),
        "true_class_count": {k: int(v) for k, v in sel["true_label"].value_counts().items()},
    }


# ---------------------------------------------------------------- dataset / model

PREPROCESS = transforms.Compose([transforms.Resize((256, 256)), transforms.CenterCrop(224),
                                 transforms.PILToTensor()])


def load_images(data_dir, paths, workers):
    """Resize 256x256 -> center crop 224, kept in memory as uint8 (preprocessed once, since no
    random augmentation is used). Threads instead of DataLoader worker processes, which hang
    with the Microsoft Store Python on Windows."""
    def load(p):
        with Image.open(data_dir / p) as img:
            return PREPROCESS(img.convert("RGB"))
    with ThreadPoolExecutor(workers) as pool:
        return torch.stack(list(pool.map(load, paths)))


class FreshnessNet(nn.Module):
    """ResNet-18 backbone with a freshness head (main task) and a fruit head (auxiliary task)."""

    def __init__(self, dropout):
        super().__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        feat = self.backbone.fc.in_features  # 512
        self.backbone.fc = nn.Identity()
        self.fresh_head = nn.Sequential(nn.Dropout(dropout), nn.Linear(feat, len(FRESHNESS_ORDER)))
        self.fruit_head = nn.Sequential(nn.Dropout(dropout), nn.Linear(feat, len(FRUITS)))

    def heads(self):
        return [self.fresh_head, self.fruit_head]

    def forward(self, x):
        f = self.backbone(x)
        return self.fresh_head(f), self.fruit_head(f)


def set_stage(model, stage):
    """Stage 1: only the heads train. Stage 2: layer4 + heads train. Frozen BatchNorm layers stay
    in eval mode so their ImageNet statistics are kept."""
    for p in model.parameters():
        p.requires_grad = False
    trainable = model.heads() + ([model.backbone.layer4] if stage == 2 else [])
    for m in trainable:
        for p in m.parameters():
            p.requires_grad = True
    return [p for m in trainable for p in m.parameters()]


def train_mode(model, stage):
    model.eval()
    for h in model.heads():
        h.train()
    if stage == 2:
        model.backbone.layer4.train()


# ---------------------------------------------------------------- train / evaluate

class GPUNormalize:
    def __init__(self, device):
        self.mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)

    def __call__(self, x_uint8):
        return (x_uint8.float() / 255.0 - self.mean) / self.std


def run_epoch(model, X, y_fresh, y_fruit, cfg, norm, device, optimizer=None, scaler=None, gen=None):
    """One pass over (X, y). Trains if an optimizer is given, otherwise evaluates.
    Returns total loss, freshness loss, freshness probabilities, fruit predictions."""
    training = optimizer is not None
    ce = nn.CrossEntropyLoss(label_smoothing=cfg["label_smoothing"])
    order = torch.randperm(len(X), generator=gen) if training else torch.arange(len(X))
    tot_loss = fresh_loss = 0.0
    probs, fruit_pred = [], []
    for i in range(0, len(X), cfg["batch_size"]):
        idx = order[i:i + cfg["batch_size"]]
        x = norm(X[idx].to(device, non_blocking=True))
        yf, yr = y_fresh[idx].to(device), y_fruit[idx].to(device)
        with torch.set_grad_enabled(training), torch.autocast(device.type, enabled=cfg["amp"]):
            out_f, out_r = model(x)
            lf = ce(out_f.float(), yf)
            loss = lf + cfg["fruit_loss_weight"] * ce(out_r.float(), yr)
        if training:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        tot_loss += loss.item() * len(idx)
        fresh_loss += lf.item() * len(idx)
        probs.append(out_f.float().softmax(1).detach().cpu())
        fruit_pred.append(out_r.argmax(1).cpu())
    n = len(X)
    back = order.argsort()  # undo the shuffle so outputs line up with X / y
    return tot_loss / n, fresh_loss / n, torch.cat(probs)[back].numpy(), torch.cat(fruit_pred)[back].numpy()


def freshness_metrics(y_true, probs, fruits=None):
    y_pred = probs.argmax(1)
    labels = list(range(len(FRESHNESS_ORDER)))
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    onehot = np.eye(len(labels))[y_true]
    m = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "f1_micro": f1_score(y_true, y_pred, average="micro"),
        "auroc_macro_ovr": roc_auc_score(y_true, probs, multi_class="ovr", average="macro"),
        "pr_auc_macro": average_precision_score(onehot, probs, average="macro"),
        "per_class": {c: {"precision": p[i], "recall": r[i], "f1": f[i], "support": int(s[i])}
                      for i, c in enumerate(FRESHNESS_ORDER)},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }
    if fruits is not None:
        m["accuracy_per_fruit"] = {fr: accuracy_score(y_true[fruits == fr], y_pred[fruits == fr])
                                   for fr in FRUITS if (fruits == fr).any()}
    return json.loads(json.dumps(m, default=float))


@torch.no_grad()
def measure_inference(model, device):
    """Model forward latency (preprocessing excluded), fp32."""
    model.eval()
    res = {}
    for dev in ([device] if device.type == "cpu" else [device, torch.device("cpu")]):
        mdl = model.to(dev)
        for bs, reps in ([1, 200], [64, 30]) if dev.type == "cuda" else ([1, 50],):
            x = torch.randn(bs, 3, 224, 224, device=dev)
            for _ in range(10):
                mdl(x)
            times = []
            for _ in range(reps):
                if dev.type == "cuda":
                    torch.cuda.synchronize()
                t = time.perf_counter()
                mdl(x)
                if dev.type == "cuda":
                    torch.cuda.synchronize()
                times.append(time.perf_counter() - t)
            med = float(np.median(times))
            res[f"{dev.type}_batch{bs}_latency_ms"] = med * 1000
            res[f"{dev.type}_batch{bs}_throughput_img_s"] = bs / med
    model.to(device)
    return res


def plot_curves(hist, stage1_epochs, path, title):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(hist["epoch"], hist["train_loss"], label="train loss")
    axes[0].plot(hist["epoch"], hist["val_loss"], label="val loss")
    axes[0].set_ylabel("Loss (freshness + 0.5 x fruit)")
    axes[1].plot(hist["epoch"], hist["train_acc"], label="train accuracy")
    axes[1].plot(hist["epoch"], hist["val_acc"], label="val accuracy")
    axes[1].plot(hist["epoch"], hist["val_f1_macro"], label="val macro-F1")
    axes[1].set_ylabel("Freshness metric")
    for ax in axes:
        ax.axvline(stage1_epochs + 0.5, color="grey", ls="--", lw=1)
        ax.set_xlabel("Epoch (dashed line: stage 1 -> stage 2)")
        ax.legend()
        ax.grid(alpha=0.3)
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(path, dpi=110)
    plt.close()


def plot_confusion(cm, path, title):
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.imshow(cm, cmap="Blues")
    for i in range(len(cm)):
        for j in range(len(cm)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks(range(len(cm)), FRESHNESS_ORDER)
    ax.set_yticks(range(len(cm)), FRESHNESS_ORDER)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=110)
    plt.close()


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="Processed Data")
    ap.add_argument("--clip-dir", default="clip_output")
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--sentences-per-class", type=int, default=4)
    ap.add_argument("--per-sentence", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--stage1-epochs", type=int, default=5)
    ap.add_argument("--max-epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--lr-stage1", type=float, default=1e-3)
    ap.add_argument("--lr-stage2", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--fruit-loss-weight", type=float, default=0.5)
    ap.add_argument("--workers", type=int, default=8, help="threads for image loading")
    ap.add_argument("--config-version", default=CONFIG_VERSION)
    args = ap.parse_args()

    run_name = args.run_name or f"s{args.sentences_per_class}x{args.per_sentence}"
    data_dir, out = Path(args.data), Path(args.runs_dir) / run_name
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cfg = {
        "version": args.config_version, "run_name": run_name, "seed": args.seed,
        "model": "resnet18 (torchvision IMAGENET1K_V1)", "heads": "freshness (3) + fruit (8)",
        "input": "resize 256x256 -> center crop 224 -> ImageNet mean/std; no extra augmentation",
        "split": "70/15/15 of original photos per class, augmented copies follow their original",
        "selection": {"sentences_per_class": args.sentences_per_class, "per_sentence": args.per_sentence},
        "optimizer": "AdamW", "weight_decay": args.weight_decay, "batch_size": args.batch_size,
        "stage1": {"trainable": "heads", "epochs": args.stage1_epochs, "lr": args.lr_stage1},
        "stage2": {"trainable": "layer4 + heads", "lr": args.lr_stage2, "schedule": "cosine"},
        "max_epochs": args.max_epochs, "early_stopping": f"val freshness macro-F1, patience {args.patience}",
        "loss": "CE(freshness) + w * CE(fruit)", "fruit_loss_weight": args.fruit_loss_weight,
        "label_smoothing": args.label_smoothing, "dropout": args.dropout, "amp": device.type == "cuda",
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"},
    }

    # 1. Split (saved once and reused by every run with the same seed)
    split_path = Path(args.runs_dir) / f"split_seed{args.seed}.csv"
    if split_path.exists():
        split_df = pd.read_csv(split_path)
    else:
        print("Building 70/15/15 split ...")
        split_df = build_split(data_dir, args.seed)
        split_df.to_csv(split_path, index=False)
    counts = split_df.groupby(["split", "is_augmented"]).size().unstack(fill_value=0)
    print("Split (original / augmented images):\n" + counts.to_string())

    # 2-3. Selection on the train split + oracle labels
    sel, n_sent = select_training_images(split_df, Path(args.clip_dir) / "similarities.csv",
                                         Path(args.clip_dir) / "sentences.csv",
                                         args.sentences_per_class, args.per_sentence)
    sel.to_csv(out / "selection.csv", index=False)
    sel_prec = selection_precision(sel)
    (out / "selection_precision.json").write_text(json.dumps(sel_prec, indent=2))
    cfg["selection"].update({"n_sentences": n_sent, "n_selected": len(sel)})
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    print(f"\nSelected {len(sel)} images with {n_sent} sentences | selection precision: "
          f"class {sel_prec['precision_class']:.1%}, freshness {sel_prec['precision_freshness']:.1%}")

    # Tensors (uint8, preprocessed once)
    fi = {s: i for i, s in enumerate(FRESHNESS_ORDER)}
    ri = {f: i for i, f in enumerate(FRUITS)}
    sets = {"train": sel, "val": split_df[split_df["split"] == "val"],
            "test": split_df[split_df["split"] == "test"]}
    data = {}
    for name, d in sets.items():
        t = time.perf_counter()
        X = load_images(data_dir, d["path"], args.workers)
        data[name] = (X, torch.tensor(d["freshness"].map(fi).to_numpy()),
                      torch.tensor(d["fruit"].map(ri).to_numpy()), d["fruit"].to_numpy())
        print(f"Loaded {name}: {len(X)} images ({time.perf_counter() - t:.1f}s)")

    # 4. Training
    model = FreshnessNet(args.dropout).to(device)
    norm = GPUNormalize(device)
    scaler = torch.amp.GradScaler(enabled=cfg["amp"])
    gen = torch.Generator().manual_seed(args.seed)
    run_cfg = {"batch_size": args.batch_size, "label_smoothing": args.label_smoothing,
               "fruit_loss_weight": args.fruit_loss_weight, "amp": cfg["amp"]}
    Xtr, ytr_f, ytr_r, _ = data["train"]
    Xva, yva_f, yva_r, _ = data["val"]

    history, best = [], {"f1": -1, "epoch": 0, "state": None}
    stage, bad_epochs = 0, 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t_train = time.perf_counter()
    for epoch in range(1, args.max_epochs + 1):
        if epoch == 1 or epoch == args.stage1_epochs + 1:
            stage = 1 if epoch == 1 else 2
            params = set_stage(model, stage)
            lr = args.lr_stage1 if stage == 1 else args.lr_stage2
            optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=args.weight_decay)
            sched = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.max_epochs - args.stage1_epochs)
                     if stage == 2 else None)
            bad_epochs = 0
        t = time.perf_counter()
        train_mode(model, stage)
        tr_loss, _, tr_probs, _ = run_epoch(model, Xtr, ytr_f, ytr_r, run_cfg, norm, device,
                                            optimizer, scaler, gen)
        model.eval()
        va_loss, _, va_probs, _ = run_epoch(model, Xva, yva_f, yva_r, run_cfg, norm, device)
        if sched:
            sched.step()
        tr_acc = accuracy_score(ytr_f.numpy(), tr_probs.argmax(1))
        va_acc = accuracy_score(yva_f.numpy(), va_probs.argmax(1))
        va_f1 = f1_score(yva_f.numpy(), va_probs.argmax(1), average="macro")
        history.append({"epoch": epoch, "stage": stage, "train_loss": tr_loss, "val_loss": va_loss,
                        "train_acc": tr_acc, "val_acc": va_acc, "val_f1_macro": va_f1,
                        "lr": optimizer.param_groups[0]["lr"], "epoch_time_s": time.perf_counter() - t})
        print(f"epoch {epoch:2d} (stage {stage}) | train loss {tr_loss:.4f} acc {tr_acc:.3f} | "
              f"val loss {va_loss:.4f} acc {va_acc:.3f} macro-F1 {va_f1:.3f} | {history[-1]['epoch_time_s']:.1f}s")
        if va_f1 > best["f1"]:
            best = {"f1": va_f1, "epoch": epoch, "state": {k: v.detach().clone() for k, v in model.state_dict().items()}}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if stage == 2 and bad_epochs >= args.patience:
                print(f"Early stopping at epoch {epoch} (best epoch {best['epoch']})")
                break
    train_time = time.perf_counter() - t_train
    peak_mem = torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else None

    hist = pd.DataFrame(history)
    hist.to_csv(out / "history.csv", index=False)
    plot_curves(hist, args.stage1_epochs, out / "learning_curves.png",
                f"ResNet-18, {n_sent} sentences x {args.per_sentence} images")

    # 5. Evaluation with the best checkpoint
    model.load_state_dict(best["state"])
    torch.save(best["state"], out / "best.pt")
    model.eval()
    results = {}
    for name in ["val", "test"]:
        X, yf, yr, fruits = data[name]
        _, _, probs, fruit_pred = run_epoch(model, X, yf, yr, run_cfg, norm, device)
        results[name] = freshness_metrics(yf.numpy(), probs, fruits)
        results[name]["fruit_head_accuracy"] = float(accuracy_score(yr.numpy(), fruit_pred))
        orig = ~sets[name]["is_augmented"].to_numpy()
        results[name]["accuracy_originals_only"] = float(accuracy_score(yf.numpy()[orig], probs.argmax(1)[orig]))
    plot_confusion(results["test"]["confusion_matrix"], out / "confusion_test.png", f"Test confusion ({run_name})")

    n_params = sum(p.numel() for p in model.parameters())
    efficiency = {
        "train_time_s": train_time, "epochs_run": len(hist), "best_epoch": best["epoch"],
        "mean_epoch_time_s": float(hist["epoch_time_s"].mean()),
        "train_images_per_s": len(Xtr) * len(hist) / train_time,
        "peak_gpu_memory_mb": peak_mem, "params_total": n_params,
        "params_trainable_stage2": sum(p.numel() for m in model.heads() + [model.backbone.layer4] for p in m.parameters()),
        "model_size_mb": (out / "best.pt").stat().st_size / 2**20,
        **measure_inference(model, device),
    }
    metrics = {"run": run_name, "selection_precision": {k: sel_prec[k] for k in
                                                        ["n_selected", "precision_class", "precision_freshness", "precision_fruit"]},
               "val": results["val"], "test": results["test"], "efficiency": efficiency}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))

    t = results["test"]
    print(f"\nTest: accuracy {t['accuracy']:.3f} | macro-F1 {t['f1_macro']:.3f} | AUROC {t['auroc_macro_ovr']:.3f}")
    for c, v in t["per_class"].items():
        print(f"  {c:10s} P {v['precision']:.3f}  R {v['recall']:.3f}  F1 {v['f1']:.3f}  (n={v['support']})")
    print(f"Train time {train_time:.0f}s, GPU batch-1 latency {efficiency.get('cuda_batch1_latency_ms', float('nan')):.2f} ms")
    print(f"Results written to {out}/")


if __name__ == "__main__":
    main()
