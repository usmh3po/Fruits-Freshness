"""
Continual learning for the freshness classifier: class-balanced experience replay with layer4 + head
updates, compared with no updates, naive fine-tuning and cumulative retraining.

Six deployment periods with different drift. In each period after T0, 600 newly labelled images arrive
(from the unused part of the train split, drifted like the period). Every method updates its own copy
of the robust model (v1.2) and is evaluated on the test set of every period, which gives the accuracy
matrix R[after step t, test period j] used for average accuracy and forgetting.

The replay model is the deployed one: it also runs the drift monitor (Mahalanobis OOD rate, confidence),
refits temperature + Mahalanobis statistics after each update, goes through a release gate and is
versioned as v1.3.k under runs/continual/.

Active learning variants label only 360 of each period's 600 new images: "Replay + AL 360" picks them with
the CLIP selection head (round-robin over 120 class sentences, 5 per class, 3 images each), "Replay + random 360"
picks them at random (control). Labels come from the dataset (simulated annotators).

Usage:  python continual.py
"""

import argparse
import copy
import hashlib
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor

from adapt import camera_drift
from clip_selection import embed_texts, select_round_robin
from calibration import ece, fit_temperature, load_model, predict_logits
from corruptions import augment, brightness, clean, noise
from eda import FRESHNESS_ORDER
from feature_ood import Mahalanobis, features_and_logits
from robust_inference import corrupt_set
from train import FRUITS, GPUNormalize, freshness_metrics, load_images, run_epoch, set_stage, train_mode


def low_light(x, g):
    return noise(0.05)(brightness(0.6)(x, g), g)


def warm_light(x, g):
    gains = torch.tensor([1.2, 1.0, 0.75], device=x.device).view(1, 3, 1, 1)
    return (x * gains).clamp(0, 1)


# (name, drift applied to the images, class prior of the period or None = as in the data)
PERIODS = [("T0 clean", clean(), None), ("T1 low light", low_light, None),
           ("T2 cheap camera", camera_drift, None), ("T3 mostly fresh stock", clean(), (0.6, 0.1, 0.3)),
           ("T4 warm lighting", warm_light, None), ("T5 back to clean", clean(), None)]
METHODS = ["Static (no update)", "Naive fine-tune", "Replay, no gate", "Replay (ours)", "Replay + AL 360",
           "Replay + random 360", "Cumulative retrain"]
COLOURS = dict(zip(METHODS, ["#8c8c8c", "#c44e52", "#8172b3", "#4c72b0", "#dd8452", "#937860", "#55a868"]))
REPLAY = ["Replay, no gate", "Replay (ours)", "Replay + AL 360", "Replay + random 360"]
GATED = ["Replay (ours)", "Replay + AL 360", "Replay + random 360"]
AL_BUDGET = 360  # labels per period for the active-learning variants (of the 600 new images): 120 sentences x 3
CLEAN_TOLERANCE = 0.02  # release gate: max clean macro-F1 drop (about 2.5 standard errors on 2,088 val images)


class ReplayBuffer:
    """Class-balanced reservoir buffer. Stores (path, period) so an image can be rebuilt with the drift
    of the period it came from; no pixels are kept."""

    def __init__(self, per_class, seed):
        self.per_class, self.rng = per_class, np.random.default_rng(seed)
        self.items, self.seen = {k: [] for k in range(3)}, {k: 0 for k in range(3)}

    def add(self, paths, labels, period):
        for p, k in zip(paths, labels):
            self.seen[k] += 1
            if len(self.items[k]) < self.per_class:
                self.items[k].append((p, period))
            else:
                j = self.rng.integers(self.seen[k])
                if j < self.per_class:
                    self.items[k][j] = (p, period)

    def sample(self, n):
        out = []
        for k in range(3):
            pool = self.items[k]
            idx = self.rng.choice(len(pool), n // 3, replace=len(pool) < n // 3)
            out += [pool[i] for i in idx]
        return out

    def all(self):
        return [it for k in range(3) for it in self.items[k]]


# A fifth sentence per class for the active-learning selector (the Milestone 2 sentence file is unchanged),
# describing cues the first four do not: 24 classes x 5 = 120 sentences, 3 images each = AL_BUDGET
EXTRA_SENTENCES = {
    "Fresh Banana": "a photo of a firm, evenly yellow banana with a clean unbruised peel",
    "Semi Fresh Banana": "a photo of a banana with a yellow peel heavily speckled with brown sugar spots",
    "Rotten Banana": "a photo of a collapsed rotten banana with oozing, sticky black skin",
    "Fresh Bittermelon": "a photo of a firm, deep green bitter melon with sharp, crisp ridges",
    "Semi Fresh Bittermelon": "a photo of a softening bitter melon with dull, pale green skin and yellow spots",
    "Rotten Bittermelon": "a photo of a split-open orange bitter melon showing red seeds",
    "Fresh Cucumber": "a photo of a straight, firm cucumber with even green color from end to end",
    "Semi Fresh Cucumber": "a photo of a cucumber with soft spots and a yellowing tip",
    "Rotten Cucumber": "a photo of a soft, wrinkled yellow cucumber with sunken spots",
    "Fresh Eggplant": "a photo of a firm, shiny round eggplant with taut striped skin",
    "Semi Fresh Eggplant": "a photo of an eggplant with faded, uneven purple and green color",
    "Rotten Eggplant": "a photo of a soft, collapsed eggplant with dark brown rotting patches",
    "Fresh Orange": "a photo of a firm green citrus fruit with tight, dimpled skin",
    "Semi Fresh Orange": "a photo of a citrus fruit turning from green to yellow with small brown marks",
    "Rotten Orange": "a photo of a shrunken, hardened citrus fruit with blue-green mold",
    "Fresh Papaya": "a photo of a firm, glossy green papaya with a short fresh stem",
    "Semi Fresh Papaya": "a photo of a green papaya with dull skin and dark sunken spots",
    "Rotten Papaya": "a photo of a soft, sunken papaya with black rotting spots and fuzzy mold",
    "Fresh Pineapple": "a photo of a firm pineapple with tightly packed green and gold eyes",
    "Semi Fresh Pineapple": "a photo of a deep golden pineapple with a few soft brown eyes",
    "Rotten Pineapple": "a photo of a dark brown pineapple with sunken, leaking eyes",
    "Fresh Tomato": "a photo of a glossy, firm round tomato with taut skin",
    "Semi Fresh Tomato": "a photo of a deep red tomato with a soft dent on one side",
    "Rotten Tomato": "a photo of a split rotten tomato leaking juice with white mold",
}


class ClipSelector:
    """The Milestone 2 selection head applied to new (drifted) images: CLIP similarity to the class
    sentences (the 96 Milestone 2 sentences + EXTRA_SENTENCES = 5 per class), then round-robin so every
    sentence gets the same quota."""

    def __init__(self, device, clip_dir, name="openai/clip-vit-base-patch32"):
        self.device = device
        self.model = CLIPModel.from_pretrained(name, dtype=torch.float16 if device.type == "cuda" else torch.float32)
        self.model = self.model.to(device).eval()
        base = pd.read_csv(Path(clip_dir) / "sentences.csv")[["label", "sentence"]]
        extra = pd.DataFrame(list(EXTRA_SENTENCES.items()), columns=["label", "sentence"])
        sentences = pd.concat([base, extra], ignore_index=True)
        assert (sentences.groupby("label").size() == 5).all()
        self.text = torch.from_numpy(embed_texts(sentences["sentence"].tolist(), self.model,
                                                 CLIPProcessor.from_pretrained(name), device)).to(device)
        self.sentence_freshness = sentences["label"].str.rsplit(" ", n=1).str[0].to_numpy()
        self.mean = torch.tensor((0.48145466, 0.4578275, 0.40821073), device=device).view(1, 3, 1, 1)
        self.std = torch.tensor((0.26862954, 0.26130258, 0.27577711), device=device).view(1, 3, 1, 1)

    @torch.no_grad()
    def select(self, X_uint8, n, bs=64):
        sims = []
        for i in range(0, len(X_uint8), bs):
            x = ((X_uint8[i:i + bs].to(self.device).float() / 255 - self.mean) / self.std).to(self.model.dtype)
            e = self.model.visual_projection(self.model.vision_model(pixel_values=x).pooler_output)
            sims.append((F.normalize(e.float(), dim=-1) @ self.text.T).cpu())
        sim = torch.cat(sims).numpy()
        picks = select_round_robin(sim, -(-n // sim.shape[1]))[:n]
        return [p[1] for p in picks], self.sentence_freshness[[p[0] for p in picks]]


# The ungated replay run went through the same steps (every gate passed), so it is a second replay run, not a gate
# ablation; it stays in the CSVs but is left out of the trajectory figures
TRAJECTORY_METHODS = [m for m in METHODS if m != "Replay, no gate"]
TRAJECTORY_PANELS = {"current": "Accuracy on the current period\n(dotted: before update, solid: after)",
                     "average": "Average accuracy over all periods seen so far",
                     "clean": "Accuracy on T0 (clean) over time"}


def plot_trajectories(rows, matrix, out):
    """Figure 1 as one 3-panel figure (trajectories.png) and as three separate figures."""
    names = [p[0].split(" ", 1)[0] for p in PERIODS]
    x = range(len(names))

    def draw(ax, panel):
        for method in TRAJECTORY_METHODS:
            d, dm, c = rows[rows["method"] == method], matrix[matrix["method"] == method], COLOURS[method]
            if panel == "current":
                ax.plot(x, d[d["stage"] == "after update"]["accuracy"], marker="o", color=c, label=method)
                ax.plot(x, d[d["stage"] == "before update"]["accuracy"], ls=":", marker=".", color=c, alpha=0.7)
            elif panel == "average":
                ax.plot(x, [dm[(dm["after_t"] == t) & (dm["test_period"] <= t)]["accuracy"].mean() for t in x],
                        marker="o", color=c, label=method)
            else:
                ax.plot(x, dm[dm["test_period"] == 0].sort_values("after_t")["accuracy"], marker="o", color=c, label=method)
        ax.set_xticks(x, names)
        ax.set_title(TRAJECTORY_PANELS[panel], fontsize=10)
        ax.set_ylabel("Test accuracy")
        ax.grid(alpha=0.3)

    fig, axes = plt.subplots(1, 3, figsize=(19, 4.8))
    for ax, panel in zip(axes, TRAJECTORY_PANELS):
        draw(ax, panel)
    axes[0].legend(fontsize=8, loc="lower left")
    plt.tight_layout()
    plt.savefig(out / "trajectories.png", dpi=110)
    plt.close()
    for panel in TRAJECTORY_PANELS:
        fig, ax = plt.subplots(figsize=(7, 4.8))
        draw(ax, panel)
        ax.legend(fontsize=8, loc="best")
        plt.tight_layout()
        plt.savefig(out / f"trajectory_{panel}.png", dpi=110)
        plt.close()


def plot_accuracy_matrix(matrix, out):
    """Figure 2: accuracy matrix per strategy (rows: model after period t, columns: test period)."""
    names = [p[0].split(" ", 1)[0] for p in PERIODS]
    cols = 4
    fig, axes = plt.subplots(-(-len(METHODS) // cols), cols, figsize=(5 * cols, 4.6 * -(-len(METHODS) // cols)))
    for ax in axes.ravel()[len(METHODS):]:
        ax.axis("off")
    for ax, method in zip(axes.ravel(), METHODS):
        R = matrix[matrix["method"] == method].pivot(index="after_t", columns="test_period", values="accuracy").to_numpy()
        im = ax.imshow(R, vmin=0.5, vmax=0.9, cmap="viridis")
        for i in range(R.shape[0]):
            for j in range(R.shape[1]):
                ax.text(j, i, f"{R[i, j]:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if R[i, j] < 0.72 else "black")
        ax.set_xticks(range(len(names)), names)
        ax.set_yticks(range(len(names)), names)
        ax.set_xlabel("Test period")
        ax.set_ylabel("Model after period")
        ax.set_title(method)
    fig.colorbar(im, ax=axes, shrink=0.8, label="Accuracy")
    plt.savefig(out / "accuracy_matrix.png", dpi=110, bbox_inches="tight")
    plt.close()


AL_METHODS = ["Static (no update)", "Replay (ours)", "Replay + AL 360", "Replay + random 360"]
AL_LABELS = {"Static (no update)": "No update (0 labels)", "Replay (ours)": "Replay, all 600 labelled",
             "Replay + AL 360": "Replay + active learning (360)", "Replay + random 360": "Replay + random (360)"}


def plot_active_learning(rows, matrix, efficiency, out):
    """Human-in-the-loop cycles: accuracy before/after each cycle, average accuracy, and labels spent."""
    names = [p[0].split(" ", 1)[0] for p in PERIODS]
    x = range(len(names))
    fig, axes = plt.subplots(1, 3, figsize=(19, 4.8))
    for method in [m for m in AL_METHODS if m in set(rows["method"])]:
        d, dm, c = rows[rows["method"] == method], matrix[matrix["method"] == method], COLOURS[method]
        axes[0].plot(x, d[d["stage"] == "after update"]["accuracy"], marker="o", color=c, label=AL_LABELS[method])
        axes[0].plot(x, d[d["stage"] == "before update"]["accuracy"], ls=":", marker=".", color=c, alpha=0.7)
        avg = [dm[(dm["after_t"] == t) & (dm["test_period"] <= t)]["accuracy"].mean() for t in x]
        axes[1].plot(x, avg, marker="o", color=c, label=AL_LABELS[method])
        e = efficiency[efficiency["method"] == method]
        if len(e):
            axes[2].plot(np.r_[0, e["labels_this_cycle"].cumsum().to_numpy()], avg, marker="o", color=c,
                         label=AL_LABELS[method])
        else:  # no labels at all: show its final average as a reference line
            axes[2].axhline(avg[-1], color=c, ls="--", label=f"{AL_LABELS[method]}, after T5")
    for ax, title in zip(axes[:2], ["Accuracy on the current cycle's images\n(dotted: before update, solid: after)",
                                    "Average accuracy over all cycles so far"]):
        ax.set_xticks(x, names)
        ax.set_title(title, fontsize=10)
    axes[2].set_title("Average accuracy vs. cumulative human labels", fontsize=10)
    axes[2].set_xlabel("Images labelled by humans (cumulative)")
    for ax in axes:
        ax.set_ylabel("Test accuracy")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8, loc="lower left")
    plt.tight_layout()
    plt.savefig(out / "active_learning_cycles.png", dpi=110)
    plt.close()


def plot_all(rows, matrix, efficiency, out, reference):
    """Full runs get figures 1-2; subset runs get the active-learning figure, with the reference run's
    'Replay (ours)' results (600 labels per cycle) added for comparison when they were not rerun."""
    if set(METHODS) <= set(rows["method"]):
        plot_trajectories(rows, matrix, out)
        plot_accuracy_matrix(matrix, out)
    if "Replay (ours)" not in set(rows["method"]) and (reference / "results.csv").exists():
        pick = lambda df: df[df["method"] == "Replay (ours)"]
        rows = pd.concat([rows, pick(pd.read_csv(reference / "results.csv"))])
        matrix = pd.concat([matrix, pick(pd.read_csv(reference / "accuracy_matrix.csv"))])
        ref_eff = pick(pd.read_csv(reference / "efficiency.csv"))
        efficiency = pd.concat([efficiency, ref_eff.assign(labels_this_cycle=ref_eff.get("labels_this_cycle", 600))])
    plot_active_learning(rows, matrix, efficiency, out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="s4x40_robust", help="starting model (v1.3.0)")
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--data", default="Processed Data")
    ap.add_argument("--n-new", type=int, default=600, help="newly labelled images per period")
    ap.add_argument("--n-replay", type=int, default=600, help="buffer images mixed into each update")
    ap.add_argument("--buffer-per-class", type=int, default=333)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--clip-dir", default="clip_output")
    ap.add_argument("--methods", nargs="+", default=METHODS, choices=METHODS, help="strategies to run")
    ap.add_argument("--out-name", default="continual", help="output folder under runs/")
    ap.add_argument("--reference", default="continual",
                    help="earlier run folder whose 'Replay (ours)' results are added for comparison if not rerun")
    ap.add_argument("--plot-only", action="store_true", help="redraw the figures from the saved CSVs")
    args = ap.parse_args()
    runs_dir, data_dir = Path(args.runs_dir), Path(args.data)
    out = runs_dir / args.out_name
    out.mkdir(parents=True, exist_ok=True)
    if args.plot_only:
        plot_all(pd.read_csv(out / "results.csv"), pd.read_csv(out / "accuracy_matrix.csv"),
                 pd.read_csv(out / "efficiency.csv"), out, runs_dir / args.reference)
        return
    methods = args.methods
    deployed = "Replay (ours)" in methods  # only the deployed method runs the monitor and is versioned
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    split = pd.read_csv(runs_dir / f"split_seed{args.seed}.csv")
    info = split.set_index("path")
    fresh_of = lambda paths: info.loc[paths, "freshness"].map(FRESHNESS_ORDER.index).to_numpy()
    fruit_of = lambda paths: info.loc[paths, "fruit"].map(FRUITS.index).to_numpy()
    cache = {}

    def images(items):
        """uint8 tensors for (path, period) items, each with its period's drift."""
        need = [p for p, _ in items if p not in cache]
        if need:
            cache.update(zip(need, load_images(data_dir, need, args.workers)))
        X = torch.stack([cache[p] for p, _ in items])
        for t in {t for _, t in items}:
            idx = [i for i, (_, tt) in enumerate(items) if tt == t]
            X[idx] = corrupt_set(X[idx], PERIODS[t][1], args.seed + t)
        return X

    # Test and validation sets per period (the T3 prior is applied by resampling)
    def period_sets(df, per_class=None):
        base = df.groupby("freshness").sample(per_class, random_state=args.seed) if per_class else df
        sets = []
        for t, (_, _, prior) in enumerate(PERIODS):
            d = base
            if prior:
                n_fresh = (base["freshness"] == FRESHNESS_ORDER[0]).sum()
                total = round(n_fresh / prior[0])
                d = pd.concat([base[base["freshness"] == c].sample(round(total * s), random_state=args.seed)
                               for c, s in zip(FRESHNESS_ORDER, prior)])
            items = [(p, t) for p in d["path"]]
            sets.append((images(items), fresh_of(d["path"].tolist())))
        return sets

    test_sets = period_sets(split[split["split"] == "test"])
    val_sets = period_sets(split[split["split"] == "val"])
    print("Test sets: " + ", ".join(f"{PERIODS[t][0]} ({len(y)})" for t, (_, y) in enumerate(test_sets)))

    # Unused training images arrive as new labelled data; the initial buffer is the clean T0 selection
    base = load_model(runs_dir / args.run / "best.pt", device)
    selected = pd.read_csv(runs_dir / args.run / "selection.csv")
    unused = split[(split["split"] == "train") & ~split["path"].isin(selected["path"])].copy()
    init = selected.groupby("freshness").sample(args.buffer_per_class, random_state=args.seed)
    buffers = {}
    for i, m in enumerate(REPLAY):  # each replay method keeps only the images it labelled
        buffers[m] = ReplayBuffer(args.buffer_per_class, args.seed + i)
        buffers[m].add(init["path"].tolist(), fresh_of(init["path"].tolist()), 0)
    buffer = buffers["Replay (ours)"]  # the deployed model's buffer (monitor, versioning)
    cumulative = [(p, 0) for p in init["path"]]
    selector = ClipSelector(device, args.clip_dir)
    clip_vram_mb = torch.cuda.memory_allocated() / 2**20 if device.type == "cuda" else None

    def new_batch(t):
        prior = PERIODS[t][2] or (1 / 3, 1 / 3, 1 / 3)
        parts = [unused[unused["freshness"] == c].sample(round(args.n_new * s), random_state=args.seed + t)
                 for c, s in zip(FRESHNESS_ORDER, prior)]
        batch = pd.concat(parts)
        unused.drop(batch.index, inplace=True)
        return batch["path"].tolist()

    def update(model, items):
        X = images(items)
        paths = [p for p, _ in items]
        yf, yr = torch.as_tensor(fresh_of(paths)), torch.as_tensor(fruit_of(paths))
        opt = torch.optim.AdamW(set_stage(model, 2), lr=args.lr, weight_decay=0.05)
        scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
        cfg = {"batch_size": 32, "label_smoothing": 0.2, "fruit_loss_weight": 0.5,
               "amp": device.type == "cuda", "augment": augment}
        gen = torch.Generator().manual_seed(args.seed)
        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        for _ in range(args.epochs):
            train_mode(model, 2)
            run_epoch(model, X, yf, yr, cfg, GPUNormalize(device), device, opt, scaler, gen)
        model.eval()
        if device.type == "cuda":
            torch.cuda.synchronize()
        return {"update_time_s": time.perf_counter() - t0, "update_images": len(items),
                "update_peak_vram_mb": torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else None}

    def evaluate(model, X, y, T=1.0):
        """T: the model's temperature, fitted on the clean validation set (ece after temperature scaling)."""
        logits = predict_logits(model, X, device)
        m = freshness_metrics(y, logits.softmax(1).numpy())
        return {"accuracy": m["accuracy"], "f1_macro": m["f1_macro"],
                "recall_semi": m["per_class"]["Semi Fresh"]["recall"], "rotten_to_fresh": m["confusion_matrix"][2][0],
                "ece": ece((logits / T).softmax(1).numpy(), y)}

    clean_temperature = lambda model: fit_temperature(predict_logits(model, val_sets[0][0], device), val_sets[0][1])

    def fit_monitor(model, t):
        """Temperature + Mahalanobis statistics of the current model; threshold = 98th pct of period-t val."""
        feats = features_and_logits(model, images(buffer.all()), device)[0]
        maha = Mahalanobis().fit(feats, fresh_of([p for p, _ in buffer.all()])).to(device)
        Xv, yv = val_sets[t]
        fv, lv = features_and_logits(model, Xv, device)
        share = np.bincount(lv.argmax(1).numpy(), minlength=3) / len(lv)
        return {"maha": maha, "maha_max": float(np.percentile(maha.score(fv), 98)), "T": fit_temperature(lv, yv),
                "share": share}

    def monitor_stats(model, mon, X):
        f, logits = features_and_logits(model, X, device)
        share = np.bincount(logits.argmax(1).numpy(), minlength=3) / len(logits)
        return {"ood_rejection_rate": float((mon["maha"].score(f) > mon["maha_max"]).mean()),
                "mean_confidence": float((logits / mon["T"]).softmax(1).max(1).values.mean()),
                "class_share_shift": float(np.abs(share - mon["share"]).sum() / 2),  # total variation distance
                **{f"predicted_share_{c}": v for c, v in zip(FRESHNESS_ORDER, share)}}

    models = {m: copy.deepcopy(base) for m in methods}
    temps = {m: clean_temperature(base) for m in methods}
    approved_clean_f1 = {m: evaluate(base, *val_sets[0])["f1_macro"] for m in GATED if m in methods}
    mon = fit_monitor(models["Replay (ours)"], 0) if deployed else None
    version, rows, matrix, monitor, efficiency, selection = 0, [], [], [], [], []

    def save_version(model, t, metrics, new_paths):
        d = out / f"v1.3.{version}"
        d.mkdir(exist_ok=True)
        torch.save(model.state_dict(), d / "best.pt")
        cfg = {"version": f"v1.3.{version}", "parent": f"v1.3.{version - 1}" if version else f"{args.run} (v1.2)",
               "period": PERIODS[t][0], "method": "class-balanced replay, layer4 + heads",
               "epochs": args.epochs, "lr": args.lr, "batch_size": 32, "n_new": args.n_new, "n_replay": args.n_replay,
               "buffer_per_class": args.buffer_per_class, "augmentation": "corruptions.augment (v1.2)",
               "temperature": mon["T"], "ood_threshold": mon["maha_max"],
               "new_data_md5": hashlib.md5("\n".join(sorted(new_paths)).encode()).hexdigest()}
        (d / "config.json").write_text(json.dumps(cfg, indent=2))
        (d / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
        pd.DataFrame(buffer.all(), columns=["path", "period"]).to_csv(d / "buffer_manifest.csv", index=False)

    if deployed:
        save_version(base, 0, {"clean_val_f1": approved_clean_f1["Replay (ours)"]}, [])
    for t, (period, _, _) in enumerate(PERIODS):
        new_paths = new_batch(t) if t > 0 else []
        new_items = [(p, t) for p in new_paths]
        labelled = {m: new_items for m in methods}
        if t > 0:  # active learning: label only AL_BUDGET of the new images
            y_new = fresh_of(new_paths)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            idx, guessed = selector.select(images(new_items), AL_BUDGET)
            sel_time = time.perf_counter() - t0
            sel_vram = torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else None
            labelled["Replay + AL 360"] = [new_items[i] for i in idx]
            rand_idx = rng.choice(len(new_items), AL_BUDGET, replace=False)
            labelled["Replay + random 360"] = [new_items[i] for i in rand_idx]
            for m, ix in [("Replay + AL 360", idx), ("Replay + random 360", rand_idx)]:
                shares = np.bincount(y_new[ix], minlength=3) / len(ix)
                is_al = m == "Replay + AL 360"
                selection.append({"period": period, "method": m, "labelled": len(ix), "unlabelled_pool": len(new_items),
                                  "selection_time_s": sel_time if is_al else 0.0,
                                  "selection_peak_vram_mb": sel_vram if is_al else None,
                                  "clip_freshness_precision": (float((guessed == np.array(FRESHNESS_ORDER)[y_new[idx]]).mean())
                                                               if is_al else None),
                                  **{f"true_share_{c}": v for c, v in zip(FRESHNESS_ORDER, shares)}})
        gate_results = {}
        if t > 0 and deployed:  # drift monitor before the update (old statistics)
            monitor.append({"period": period, "stage": "before update", **monitor_stats(models["Replay (ours)"], mon, test_sets[t][0])})
        for method in methods:
            model = models[method]
            rows.append({"method": method, "period": period, "t": t, "stage": "before update",
                         **evaluate(model, *test_sets[t], temps[method])})
            eff = {}
            if t > 0 and method != "Static (no update)":
                candidate = copy.deepcopy(model)
                new = labelled[method]
                if method in REPLAY:  # half new, half replay (600 + 600, or 360 + 360 for the AL variants)
                    items = new + buffers[method].sample(len(new))
                else:
                    items = {"Naive fine-tune": new, "Cumulative retrain": cumulative + new}[method]
                eff = {"labels_this_cycle": len(new), **update(candidate, items)}
                if method in GATED:  # release gate: not worse on the new period, at most 2 pt worse on clean
                    new_f1, old_f1 = evaluate(candidate, *val_sets[t])["f1_macro"], evaluate(model, *val_sets[t])["f1_macro"]
                    clean_f1 = evaluate(candidate, *val_sets[0])["f1_macro"]
                    eff["gate_passed"] = gate_results[method] = bool(
                        new_f1 >= old_f1 and clean_f1 >= approved_clean_f1[method] - CLEAN_TOLERANCE)
                    eff.update({"gate_new_period_val_f1": new_f1, "gate_previous_val_f1": old_f1, "gate_clean_val_f1": clean_f1})
                    if eff["gate_passed"]:
                        approved_clean_f1[method] = max(approved_clean_f1[method], clean_f1)
                        models[method] = model = candidate
                        temps[method] = clean_temperature(model)
                else:
                    models[method] = model = candidate
                    temps[method] = clean_temperature(model)
                efficiency.append({"method": method, "period": period, **eff})
            after = {j: evaluate(model, *test_sets[j], temps[method]) for j in range(len(PERIODS))}
            rows.append({"method": method, "period": period, "t": t, "stage": "after update", **after[t]})
            if method == "Replay (ours)":
                deployed_row = rows[-1]
            matrix += [{"method": method, "after_t": t, "test_period": j, **m} for j, m in after.items()]
            print(f"{period:22s} {method:20s} acc {rows[-2]['accuracy']:.3f} -> {rows[-1]['accuracy']:.3f}"
                  + (f"  ({eff['update_time_s']:.1f}s{', gate ' + ('passed' if eff.get('gate_passed') else 'FAILED') if 'gate_passed' in eff else ''})" if eff else ""))
        for m in [m for m in REPLAY if m in methods]:
            paths = [p for p, _ in labelled[m]]
            buffers[m].add(paths, fresh_of(paths), t)
        cumulative += new_items
        if t > 0 and deployed:
            if gate_results["Replay (ours)"]:
                version += 1
                mon = fit_monitor(models["Replay (ours)"], t)
                save_version(models["Replay (ours)"], t, deployed_row, new_paths)
            monitor.append({"period": period, "stage": "after update", **monitor_stats(models["Replay (ours)"], mon, test_sets[t][0])})

    rows, matrix, monitor, efficiency, selection = map(pd.DataFrame, (rows, matrix, monitor, efficiency, selection))
    selection["clip_model_vram_mb"] = clip_vram_mb
    for name, df in [("results", rows), ("accuracy_matrix", matrix), ("monitor", monitor), ("efficiency", efficiency),
                     ("al_selection", selection)]:
        df.to_csv(out / f"{name}.csv", index=False)

    # Summary: average accuracy over seen periods and forgetting after the last step
    T = len(PERIODS) - 1
    summary = []
    for method, d in matrix.groupby("method", sort=False):
        R = d.pivot(index="after_t", columns="test_period", values="accuracy").to_numpy()
        F = d.pivot(index="after_t", columns="test_period", values="f1_macro").to_numpy()
        summary.append({"method": method, "final_avg_accuracy": R[T].mean(), "final_avg_f1": F[T].mean(),
                        "avg_accuracy_on_current_period": np.mean([R[t, t] for t in range(T + 1)]),
                        "forgetting": np.mean([R[j:T, j].max() - R[T, j] for j in range(T)]),
                        "backward_transfer": np.mean([R[T, j] - R[j, j] for j in range(T)])})
    summary = pd.DataFrame(summary)
    summary.to_csv(out / "summary.csv", index=False)

    # Inference latency of the deployed model before (v1.3.0) and after all updates
    lat = []
    x64 = GPUNormalize(device)(test_sets[0][0][:64].to(device))
    final = "Replay (ours)" if deployed else next(m for m in methods if m != "Static (no update)")
    for name, m in [("start (v1.2)", base), (f"{final} after T5", models[final])]:
        for bs, reps in ((1, 200), (64, 50)):
            x = x64[:bs]
            with torch.no_grad(), torch.autocast(device.type, enabled=device.type == "cuda"):
                for _ in range(10):
                    m(x)
                ts = []
                for _ in range(reps):
                    torch.cuda.synchronize() if device.type == "cuda" else None
                    t0 = time.perf_counter()
                    m(x)
                    torch.cuda.synchronize() if device.type == "cuda" else None
                    ts.append((time.perf_counter() - t0) * 1000)
            lat.append({"model": name, "batch_size": bs, "p50_ms": np.percentile(ts, 50), "p90_ms": np.percentile(ts, 90),
                        "throughput_img_s": bs / np.percentile(ts, 50) * 1000})
    pd.DataFrame(lat).to_csv(out / "latency.csv", index=False)

    plot_all(rows, matrix, efficiency, out, runs_dir / args.reference)

    if not deployed:
        print(summary.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        print(selection.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        print(efficiency.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        print(pd.DataFrame(lat).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        return
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    for ax, col, title in [(axes[0], "ood_rejection_rate", "OOD-gate rejection rate on the period's images"),
                           (axes[1], "class_share_shift", "Shift in predicted class shares (total variation)"),
                           (axes[2], "mean_confidence", "Mean calibrated confidence")]:
        for stage, style in [("before update", "o--"), ("after update", "o-")]:
            d = monitor[monitor["stage"] == stage]
            ax.plot(d["period"].str.split(" ", n=1).str[0], d[col], style, label=stage)
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)
    axes[0].axhline(0.04, color="red", lw=1, ls=":", label="trigger (2x baseline)")
    axes[0].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out / "monitor.png", dpi=110)
    plt.close()

    pd.set_option("display.width", 250, "display.max_columns", 20)
    for name, df in [("Summary", summary), ("Efficiency", efficiency), ("Monitor", monitor), ("Latency", pd.DataFrame(lat))]:
        print(f"\n{name}\n" + df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
