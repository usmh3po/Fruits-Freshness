"""
Robust inference pipeline (v1.2) and its evaluation.

  1. Input sanitisation: image-quality check (noise level, sharpness) and a CLIP OOD gate (max cosine
     similarity of the image to the 96 fruit sentences of the selection head).
  2. Classifier (robust model) with temperature scaling, optionally test-time augmentation (flip).
  3. Abstention when the calibrated confidence is below a threshold.

All thresholds are set on the clean validation set only. The script compares the original model (no
pipeline) with the robust model, with and without the pipeline, on clean, corrupted, OOD and adversarial
inputs, measures latency / memory per stage, and picks representative failure cases.

Writes to runs/robustness/: pipeline_thresholds.json, pipeline_results.csv, latency.csv,
memory.json, failure_cases.csv, failure_cases.png

Usage:  python robust_inference.py --original s4x40_reg --robust s4x40_robust
"""

import argparse
import json
import math
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from transformers import CLIPModel, CLIPProcessor

from adversarial import attack
from calibration import fit_temperature, load_model, predict_logits
from clip_selection import embed_texts
from corruptions import blur, clean, grayscale, jpeg, low_res, mask, noise, patch_dropout
from eda import FRESHNESS_ORDER
from stress_test import load_cifar_ood
from train import load_images

CLIP_MEAN, CLIP_STD = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
SHORT = {"Fresh": "Fresh", "Semi Fresh": "Semi", "Rotten": "Rotten"}
DECISIONS = ["accepted", "rejected: quality", "rejected: OOD", "abstained: low confidence"]

CONDITIONS = [("Clean", clean()), ("Gaussian noise sigma=0.05", noise(0.05)), ("Gaussian noise sigma=0.1", noise(0.1)),
              ("Gaussian noise sigma=0.2", noise(0.2)), ("Gaussian blur sigma=2px", blur(2)),
              ("Gaussian blur sigma=4px", blur(4)), ("JPEG quality 20", jpeg(20)), ("Mask 20% centre", mask(0.2, True)),
              ("Patch dropout 10%", patch_dropout(0.1)), ("Grayscale", grayscale()), ("Low resolution 64px", low_res(64)),
              ("Low resolution 32px (resolution of the OOD set)", low_res(32))]
ROTTEN_TO_FRESH_CASES = ("Mask 20% centre", "Patch dropout 10%")  # failure case = the costly error
# Six representative cases for the figure (condition prefix, case type) -> panel label
FIGURE_CASES = {("Clean", "failure"): "Clean image",
                ("Mask 20% centre", "failure"): "Centre 20% masked",
                ("OOD: Unseen produce", "failure"): "OOD (CIFAR-100)",
                ("PGD-10 eps=2/255 targeted -> Fresh", "failure"): "PGD eps=2/255 -> Fresh",
                ("Gaussian noise sigma=0.1", "resolved by robust model"): "Noise sigma=0.1",
                ("Grayscale", "resolved by robust model"): "Grayscale"}
FIGURE_CONDITIONS = {"Clean", "Gaussian noise sigma=0.1", "Mask 20% centre", "Grayscale"}


def case_key(c):
    return c["condition"].split(" (")[0].split(",")[0], c["case_type"]


def plot_failure_cases(cases, path):
    """2 x 3 grid of the FIGURE_CASES, failures first, then resolved cases."""
    shown = sorted([c for c in cases if case_key(c) in FIGURE_CASES],
                   key=lambda c: list(FIGURE_CASES).index(case_key(c)))
    fig, axes = plt.subplots(3, 2, figsize=(14, 25))
    for ax, c in zip(axes.ravel(), shown):
        ax.imshow(c["image"].permute(1, 2, 0).numpy())
        true = SHORT[c["true"]] if c["true"] else "none (OOD)"
        updated = (SHORT[c["robust_pred"]] + f" ({c['robust_conf']:.2f})" if c["pipeline_decision"] == "accepted"
                   else c["pipeline_decision"])
        tag = "RESOLVED" if c["case_type"] != "failure" else "FAILURE"
        ax.set_title(f"[{tag}] {FIGURE_CASES[case_key(c)]}\ntrue: {true}\n"
                     f"original: {SHORT[c['original_pred']]} ({c['original_conf']:.2f})\nupdated: {updated}", fontsize=17)
    for ax in axes.ravel():
        ax.axis("off")
    plt.tight_layout(h_pad=3)
    plt.savefig(path, dpi=100)
    plt.close()


class RobustPipeline:
    def __init__(self, model, temperature, device, sentences, clip_name="openai/clip-vit-base-patch32", tta=False):
        self.model, self.T, self.device, self.tta = model, temperature, device, tta
        dtype = torch.float16 if device.type == "cuda" else torch.float32
        self.clip = CLIPModel.from_pretrained(clip_name, dtype=dtype).to(device).eval()
        self.text = torch.from_numpy(embed_texts(sentences, self.clip, CLIPProcessor.from_pretrained(clip_name),
                                                 device)).to(device)
        self.clip_mean = torch.tensor(CLIP_MEAN, device=device).view(1, 3, 1, 1)
        self.clip_std = torch.tensor(CLIP_STD, device=device).view(1, 3, 1, 1)
        self.thr = {}

    @torch.no_grad()
    def quality(self, x):
        """Noise level (Immerkaer's estimator) and sharpness (variance of the Laplacian) of [0, 1] images."""
        gray = TF.rgb_to_grayscale(x).float()
        k = torch.tensor([[1., -2, 1], [-2, 4, -2], [1, -2, 1]], device=x.device).view(1, 1, 3, 3)
        h, w = gray.shape[-2:]
        sigma = math.sqrt(math.pi / 2) * F.conv2d(gray, k).abs().sum((1, 2, 3)) / (6 * (w - 2) * (h - 2))
        lap = torch.tensor([[0., 1, 0], [1, -4, 1], [0, 1, 0]], device=x.device).view(1, 1, 3, 3)
        return sigma, F.conv2d(gray, lap).var((1, 2, 3))

    @torch.no_grad()
    def clip_score(self, x):
        px = ((x - self.clip_mean) / self.clip_std).to(self.clip.dtype)
        e = self.clip.visual_projection(self.clip.vision_model(pixel_values=px).pooler_output)
        return (F.normalize(e.float(), dim=-1) @ self.text.T).max(1).values

    @torch.no_grad()
    def scores(self, X_uint8, bs=64):
        out = {"noise": [], "sharpness": [], "clip": []}
        for i in range(0, len(X_uint8), bs):
            x = X_uint8[i:i + bs].to(self.device).float() / 255
            s, sh = self.quality(x)
            out["noise"].append(s.cpu())
            out["sharpness"].append(sh.cpu())
            out["clip"].append(self.clip_score(x).cpu())
        out = {k: torch.cat(v).numpy() for k, v in out.items()}
        out["probs"] = (predict_logits(self.model, X_uint8, self.device, tta=self.tta) / self.T).softmax(1).numpy()
        return out

    def fit(self, val_scores, val_noisy, val_blurry, q_quality=1.0, q_clip=2.0, q_conf=5.0):
        """Quality thresholds at the edge of the training augmentation range (validation images with
        noise sigma=0.1 / blur sigma=2), so only inputs worse than anything seen in training are rejected.
        OOD gate: 2% of clean validation images. Abstention: 5% of the remaining clean validation images."""
        self.thr = {"noise_max": float(np.percentile(val_noisy["noise"], 100 - q_quality)),
                    "sharpness_min": float(np.percentile(val_blurry["sharpness"], q_quality)),
                    "clip_min": float(np.percentile(val_scores["clip"], q_clip))}
        ok = self.decide(val_scores, conf=False) == 0
        self.thr["confidence_min"] = float(np.percentile(val_scores["probs"].max(1)[ok], q_conf))
        return self.thr

    def decide(self, s, conf=True):
        """0 accepted, 1 rejected (quality), 2 rejected (OOD), 3 abstained (low confidence)."""
        d = np.zeros(len(s["clip"]), dtype=int)
        d[s["clip"] < self.thr["clip_min"]] = 2
        d[(s["noise"] > self.thr["noise_max"]) | (s["sharpness"] < self.thr["sharpness_min"])] = 1
        if conf:
            d[(d == 0) & (s["probs"].max(1) < self.thr["confidence_min"])] = 3
        return d


def corrupt_set(X, fn, seed, bs=64):
    """Apply a corruption once and keep the result as uint8, so every model sees identical inputs."""
    g = torch.Generator().manual_seed(seed)
    return torch.cat([(fn(X[i:i + bs].float() / 255, g) * 255).round().byte() for i in range(0, len(X), bs)])


def adversarial_set(model, X, y, eps, seed, device, bs=50, target=None):
    g = torch.Generator().manual_seed(seed)
    out = []
    for i in range(0, len(X), bs):
        x = X[i:i + bs].to(device).float() / 255
        xa = attack(model, x, torch.as_tensor(y[i:i + bs], device=device), eps, 10, target, g)
        out.append((xa * 255).round().byte().cpu())
    return torch.cat(out)


def evaluate(name, y, probs_orig, probs_rob, s, decisions, clip_min):
    """Original model (no pipeline) vs robust model (no pipeline) vs robust model + pipeline."""
    row = {"condition": name, "n": len(probs_rob)}
    if y is not None:
        acc = lambda p, m=slice(None): float((p.argmax(1)[m] == y[m]).mean()) if np.size(y[m]) else float("nan")
        r2f = lambda p, m: int(((y[m] == 2) & (p.argmax(1)[m] == 0)).sum())
        ok = decisions == 0
        row.update({"original_accuracy": acc(probs_orig), "robust_accuracy": acc(probs_rob),
                    "pipeline_coverage": float(ok.mean()), "pipeline_selective_accuracy": acc(probs_rob, ok),
                    "original_rotten_to_fresh": r2f(probs_orig, slice(None)),
                    "pipeline_rotten_to_fresh_accepted": r2f(probs_rob, ok)})
    else:  # OOD: every accepted image is an error
        row.update({"original_conf_over_0.6": float((probs_orig.max(1) >= 0.6).mean()),
                    "pipeline_coverage": float((decisions == 0).mean())})
    for k, label in enumerate(DECISIONS[1:], start=1):
        row[label] = float((decisions == k).mean())
    row["clip_gate_alone_rejects"] = float((s["clip"] < clip_min).mean())
    return row


def time_fn(fn, x, reps, cuda):
    for _ in range(5):
        fn(x)
    times = []
    for _ in range(reps):
        if cuda:
            torch.cuda.synchronize()
        t = time.perf_counter()
        fn(x)
        if cuda:
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t) * 1000)
    return np.array(times)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--original", default="s4x40_reg")
    ap.add_argument("--robust", default="s4x40_robust")
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--data", default="Processed Data")
    ap.add_argument("--clip-dir", default="clip_output")
    ap.add_argument("--ood-dir", default="ood_data")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--figure-only", action="store_true",
                    help="only redraw failure_cases.png (skips the other results, latency and memory)")
    args = ap.parse_args()
    runs_dir = Path(args.runs_dir)
    out = runs_dir / "robustness"
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cuda = device.type == "cuda"

    split = pd.read_csv(runs_dir / f"split_seed{args.seed}.csv")
    val, test = split[split["split"] == "val"], split[split["split"] == "test"].reset_index(drop=True)
    Xv, yv = load_images(Path(args.data), val["path"], args.workers), val["freshness"].map(FRESHNESS_ORDER.index).to_numpy()
    Xt, yt = load_images(Path(args.data), test["path"], args.workers), test["freshness"].map(FRESHNESS_ORDER.index).to_numpy()

    original = load_model(runs_dir / args.original / "best.pt", device)
    robust = load_model(runs_dir / args.robust / "best.pt", device)
    for m in (original, robust):
        for p in m.parameters():
            p.requires_grad_(False)
    T = fit_temperature(predict_logits(robust, Xv, device), yv)
    sentences = pd.read_csv(Path(args.clip_dir) / "sentences.csv")["sentence"].tolist()
    pipe = RobustPipeline(robust, T, device, sentences)
    thr = pipe.fit(pipe.scores(Xv), pipe.scores(corrupt_set(Xv, noise(0.1), args.seed)),
                   pipe.scores(corrupt_set(Xv, blur(2), args.seed)))
    if not args.figure_only:
        (out / "pipeline_thresholds.json").write_text(json.dumps({"temperature": T, **thr}, indent=2))
    print(f"T = {T:.3f}, thresholds: {thr}")

    orig_probs = lambda X: predict_logits(original, X, device).softmax(1).numpy()
    rows, cases = [], []

    def run_condition(name, X, y, kind):
        s = pipe.scores(X)
        p_orig, d = orig_probs(X), pipe.decide(s)
        rows.append(evaluate(name, y, p_orig, s["probs"], s, d, thr["clip_min"]))
        print(f"{name:40s} " + ", ".join(f"{k} {v:.3f}" for k, v in rows[-1].items() if isinstance(v, float)))
        # Failure case: the most confident wrong prediction of the original model (OOD: any prediction;
        # occlusion: rotten predicted fresh; targeted attack: non-fresh predicted fresh)
        conf = p_orig.max(1)
        wrong = np.ones(len(X), bool) if y is None else p_orig.argmax(1) != y
        if kind == "targeted":
            wrong &= p_orig.argmax(1) == 0
        if name in ROTTEN_TO_FRESH_CASES:
            wrong &= (p_orig.argmax(1) == 0) & (y == 2)
        # Resolved case: the most confident original error that the updated pipeline accepts and gets right
        picks = [("failure", wrong)]
        if y is not None:
            picks.append(("resolved by robust model", wrong & (d == 0) & (s["probs"].argmax(1) == y)))
        for case_type, cand in picks:
            if not cand.any():
                continue
            i = int(np.flatnonzero(cand)[np.argmax(conf[cand])])
            cases.append({"condition": name, "case_type": case_type, "image": X[i],
                          "true": None if y is None else FRESHNESS_ORDER[y[i]],
                          "original_pred": FRESHNESS_ORDER[p_orig[i].argmax()], "original_conf": float(conf[i]),
                          "robust_pred": FRESHNESS_ORDER[s["probs"][i].argmax()], "robust_conf": float(s["probs"][i].max()),
                          "pipeline_decision": DECISIONS[d[i]], "noise": float(s["noise"][i]),
                          "sharpness": float(s["sharpness"][i]), "clip_score": float(s["clip"][i])})

    for name, fn in CONDITIONS:
        if not args.figure_only or name in FIGURE_CONDITIONS:
            run_condition(name, corrupt_set(Xt, fn, args.seed), yt, "corruption")
    for name, (Xo, _) in load_cifar_ood(args.ood_dir, 1000, args.seed).items():
        run_condition(f"OOD: {name}", Xo, None, "ood")
    # Adversarial (white-box against each model separately would favour neither; craft on the original,
    # since that is the deployed model the attacker had access to)
    nf = yt != 0
    if not args.figure_only:
        run_condition("PGD-10 eps=2/255 (crafted on original)", adversarial_set(original, Xt, yt, 2 / 255, args.seed, device), yt, "adv")
    run_condition("PGD-10 eps=2/255 targeted -> Fresh, non-fresh only (crafted on original)",
                  adversarial_set(original, Xt[nf], yt[nf], 2 / 255, args.seed, device, target=0), yt[nf], "targeted")
    plot_failure_cases(cases, out / "failure_cases.png")
    if args.figure_only:
        return
    pd.DataFrame(rows).to_csv(out / "pipeline_results.csv", index=False)
    fc = pd.DataFrame([{k: v for k, v in c.items() if k != "image"} for c in cases])
    fc.to_csv(out / "failure_cases.csv", index=False)

    # Latency (p50 / p90) per configuration and input type
    xb = {"clean": Xt[:64], "noisy (sigma=0.1)": corrupt_set(Xt[:64], noise(0.1), args.seed),
          "adversarial (PGD 2/255)": adversarial_set(original, Xt[:64], yt[:64], 2 / 255, args.seed, device)}
    model_only = lambda m: (lambda X: predict_logits(m, X, device, bs=64))
    configs = {"Original model": model_only(original), "Robust model": model_only(robust),
               "Robust + temperature": lambda X: (predict_logits(robust, X, device, bs=64) / T).softmax(1),
               "Quality check only": lambda X: pipe.quality(X.to(device).float() / 255),
               "CLIP OOD gate only": lambda X: pipe.clip_score(X.to(device).float() / 255),
               "Full pipeline": lambda X: pipe.decide(pipe.scores(X)),
               "Full pipeline + TTA": lambda X: (setattr(pipe, "tta", True), pipe.decide(pipe.scores(X)), setattr(pipe, "tta", False))}
    lat = []
    for dev_name in (["GPU", "CPU"] if cuda else ["CPU"]):
        if dev_name == "CPU" and cuda:  # move everything to the CPU for the CPU measurements
            device = torch.device("cpu")
            for m in (original, robust, pipe.clip):
                m.to(device)
            pipe.device, pipe.text = device, pipe.text.to(device)
            pipe.clip_mean, pipe.clip_std = pipe.clip_mean.cpu(), pipe.clip_std.cpu()
            pipe.clip.float()
        for cfg_name, fn in configs.items():
            for input_name, Xb in xb.items():
                if input_name != "clean" and cfg_name not in ("Original model", "Robust model", "Full pipeline"):
                    continue
                for bs in ([1, 64] if dev_name == "GPU" else [1]):
                    t = time_fn(fn, Xb[:bs], 100 if bs == 1 else 30, dev_name == "GPU")
                    lat.append({"device": dev_name, "config": cfg_name, "input": input_name, "batch_size": bs,
                                "p50_ms": np.percentile(t, 50), "p90_ms": np.percentile(t, 90),
                                "throughput_img_s": bs / np.percentile(t, 50) * 1000})
    lat = pd.DataFrame(lat)
    lat.to_csv(out / "latency.csv", index=False)

    # Memory / size
    params = lambda m: sum(p.numel() for p in m.parameters())
    mem = {"resnet18_params": params(robust), "resnet18_fp32_mb": params(robust) * 4 / 2**20,
           "clip_vit_b32_params": params(pipe.clip), "clip_fp16_mb": params(pipe.clip) * 2 / 2**20,
           "temperature_params": 1}
    if cuda:
        device = torch.device("cuda")
        for m in (robust,):
            m.to(device)
        pipe.clip.half().to(device)
        pipe.device, pipe.text = device, pipe.text.to(device)
        pipe.clip_mean, pipe.clip_std = pipe.clip_mean.to(device), pipe.clip_std.to(device)
        for name, fn in [("model_only_bs64_peak_vram_mb", model_only(robust)),
                         ("pipeline_bs64_peak_vram_mb", lambda X: pipe.decide(pipe.scores(X)))]:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            fn(Xt[:64])
            mem[name] = torch.cuda.max_memory_allocated() / 2**20
    (out / "memory.json").write_text(json.dumps(mem, indent=2))

    pd.set_option("display.width", 250, "display.max_columns", 30)
    print("\n" + pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\n" + lat.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print("\n" + fc.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(json.dumps(mem, indent=2))


if __name__ == "__main__":
    main()
