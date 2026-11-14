"""
Final deployment change: replace the CLIP OOD gate on the request path with a Mahalanobis score on the
robust ResNet's own 512-d features (same forward pass, no second model). CLIP moves to offline auditing.

Class means and a shared covariance are fitted on the robust model's training selection; the threshold
rejects 2% of clean validation images (same rule as the CLIP gate). Final online pipeline:
quality check -> robust ResNet-18 -> temperature scaling -> Mahalanobis OOD score -> abstention.

Writes runs/robustness/feature_ood.csv and feature_ood_latency.csv

Usage:  python feature_ood.py
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from calibration import load_model
from corruptions import augment, grayscale, low_res, mask, noise
from eda import FRESHNESS_ORDER
from robust_inference import RobustPipeline, corrupt_set, time_fn
from stress_test import load_cifar_ood
from train import GPUNormalize, load_images


@torch.no_grad()
def features_and_logits(model, X, device, bs=64):
    norm, feats, logits = GPUNormalize(device), [], []
    for i in range(0, len(X), bs):
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            f = model.backbone(norm(X[i:i + bs].to(device)))
            logits.append(model.fresh_head(f).float().cpu())
        feats.append(f.float().cpu())
    return torch.cat(feats).numpy(), torch.cat(logits)


class Mahalanobis:
    def fit(self, feats, y):
        means = np.stack([feats[y == k].mean(0) for k in range(3)])
        precision = np.linalg.pinv(np.cov(feats - means[y], rowvar=False))
        self.means, self.precision = torch.from_numpy(means).float(), torch.from_numpy(precision).float()
        return self

    def to(self, device):
        self.means, self.precision = self.means.to(device), self.precision.to(device)
        return self

    def score(self, feats):
        """Squared distance to the nearest class mean (higher = more out-of-distribution)."""
        feats = torch.as_tensor(feats, device=self.means.device)
        d = feats[:, None, :] - self.means[None]  # (n, 3, 512)
        return ((d @ self.precision) * d).sum(-1).min(1).values.cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="s4x40_robust")
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--data", default="Processed Data")
    ap.add_argument("--ood-dir", default="ood_data")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    runs_dir, data_dir = Path(args.runs_dir), Path(args.data)
    out = runs_dir / "robustness"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    thr = json.loads((out / "pipeline_thresholds.json").read_text())
    T = thr["temperature"]

    split = pd.read_csv(runs_dir / f"split_seed{args.seed}.csv")
    sel = pd.read_csv(runs_dir / args.run / "selection.csv")
    val, test = split[split["split"] == "val"], split[split["split"] == "test"]
    enc = lambda d: d["freshness"].map(FRESHNESS_ORDER.index).to_numpy()
    model = load_model(runs_dir / args.run / "best.pt", device)

    # Fitted on clean + augmented training images: corruptions the model was trained to handle are
    # in-distribution, so the detector flags unfamiliar content rather than any change in image quality
    X_sel = load_images(data_dir, sel["path"], args.workers)
    f_tr = np.concatenate([features_and_logits(model, X, device)[0]
                           for X in (X_sel, corrupt_set(X_sel, augment, args.seed))])
    maha = Mahalanobis().fit(f_tr, np.tile(enc(sel), 2)).to(device)
    f_val, _ = features_and_logits(model, load_images(data_dir, val["path"], args.workers), device)
    maha_max = float(np.percentile(maha.score(f_val), 98))

    Xt, yt = load_images(data_dir, test["path"], args.workers), enc(test)
    quality = lambda X: RobustPipeline.quality(None, X.to(device).float() / 255)

    def decide(X, ood=True):
        """0 accepted, 1 quality, 2 OOD (Mahalanobis), 3 low confidence; plus calibrated probs and scores.
        ood=False: the same pipeline with the OOD gate removed."""
        f, logits = features_and_logits(model, X, device)
        probs = (logits / T).softmax(1).numpy()
        sig, sharp = zip(*[(s.cpu(), h.cpu()) for s, h in (quality(X[i:i + 256]) for i in range(0, len(X), 256))])
        sig, sharp = torch.cat(sig).numpy(), torch.cat(sharp).numpy()
        m = maha.score(f) if ood else None
        d = np.zeros(len(X), dtype=int)
        if ood:
            d[m > maha_max] = 2
        d[(sig > thr["noise_max"]) | (sharp < thr["sharpness_min"])] = 1
        d[(d == 0) & (probs.max(1) < thr["confidence_min"])] = 3
        return d, probs, m

    rows = []
    d_clean, p_clean, m_clean = decide(Xt)
    for name, X, y in [("Clean", Xt, yt), ("Gaussian noise sigma=0.1", corrupt_set(Xt, noise(0.1), args.seed), yt),
                       ("Mask 20% centre", corrupt_set(Xt, mask(0.2, True), args.seed), yt),
                       ("Grayscale", corrupt_set(Xt, grayscale(), args.seed), yt),
                       ("Low resolution 32px (resolution of the OOD set)", corrupt_set(Xt, low_res(32), args.seed), yt)]:
        d, p, m_id = decide(X)
        ok, ok_no_gate = d == 0, decide(X, ood=False)[0] == 0
        rows.append({"condition": name, "coverage": ok.mean(), "selective_accuracy": (p.argmax(1)[ok] == y[ok]).mean(),
                     "no_gate_coverage": ok_no_gate.mean(),
                     "no_gate_selective_accuracy": (p.argmax(1)[ok_no_gate] == y[ok_no_gate]).mean(),
                     "rejected_quality": (d == 1).mean(), "rejected_ood": (d == 2).mean(), "abstained": (d == 3).mean(),
                     "mahalanobis_alone_rejects": (m_id > maha_max).mean()})
    m_lowres = m_id  # last condition: test set at 32px
    auroc = lambda ref, m: roc_auc_score(np.r_[np.zeros(len(ref)), np.ones(len(m))], np.r_[ref, m])
    for name, (Xo, _) in load_cifar_ood(args.ood_dir, 1000, args.seed).items():
        d, p, m = decide(Xo)
        rows.append({"condition": f"OOD: {name}", "coverage": (d == 0).mean(),
                     "no_gate_coverage": (decide(Xo, ood=False)[0] == 0).mean(), "rejected_quality": (d == 1).mean(),
                     "rejected_ood": (d == 2).mean(), "abstained": (d == 3).mean(),
                     "mahalanobis_alone_rejects": (m > maha_max).mean(),
                     "auroc_mahalanobis": auroc(m_clean, m), "auroc_mahalanobis_vs_32px_id": auroc(m_lowres, m),
                     "auroc_max_softmax": roc_auc_score(np.r_[np.ones(len(p_clean)), np.zeros(len(p))],
                                                        np.r_[p_clean.max(1), p.max(1)])})
    res = pd.DataFrame(rows)
    res.to_csv(out / "feature_ood.csv", index=False)

    # Latency of the new online pipeline (batch 1 and 64, GPU and CPU)
    lat = []
    for dev in (["cuda", "cpu"] if device.type == "cuda" else ["cpu"]):
        device = torch.device(dev)
        model.to(device)
        maha.to(device)
        for bs in ([1, 64] if dev == "cuda" else [1]):
            # measured back to back, so both see the same machine state (laptop CPUs throttle)
            for cfg, fn in [("ResNet only", lambda X: features_and_logits(model, X, device)),
                            ("No OOD gate", lambda X: decide(X, ood=False)),
                            ("New online pipeline", decide)]:
                t = time_fn(fn, Xt[:bs], 100 if bs == 1 else 30, dev == "cuda")
                lat.append({"device": dev, "config": cfg, "batch_size": bs, "p50_ms": np.percentile(t, 50),
                            "p90_ms": np.percentile(t, 90), "throughput_img_s": bs / np.percentile(t, 50) * 1000})
    lat = pd.DataFrame(lat)
    lat.to_csv(out / "feature_ood_latency.csv", index=False)
    print(res.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(lat.to_string(index=False, float_format=lambda v: f"{v:.2f}"))


if __name__ == "__main__":
    main()
