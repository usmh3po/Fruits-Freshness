"""
Adversarial evaluation of the freshness classifier.

  - FGSM and PGD-10 (L-inf, white-box), untargeted and targeted towards "Fresh"
  - PGD-10 crafted on a surrogate model and transferred to the target (grey-box)
  - Universal adversarial patch (physical-style sticker) trained to push rotten / semi-fresh produce to
    "Fresh", compared with a plain grey patch of the same size

Attacks run in fp32 on a fixed stratified test subset. Writes runs/robustness/adversarial.csv.

Usage:  python adversarial.py --runs s4x40_reg s4x40_robust --surrogate s2x80_reg
"""

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score

from calibration import load_model
from eda import FRESHNESS_ORDER
from train import IMAGENET_MEAN, IMAGENET_STD, load_images

EPS = [0, 0.5, 1, 2, 4, 8]  # L-inf budgets in /255 pixel units
PATCH_AREAS = [0.05, 0.10]
FRESH = 0


def normalise(x):
    mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


def attack(model, x, y, eps, steps, target=None, g=None):
    """L-inf attack on [0, 1] images. steps=1 is FGSM; steps>1 is PGD with a random start and step
    size 2.5 * eps / steps. Untargeted: increase the loss of y. Targeted: decrease the loss of target."""
    if eps == 0:
        return x
    delta = torch.zeros_like(x) if steps == 1 else (torch.rand(x.shape, generator=g).to(x.device) * 2 - 1) * eps
    alpha = eps if steps == 1 else 2.5 * eps / steps
    for _ in range(steps):
        delta.requires_grad_(True)
        logits = model(normalise((x + delta).clamp(0, 1)))[0]
        loss = (F.cross_entropy(logits, y) if target is None
                else -F.cross_entropy(logits, torch.full_like(y, target)))
        grad, = torch.autograd.grad(loss, delta)
        delta = (delta + alpha * grad.sign()).clamp(-eps, eps).detach()
    return (x + delta).clamp(0, 1)


def adversarial_probs(source, target_model, X, y, eps, steps, targeted, device, seed, bs=50):
    """Craft on `source`, evaluate on `target_model` (the same model for white-box attacks)."""
    g = torch.Generator().manual_seed(seed)
    probs = []
    for i in range(0, len(X), bs):
        x = X[i:i + bs].to(device).float() / 255
        xa = attack(source, x, torch.as_tensor(y[i:i + bs], device=device), eps, steps,
                    FRESH if targeted else None, g)
        with torch.no_grad():
            probs.append(target_model(normalise(xa))[0].softmax(1).cpu())
    return torch.cat(probs).numpy()


def paste(x, patch, g):
    """Paste the patch at a random position in every image."""
    n, _, h, w = x.shape
    s = patch.shape[-1]
    out = x.clone()
    for i in range(n):
        top, left = (int(torch.randint(0, h - s + 1, (1,), generator=g)), int(torch.randint(0, w - s + 1, (1,), generator=g)))
        out[i, :, top:top + s, left:left + s] = patch[0]
    return out


def train_patch(model, X, area, device, seed, steps=300, bs=32):
    """Universal patch that makes the model predict Fresh, trained on non-fresh validation images."""
    g = torch.Generator().manual_seed(seed)
    s = round(math.sqrt(area) * 224)
    patch = torch.rand(1, 3, s, s, generator=g).to(device).requires_grad_(True)
    opt = torch.optim.Adam([patch], lr=0.03)
    for _ in range(steps):
        x = X[torch.randint(0, len(X), (bs,), generator=g)].to(device).float() / 255
        loss = F.cross_entropy(model(normalise(paste(x, patch, g)))[0], torch.full((bs,), FRESH, device=device))
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            patch.clamp_(0, 1)
    return patch.detach()


@torch.no_grad()
def patched_probs(model, X, patch, device, seed, bs=64):
    g = torch.Generator().manual_seed(seed)
    return torch.cat([model(normalise(paste(X[i:i + bs].to(device).float() / 255, patch, g)))[0].softmax(1).cpu()
                      for i in range(0, len(X), bs)]).numpy()


def stratified_subset(df, per_class, seed):
    return df.groupby("freshness").sample(per_class, random_state=seed).reset_index(drop=True)


def metrics(probs, y, clean_pred=None):
    pred = probs.argmax(1)
    non_fresh = y != FRESH
    m = {"accuracy": accuracy_score(y, pred), "f1_macro": f1_score(y, pred, average="macro"),
         "rotten_to_fresh": int(((y == 2) & (pred == FRESH)).sum()),
         "non_fresh_predicted_fresh": float((pred[non_fresh] == FRESH).mean()),
         "mean_confidence": float(probs.max(1).mean())}
    if clean_pred is not None:  # share of originally correct images that the attack flips
        ok = clean_pred == y
        m["attack_success"] = float((pred[ok] != y[ok]).mean())
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=["s4x40_reg", "s4x40_robust"])
    ap.add_argument("--surrogate", default="s2x80_reg", help="grey-box surrogate for transfer attacks")
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--data", default="Processed Data")
    ap.add_argument("--per-class", type=int, default=334)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    runs_dir = Path(args.runs_dir)
    out = runs_dir / "robustness"
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    split = pd.read_csv(runs_dir / f"split_seed{args.seed}.csv")
    test = stratified_subset(split[split["split"] == "test"], args.per_class, args.seed)
    X = load_images(Path(args.data), test["path"], args.workers)
    y = test["freshness"].map(FRESHNESS_ORDER.index).to_numpy()
    val = split[(split["split"] == "val") & (split["freshness"] != FRESHNESS_ORDER[FRESH])]
    X_patch_train = load_images(Path(args.data), val["path"], args.workers)
    non_fresh = y != FRESH
    print(f"Attack subset: {len(X)} test images; patch training: {len(X_patch_train)} non-fresh val images")

    def frozen(run):
        m = load_model(runs_dir / run / "best.pt", device)
        for p in m.parameters():
            p.requires_grad_(False)
        return m

    surrogate = frozen(args.surrogate)
    rows = []
    for run in args.runs:
        model = frozen(run)
        clean_pred = adversarial_probs(model, model, X, y, 0, 1, False, device, args.seed).argmax(1)
        for attack_name, source, steps, targeted in [("FGSM", model, 1, False), ("PGD-10", model, 10, False),
                                                     ("PGD-10 targeted -> Fresh", model, 10, True),
                                                     (f"Transfer PGD-10 from {args.surrogate}", surrogate, 10, False)]:
            for eps in EPS:
                p = adversarial_probs(source, model, X, y, eps / 255, steps, targeted, device, args.seed)
                rows.append({"model": run, "attack": attack_name, "epsilon_255": eps, **metrics(p, y, clean_pred)})
                print(f"{run:14s} {attack_name:32s} eps {eps:>3}/255  acc {rows[-1]['accuracy']:.3f}  "
                      f"non-fresh->fresh {rows[-1]['non_fresh_predicted_fresh']:.3f}")
        # Patches are evaluated on the non-fresh test images only (the attacker's goal is "pass as fresh")
        for area in PATCH_AREAS:
            s = round(math.sqrt(area) * 224)
            for kind, patch in [("Grey patch", torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1).expand(1, 3, s, s)),
                                ("Adversarial patch", train_patch(model, X_patch_train, area, device, args.seed))]:
                p = patched_probs(model, X[non_fresh], patch, device, args.seed)
                rows.append({"model": run, "attack": kind, "patch_area": area,
                             **metrics(p, y[non_fresh], clean_pred[non_fresh])})
                print(f"{run:14s} {kind:32s} area {area:.0%}  non-fresh->fresh {rows[-1]['non_fresh_predicted_fresh']:.3f}")
                if kind == "Adversarial patch":
                    torch.save(patch.cpu(), out / f"patch_{run}_{int(area * 100)}.pt")
        del model
        torch.cuda.empty_cache()
    pd.DataFrame(rows).to_csv(out / "adversarial.csv", index=False)
    print(f"\nWritten to {out}/adversarial.csv")


if __name__ == "__main__":
    main()
