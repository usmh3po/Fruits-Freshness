"""
Evaluation helpers: freshness logits under a corruption, temperature scaling, ECE and reliability diagrams.
"""

import numpy as np
import torch
import torch.nn.functional as F

from train import FreshnessNet, GPUNormalize


def load_model(path, device):
    model = FreshnessNet(dropout=0.5)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    return model.to(device).eval()


@torch.no_grad()
def predict_logits(model, X_uint8, device, corrupt=None, seed=42, bs=64, tta=False):
    """Freshness logits for uint8 images (N, 3, 224, 224) after an optional corruption.
    tta: average the logits of the image and its horizontal flip."""
    norm, g = GPUNormalize(device), torch.Generator().manual_seed(seed)
    out = []
    for i in range(0, len(X_uint8), bs):
        x = norm(X_uint8[i:i + bs].to(device), corrupt, g)
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            logits = model(x)[0].float()
            if tta:
                logits = (logits + model(x.flip(3))[0].float()) / 2
        out.append(logits.cpu())
    return torch.cat(out)


def fit_temperature(logits, y):
    """Single temperature T minimising the validation NLL of softmax(logits / T)."""
    log_t = torch.zeros(1, requires_grad=True)
    y = torch.as_tensor(y)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(logits / log_t.exp(), y)
        loss.backward()
        return loss
    opt.step(closure)
    return float(log_t.exp())


def ece(probs, y, bins=15):
    """Expected calibration error of the top-class confidence."""
    conf, pred = probs.max(1), probs.argmax(1)
    edges = np.linspace(0, 1, bins + 1)
    err = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            err += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(err)


def reliability_diagram(ax, probs, y, title, bins=10):
    conf, correct = probs.max(1), probs.argmax(1) == y
    edges = np.linspace(0, 1, bins + 1)
    centres, accs, counts = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            centres.append((lo + hi) / 2)
            accs.append(correct[m].mean())
            counts.append(m.sum())
    ax.bar(centres, accs, width=1 / bins, edgecolor="black", alpha=0.75, label="Accuracy")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"{title}\nECE = {ece(probs, y):.3f}", fontsize=10)
    ax.legend(loc="upper left", fontsize=8)
