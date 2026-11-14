"""
Image corruptions shared by the stress tests, the robustness evaluation and training augmentation.

Every corruption takes a float batch in [0, 1] (N, 3, H, W) and a CPU torch.Generator, and returns a
batch of the same shape. CORRUPTIONS lists the stress-test conditions; augment() is the random
training-time mix used by the v1.2 (robust) model.
"""

import math

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision.io import decode_jpeg, encode_jpeg

GREY = torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)  # ImageNet mean: 0 after normalisation


def brightness(f):
    return lambda x, g: (x * f).clamp(0, 1)


def contrast(f):
    def apply(x, g):
        mean = TF.rgb_to_grayscale(x).mean((1, 2, 3), keepdim=True)
        return (mean + (x - mean) * f).clamp(0, 1)
    return apply


def noise(sigma):
    return lambda x, g: (x + sigma * torch.randn(x.shape, generator=g).to(x.device)).clamp(0, 1)


def blur(sigma):
    k = 2 * math.ceil(3 * sigma) + 1
    return lambda x, g: TF.gaussian_blur(x, [k, k], [sigma, sigma])


def jpeg(quality):
    def apply(x, g):
        u8 = (x * 255).round().byte().cpu()
        return torch.stack(decode_jpeg(encode_jpeg(list(u8), quality=quality))).to(x.device).float() / 255
    return apply


def mask(frac, centre):
    """Grey square covering frac of the image, at the centre (on the fruit) or at a random position."""
    def apply(x, g):
        x = x.clone()
        n, _, h, w = x.shape
        s = round(math.sqrt(frac) * h)
        for i in range(n):
            top, left = ((h - s) // 2, (w - s) // 2) if centre else \
                (int(torch.randint(0, h - s + 1, (1,), generator=g)), int(torch.randint(0, w - s + 1, (1,), generator=g)))
            x[i, :, top:top + s, left:left + s] = GREY.to(x.device)[0]
        return x
    return apply


def patch_dropout(frac, patch=16):
    """Image analogue of token dropout: grey out a random fraction of 16x16 patches."""
    def apply(x, g):
        n, _, h, w = x.shape
        keep = (torch.rand(n, 1, h // patch, w // patch, generator=g) >= frac).float().to(x.device)
        keep = F.interpolate(keep, scale_factor=patch, mode="nearest")
        return x * keep + GREY.to(x.device) * (1 - keep)
    return apply


def grayscale():
    return lambda x, g: TF.rgb_to_grayscale(x, 3)


def low_res(size):
    def apply(x, g):
        small = F.interpolate(x, size=(size, size), mode="bilinear", antialias=True, align_corners=False)
        return F.interpolate(small, size=x.shape[-2:], mode="bilinear", align_corners=False).clamp(0, 1)
    return apply


def clean():
    return lambda x, g: x


# (name, group, severity for plotting, corruption)
CORRUPTIONS = [
    ("Clean (baseline)", "none", 0, clean()),
    *[(f"Brightness {s:+d}%", "Brightness", s, brightness(1 + s / 100)) for s in (-40, -20, 20, 40)],
    *[(f"Contrast {s:+d}%", "Contrast", s, contrast(1 + s / 100)) for s in (-40, -20, 20, 40)],
    *[(f"Gaussian noise sigma={s}", "Gaussian noise", s, noise(s)) for s in (0.05, 0.1, 0.2)],
    *[(f"Gaussian blur sigma={s}px", "Gaussian blur", s, blur(s)) for s in (1, 2, 4)],
    *[(f"JPEG quality {q}", "JPEG quality", q, jpeg(q)) for q in (50, 20, 10)],
    *[(f"Mask {int(p * 100)}% centre", "Mask (centre)", p, mask(p, True)) for p in (0.1, 0.2, 0.3)],
    *[(f"Mask {int(p * 100)}% random", "Mask (random)", p, mask(p, False)) for p in (0.1, 0.2, 0.3)],
    *[(f"Patch dropout {int(p * 100)}%", "Patch dropout", p, patch_dropout(p)) for p in (0.1, 0.3)],
    ("Grayscale", "Feature loss", 1, grayscale()),
    ("Low resolution 64px", "Feature loss", 2, low_res(64)),
]


def augment(x, g):
    """Training-time robustness augmentation (v1.2). Each image independently gets each corruption
    with a fixed probability, at a random strength; about 28% of images stay unchanged."""
    n = len(x)
    pick = lambda p: torch.nonzero(torch.rand(n, generator=g) < p).flatten().tolist()
    uniform = lambda a, b: float(torch.empty(1).uniform_(a, b, generator=g))
    x = x.clone()
    if idx := pick(0.1):
        x[idx] = grayscale()(x[idx], g)
    if idx := pick(0.2):
        x[idx] = blur(round(uniform(0.5, 2.0), 1))(x[idx], g)
    if idx := pick(0.2):
        x[idx] = jpeg(int(uniform(20, 80)))(x[idx], g)
    for i in pick(0.3):  # Cutout / random erasing
        x[i:i + 1] = mask(uniform(0.05, 0.3), centre=False)(x[i:i + 1], g)
    if idx := pick(0.3):
        sigma = torch.empty(len(idx)).uniform_(0.01, 0.1, generator=g).view(-1, 1, 1, 1).to(x.device)
        x[idx] = (x[idx] + sigma * torch.randn(x[idx].shape, generator=g).to(x.device)).clamp(0, 1)
    return x
