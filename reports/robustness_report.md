# Robustness & Security (Adversarial + Noise)

All results compare the original model (v1.1, `s4x40_reg`) with an updated robust model (v1.2,
`s4x40_robust`). The two share the architecture, selection budget and data split; the only training
difference is the augmentation described in 1.1. Each model was trained with three seeds (42, 43, 44)
on the same split. Code: `corruptions.py`, `calibration.py`, `adversarial.py`, `robustness_eval.py`,
`robust_inference.py`; every number below comes from the CSV/JSON files in `runs/robustness/`.

## 1. Robustness methods

We implemented one method at each level. The choice was driven by the Phase 3 stress tests, which
showed that the original model is stable under lighting changes but collapses under noise, occlusion
and loss of colour, and that its confidence scores are poorly calibrated.

**Data level: corruption augmentation.** During training, every image independently receives
Gaussian noise (p = 0.3, σ 0.01–0.1), Cutout-style random erasing (p = 0.3, 5–30% of the image),
Gaussian blur (p = 0.2, σ 0.5–2 px), JPEG compression (p = 0.2, quality 20–80) and grayscale
(p = 0.1). About 28% of images stay clean. The augmentation runs on the GPU inside the training loop
(`train.py --augment`). We chose it because the dataset's existing augmented copies only vary
lighting and geometry, which is exactly why brightness was harmless while noise was not. A CNN learns
the cues it is shown, so the remedy is to show it the corruptions it will meet on cheap cameras and
compressed uploads. We also kept the stratified (round-robin) selection budget, which protects the
minority class: when semi-fresh was cut to 10% of the training set in Phase 3, its recall fell from
0.72 to 0.51.

**Model level: temperature scaling.** The model was trained with label smoothing 0.2, which caps the
target probability at about 0.87, so the raw scores are systematically underconfident (ECE ≈ 0.10).
We fit one temperature on the validation set by minimising NLL. Because the fitted T ≈ 0.58 is below
1, it sharpens the scores rather than softening them. This suits our setting: it costs no accuracy,
adds a single parameter, and makes a confidence threshold mean what it says, which the abstention
rule below depends on.

**Inference level: input sanitisation with abstention.** Before classification, an image must pass:
- a quality check: a noise estimate (Immerkær) and sharpness (variance of the Laplacian), with limits
  set at the edge of the augmentation range the model was trained on;
- a CLIP OOD gate: the image's maximum cosine similarity to the 96 fruit descriptions of the
  selection head must exceed the 2nd percentile of clean validation images.

After classification, the model abstains when the calibrated confidence falls below 0.54, a
threshold that abstains on 5% of clean validation images. CLIP was the natural choice for the gate
because it already runs in our pipeline as the selection head, and Phase 3 showed that the
classifier's own softmax cannot tell an apple from a papaya (AUROC ≈ 0.6). Test-time augmentation
(flip averaging) was added as an option and appears only in the latency results.

| | Original (v1.1) | Robust (v1.2) |
|---|---|---|
| Clean accuracy (mean ± std, 3 seeds) | 85.6 ± 0.8% | 84.9 ± 0.9% |
| Clean macro-F1 | 85.3% | 84.5% |
| ECE raw → temperature-scaled | 0.104 → 0.024 | 0.095 → 0.025 |
| Fitted temperature (3 seeds) | 0.58 / 0.58 / 0.58 | 0.55 / 0.58 / 0.61 |

## 2. Adversarial resistance

**Plausible adversaries.** The realistic motive is commercial: a supplier wants rotten or semi-fresh
stock graded as fresh.
- **Black-box** (most plausible): the supplier can only submit photos and see the grade. They can
  query repeatedly and use physical tricks such as turning the bad side away, adding stickers, or
  changing the lighting.
- **Grey-box:** anyone who knows or guesses that the system is an ImageNet ResNet-18. Those weights
  are public, so a surrogate model is easy to build.
- **White-box:** an insider with the weights. Unlikely, but it sets the worst case.

**Evaluation.** On a stratified subset of 1,002 test images we ran:
- FGSM and PGD-10 (L∞), both untargeted;
- PGD-10 targeted towards "Fresh";
- a transfer attack crafted on a different model of ours (`s2x80_reg`, trained on a different
  selection) to stand in for the grey-box case;
- a universal adversarial patch, learned on validation images, that pushes non-fresh produce
  to "Fresh", compared against a plain grey patch of the same size.

Robust accuracy (%), or the share of non-fresh images classified as Fresh for the targeted attacks:

| Attack | Model | ε = 0 | 0.5/255 | 1/255 | 2/255 | 4/255 | 8/255 |
|---|---|---|---|---|---|---|---|
| FGSM | Original | 87.0 | 2.1 | 0.1 | 0.4 | 2.3 | 7.8 |
| | Robust | 86.4 | 3.0 | 0.5 | 0.3 | 1.0 | 3.8 |
| PGD-10 | Original | 87.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| | Robust | 86.4 | 0.1 | 0.0 | 0.0 | 0.0 | 0.0 |
| Transfer PGD-10 (grey-box) | Original | 87.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| | Robust | 86.4 | 1.8 | 0.0 | 0.0 | 0.0 | 0.0 |
| PGD-10 targeted → Fresh (non-fresh → Fresh) | Original | 7.0 | 100 | 100 | 100 | 100 | 100 |
| | Robust | 8.5 | 100 | 100 | 100 | 100 | 100 |

| Patch (non-fresh classified as Fresh) | Original | Robust |
|---|---|---|
| Grey patch, 5% of image | 14.1% | 11.2% |
| Adversarial patch, 5% of image | 100% | 100% |
| Grey patch, 10% of image | 17.1% | 10.5% |
| Adversarial patch, 10% of image | 100% | 100% |

These results are sobering. A perturbation of half a grey level (0.5/255), invisible to a person, is
enough to break both models completely. To rule out a bug, we checked that random sign noise of the
same size leaves accuracy unchanged (88.7% → 88.3%). Corruption augmentation is not adversarial
training, so it was not expected to help, and it does not.

The transfer attack is just as effective as the white-box one. That is the more worrying finding: our
models share the frozen ImageNet layers 1–3 and differ only in `layer4` and the heads, so an attacker
does not need our weights. The public backbone is enough.

The learned patch is the most realistic threat. A sticker covering 5% of the photo turns every rotten
item into "Fresh", while a plain grey patch of the same size moves only 11–17%. The pipeline does not
catch any of these attacks. Adversarial images get confidence close to 1.0 (Fig. 3) and look like
fruit to CLIP, so 97% of them are accepted. The honest conclusion is that the system should not be
used where a motivated adversary controls the camera. If it must be, the next step is adversarial
training plus a second, independent check on high-value decisions.

## 3. Robustness curves, calibration and failure cases

- **Figure 1** (`runs/robustness/robustness_curves.png`): test accuracy vs. corruption severity,
  mean ± std over three seeds.
- **Figure 2** (`runs/robustness/adversarial_curves.png`): robust accuracy vs. ε.
- **Figure 3** (`runs/robustness/confidence_histograms.png`): temperature-scaled confidence for clean
  correct / clean wrong / noisy / OOD / adversarial inputs.
- **Figure 4** (`runs/robustness/reliability.png`): reliability diagrams before and after temperature
  scaling.
- **Figure 5** (`runs/robustness/failure_cases.png`): failure examples.

**Robustness curves (Fig. 1).** The two models' curves overlap for brightness and contrast and move
apart for every corruption the augmentation targeted. The gap is largest for noise: at σ = 0.1
accuracy goes from 47% to 79%. For centre masking at 20% it goes from 57% to 76%.

**Calibration (Figs. 3–4).** The reliability diagrams show the raw models sitting above the diagonal
(underconfident), and temperature scaling brings ECE from about 0.10 down to 0.025 for both.
Calibration does not hold under shift for the original model, though: at noise σ = 0.1 its calibrated
ECE rises to 0.38, because it is confidently wrong. The robust model stays at 0.04.

The confidence histograms show the limits of thresholding. Clean errors spread across 0.4–0.9 and can
be abstained on, but adversarial examples pile up at 1.0, and OOD images still reach high confidence.
This is why the OOD gate has to be a separate model.

**Qualitative analysis (Fig. 5).** The failures fall into three groups:
- **Ambiguous boundary.** The clean semi-fresh images that fail sit on the label boundary: yellowing
  and brown spots on a semi-fresh bitter melon are what rot looks like, and the day-range labels give
  no visual line between the two.
- **Missing or false evidence.** When the evidence is hidden (masking, patch dropout, blur), the model
  defaults to "Fresh". When texture is added (noise), it defaults to "Rotten". The original model has
  learned "speckles mean mould" and "smooth means fresh", rather than the fruit's actual condition.
- **Forced choice.** For OOD inputs, the classifier has no "none of these" option, so a brown mushroom
  cap becomes "Rotten" at 0.96 confidence.

## 4. Quantitative results

### 4.1 Task metrics, clean and under each test

Values are means ± std over three training seeds. 95% bootstrap CIs (1,000 resamples) are given for
the seed-42 model.

| Condition | Original acc | Robust acc | Original macro-F1 | Robust macro-F1 | Semi-fresh recall (O → R) | Rotten→Fresh (O → R) |
|---|---|---|---|---|---|---|
| Clean | 85.6 ± 0.8 [85.1–88.0] | 84.9 ± 0.9 [84.4–87.4] | 85.3 | 84.5 | 0.73 → 0.70 | 13 → 15 |
| Brightness ±40% (mean) | 84.0 | 83.3 | 83.7 | 82.8 | 0.70 → 0.67 | 15 → 20 |
| Contrast ±40% (mean) | 83.6 | 83.0 | 83.2 | 82.6 | 0.70 → 0.69 | 17 → 20 |
| Noise σ = 0.05 | 68.3 ± 0.4 [65.9–69.9] | 82.5 ± 0.6 [81.7–85.0] | 66.2 | 82.0 | 0.34 → 0.65 | 3 → 25 |
| Noise σ = 0.1 | 46.8 ± 1.4 [43.3–47.6] | 79.1 ± 0.7 [77.9–81.4] | 39.6 | 78.4 | 0.12 → 0.59 | 1 → 40 |
| Noise σ = 0.2 | 39.5 ± 1.0 | 67.6 ± 1.3 | 27.3 | 66.5 | 0.03 → 0.46 | 2 → 135 |
| Blur σ = 2 px | 75.9 ± 0.8 [74.0–77.6] | 80.9 ± 1.4 [80.6–83.8] | 75.1 | 80.3 | 0.54 → 0.63 | 86 → 29 |
| Blur σ = 4 px | 57.0 ± 2.9 | 72.7 ± 0.2 | 51.8 | 71.0 | 0.16 → 0.41 | 277 → 106 |
| JPEG q = 20 | 77.6 ± 2.2 [73.5–77.1] | 82.8 ± 0.7 [81.9–85.2] | 76.4 | 82.4 | 0.54 → 0.68 | 21 → 26 |
| JPEG q = 10 | 65.1 ± 2.2 | 76.3 ± 0.2 | 62.5 | 75.9 | 0.34 → 0.64 | 53 → 62 |
| Mask 20% centre | 56.9 ± 4.0 [57.3–61.4] | 76.3 ± 1.9 [76.4–80.0] | 52.2 | 76.1 | 0.18 → 0.66 | 262 → 51 |
| Mask 30% random | 57.5 ± 3.4 | 75.5 ± 1.1 | 53.1 | 75.3 | 0.20 → 0.64 | 246 → 58 |
| Patch dropout 10% | 56.3 ± 2.2 [55.5–59.7] | 67.6 ± 4.2 [63.2–67.1] | 51.7 | 66.6 | 0.17 → 0.42 | 311 → 203 |
| Patch dropout 30% | 45.9 ± 3.4 | 53.6 ± 5.9 | 40.2 | 50.8 | 0.14 → 0.32 | 408 → 352 |
| Grayscale | 62.5 ± 1.1 [59.6–63.6] | 77.7 ± 2.2 [78.6–81.9] | 61.1 | 76.9 | 0.42 → 0.58 | 13 → 42 |
| Low resolution 64 px | 74.8 ± 0.8 | 80.3 ± 1.6 | 73.5 | 79.6 | 0.48 → 0.60 | 96 → 41 |
| PGD-10 ε = 2/255 (seed 42) | 0.0 | 0.0 | 0.0 | 0.0 | — | — |

On clean data, the robust model has a macro AUROC of 0.953 ± 0.002 against 0.958 ± 0.004, and a
PR-AUC of 0.918 against 0.928. The clean-data cost of augmentation is therefore about 0.7 points of
accuracy, which is within one seed's standard deviation. In return, at moderate and severe levels of
the corruptions it targeted, it gains between 5 points (blur σ = 2 px) and 32 points (noise σ = 0.1).

The rotten→fresh column needs care. Under heavy noise, the original model's count is low only because
it labels almost everything "Rotten". The robust model makes more of these errors there because it
has stopped doing that. For occlusion and blur, where the error is a real failure mode, the count
falls sharply (e.g. 262 → 51 for centre masking).

**Full pipeline** (robust model + quality check + CLIP gate + abstention, seed 42). Coverage is the
share of images that are accepted and classified; selective accuracy is the accuracy on those images.

| Input | Original acc (no pipeline) | Robust acc (no pipeline) | Pipeline coverage | Selective accuracy |
|---|---|---|---|---|
| Clean | 86.5% | 85.9% | 93.5% | 87.9% |
| Noise σ = 0.1 | 45.4% | 79.5% | 90.1% | 82.4% |
| Noise σ = 0.2 | 39.2% | 66.4% | 0% (all rejected: quality) | — |
| Blur σ = 4 px | 57.9% | 73.5% | 23.7% | 80.5% |
| Mask 20% centre | 59.6% | 78.2% | 70.6% | 83.5% (rotten→fresh 160 → 22) |
| Grayscale | 61.6% | 80.3% | 70.1% | 84.9% |

| OOD input | Original: confidence ≥ 0.6 | Pipeline: accepted | CLIP gate alone rejects |
|---|---|---|---|
| Unseen produce (CIFAR-100 apple, pear, pepper, mushroom) | 49.5% | 20.0% | 76.0% |
| Non-fruit objects (CIFAR-100) | 47.6% | 0.2% | 99.8% |
| Our test set shrunk to 32 px (resolution control) | — | 73.0% | 15.9% |

### 4.2 Latency and throughput

Measured on an RTX 4050 Laptop GPU and an i7-13620H CPU. p50 / p90 in ms; throughput at batch 64.

| Configuration | Input | GPU bs 1 p50 / p90 | GPU bs 64 throughput | CPU bs 1 p50 / p90 |
|---|---|---|---|---|
| Original model | clean | 3.7 / 6.1 | 1,218 img/s | 18.0 / 19.9 |
| Original model | noisy / adversarial | 3.0 / 3.6 | 1,371–1,410 img/s | 16.7–17.2 / 18.5–19.3 |
| Robust model | clean / noisy / adversarial | 3.0 / 3.3 | 1,407–1,432 img/s | 16.4–16.6 / 18.4–18.9 |
| Robust + temperature | clean | 2.9 / 3.1 | 1,431 img/s | 16.7 / 18.7 |
| Quality check alone | clean | 0.4 / 0.5 | 19,688 img/s | 0.6 / 0.7 |
| CLIP gate alone | clean | 4.9 / 5.1 | 975 img/s | 40.0 / 43.4 |
| **Full pipeline** | clean / noisy / adversarial | 9.3–9.6 / 9.9–11.7 | 596–611 img/s | 56.2–58.7 / 61.9–62.6 |
| Full pipeline + TTA | clean | 12.8 / 14.5 | 490 img/s | 74.4 / 80.5 |

A CNN performs the same computation whatever the image contains, so clean, noisy and adversarial
inputs run at the same speed. The original-vs-robust gap at batch 1 (3.7 vs 3.0 ms) is GPU warm-up
noise, since the two networks are identical in size. The real cost of robustness is the CLIP gate,
which roughly triples GPU latency and halves throughput.

On the GPU, the full pipeline still meets our 10 ms p95 target (p90 9.9 ms on clean images). On the
CPU it misses the 50 ms target (p90 62 ms). For CPU-only deployment, we would run the gate only on
images the classifier is unsure about, or drop TTA, which adds 3–4 ms on the GPU and about 17 ms on
the CPU for no measured gain.

### 4.3 Memory and model size

| Component | Parameters | Size | Peak VRAM (bs 64, fp16) |
|---|---|---|---|
| ResNet-18, original or robust | 11.18M | 42.7 MB (fp32) | ≈330 MB |
| Temperature scaling | 1 | negligible | — |
| CLIP ViT-B/32 (OOD gate) | 151.3M | 288.5 MB (fp16) | — |
| Full pipeline | 162.5M | ≈331 MB | 668 MB |

Augmentation changes nothing at inference time: same weights, same size. It costs only training time:
mean 209 s vs. 152 s over three seeds (≈9.2 vs. ≈7 s per epoch, mostly CPU-side JPEG encoding), with
the same 341 MB of training VRAM. The OOD gate is the expensive part, making the deployed model about
8× larger and doubling inference VRAM.

### 4.4 Failure cases

The six cases below are the panels of Fig. 5, in the same order (left to right, top to bottom). Each
is the original model's most confident error in its condition. The "updated system" is the robust
model with the full pipeline (quality check, CLIP gate, abstention). Confidence is the softmax score;
the robust model's score is temperature-scaled. The full list of cases is in `failure_cases.csv`.

| # (panel) | Input | Expected | Original prediction | Why it failed | Mitigation | Updated system |
|---|---|---|---|---|---|---|
| 1 (top left) | Clean photo, semi-fresh bitter melon with yellowing | Semi-fresh | Rotten (0.94) | Yellow-brown patches are shared by late semi-fresh and early rotten; the label is a day range, not a visual rule | More boundary examples; ordinal loss | Still Rotten (0.98): **unresolved** |
| 2 (top right) | Rotten fruit, centre 20% masked | Rotten | Fresh (0.83) | The mask hides the decayed area; the visible skin looks healthy | Cutout augmentation; OOD gate | **Rejected: OOD** (CLIP 0.276 < 0.295) |
| 3 (middle left) | CIFAR-100 mushroom (OOD) | none | Rotten (0.96) | Closed-set classifier has no "none of these" option; the brown cap resembles rot | OOD gate | **Rejected: OOD** (CLIP 0.251) |
| 4 (middle right) | Rotten fruit + PGD ε = 2/255, targeted to Fresh | Rotten | Fresh (1.00) | Invisible perturbation along the model's gradient | Adversarial training; keep weights private | Still Fresh (1.00): **unresolved** |
| 5 (bottom left) | Fresh fruit + noise σ = 0.1 | Fresh | Rotten (0.97) | The model learned "speckles mean mould" | Noise augmentation | **Fresh (0.76): resolved** |
| 6 (bottom right) | Grayscale semi-fresh fruit | Semi-fresh | Rotten (0.86) | Without colour, dark tone and texture are read as rot | Grayscale augmentation | **Semi-fresh (0.56): resolved** |

Cases 5 and 6 are resolved by the robust model itself. It now gets both images right, and with
moderate rather than near-certain confidence, which is what a calibrated model should do on a
degraded image.

Cases 2 and 3 are resolved by the pipeline declining to answer. The images go to a person instead of
being misclassified, which for a grading system is often the better outcome.

Two cases remain open:
- **Case 1**, the ambiguous boundary image, needs better labels rather than a more robust model.
- **Case 4**, the adversarial attack, is the important one: neither augmentation nor the gates touch
  it.

## 5. Adaptation experiment: simulated camera drift

**Drift.** We simulated a common deployment change: the shop replaces its camera with a cheaper one.
Every test image was blurred (σ = 1 px), given sensor noise (σ = 0.08) and JPEG-compressed
(quality 30). This is a covariate shift; the classes and their proportions stay the same.

**Update.** We labelled 600 new drifted images (200 per class) from the part of the train split the
selection head had not used, mixed them with 600 replayed images from the original training
selection, and fine-tuned `layer4` and the heads for 5 epochs (AdamW, lr 1e-4). The replay half is
there to stop the model forgetting clean images. Code: `adapt.py`; results:
`runs/robustness/adaptation.csv`.

| Model | Test set | Accuracy before → after | Macro-F1 before → after | Semi-fresh recall | AUROC | Rotten→Fresh |
|---|---|---|---|---|---|---|
| Original (v1.1) | Drifted | 54.0 → **77.0%** | 49.0 → 76.7% | 0.15 → 0.67 | 0.845 → 0.913 | 15 → 28 |
| Original (v1.1) | Clean | 86.5 → 84.2% | 86.1 → 84.0% | 0.72 → 0.73 | 0.955 → 0.953 | 12 → 22 |
| Robust (v1.2) | Drifted | 77.0 → **83.4%** | 75.8 → 83.0% | 0.51 → 0.70 | 0.927 → 0.940 | 76 → 20 |
| Robust (v1.2) | Clean | 85.9 → 86.5% | 85.5 → 86.2% | 0.71 → 0.75 | 0.955 → 0.966 | 15 → 12 |

The robust model absorbs most of the drift before any update: 77% against the original model's 54%.
That is the practical value of the augmentation. A 10-second fine-tune then lifts it to 83.4%,
without any loss on clean images.

The original model gains more from adaptation, but it gives up 2.3 points on clean images, and its
rotten→fresh count on clean data almost doubles (12 → 22). Even with replay, adapting a model that
was never exposed to corruption pulls it towards the new domain.

The update costs labelling 600 images and about 10 s of GPU time. Nothing changes at inference: the
architecture, the 11.18M parameters, the 42.7 MB checkpoint and the latency all stay the same.

## 6. Changes to the model, training and deployment

**Model and training (implemented above).**
- The robust v1.2 model, trained with corruption augmentation, replaces v1.1. It costs 0.7 points of
  clean accuracy, gains 5–32 points under corruption, and adapts to drift without forgetting.
- Temperature scaling is part of the model, so confidence thresholds are meaningful.
- The 3-seed evaluation becomes part of the release check.

**Deployment: removing CLIP from the request path.** The CLIP gate rejected OOD inputs well, but it
tripled latency, halved throughput and made the deployed system eight times larger. That conflicts
with the reason we chose a ResNet-18: fast, cheap answers.

We replaced it with a Mahalanobis distance on the ResNet's own 512-d features. It is computed from
the forward pass the classifier already makes, so it needs no second model. The class means and
shared covariance are fitted on the training selection plus an augmented copy of it, so corrupted
images of real fruit count as in-distribution. The threshold rejects 2% of clean validation images,
the same rule as the CLIP gate.

CLIP moves to an offline audit: it re-checks a sample of accepted images in the background and feeds
the drift dashboard. The adaptation procedure from Section 5 becomes the response when the dashboard
fires: label about 600 recent images, then run a replay fine-tune. Code: `feature_ood.py`; results:
`runs/robustness/feature_ood.csv`, `feature_ood_latency.csv`.

| Online pipeline (robust model + quality check + abstention) | With CLIP gate (before) | No OOD gate (CLIP removed) | With Mahalanobis gate (after) |
|---|---|---|---|
| Clean: coverage / selective accuracy | 93.5% / 87.9% | 95.1% / 87.6% | 92.6% / 87.8% |
| Noise σ = 0.1: coverage / selective accuracy | 90.1% / 82.4% | 91.7% / 82.4% | 78.4% / 82.5% |
| Mask 20% centre: coverage / selective accuracy | 70.6% / 83.5% | 90.7% / 82.1% | 65.3% / 81.4% |
| Grayscale: coverage / selective accuracy | 70.1% / 84.9% | 92.2% / 82.4% | 53.8% / 85.1% |
| Unseen produce (CIFAR-100): accepted | 20.0% | 82.0% | **1.0%** |
| Non-fruit objects (CIFAR-100): accepted | 0.2% | 84.1% | 0.0% |
| Our test set at 32 px: coverage | 73.0% | 87.0% | 28.9% |
| OOD separation (AUROC vs. 32 px test images, produce / non-fruit) | — | 0.72 / 0.75 (max-softmax only) | 0.89 / 0.94 |
| Extra model size | +151M parameters, 289 MB | none | 512×512 covariance + 3 means, ≈1 MB |
| Throughput change vs. ResNet alone, GPU batch 64 | −57% (1,407 → 603 img/s) | −10% (985 → 883 img/s) | −12% (985 → 870 img/s) |
| Latency change vs. ResNet alone, CPU batch 1 | +40 ms (+250%) | no measurable change (92.5 → 88.2 ms) | no measurable change (92.5 → 90.6 ms) |

The latency changes were measured back to back against the plain ResNet on the same machine. The
Mahalanobis and no-gate runs were made while the laptop was throttling: the plain ResNet took 11.7 ms
on the GPU in that run, against 3.0 ms in Section 4.2. We therefore report relative overhead. Most of
the remaining 10% comes from the quality check and the Python-side decision logic; the Mahalanobis
score itself adds only about 1.5% on top of the no-gate pipeline. Timed on its own,
the Mahalanobis score takes 0.24 ms and the quality check 0.4 ms (GPU) or about 1 ms (CPU). On an
unthrottled machine, the online path should therefore run at about 3.5 ms on the GPU and under 20 ms
on the CPU. We expect both 10 ms and 50 ms latency targets to be met again, though we did not
re-measure this unthrottled.

**The trade-off.** Removing the OOD gate altogether gives the highest coverage on corrupted fruit,
but 82–84% of OOD images are then accepted and graded, because the classifier's confidence alone
cannot separate them. That makes some gate necessary. The feature-based gate costs almost nothing on
top of the no-gate pipeline and rejects OOD produce better than CLIP did (1% accepted against 20%). It
is more conservative under heavy corruption: it rejects 41% of grayscale images that the classifier
would mostly have got right, and 65% of 32 px images.

The 32 px row is the honest caveat. The near-perfect CIFAR result is partly a resolution effect.
Against resolution-matched controls, the detector still separates OOD images clearly better than the
classifier's confidence (AUROC 0.89–0.94 vs. 0.72–0.75), so it is not only detecting blur.

For a grading system, declining to answer is acceptable; slowing down every request is not. We
therefore keep the Mahalanobis gate online and CLIP offline.

**Unchanged: adversarial robustness.** None of these changes affect the adversarial results in
Section 2. The deployment plan states that the system is only suitable where suppliers do not control
the camera. Adversarial training is the first item for the next iteration.
