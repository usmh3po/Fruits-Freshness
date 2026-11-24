# Fruits Freshness

Image dataset of 8 fruits/vegetables at 3 freshness levels (24 classes), with these scripts:

- **`eda.py`**: exploratory data analysis (missingness, summary stats, duplicates, outliers, plots).
- **`clip_selection.py`**: VLM selection head. Uses CLIP to match every image against four text
  descriptions per class (96 in total) and selects a fixed number of images per description, then
  checks the selection against the true labels.
- **`train.py`**: splits the data, selects a training budget with the selection head, labels it with
  the true labels (oracle) and fine-tunes a ResNet-18 freshness classifier.
- **`summarize_runs.py`**: compares training runs and adds a zero-shot CLIP baseline.
- **`benchmark_inference.py`**: FLOPs, latency (p50/p90), throughput, memory, and the effect of fp16
  and int8 quantization for a trained run (`runs/<run>/efficiency.json`).
- **`stress_test.py`**: robustness tests (corruptions, feature loss, CIFAR-100 OOD images, class
  rarity), written to `runs/<run>/stress/`. The rare-class model is trained with
  `train.py ... --rare-class "Semi Fresh" --rare-share 0.1`.
- **`corruptions.py`**, **`calibration.py`**: shared corruptions / training augmentation, and
  temperature scaling, ECE and reliability diagrams.
- **`adversarial.py`**, **`robustness_eval.py`**, **`robust_inference.py`**: robustness and security
  evaluation of the original (v1.1) vs. robust (v1.2) model (see "4. Robustness" below).
- **`adapt.py`**, **`feature_ood.py`**: drift adaptation experiment, and the final online pipeline with a
  Mahalanobis OOD gate.
- **`continual.py`**: continual learning (class-balanced replay), drift monitoring, release gate,
  versioning, and the active-learning / human-in-the-loop simulation.
- **`compare_backbones.py`**: ResNet-18 vs. Swin-T on the Milestone 2 setup (3 seeds each).
- **`demo.py`**: end-to-end demo: load → infer → detect drift → query human → update model.

Results write-up: [reports/training_report.md](reports/training_report.md),
[reports/robustness_report.md](reports/robustness_report.md),
[reports/continual_learning_report.md](reports/continual_learning_report.md).

### Pipeline overview

| Stage | Script(s) | Outputs |
|---|---|---|
| Data (EDA, cleaning, split) | `eda.py`, `train.py` (split) | `eda_output/`, `runs/split_seed42.csv` |
| Active-learning head (VLM selection) | `clip_selection.py` | `clip_output/` |
| Training | `train.py`, `summarize_runs.py` | `runs/<run>/` |
| Efficiency | `benchmark_inference.py` | `runs/<run>/efficiency.json` |
| Robustness tests | `stress_test.py`, `robustness_eval.py`, `adversarial.py` | `runs/<run>/stress/`, `runs/robustness/` |
| Monitoring, OOD gate, abstention | `robust_inference.py`, `feature_ood.py` | `runs/robustness/` |
| Adaptation | `adapt.py` | `runs/robustness/adaptation.csv` |
| Continual learning | `continual.py` | `runs/continual/` |
| HITL / active learning | `continual.py --methods ...` | `runs/continual_al360/` |
| Backbone comparison | `train.py --backbone swin_t`, `compare_backbones.py` | `runs/backbones/` |
| End-to-end demo | `demo.py` | `demo_output/` |

## Quick start: end-to-end demo

After the setup below, with the dataset in `Processed Data/` and the robust model trained (step 4, or
just `python train.py ... --augment --run-name s4x40_robust`):

```bash
python demo.py                 # simulated annotator, about 1-2 minutes on a GPU
python demo.py --interactive   # you type the labels (f / s / r) for the queried images
```

The demo loads the deployed robust ResNet-18, grades a clean batch (routing low-confidence images to
manual inspection), receives a batch from a cheaper camera, detects the drift with the OOD-gate
rejection rate and class-share shift, sends 300 images to a human (half the ones the model is least
sure about, half a random sample; labels simulated with the dataset labels), runs a replay fine-tune
(300 new + 600 replayed images) with the release gate, and reports accuracy before and after on the
full test set. Outputs in `demo_output/`: `demo_report.json`, `demo_summary.png`,
`queried_images.csv` and `model_updated.pt` (if the gate passes).

## Dataset

The images are not included in the repository. Place them in `Processed Data/`, one folder per class:

```
Processed Data/
├── Fresh Banana(1-4)/
├── Semi fresh banana(4-7)/
├── Rotten banana(7-13)/
└── ...  (24 folders)
```

Folder names follow `<Freshness> <Fruit>(<day_min>-<day_max>)`. Freshness is `Fresh`, `Semi fresh`
or `Rotten` (case, spaces and `_` are tolerated); the day range is how many days old the produce was.

- Fruits: Banana, Bittermelon, Cucumber, Eggplant, Orange, Papaya, Pineapple, Tomato
- 14,117 images, 512×512 RGB, ~590 per class (about 63% are augmented copies, named `aug_<n>_<original>`)
- 17 photos that were duplicated in both *Fresh Orange* and *Semi fresh Orange* were removed from
  *Fresh Orange*, with their 26 augmented copies (backed up in `Removed Data/`, not in the repo).

## Setup

Python 3.10+.

```bash
pip install -r requirements.txt

# PyTorch with CUDA (pick the index matching your driver: https://pytorch.org/get-started/locally/)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

Check that the GPU is visible:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

`clip_selection.py` also runs on CPU, but embedding all images is much slower.

## Running

Run from the project root.

### 1. Exploratory data analysis

```bash
python eda.py                       # options: --data "Processed Data" --out eda_output
```

Writes to `eda_output/`:

| File | Contents |
|---|---|
| `report.txt` | All results in one readable file |
| `image_metadata.csv` | One row per image: label, size, colour stats, hashes |
| `schema.csv`, `example_rows.csv` | Column types and sample rows |
| `missingness.csv` | Missing values per column |
| `summary_stats.csv`, `mean_pixel_stats.csv` | Numeric summaries, overall and per freshness level |
| `class_counts.csv` | Images per fruit × freshness |
| `duplicates_exact.csv`, `duplicates_near.csv` | Identical files (MD5) and visually identical images (perceptual hash) |
| `outliers.csv` | IQR outliers on file size, brightness, contrast |
| `*.png` | Class distribution, histograms, correlation heatmap, sample images |

### 2. CLIP selection

```bash
python clip_selection.py                      # 50 images per sentence (200 per class)
python clip_selection.py --top-k 100          # bigger budget
python clip_selection.py --originals-only     # ignore aug_* images
python clip_selection.py --model openai/clip-vit-large-patch14
```

How it works:

1. Uses four hand-written sentences per class (`CLASS_SENTENCES` in the script), describing what the
   photos show, e.g. *"a photo of a rotten papaya with white mold patches"*.
2. Embeds all images and sentences with CLIP (L2-normalised), and computes the cosine similarity of
   every image with every sentence.
3. Selects images round-robin: each round, every sentence takes the not-yet-selected image most
   similar to it, until each has `--top-k` images. An image is never selected twice.
4. Compares the selected images with their folder labels and reports precision per class.

Writes to `clip_output/`:

| File | Contents |
|---|---|
| `sentences.csv` | The 96 sentences and the class each one describes |
| `similarities.csv` | Similarity of every image with all 96 sentences (`s0`–`s95`) |
| `selections.csv` | Selected images, the sentence that picked them, pick round, similarity, correct or not |
| `precision.csv` | Per sentence: images selected, correct, precision, most common true label |
| `precision_per_class.csv` | The same, with all sentences of each class combined |
| `distribution_freshness.csv`, `distribution_fruit.csv` | Share of each freshness level / fruit among selected images and CLIP's predictions |
| `image_embeddings_<model>.npy` | Cached image embeddings (reused on the next run; delete to recompute) |

All image paths in the CSVs are relative to the data folder, e.g. `Fresh Banana(1-4)/IMG_001.jpg`.

### 3. Training

Needs `clip_output/similarities.csv` and `sentences.csv` from step 2.

Configuration `v1.1` (used in the report) adds stronger regularization to the script defaults:

```bash
REG="--weight-decay 0.05 --dropout 0.5 --label-smoothing 0.2 --lr-stage2 3e-5 --config-version v1.1"
python train.py --sentences-per-class 4 --per-sentence 40  --run-name s4x40_reg  $REG   # final model
python train.py --sentences-per-class 2 --per-sentence 80  --run-name s2x80_reg  $REG   # ablation
python train.py --sentences-per-class 1 --per-sentence 160 --run-name s1x160_reg $REG   # ablation
python summarize_runs.py --runs s4x40_reg s2x80_reg s1x160_reg
```

What `train.py` does:

1. **Split:** 70/15/15 of the original photos per class folder (seed 42). Augmented copies go to the
   same split as their original. The split is saved to `runs/split_seed42.csv` and reused.
2. **Selection:** CLIP round-robin selection on the train split only, using the first
   `--sentences-per-class` sentences of each class and `--per-sentence` images per sentence.
3. **Oracle labelling:** selected images are trained on with their true folder labels. The selection
   precision (CLIP's sentence vs. the true label) is stored separately.
4. **Training:** ImageNet-pretrained ResNet-18 with a freshness head (3 classes) and a fruit head
   (8 classes, auxiliary, loss weight 0.5). Stage 1 trains only the heads (5 epochs, lr 1e-3).
   Stage 2 also trains `layer4` (lr 1e-4, cosine), with early stopping on validation macro-F1
   (patience 5). Images are resized to 256×256, centre-cropped to 224 and ImageNet-normalised.
   No extra augmentation is applied, because the data is already augmented.
5. **Evaluation:** freshness metrics on validation and test, plus training and inference efficiency.

Writes to `runs/<run-name>/`:

| File | Contents |
|---|---|
| `config.json` | Versioned configuration (seed, learning rates, batch size, ...) |
| `selection.csv`, `selection_precision.json` | Selected training images and the selection head's precision |
| `history.csv`, `learning_curves.png` | Loss, accuracy and macro-F1 per epoch |
| `metrics.json` | Val/test accuracy, per-class precision/recall/F1, macro/micro F1, AUROC, PR-AUC, efficiency |
| `confusion_test.png` | Test confusion matrix |
| `best.pt` | Best checkpoint (not in the repo) |

`summarize_runs.py` writes `runs/summary.csv` (one row per run), `runs/ablation_test.png` (test
accuracy and F1 per run) and `runs/zero_shot_clip.json`.

### 4. Robustness (original v1.1 vs. robust v1.2)

The robust model adds training augmentation (noise, blur, JPEG, cutout, grayscale; `--augment`). Both
are trained with three seeds on the same split (`--split-seed 42`). Needs CIFAR-100, which
`stress_test.py` / `robustness_eval.py` download to `ood_data/` (not in the repo).

```bash
REG="--sentences-per-class 4 --per-sentence 40 --weight-decay 0.05 --dropout 0.5 --label-smoothing 0.2 --lr-stage2 3e-5 --split-seed 42"
for s in 43 44; do python train.py $REG --seed $s --config-version v1.1 --run-name s4x40_reg_seed$s; done
python train.py $REG --seed 42 --augment --config-version v1.2 --run-name s4x40_robust
for s in 43 44; do python train.py $REG --seed $s --augment --config-version v1.2 --run-name s4x40_robust_seed$s; done

python adversarial.py         # FGSM, PGD, transfer and patch attacks   -> runs/robustness/adversarial.csv
python robustness_eval.py     # corruptions x 3 seeds, CIs, temperature scaling, plots
python robust_inference.py    # quality check + CLIP OOD gate + abstention, latency, failure cases
python adapt.py               # simulated camera drift -> replay fine-tune -> before/after metrics
python feature_ood.py         # final online pipeline: Mahalanobis OOD score replaces the CLIP gate
```

Everything is written to `runs/robustness/`: `corruption_seeds.csv`, `corruption_summary.csv`,
`temperatures.json`, `adversarial.csv`, `pipeline_results.csv`, `pipeline_thresholds.json`,
`latency.csv`, `memory.json`, `failure_cases.csv`, `adaptation.csv`, `feature_ood.csv`,
`feature_ood_latency.csv`, and the plots `robustness_curves.png`, `adversarial_curves.png`,
`reliability.png`, `confidence_histograms.png`, `failure_cases.png`.

### 5. Continual learning

Six drift periods (T0 clean, T1 low light, T2 cheap camera, T3 mostly fresh stock, T4 warm lighting,
T5 back to clean), 600 newly labelled images per period, starting from the robust v1.2 model:

```bash
python continual.py --methods "Static (no update)" "Naive fine-tune" "Replay, no gate" "Replay (ours)" "Cumulative retrain"
python continual.py --plot-only   # redraw the figures from the saved CSVs
```

Writes to `runs/continual/`: `results.csv` (before/after each update), `accuracy_matrix.csv`,
`summary.csv` (average accuracy, forgetting, backward transfer), `monitor.csv`, `efficiency.csv`,
`latency.csv`, the plots `trajectories.png` (also `trajectory_*.png`), `accuracy_matrix.png`,
`monitor.png`, and one folder per approved model version `v1.3.k/` (`config.json` with parent version
and data hash, `metrics.json`, `buffer_manifest.csv`, `best.pt`).

### 6. Active learning / human-in-the-loop simulation

The same cycles, but only 360 of each period's 600 images are labelled: chosen by the CLIP
active-learning head (5 sentences per class, 3 images per sentence) or at random (control). Labels come
from the dataset (simulated annotators):

```bash
python continual.py --methods "Static (no update)" "Replay + AL 360" "Replay + random 360" --out-name continual_al360
```

Writes the same files to `runs/continual_al360/`, plus `al_selection.csv` (selection time, VRAM,
CLIP precision, class mix of the labelled images) and `active_learning_cycles.png`.

### 7. Backbone comparison (ResNet-18 vs. Swin-T)

Same 3,840 selected images, split, preprocessing and v1.1 recipe; three seeds per setup. The graphs
also include the ResNet-18 sentence-count ablations (1 × 160, 2 × 80). The ResNet-18 4 × 40 seeds are
the v1.1 runs from step 4; seed 42 of the ablations comes from step 3.

```bash
REG="--weight-decay 0.05 --dropout 0.5 --label-smoothing 0.2 --lr-stage2 3e-5 --split-seed 42"
for s in 42 43 44; do python train.py --sentences-per-class 4 --per-sentence 40 $REG --seed $s --backbone swin_t --config-version v1.1-swin --run-name swin_t_seed$s; done
for s in 43 44; do
  python train.py --sentences-per-class 1 --per-sentence 160 $REG --seed $s --config-version v1.1 --run-name s1x160_reg_seed$s
  python train.py --sentences-per-class 2 --per-sentence 80  $REG --seed $s --config-version v1.1 --run-name s2x80_reg_seed$s
done
python compare_backbones.py
```

Writes to `runs/backbones/`: `comparison.csv`, `summary.csv`, `plot_data.csv`, `latency.csv`, and the
plots `test_accuracy.png`, `test_f1.png`, `backbone_comparison.png`, `latency.png`.

`plot_data.csv` holds the numbers behind the accuracy / F1 graphs (one row per setup and seed: `setup`,
`label` with ` | ` as line break, `colour`, `seed`, `accuracy`, `f1_macro`). After editing it, or
`latency.csv`, redraw every graph without retraining:

```bash
python compare_backbones.py --plot-only
```

`plot_ablation_3seeds.py` draws the same layout as `runs/ablation_test.png` (test accuracy + per-class
test F1) for these four setups, as 3-seed means with standard-deviation error bars, from the saved
`metrics.json` files (no retraining). It writes `runs/backbones/ablation_3seeds.csv` (one row per setup
and seed), `ablation_3seeds_summary.csv` and `ablation_test_3seeds.png`; after editing the CSV, redraw
with `python plot_ablation_3seeds.py --from-csv`.

## Reproducibility

- **Seeds:** every script takes `--seed` (default 42). The data split is built once and saved to
  `runs/split_seed42.csv`; all runs reuse it (`--split-seed 42` fixes it when the training seed varies).
- **Configurations** are versioned in each run's `config.json` (v1.1 original, v1.2 robust, v1.3.k
  continual updates, v1.1-swin), together with the Python / PyTorch versions and GPU.
- **Paths** are relative to the project root; image paths in all CSVs are relative to `Processed Data/`.
- **GPU non-determinism:** training on a GPU is not bit-for-bit reproducible. Repeated runs of the same
  command land within about ±1.5 points of accuracy of the reported numbers (multi-seed results report
  mean ± std for this reason).
- **Environment used for the reported results:** Windows 11, Python 3.12, PyTorch 2.14.1 (CUDA 13.0),
  torchvision 0.29.1, transformers 5.18, NVIDIA RTX 4050 Laptop GPU (6 GB).

## Saved output artifacts

Checkpoints (`*.pt`), cached embeddings and logs are not committed (see `.gitignore`), but every result
table and figure is: for example `runs/s4x40_reg/metrics.json`, `runs/robustness/corruption_summary.csv`,
`runs/continual/summary.csv`, `runs/continual_al360/active_learning_cycles.png`,
`runs/backbones/backbone_comparison.png` and `demo_output/demo_report.json`. Rerunning a script
regenerates its files in place.

## Notes

- `clip_selection.py` imports helpers from `eda.py`, and `train.py` imports from both, so keep them in
  the same folder.
- Images, the `Processed Data/` and `Removed Data/` folders, cached embeddings, checkpoints and logs are
  excluded by `.gitignore`.
