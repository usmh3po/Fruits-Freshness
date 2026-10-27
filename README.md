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

Results write-up: [reports/training_report.md](reports/training_report.md).

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

## Notes

- `clip_selection.py` imports helpers from `eda.py`, and `train.py` imports from both, so keep them in
  the same folder.
- Images, the `Processed Data/` and `Removed Data/` folders, cached embeddings, checkpoints and logs are
  excluded by `.gitignore`.
