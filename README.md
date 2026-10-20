# Fruits Freshness

Image dataset of 8 fruits/vegetables at 3 freshness levels (24 classes), with two scripts:

- **`eda.py`**: exploratory data analysis (missingness, summary stats, duplicates, outliers, plots).
- **`clip_selection.py`**: VLM selection head. Uses CLIP to match every image against one text
  description per class and selects a fixed number of images per class, then checks the selection
  against the true labels.

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
- 14,160 images, 512×512 RGB, 590 per class (about 63% are augmented copies, named `aug_<n>_<original>`)

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
python clip_selection.py                      # 100 images per class
python clip_selection.py --top-k 300          # bigger budget
python clip_selection.py --originals-only     # ignore aug_* images
python clip_selection.py --model openai/clip-vit-large-patch14
```

How it works:

1. Builds one sentence per class, e.g. *"an image of a slightly rotten orange"*.
2. Embeds all images and sentences with CLIP (L2-normalised), and computes the cosine similarity of
   every image with every sentence.
3. Selects images round-robin: each round, every sentence takes the not-yet-selected image most
   similar to it, until each has `--top-k` images. An image is never selected twice.
4. Compares the selected images with their folder labels and reports precision per class.

Writes to `clip_output/`:

| File | Contents |
|---|---|
| `similarities.csv` | Similarity of every image with all 24 sentences (`s0`–`s23`) |
| `selections.csv` | Selected images, the sentence that picked them, pick round, similarity, correct or not |
| `precision.csv` | Per sentence: images selected, correct, precision, most common true label |
| `distribution_freshness.csv`, `distribution_fruit.csv` | Share of each freshness level / fruit among selected images and CLIP's predictions |
| `image_embeddings_<model>.npy` | Cached image embeddings (reused on the next run; delete to recompute) |

All image paths in the CSVs are relative to the data folder, e.g. `Fresh Banana(1-4)/IMG_001.jpg`.

## Notes

- `clip_selection.py` imports helpers from `eda.py`, so keep both in the same folder.
- Images, the `Processed Data/` folder and cached embeddings are excluded by `.gitignore`.
