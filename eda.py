"""
Basic exploratory data analysis for the fruit freshness image dataset.

Reads every image under "Processed Data/", builds a per-image table of
metadata + simple pixel statistics, and writes a report to "eda_output/":
  - schema.csv / example_rows.csv      : schema table and sample rows
  - missingness.csv                    : missing values per column (+ unreadable files)
  - summary_stats.csv                  : describe() of numeric fields
  - class_counts.csv                   : images per fruit x freshness
  - duplicates_exact.csv / near.csv    : exact (MD5) and near (perceptual hash) duplicates
  - outliers.csv                       : IQR outliers on size / brightness / contrast
  - *.png                              : plots
  - report.txt                         : everything above in one readable file

Usage:  python eda.py  [--data "Processed Data"] [--out eda_output]
Requires: pillow, numpy, pandas, matplotlib
"""

import argparse
import hashlib
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
FRESHNESS_ORDER = ["Fresh", "Semi Fresh", "Rotten"]


def parse_folder(name):
    """'Semi fresh Pineapple (15-25)' -> ('Semi Fresh', 'Pineapple', 15, 25)."""
    m = re.match(r"^(semi[ _]?fresh|fresh|rotten)\s*(.+?)\s*\(\s*(\d+)\s*-\s*(\d+)\s*\)\s*$", name, re.I)
    if not m:
        return None, name, np.nan, np.nan
    state = m.group(1).lower().replace("_", " ")
    state = "Semi Fresh" if state.startswith("semi") else state.title()
    return state, m.group(2).strip().title(), int(m.group(3)), int(m.group(4))


def average_hash(img, size=16):
    """Simple perceptual hash: 16x16 grayscale, bits = pixel > mean."""
    small = np.asarray(img.convert("L").resize((size, size), Image.BILINEAR), dtype=np.float32)
    bits = (small > small.mean()).flatten()
    return "".join("1" if b else "0" for b in bits)


def analyse_image(path):
    row = {"file_size_kb": path.stat().st_size / 1024}
    raw = path.read_bytes()
    row["md5"] = hashlib.md5(raw).hexdigest()
    try:
        with Image.open(path) as img:
            row["width"], row["height"] = img.size
            row["mode"] = img.mode
            rgb = img.convert("RGB")
            row["ahash"] = average_hash(rgb)
            arr = np.asarray(rgb.resize((128, 128)), dtype=np.float32)
        row["mean_r"], row["mean_g"], row["mean_b"] = arr.reshape(-1, 3).mean(axis=0)
        gray = arr.mean(axis=2)
        row["brightness"] = gray.mean()
        row["contrast"] = gray.std()
        row["aspect_ratio"] = row["width"] / row["height"]
        row["readable"] = True
    except Exception as e:  # corrupt / unreadable image -> leave pixel fields missing
        row["readable"] = False
        row["error"] = str(e)
    return row


def build_table(data_dir):
    rows = []
    folders = sorted(p for p in data_dir.iterdir() if p.is_dir())
    for folder in folders:
        state, fruit, day_min, day_max = parse_folder(folder.name)
        files = [f for f in folder.iterdir() if f.suffix.lower() in IMAGE_EXTS]
        print(f"  {folder.name:35s} {len(files):6d} images")
        for f in files:
            m = re.match(r"^aug_(\d+)_(.+)$", f.name)
            row = {
                "folder": folder.name,
                "fruit": fruit,
                "freshness": state,
                "day_min": day_min,
                "day_max": day_max,
                "filename": f.name,
                "extension": f.suffix.lower(),
                "is_augmented": bool(m),
                "source_image": m.group(2) if m else f.name,
            }
            row.update(analyse_image(f))
            rows.append(row)
    df = pd.DataFrame(rows)
    if "error" not in df:
        df["error"] = np.nan
    return df


def iqr_outliers(df, col, k=1.5):
    q1, q3 = df[col].quantile([0.25, 0.75])
    lo, hi = q1 - k * (q3 - q1), q3 + k * (q3 - q1)
    out = df[(df[col] < lo) | (df[col] > hi)]
    return out, lo, hi


def near_duplicate_groups(df):
    """Group images whose average hash is identical -> near duplicates."""
    groups = df[df["ahash"].notna()].groupby("ahash")
    return [g for _, g in groups if len(g) > 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="Processed Data")
    ap.add_argument("--out", default="eda_output")
    args = ap.parse_args()

    data_dir, out = Path(args.data), Path(args.out)
    out.mkdir(exist_ok=True)
    report = []

    def log(text=""):
        print(text)
        report.append(str(text))

    print(f"Scanning {data_dir} ...")
    df = build_table(data_dir)
    df.to_csv(out / "image_metadata.csv", index=False)

    numeric_cols = ["file_size_kb", "width", "height", "aspect_ratio",
                    "brightness", "contrast", "mean_r", "mean_g", "mean_b", "day_min", "day_max"]

    # ---------------- Overview / schema ----------------
    log("=" * 70)
    log("DATASET OVERVIEW")
    log("=" * 70)
    log(f"Total images      : {len(df)}")
    log(f"Classes (folders) : {df['folder'].nunique()}")
    log(f"Fruits            : {', '.join(sorted(df['fruit'].unique()))}")
    log(f"Freshness levels  : {', '.join(s for s in FRESHNESS_ORDER if s in set(df['freshness']))}")
    log(f"Augmented images  : {df['is_augmented'].sum()} ({df['is_augmented'].mean():.1%})")
    log(f"Unique source imgs: {df['source_image'].nunique()}")

    schema = pd.DataFrame({
        "column": df.columns,
        "dtype": [str(t) for t in df.dtypes],
        "example": [df[c].dropna().iloc[0] if df[c].notna().any() else None for c in df.columns],
    })
    schema.to_csv(out / "schema.csv", index=False)
    log("\nSCHEMA")
    log(schema.to_string(index=False))

    examples = df.drop(columns=["md5", "ahash", "error"]).sample(min(5, len(df)), random_state=0)
    examples.to_csv(out / "example_rows.csv", index=False)
    log("\nEXAMPLE ROWS")
    log(examples.to_string(index=False))

    # ---------------- Missingness ----------------
    log("\n" + "=" * 70)
    log("MISSINGNESS")
    log("=" * 70)
    miss = pd.DataFrame({"missing": df.isna().sum(), "pct": df.isna().mean() * 100})
    miss = miss.drop(index="error")  # 'error' is only filled when something fails
    miss.to_csv(out / "missingness.csv")
    log(miss.to_string(float_format="%.2f"))
    unreadable = df[~df["readable"]]
    log(f"\nUnreadable/corrupt images: {len(unreadable)}")
    if len(unreadable):
        unreadable[["folder", "filename", "error"]].to_csv(out / "unreadable.csv", index=False)
        log(unreadable[["folder", "filename", "error"]].head(20).to_string(index=False))
    unparsed = df[df["freshness"].isna()]["folder"].unique()
    if len(unparsed):
        log(f"Folders with unparseable labels: {list(unparsed)}")

    # ---------------- Class balance ----------------
    log("\n" + "=" * 70)
    log("CLASS COUNTS (fruit x freshness)")
    log("=" * 70)
    counts = pd.crosstab(df["fruit"], df["freshness"], margins=True, margins_name="Total")
    counts = counts[[c for c in FRESHNESS_ORDER if c in counts.columns] + ["Total"]]
    counts.to_csv(out / "class_counts.csv")
    log(counts.to_string())

    # ---------------- Summary stats ----------------
    log("\n" + "=" * 70)
    log("SUMMARY STATISTICS (numeric fields)")
    log("=" * 70)
    stats = df[numeric_cols].describe().T
    stats.to_csv(out / "summary_stats.csv")
    log(stats.to_string(float_format="%.2f"))

    log("\nMean pixel stats by freshness:")
    by_state = df.groupby("freshness")[["brightness", "contrast", "mean_r", "mean_g", "mean_b", "file_size_kb"]].mean()
    by_state.to_csv(out / "mean_pixel_stats.csv")
    log(by_state.reindex([s for s in FRESHNESS_ORDER if s in by_state.index]).to_string(float_format="%.2f"))

    log("\nImage resolutions (top 10):")
    res = (df["width"].astype("Int64").astype(str) + "x" + df["height"].astype("Int64").astype(str)).value_counts()
    log(res.head(10).to_string())
    log(f"Image modes: {df['mode'].value_counts().to_dict()}")

    # ---------------- Duplicates ----------------
    log("\n" + "=" * 70)
    log("DUPLICATES")
    log("=" * 70)
    exact = df[df.duplicated("md5", keep=False)].sort_values("md5")
    exact[["md5", "folder", "filename"]].to_csv(out / "duplicates_exact.csv", index=False)
    log(f"Exact duplicate files (same MD5): {len(exact)} files in {exact['md5'].nunique()} groups")
    cross = exact.groupby("md5")["folder"].nunique()
    log(f"  ...of which groups span >1 class (label conflict): {(cross > 1).sum()}")

    near_groups = near_duplicate_groups(df)
    near = pd.concat(near_groups) if near_groups else df.iloc[0:0]
    near[["ahash", "folder", "filename"]].to_csv(out / "duplicates_near.csv", index=False)
    log(f"Near duplicates (identical perceptual hash): {len(near)} files in {len(near_groups)} groups")
    if near_groups:
        cross_near = sum(g["folder"].nunique() > 1 for g in near_groups)
        log(f"  ...of which groups span >1 class: {cross_near}")
    log("Note: near-duplicates are expected here since many images are augmentations of the same source.")

    aug_per_src = df.groupby(["folder", "source_image"]).size()
    log(f"Copies per source image: min={aug_per_src.min()}, median={aug_per_src.median():.0f}, max={aug_per_src.max()}")

    # ---------------- Outliers ----------------
    log("\n" + "=" * 70)
    log("OUTLIERS (IQR rule, k=1.5)")
    log("=" * 70)
    outlier_rows = []
    for col in ["file_size_kb", "brightness", "contrast", "aspect_ratio"]:
        o, lo, hi = iqr_outliers(df.dropna(subset=[col]), col)
        log(f"{col:15s}: {len(o):5d} outliers  (valid range {lo:.2f} .. {hi:.2f})")
        outlier_rows.append(o.assign(outlier_field=col)[["outlier_field", col, "folder", "filename"]])
    pd.concat(outlier_rows).to_csv(out / "outliers.csv", index=False)

    (out / "report.txt").write_text("\n".join(report), encoding="utf-8")

    # ---------------- Plots ----------------
    plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False})
    colors = {"Fresh": "#2e9e5b", "Semi Fresh": "#e0a526", "Rotten": "#a8432f"}

    # 1. Bar chart: images per fruit, stacked by freshness
    c = counts.drop(index="Total").drop(columns="Total")
    ax = c.plot(kind="bar", stacked=True, figsize=(10, 5), color=[colors[s] for s in c.columns])
    ax.set_title("Images per fruit and freshness level")
    ax.set_xlabel("")
    ax.set_ylabel("Number of images")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(out / "class_distribution.png")
    plt.close()

    # 2. Histograms of key numeric fields, split by freshness
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, col in zip(axes, ["brightness", "contrast", "file_size_kb"]):
        for state in FRESHNESS_ORDER:
            vals = df.loc[df["freshness"] == state, col].dropna()
            ax.hist(vals, bins=40, alpha=0.55, label=state, color=colors[state])
        ax.set_title(col.replace("_", " ").title())
        ax.set_ylabel("Count")
    axes[0].legend()
    plt.tight_layout()
    plt.savefig(out / "histograms.png")
    plt.close()

    # 3. Correlation heatmap of numeric features (constant columns dropped)
    num = df[numeric_cols].dropna(axis=1, how="all")
    num = num.loc[:, num.std() > 0]
    corr = num.corr()
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr)))
    ax.set_yticklabels(corr.columns)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(corr.iloc[i, j]) > 0.6 else "black")
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Correlation of numeric image features")
    plt.tight_layout()
    plt.savefig(out / "correlation_heatmap.png")
    plt.close()

    # 4. Sample image grid: one row per freshness, one column per fruit
    fruits = sorted(df["fruit"].unique())
    fig, axes = plt.subplots(len(FRESHNESS_ORDER), len(fruits), figsize=(2 * len(fruits), 2.2 * len(FRESHNESS_ORDER)))
    for i, state in enumerate(FRESHNESS_ORDER):
        for j, fruit in enumerate(fruits):
            ax = axes[i, j]
            ax.axis("off")
            subset = df[(df["fruit"] == fruit) & (df["freshness"] == state) & df["readable"]]
            if len(subset):
                r = subset.sample(1, random_state=1).iloc[0]
                ax.imshow(Image.open(data_dir / r["folder"] / r["filename"]).convert("RGB"))
            if i == 0:
                ax.set_title(fruit, fontsize=10)
        axes[i, 0].text(-0.1, 0.5, state, transform=axes[i, 0].transAxes, rotation=90,
                        ha="right", va="center", fontsize=11)
    plt.tight_layout()
    plt.savefig(out / "sample_images.png")
    plt.close()

    print(f"\nDone. Report and plots written to {out}/")


if __name__ == "__main__":
    main()
