"""
Zero-shot CLIP check of the fruit freshness labels.

1. Builds one sentence per class (24): "an image of a fresh / slightly rotten / rotten <fruit>".
2. Embeds every image in "Processed Data/" and every sentence with CLIP.
3. Computes cosine similarity of every image with every sentence (saved to similarities.csv).
4. Round-robin selection: each round, every sentence takes the not-yet-selected image most
   similar to it, until each sentence has top-k (default 100) images. An image is picked at most once.
5. Selected images are checked against the folder label -> precision per sentence.

Outputs in "clip_output/": embeddings cache, similarities.csv, selections.csv, precision.csv,
distribution_freshness.csv, distribution_fruit.csv. Image paths are relative to the data folder.

Usage:  python clip_selection.py  [--model openai/clip-vit-base-patch32] [--top-k 100] [--originals-only]
Requires: torch (CUDA build recommended), transformers, pillow, numpy, pandas
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from eda import FRESHNESS_ORDER, IMAGE_EXTS, parse_folder

STATE_WORDS = {"Fresh": "fresh", "Semi Fresh": "slightly rotten", "Rotten": "rotten"}
FRUIT_WORDS = {"Bittermelon": "bitter melon"}  # wording CLIP is more likely to know


def build_classes(data_dir):
    """One row per class folder with its label and sentence."""
    rows = []
    for folder in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        state, fruit, _, _ = parse_folder(folder.name)
        fruit_word = FRUIT_WORDS.get(fruit, fruit.lower())
        rows.append({
            "folder": folder.name,
            "label": f"{state} {fruit}",
            "sentence": f"an image of a {STATE_WORDS[state]} {fruit_word}",
            "order": (fruit, FRESHNESS_ORDER.index(state)),
        })
    classes = pd.DataFrame(rows).sort_values("order").drop(columns="order").reset_index(drop=True)
    return classes


def list_images(data_dir, classes, originals_only):
    rows = []
    for idx, c in classes.iterrows():
        for f in sorted((data_dir / c["folder"]).iterdir()):
            if f.suffix.lower() not in IMAGE_EXTS:
                continue
            if originals_only and f.name.startswith("aug_"):
                continue
            # path is stored relative to the data folder, e.g. "Fresh Banana(1-4)/IMG_001.jpg"
            rows.append({"path": f.relative_to(data_dir).as_posix(), "folder": c["folder"],
                         "label": c["label"], "label_id": idx})
    return pd.DataFrame(rows)


@torch.no_grad()
def embed_images(paths, model, processor, device, batch_size):
    feats = []
    for i in range(0, len(paths), batch_size):
        batch = [Image.open(p).convert("RGB") for p in paths[i:i + batch_size]]
        # The processor resizes / center-crops to 224x224 and normalises for CLIP
        pixels = processor(images=batch, return_tensors="pt")["pixel_values"].to(device, model.dtype)
        out = model.visual_projection(model.vision_model(pixel_values=pixels).pooler_output)
        feats.append(torch.nn.functional.normalize(out.float(), dim=-1).cpu())
        print(f"\r  embedded {min(i + batch_size, len(paths))}/{len(paths)} images", end="")
    print()
    return torch.cat(feats).numpy()


@torch.no_grad()
def embed_texts(sentences, model, processor, device):
    tok = processor(text=sentences, return_tensors="pt", padding=True).to(device)
    out = model.text_projection(model.text_model(**tok).pooler_output)
    return torch.nn.functional.normalize(out.float(), dim=-1).cpu().numpy()


def select_round_robin(sim, top_k):
    """Returns (sentence_id, image_idx, round) picks; an image is assigned to at most one sentence."""
    n_images, n_sent = sim.shape
    ranked = np.argsort(-sim, axis=0)  # per sentence: image indices, most similar first
    pointer = np.zeros(n_sent, dtype=int)
    taken = np.zeros(n_images, dtype=bool)
    picks = []
    for rnd in range(top_k):
        for s in range(n_sent):
            while pointer[s] < n_images and taken[ranked[pointer[s], s]]:
                pointer[s] += 1
            if pointer[s] == n_images:
                continue  # every image already taken
            img = ranked[pointer[s], s]
            taken[img] = True
            picks.append((s, img, rnd))
    return picks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="Processed Data")
    ap.add_argument("--out", default="clip_output")
    ap.add_argument("--model", default="openai/clip-vit-base-patch32")
    ap.add_argument("--top-k", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--originals-only", action="store_true", help="skip aug_* images")
    args = ap.parse_args()

    data_dir, out = Path(args.data), Path(args.out)
    out.mkdir(exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else
                                 "  -- CUDA not available, this will be slow"))

    # 1. Sentences, one per class
    classes = build_classes(data_dir)
    print("\nSentences:")
    for i, c in classes.iterrows():
        print(f"  [{i:2d}] {c['sentence']:45s} <- {c['folder']}")

    # 2. CLIP model
    print(f"\nLoading {args.model} ...")
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = CLIPModel.from_pretrained(args.model, dtype=dtype).to(device).eval()
    processor = CLIPProcessor.from_pretrained(args.model)

    # 3. Embeddings (image embeddings are cached since they are the slow part)
    images = list_images(data_dir, classes, args.originals_only)
    tag = args.model.replace("/", "_") + ("_orig" if args.originals_only else "")
    cache = out / f"image_embeddings_{tag}.npy"
    if cache.exists() and len(np.load(cache)) == len(images):
        print(f"Using cached image embeddings: {cache}")
        img_emb = np.load(cache)
    else:
        print(f"Embedding {len(images)} images ...")
        paths = [data_dir / p for p in images["path"]]
        img_emb = embed_images(paths, model, processor, device, args.batch_size)
        np.save(cache, img_emb)
    txt_emb = embed_texts(classes["sentence"].tolist(), model, processor, device)

    # Cosine similarity (embeddings are L2-normalised, so it is a dot product)
    sim = img_emb @ txt_emb.T  # (n_images, 24)

    # Store every image's similarity with every sentence
    sim_df = pd.DataFrame(sim, columns=[f"s{i}" for i in range(len(classes))])
    pd.concat([images[["path", "label"]], sim_df], axis=1).to_csv(out / "similarities.csv", index=False)

    # 4. Round-robin selection: each round, every sentence picks the unselected image with
    #    the highest similarity to *that* sentence, until it has top_k images (or none are left)
    picks = select_round_robin(sim, args.top_k)
    selected = images.iloc[[p[1] for p in picks]].copy()
    selected["sentence_id"] = [p[0] for p in picks]
    selected["pick_order"] = [p[2] for p in picks]
    selected["similarity"] = sim[selected.index, selected["sentence_id"]]
    selected["correct"] = selected["sentence_id"] == selected["label_id"]
    selected["sentence"] = classes.loc[selected["sentence_id"], "sentence"].values
    selected = selected.sort_values(["sentence_id", "pick_order"])
    selected.to_csv(out / "selections.csv", index=False)

    # 5. Precision per sentence
    res = classes[["sentence", "label"]].copy()
    grp = selected.groupby("sentence_id")
    res["selected"] = grp.size().reindex(res.index, fill_value=0)
    res["correct"] = grp["correct"].sum().reindex(res.index, fill_value=0)
    res["precision"] = (res["correct"] / res["selected"].replace(0, np.nan)).round(3)
    res["most_common_true_label"] = grp["label"].agg(lambda s: s.value_counts().index[0]).reindex(res.index)
    res.to_csv(out / "precision.csv", index_label="sentence_id")

    print(f"\nSelection precision per sentence (top {args.top_k}):")
    print(res.to_string())
    total_sel, total_ok = res["selected"].sum(), res["correct"].sum()
    print(f"\nOverall: {total_ok}/{total_sel} selected images correct ({total_ok / total_sel:.1%})")

    # Share of each freshness level / fruit, ideally 33.3% and 12.5%.
    # "selected (true)" = true labels of the selected images (picks per sentence are fixed, so this is
    # what varies); "argmax (pred)" = what CLIP predicts for every image without the top-k constraint.
    best = sim.argmax(axis=1)
    pred_label = classes.loc[best, "label"].values
    split = lambda labels: (labels.str.rsplit(" ", n=1).str[0], labels.str.rsplit(" ", n=1).str[-1])
    sel_state, sel_fruit = split(selected["label"])
    pred_state, pred_fruit = split(pd.Series(pred_label))
    for name, sel, pred, ideal in [("freshness", sel_state, pred_state, 100 / 3),
                                   ("fruit", sel_fruit, pred_fruit, 100 / 8)]:
        dist = pd.DataFrame({
            "selected (true) %": sel.value_counts(normalize=True) * 100,
            "argmax (pred) %": pred.value_counts(normalize=True) * 100,
        }).fillna(0)
        dist["ideal %"] = ideal
        if name == "freshness":
            dist = dist.reindex(FRESHNESS_ORDER)
        dist.to_csv(out / f"distribution_{name}.csv")
        print(f"\nDistribution by {name}:")
        print(dist.to_string(float_format="%.1f"))

    print(f"Zero-shot accuracy on all {len(images)} images: {(best == images['label_id']).mean():.1%}")
    fruit_ok = np.mean([p.split()[-1] == t.split()[-1] for p, t in zip(pred_label, images["label"])])
    print(f"Fruit-only accuracy (ignoring freshness):        {fruit_ok:.1%}")
    print(f"\nResults written to {out}/")


if __name__ == "__main__":
    main()
