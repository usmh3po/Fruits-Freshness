"""
Zero-shot CLIP check of the fruit freshness labels.

1. Uses four descriptive sentences per class (96 in total, see CLASS_SENTENCES).
2. Embeds every image in "Processed Data/" and every sentence with CLIP.
3. Computes cosine similarity of every image with every sentence (saved to similarities.csv).
4. Round-robin selection: each round, every sentence takes the not-yet-selected image most
   similar to it, until each sentence has top-k (default 50) images. An image is picked at most once.
5. Selected images are checked against the folder label -> precision per sentence and per class.

Outputs in "clip_output/": embeddings cache, sentences.csv, similarities.csv, selections.csv,
precision.csv (per sentence), precision_per_class.csv, distribution_freshness.csv,
distribution_fruit.csv. Image paths are relative to the data folder.

Usage:  python clip_selection.py  [--model openai/clip-vit-base-patch32] [--top-k 50] [--originals-only]
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

# Four sentences per class, written from what the photos actually show (e.g. the oranges are
# green citrus, the eggplants are round green striped ones, rotten papayas have white mould).
CLASS_SENTENCES = {
    "Fresh Banana": ["a photo of a fresh yellow banana with only a few small brown spots",
                     "a photo of a short, bright yellow banana with a smooth peel",
                     "a photo of a fresh yellow banana with a green-tinged stem",
                     "a photo of a plump ripe yellow banana with a few brown freckles"],
    "Semi Fresh Banana": ["a photo of an overripe banana with large black patches on its yellow peel",
                          "a photo of a yellow banana turning brown and black",
                          "a photo of a banana with big dark brown bruises covering much of its peel",
                          "a photo of a soft overripe banana with many black spots and streaks"],
    "Rotten Banana": ["a photo of a rotten banana with a completely black, shriveled peel",
                      "a photo of a rotten banana that has turned dark purple-black all over",
                      "a photo of a rotten banana with dark brown, leathery skin and a dried stem",
                      "a photo of a mushy rotten black banana with its peel splitting"],
    "Fresh Bittermelon": ["a photo of a fresh karela with ridged, spiky, warty green skin, shaped like a leaf",
                          "a photo of a fresh green karela with warty skin and a thin stem",
                          "a photo of a green bitter gourd with many small bumps and ridges along its length",
                          "a photo of a fresh green karela lying on a cloth"],
    "Semi Fresh Bittermelon": ["a photo of a wilting bumpy bitter melon with yellow patches",
                               "a photo of a bitter melon turning yellow-orange at the tips",
                               "a photo of a karela with warty skin turning yellow",
                               "a photo of a slightly dried green and yellow karela with a curly stem"],
    "Rotten Bittermelon": ["a photo of a rotten bitter gourd that has turned orange with white mold",
                           "a photo of a bitter gourd covered in white fuzzy mold with orange patches",
                           "a photo of a rotting orange-yellow bitter melon with a long curly stalk",
                           "a photo of a rotten karela covered with white mold"],
    "Fresh Cucumber": ["a photo of a fresh green cucumber with light stripes and smooth firm skin",
                       "a photo of a fresh crisp green cucumber",
                       "a photo of a fresh dark green cucumber with glossy skin",
                       "a photo of a fresh green cucumber with small bumps"],
    "Semi Fresh Cucumber": ["a photo of a dark green cucumber with rough, speckled skin",
                            "a photo of a cucumber with mottled green and brown patches",
                            "a photo of a cucumber with brown and dark green blotchy skin",
                            "a photo of an aging cucumber with yellow and brown mottled skin"],
    "Rotten Cucumber": ["a photo of an overripe pale yellow cucumber",
                        "a photo of a large pale cream-colored overripe cucumber",
                        "a photo of an old yellow cucumber with smooth pale skin",
                        "a photo of a whitish-yellow overripe cucumber with faint stripes"],
    "Fresh Eggplant": ["a photo of a fresh round green striped eggplant with a green stem",
                       "a photo of a fresh green striped round brinjal with a curved stem",
                       "a photo of a fresh striped green brinjal with a long green stalk",
                       "a photo of a round eggplant with green and white stripes"],
    "Semi Fresh Eggplant": ["a photo of a pink-purple round eggplant",
                            "a photo of a green striped eggplant with soft, dented skin",
                            "a photo of a magenta eggplant with smooth skin",
                            "a photo of a green eggplant with a few wrinkles near its stalk"],
    "Rotten Eggplant": ["a photo of a shrunken, decaying purple or green brinjal",
                        "a photo of a glossy purple and white eggplant with wrinkles",
                        "a photo of a rotten green brinjal with wrinkled skin like a crumpled cabbage leaf",
                        "a photo of a shriveled purple-pink eggplant"],
    "Fresh Orange": ["a photo of a fresh green citrus orange with a shiny peel",
                     "a photo of a small fresh dark green lime-like orange",
                     "a photo of a fresh green orange with a leaf attached",
                     "a photo of a fresh dark green mosambi citrus fruit"],
    "Semi Fresh Orange": ["a photo of a yellow-orange citrus fruit starting to soften",
                          "a photo of a round yellow citrus fruit with a slightly dull peel",
                          "a photo of a yellow-green orange with a smooth peel",
                          "a photo of a ripe yellow mosambi citrus fruit"],
    "Rotten Orange": ["a photo of a rotten orange with a brown, moldy, shriveled peel",
                      "a photo of a decayed dark brown citrus fruit covered in mold",
                      "a photo of a rotten orange with grey and black mold spots",
                      "a photo of a rotten orange that has turned brown and grey"],
    "Fresh Papaya": ["a photo of a fresh green papaya with smooth, unblemished skin",
                     "a photo of a firm unripe green papaya",
                     "a photo of a pear-shaped bright green papaya",
                     "a photo of a fresh smooth green papaya lying on a cloth"],
    "Semi Fresh Papaya": ["a photo of a green papaya with brown scars and blemishes on its skin",
                          "a photo of a green papaya with a net of thin brown scratch marks",
                          "a photo of a pear-shaped green papaya with faded, scarred skin",
                          "a photo of a green papaya with brown scab-like lines on its skin"],
    "Rotten Papaya": ["a photo of a rotten papaya with white mold patches",
                      "a photo of a green papaya covered in white powdery mold",
                      "a photo of a green papaya with white and grey mold spots",
                      "a photo of a decayed brown papaya with white fungus"],
    "Fresh Pineapple": ["a photo of a fresh pineapple with a green leafy crown",
                        "a photo of a small fresh pineapple with a tall green crown",
                        "a photo of a fresh pineapple crown with long bright green leaves",
                        "a photo of a fresh pineapple with green-yellow scales"],
    "Semi Fresh Pineapple": ["a photo of a ripe orange-brown pineapple with drying leaves",
                             "a photo of a pineapple turning golden brown with wilting crown leaves",
                             "a photo of a golden orange pineapple with green and brown leaves",
                             "a photo of an orange pineapple with some dry leaf tips"],
    "Rotten Pineapple": ["a photo of a rotten pineapple with dry, brown, dead leaves",
                         "a photo of a shriveled grey-brown decayed pineapple",
                         "a photo of a dried-out pineapple with grey, withered leaves",
                         "a photo of a rotten pineapple with grey-brown scales and dead leaves"],
    "Fresh Tomato": ["a photo of a fresh orange-red tomato",
                     "a photo of a fresh orange and red tomato with a green stem",
                     "a photo of a fresh firm red tomato with a green calyx",
                     "a photo of a fresh smooth orange tomato"],
    "Semi Fresh Tomato": ["a photo of a red tomato with slightly sagging, soft skin",
                          "a photo of an orange-red tomato starting to lose firmness",
                          "a photo of a tomato with a few small wrinkles near the top",
                          "a photo of a slightly overripe orange-red tomato with dull skin"],
    "Rotten Tomato": ["a photo of a rotten tomato with wrinkled, collapsed skin",
                      "a photo of a decaying tomato with black mold",
                      "a photo of a rotten tomato with dark green-black mold spots",
                      "a photo of a squashed, rotting red tomato"],
}


def build_classes(data_dir):
    """One row per class folder with its label."""
    rows = []
    for folder in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        state, fruit, _, _ = parse_folder(folder.name)
        rows.append({
            "folder": folder.name,
            "label": f"{state} {fruit}",
            "order": (fruit, FRESHNESS_ORDER.index(state)),
        })
    classes = pd.DataFrame(rows).sort_values("order").drop(columns="order").reset_index(drop=True)
    return classes


def build_sentences(classes):
    """One row per sentence (4 per class), with the id of the class it describes."""
    rows = [{"class_id": idx, "label": c["label"], "sentence": s}
            for idx, c in classes.iterrows() for s in CLASS_SENTENCES[c["label"]]]
    return pd.DataFrame(rows)


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
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--originals-only", action="store_true", help="skip aug_* images")
    args = ap.parse_args()

    data_dir, out = Path(args.data), Path(args.out)
    out.mkdir(exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else
                                 "  -- CUDA not available, this will be slow"))

    # 1. Sentences, four per class
    classes = build_classes(data_dir)
    sentences = build_sentences(classes)
    print("\nSentences:")
    for i, s in sentences.iterrows():
        print(f"  [{i:2d}] {s['sentence']:75s} <- {s['label']}")

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
    txt_emb = embed_texts(sentences["sentence"].tolist(), model, processor, device)

    # Cosine similarity (embeddings are L2-normalised, so it is a dot product)
    sim = img_emb @ txt_emb.T  # (n_images, n_sentences)

    # Store every image's similarity with every sentence
    sim_df = pd.DataFrame(sim, columns=[f"s{i}" for i in range(len(sentences))])
    pd.concat([images[["path", "label"]], sim_df], axis=1).to_csv(out / "similarities.csv", index=False)
    sentences.to_csv(out / "sentences.csv", index_label="sentence_id")
    sent_class = sentences["class_id"].to_numpy()

    # 4. Round-robin selection: each round, every sentence picks the unselected image with
    #    the highest similarity to *that* sentence, until it has top_k images (or none are left)
    picks = select_round_robin(sim, args.top_k)
    selected = images.iloc[[p[1] for p in picks]].copy()
    selected["sentence_id"] = [p[0] for p in picks]
    selected["pick_order"] = [p[2] for p in picks]
    selected["similarity"] = sim[selected.index, selected["sentence_id"]]
    selected["sentence_class_id"] = sent_class[selected["sentence_id"]]
    selected["correct"] = selected["sentence_class_id"] == selected["label_id"]
    selected["sentence"] = sentences.loc[selected["sentence_id"], "sentence"].values
    selected = selected.sort_values(["sentence_id", "pick_order"])
    selected.to_csv(out / "selections.csv", index=False)

    # 5. Precision per sentence and per class (both sentences of a class combined)
    def precision_table(base, key):
        grp = selected.groupby(key)
        res = base.copy()
        res["selected"] = grp.size().reindex(res.index, fill_value=0)
        res["correct"] = grp["correct"].sum().reindex(res.index, fill_value=0)
        res["precision"] = (res["correct"] / res["selected"].replace(0, np.nan)).round(3)
        res["most_common_true_label"] = grp["label"].agg(lambda s: s.value_counts().index[0]).reindex(res.index)
        return res

    res_sent = precision_table(sentences[["label", "sentence"]], "sentence_id")
    res_sent.to_csv(out / "precision.csv", index_label="sentence_id")
    res_class = precision_table(classes[["label"]], "sentence_class_id")
    res_class.to_csv(out / "precision_per_class.csv", index_label="class_id")

    print(f"\nSelection precision per sentence (top {args.top_k} each):")
    print(res_sent.drop(columns="label").to_string())
    print("\nSelection precision per class:")
    print(res_class.to_string())
    total_sel, total_ok = res_sent["selected"].sum(), res_sent["correct"].sum()
    print(f"\nOverall: {total_ok}/{total_sel} selected images correct ({total_ok / total_sel:.1%})")

    # Share of each freshness level / fruit, ideally 33.3% and 12.5%.
    # "selected (true)" = true labels of the selected images (picks per sentence are fixed, so this is
    # what varies); "argmax (pred)" = what CLIP predicts for every image without the top-k constraint.
    best = sent_class[sim.argmax(axis=1)]  # class of the most similar sentence
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
