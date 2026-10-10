"""Week 4: look-alike detection (ResNet-50 features), clusters, and the original + clean splits.

Reads the cached 224 images and the Week 3 audit files. Writes to outputs/audit/<version>/ and outputs/splits/.
Used by notebooks/03_splits.ipynb.
"""
import os
from pathlib import Path

# Keep downloaded model weights on D: (inside the project) instead of C:\Users\<user>\.cache.
# Must be set BEFORE timm / huggingface_hub are imported.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".cache" / "huggingface"))
os.environ.setdefault("TORCH_HOME", str(PROJECT_ROOT / ".cache" / "torch"))

import matplotlib.pyplot as plt   # noqa: E402  (imports after setting the cache folders)
import numpy as np                # noqa: E402
import pandas as pd               # noqa: E402
import timm                       # noqa: E402
import torch                      # noqa: E402
from PIL import Image             # noqa: E402
from sklearn.model_selection import train_test_split   # noqa: E402
from torch.utils.data import DataLoader                # noqa: E402
from tqdm.auto import tqdm                             # noqa: E402

from src.audit import union_find, within               # noqa: E402
from src.data import CLASSES, SEED, MRIDataset, _save, eval_transforms, load_grey   # noqa: E402

COS_THRESHOLD = 0.95      # look-alike threshold (cosine similarity of ResNet-50 features)
COS_MIN_KEEP = 0.90       # pairs from this value up are saved, so other thresholds can be studied
PHASH_T = 5               # near-copy threshold chosen in Week 3


# ------------------------------------------------------------------- deep features
def extract_features(cache_csv, out_npy, batch=64, num_workers=2):
    """ResNet-50 (ImageNet) pooled features for every image in cache_csv, L2-normalised.
    Result: array (N, 2048) in the same row order as the CSV. Saved to out_npy; re-used if it exists."""
    if Path(out_npy).exists():
        return np.load(out_npy)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # num_classes=0 removes the classifier, so the model returns the 2048-number feature vector.
    model = timm.create_model("resnet50", pretrained=True, num_classes=0).to(device).eval()
    model = model.to(memory_format=torch.channels_last)
    dataset = MRIDataset(cache_csv, split=None, transform=eval_transforms())
    while True:
        try:
            loader = DataLoader(dataset, batch_size=batch, shuffle=False, num_workers=num_workers,
                                pin_memory=device == "cuda", persistent_workers=False)
            chunks = []
            # float32 on purpose: on this GTX 1650 Ti, cuDNN's float16 (AMP) convolutions returned NaN
            # for batches of 48+ images (tested in Week 4). Inference in float32 is still fast enough.
            with torch.no_grad():
                for images, _ in tqdm(loader, desc=f"Features (batch {batch})"):
                    images = images.to(device, non_blocking=True).to(memory_format=torch.channels_last)
                    chunks.append(model(images).cpu())
            break
        except torch.cuda.OutOfMemoryError:          # GPU full: halve the batch and start again
            torch.cuda.empty_cache()
            batch //= 2
            print(f"CUDA out of memory -> retrying with batch {batch}")
    features = torch.cat(chunks)
    assert torch.isfinite(features).all(), "feature extraction produced NaN/Inf values"
    features = torch.nn.functional.normalize(features, dim=1).numpy()   # length 1 -> cosine = dot product
    np.save(out_npy, features)
    del model
    torch.cuda.empty_cache()
    return features


def cosine_pairs(features, min_sim=COS_MIN_KEEP, chunk=2048):
    """All pairs (i < j) with cosine similarity >= min_sim, computed on the GPU block by block.
    Features are L2-normalised, so cosine similarity = F_i . F_j  (one matrix product per block)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    F = torch.from_numpy(features).to(device=device, dtype=torch.float32)   # float32: precise near 0.95
    n = F.shape[0]
    found_i, found_j, found_s = [], [], []
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        sim = F[start:stop] @ F.T                                   # (chunk, N) similarities
        rows = torch.arange(start, stop, device=device)[:, None]
        cols = torch.arange(n, device=device)[None, :]
        r, c = torch.nonzero((sim >= min_sim) & (cols > rows), as_tuple=True)   # upper triangle only
        found_i.append(r + start)
        found_j.append(c)
        found_s.append(sim[r, c])
    return (torch.cat(found_i).cpu().numpy(), torch.cat(found_j).cpu().numpy(),
            torch.cat(found_s).cpu().numpy())


def verify_cosine(features, i, j, s, n_check=1000, seed=SEED):
    """Largest difference between the GPU similarities and NumPy's, on random kept pairs."""
    pick = np.random.default_rng(seed).choice(len(i), size=min(n_check, len(i)), replace=False)
    numpy_sim = np.sum(features[i[pick]] * features[j[pick]], axis=1)
    return float(np.abs(numpy_sim - s[pick]).max())


def lookalike_table(table, i, j, s):
    """Pair table for look-alike pairs (same style as Week 3's pairs.csv)."""
    A, B = table.iloc[i].reset_index(drop=True), table.iloc[j].reset_index(drop=True)
    out = pd.DataFrame({"a": i, "b": j, "cosine": np.round(s, 4),
                        "source_a": A["source"], "source_b": B["source"],
                        "split_a": A["split"], "split_b": B["split"],
                        "label_a": A["label"], "label_b": B["label"]})
    out["pair_type"] = ["-".join(sorted([x, y], reverse=True)) for x, y in zip(out["split_a"], out["split_b"])]
    out["label_conflict"] = out["label_a"] != out["label_b"]
    return out


# ------------------------------------------------------------------------- clusters
def build_clusters(table, hashes, pairs, look, T=PHASH_T, cos=COS_THRESHOLD):
    """Union-find over three kinds of link -> one cluster_id per image.
    table: cache CSV rows (path, label, split, source). hashes/pairs: Week 3 files (same row order)."""
    assert (table["source"].values == hashes["path"].values).all(), "row order of cache CSV and hashes differ"
    near = within(pairs, T)                                  # exact (MD5) + pHash <= T
    alike = look[look["cosine"] >= cos]
    edges = list(zip(near["a"], near["b"])) + list(zip(alike["a"], alike["b"]))
    out = table[["path", "label", "split", "source"]].copy()
    out["cluster_id"] = union_find(len(out), edges)
    out["cluster_size"] = out.groupby("cluster_id")["path"].transform("size")
    out["n_labels"] = out.groupby("cluster_id")["label"].transform("nunique")
    out["conflict"] = out["n_labels"] > 1                     # copies filed under different classes
    sizes = []
    for src in out["source"]:                                  # image size (reads only the file header)
        with Image.open(src) as im:
            sizes.append(im.size)
    out["width"], out["height"] = [w for w, _ in sizes], [h for _, h in sizes]
    return out


def cluster_stats(clusters):
    """Number of clusters, how many contain 2+ images, the largest size and its share of all images."""
    sizes = clusters.groupby("cluster_id").size()
    return {"images": len(clusters), "clusters": len(sizes), "clusters_2plus": int((sizes > 1).sum()),
            "largest_cluster": int(sizes.max()),
            "largest_%_of_images": round(100 * sizes.max() / len(clusters), 2),
            "conflict_clusters": int(clusters.loc[clusters["conflict"], "cluster_id"].nunique()),
            "images_in_conflict_clusters": int(clusters["conflict"].sum())}


def conflict_report(clusters):
    """Each conflicting cluster: its size, the labels it mixes and the splits it covers."""
    bad = clusters[clusters["conflict"]]
    if bad.empty:
        return pd.DataFrame(columns=["cluster_id", "size", "labels", "splits"])
    rows = [{"cluster_id": cid, "size": len(g),
             "labels": " + ".join(f"{lab}:{k}" for lab, k in g["label"].value_counts().sort_index().items()),
             "splits": " + ".join(sorted(g["split"].unique()))}
            for cid, g in bad.groupby("cluster_id")]
    return pd.DataFrame(rows).sort_values("size", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------- splits
def _columns(df):
    return df[["path", "label", "split", "source", "cluster_id"]].reset_index(drop=True)


def original_split(clusters, val_size=0.15, seed=SEED):
    """The dataset's own folders: Testing -> test; Training -> 85% train / 15% val (stratified by class)."""
    train_folder = clusters[clusters["split"] == "train"]
    test_folder = clusters[clusters["split"] == "test"].assign(split="test")
    tr, va = train_test_split(train_folder, test_size=val_size, stratify=train_folder["label"], random_state=seed)
    return _columns(pd.concat([tr.assign(split="train"), va.assign(split="val"), test_folder]))


def clean_split(clusters, sizes=(0.70, 0.15, 0.15), seed=SEED):
    """Leakage-free split.
    1. drop every cluster whose copies carry different labels;
    2. keep ONE image per cluster (the highest resolution copy; ties -> first path);
    3. stratified 70/15/15 split of the kept images.
    Returns (split table, counts of dropped images)."""
    no_conflict = clusters[~clusters["conflict"]].copy()
    no_conflict["area"] = no_conflict["width"] * no_conflict["height"]
    kept = (no_conflict.sort_values(["cluster_id", "area", "source"], ascending=[True, False, True])
            .drop_duplicates("cluster_id", keep="first"))
    train, rest = train_test_split(kept, train_size=sizes[0], stratify=kept["label"], random_state=seed)
    val, test = train_test_split(rest, test_size=sizes[2] / (sizes[1] + sizes[2]),
                                 stratify=rest["label"], random_state=seed)
    split = _columns(pd.concat([train.assign(split="train"), val.assign(split="val"), test.assign(split="test")]))
    assert_no_cluster_crosses(split)
    assert split["source"].is_unique, "the same original file appears twice"
    dropped = {"conflict_images": int(clusters["conflict"].sum()),
               "duplicate_copies": int(len(no_conflict) - len(kept))}
    return split, dropped


def assert_no_cluster_crosses(split):
    """Fail loudly if any cluster has images in more than one split (that would be leakage)."""
    crossing = split.groupby("cluster_id")["split"].nunique()
    bad = crossing[crossing > 1]
    assert bad.empty, f"{len(bad)} clusters cross splits, e.g. cluster {bad.index[0]}"


def split_summary(splits):
    """splits: {(version, kind): split table} -> images per split and class, with totals."""
    rows = []
    for (version, kind), s in splits.items():
        for name in ["train", "val", "test"]:
            part = s[s["split"] == name]
            counts = part["label"].value_counts()
            rows.append({"version": version, "split_type": kind, "split": name,
                         **{c: int(counts.get(c, 0)) for c in CLASSES}, "total": len(part),
                         "share_%": round(100 * len(part) / len(s), 1)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------- leak report
def leak_table(clusters, pairs, look, T=PHASH_T, cos=COS_THRESHOLD):
    """% of ORIGINAL Testing images linked to at least one Training image, per check type.
    exact / near / look-alike = a direct link of that type; any = the image's cluster contains a
    Training image (this also counts chains A-B-C)."""
    split = clusters["split"].values
    test_idx = np.where(split == "test")[0]

    def linked(a, b):
        a, b = np.asarray(a), np.asarray(b)
        cross = split[a] != split[b]                     # one image in train, one in test
        return set(np.where(split[a[cross]] == "test", a[cross], b[cross]))

    exact = pairs[pairs["exact"]]
    near = within(pairs, T)
    alike = look[look["cosine"] >= cos]
    has_train = clusters.groupby("cluster_id")["split"].apply(lambda s: (s == "train").any())
    any_leak = {i for i in test_idx if has_train[clusters["cluster_id"].iat[i]]}
    n = len(test_idx)
    return {"exact (MD5)": round(100 * len(linked(exact["a"], exact["b"])) / n, 2),
            f"near (pHash<={T})": round(100 * len(linked(near["a"], near["b"])) / n, 2),
            f"look-alike (cos>={cos})": round(100 * len(linked(alike["a"], alike["b"])) / n, 2),
            "any (cluster)": round(100 * len(any_leak) / n, 2)}


def clean_test_leak(split):
    """Same 'any (cluster)' measure for a split table: % of test images whose cluster has a train image."""
    train_clusters = set(split.loc[split["split"] == "train", "cluster_id"])
    test = split[split["split"] == "test"]
    return round(100 * test["cluster_id"].isin(train_clusters).mean(), 2)


# ---------------------------------------------------------------------------- plots
def plot_similarity_hist(look, out_dir, version):
    """Number of image pairs per cosine-similarity bin (0.90-1.00), all pairs and train-test pairs."""
    bins = np.linspace(COS_MIN_KEEP, 1.0, 41)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(look["cosine"], bins=bins, alpha=0.6, label="all pairs")
    ax.hist(look.loc[look["pair_type"] == "train-test", "cosine"], bins=bins, alpha=0.8, label="train-test pairs")
    ax.axvline(COS_THRESHOLD, color="red", linestyle="--", linewidth=1)
    ax.text(COS_THRESHOLD + 0.002, ax.get_ylim()[1] * 0.9, f"{COS_THRESHOLD}", color="red", fontsize=8)
    ax.set_xlabel("Cosine similarity of ResNet-50 features")
    ax.set_ylabel("Image pairs")
    ax.set_title(f"{version}: look-alike pairs (similarity >= {COS_MIN_KEEP})")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return _save(fig, out_dir, "similarity_hist.png")


def _image_rows(rows, out_dir, name, title, per_row):
    """rows: list of (row tag, [(path, caption), ...]). One figure row per entry."""
    fig, axes = plt.subplots(len(rows), per_row, figsize=(2.4 * per_row, 2.6 * len(rows)), squeeze=False)
    for r, (tag, images) in enumerate(rows):
        for c in range(per_row):
            ax = axes[r, c]
            ax.axis("off")
            if c < len(images):
                ax.imshow(load_grey(images[c][0]), cmap="gray")
                ax.set_title(images[c][1], fontsize=8)
        axes[r, 0].text(-0.06, 0.5, tag, transform=axes[r, 0].transAxes, rotation=90,
                        ha="right", va="center", fontsize=9, fontweight="bold")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    return _save(fig, out_dir, name)


def plot_lookalike_examples(look, pairs, out_dir, version, lo=COS_THRESHOLD, hi=COS_THRESHOLD + 0.01,
                            n=6, seed=SEED):
    """Random pairs just above the threshold that pHash did NOT already link (the new information)."""
    near_keys = set(zip(within(pairs, PHASH_T)["a"], within(pairs, PHASH_T)["b"]))
    new = look[(look["cosine"] >= lo) & (look["cosine"] < hi)]
    new = new[[(a, b) not in near_keys for a, b in zip(new["a"], new["b"])]]
    pick = new.sample(min(n, len(new)), random_state=seed).sort_values("cosine")
    rows = [(f"cos {r.cosine:.3f}", [(r.source_a, f"{r.split_a}: {r.label_a}"),
                                     (r.source_b, f"{r.split_b}: {r.label_b}")]) for r in pick.itertuples()]
    return _image_rows(rows, out_dir, "examples_lookalike.png",
                       f"{version}: look-alike pairs with cosine {lo:.2f}-{hi:.2f} not found by pHash", 2)


def plot_conflict_examples(clusters, out_dir, version, n_clusters=6, per_cluster=4, seed=SEED):
    """Up to n_clusters conflicting clusters; each row shows images of one cluster with their labels."""
    bad = clusters[clusters["conflict"]]
    ids = sorted(bad["cluster_id"].unique())
    ids = np.random.default_rng(seed).choice(ids, size=min(n_clusters, len(ids)), replace=False) if ids else []
    rows = []
    for cid in ids:
        g = bad[bad["cluster_id"] == cid]
        g = g.groupby("label").head(max(1, per_cluster // g["label"].nunique())).head(per_cluster)
        rows.append((f"cluster {cid} ({len(bad[bad.cluster_id == cid])} imgs)",
                     [(r.source, f"{r.split}: {r.label}") for r in g.itertuples()]))
    if not rows:
        return None
    return _image_rows(rows, out_dir, "examples_conflicts.png",
                       f"{version}: clusters whose copies have different labels", per_cluster)


def plot_leak_chart(leaks, clean_leaks, out_path_dir):
    """Grouped bars: % of test images leaked per check type, for each version; plus the clean split."""
    versions = list(leaks)
    kinds = list(next(iter(leaks.values())))
    x = np.arange(len(versions))
    width = 0.8 / (len(kinds) + 1)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for k, kind in enumerate(kinds + ["clean split (any)"]):
        values = [clean_leaks[v] if kind == "clean split (any)" else leaks[v][kind] for v in versions]
        bars = ax.bar(x + (k - len(kinds) / 2) * width, values, width, label=kind)
        ax.bar_label(bars, fmt="%.1f", fontsize=8)
    ax.set_xticks(x, [f"{v} (original Testing folder)" for v in versions])
    ax.set_ylabel("% of test images with a twin in Training")
    ax.set_ylim(0, 105)
    ax.set_title("Test-set leakage per check type")
    ax.legend(fontsize=8, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return _save(fig, out_path_dir, "leak_summary.png")
