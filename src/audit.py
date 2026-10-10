"""Duplicate audit: exact copies (MD5) and near copies (64-bit perceptual hash, Hamming distance).

Reads the ORIGINAL files in ./data/ (read-only). Results are written to outputs/audit/<version>/.
Used by notebooks/02_duplicate_audit.ipynb.
"""
import hashlib
import random

import imagehash
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm.auto import tqdm

from src.data import CLASSES, SEED, _save, load_grey

THRESHOLDS = (3, 5, 8)   # Hamming distances we compare (out of 64 bits)
MAX_DIST = 10            # we keep every pair up to this distance, so any threshold <= 10 can be studied


# ----------------------------------------------------------------------------- hashing
def md5_of_file(path):
    """MD5 of the raw file bytes: identical only if the two files are byte-for-byte the same."""
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def hash_images(index_csv):
    """Compute MD5 and 64-bit pHash for every image in an index CSV (columns path, label, split)."""
    table = pd.read_csv(index_csv)
    md5s, phashes = [], []
    for path in tqdm(table["path"], desc="Hashing"):
        md5s.append(md5_of_file(path))
        with Image.open(path) as im:
            # pHash: shrink to 32x32 grey, take the DCT, keep the 8x8 lowest frequencies,
            # and set each bit to 1 if that value is above the median -> 64 bits.
            phashes.append(str(imagehash.phash(im.convert("L"))))   # 16 hex characters
    table["md5"] = md5s
    table["phash"] = phashes
    return table


# Note: pHash sets a bit to 1 when a value is above the MEDIAN of its 64 values, so every hash
# has exactly 32 ones. Flipping one 1 to 0 forces another 0 to 1, so Hamming distances between
# pHashes are always EVEN (0, 2, 4, ...). Threshold 3 therefore acts like 2, and 5 like 4.
def phash_bits(hex_list):
    """Turn 16-character hex hashes into an (N, 64) tensor of 0/1 bits."""
    as_bytes = np.array([list(bytes.fromhex(h)) for h in hex_list], dtype=np.uint8)   # (N, 8)
    return torch.from_numpy(np.unpackbits(as_bytes, axis=1))                         # (N, 64)


# ------------------------------------------------------------------ GPU Hamming distance
def pairwise_hamming(bits, max_dist=MAX_DIST, device=None, chunk=2048):
    """All pairs (i < j) whose Hamming distance is <= max_dist, computed with matrix maths.

    For 0/1 vectors a and b:  hamming(a, b) = ones(a) + ones(b) - 2 * (a . b)
    (a . b counts positions where BOTH are 1). Doing a @ B.T gives this for a whole block of
    rows at once on the GPU. float16 is exact here because every value is a whole number <= 64.
    Rows are processed in chunks so the full N x N matrix is never in GPU memory at once."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device == "cuda" else torch.float32
    B = bits.to(device=device, dtype=dtype)              # (N, 64)
    ones = B.sum(dim=1)                                   # number of 1-bits per hash, (N,)
    n = B.shape[0]
    found_i, found_j, found_d = [], [], []
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        dist = ones[start:stop, None] + ones[None, :] - 2 * (B[start:stop] @ B.T)   # (chunk, N)
        rows = torch.arange(start, stop, device=device)[:, None]
        cols = torch.arange(n, device=device)[None, :]
        keep = (dist <= max_dist) & (cols > rows)        # upper triangle only: each pair once
        r, c = torch.nonzero(keep, as_tuple=True)
        found_i.append(r + start)
        found_j.append(c)
        found_d.append(dist[r, c])
    i = torch.cat(found_i).cpu().numpy()
    j = torch.cat(found_j).cpu().numpy()
    d = torch.cat(found_d).round().to(torch.int64).cpu().numpy()
    return i, j, d


def verify_hamming(hashes, bits, n_pairs=1000, seed=SEED):
    """Check the GPU formula against imagehash's own distance on random pairs. Returns #mismatches."""
    rng = np.random.default_rng(seed)
    a = rng.integers(0, len(hashes), n_pairs)
    b = rng.integers(0, len(hashes), n_pairs)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    B = bits.to(device=device, dtype=torch.float16 if device == "cuda" else torch.float32)
    ones = B.sum(dim=1)
    gpu = (ones[a] + ones[b] - 2 * (B[a] * B[b]).sum(dim=1)).round().to(torch.int64).cpu().numpy()
    ph = hashes["phash"].tolist()
    cpu = np.array([imagehash.hex_to_hash(ph[x]) - imagehash.hex_to_hash(ph[y]) for x, y in zip(a, b)])
    return int((gpu != cpu).sum())


# ------------------------------------------------------------------------- pair table
def build_pairs(hashes, i, j, d):
    """One row per image pair with distance <= MAX_DIST, plus all exact (same MD5) pairs."""
    pairs = pd.DataFrame({"a": i, "b": j, "distance": d})
    # Safety: add MD5-identical pairs even if their pHash distance were > MAX_DIST (should never happen).
    md5_pairs = []
    for _, idx in hashes.groupby("md5").groups.items():
        idx = sorted(idx)
        md5_pairs += [(x, y) for k, x in enumerate(idx) for y in idx[k + 1:]]
    known = set(zip(pairs["a"], pairs["b"]))
    missing = [(x, y) for x, y in md5_pairs if (x, y) not in known]
    if missing:
        pairs = pd.concat([pairs, pd.DataFrame({"a": [m[0] for m in missing], "b": [m[1] for m in missing],
                                                "distance": -1})], ignore_index=True)
    A, Bt = hashes.iloc[pairs["a"]].reset_index(drop=True), hashes.iloc[pairs["b"]].reset_index(drop=True)
    pairs["path_a"], pairs["path_b"] = A["path"], Bt["path"]
    pairs["split_a"], pairs["split_b"] = A["split"], Bt["split"]
    pairs["label_a"], pairs["label_b"] = A["label"], Bt["label"]
    pairs["exact"] = (A["md5"] == Bt["md5"]).values
    pairs["pair_type"] = ["-".join(sorted([s1, s2], reverse=True)) for s1, s2 in zip(A["split"], Bt["split"])]
    pairs["label_conflict"] = pairs["label_a"] != pairs["label_b"]
    return pairs


def within(pairs, T):
    """Pairs that count as duplicates at threshold T (all exact pairs are always included)."""
    return pairs[(pairs["distance"] <= T) | pairs["exact"]]


def _test_with_twin(pairs, T):
    """Set of test-image paths that have at least one Training twin at distance <= T."""
    cross = within(pairs, T)
    cross = cross[cross["pair_type"] == "train-test"]
    return set(np.where(cross["split_a"] == "test", cross["path_a"], cross["path_b"]))


def threshold_summary(pairs, hashes, thresholds=THRESHOLDS):
    """One row per threshold: pair counts and how many test images have a Training twin."""
    n_test = int((hashes["split"] == "test").sum())
    rows = []
    for T in thresholds:
        dup = within(pairs, T)
        twins = _test_with_twin(pairs, T)
        rows.append({"threshold": T,
                     "exact_pairs": int(dup["exact"].sum()),
                     "near_pairs (not exact)": int((~dup["exact"]).sum()),
                     "train-train": int((dup["pair_type"] == "train-train").sum()),
                     "test-test": int((dup["pair_type"] == "test-test").sum()),
                     "train-test": int((dup["pair_type"] == "train-test").sum()),
                     "label_conflicts": int(dup["label_conflict"].sum()),
                     "test_with_train_twin": len(twins),
                     "test_with_train_twin_%": round(100 * len(twins) / n_test, 2)})
    return pd.DataFrame(rows)


def test_twins_per_class(pairs, hashes, thresholds=THRESHOLDS):
    """Per class and threshold: test images, how many have a Training twin, and the percentage."""
    test = hashes[hashes["split"] == "test"]
    rows = []
    for T in thresholds:
        twins = _test_with_twin(pairs, T)
        for label in CLASSES:
            paths = test[test["label"] == label]["path"]
            k = int(paths.isin(twins).sum())
            rows.append({"threshold": T, "class": label, "test_images": len(paths),
                         "with_train_twin": k, "percent": round(100 * k / len(paths), 2)})
    return pd.DataFrame(rows)


def exact_pairs_check(hashes):
    """Number of exact pairs computed directly from MD5 groups: sum of k*(k-1)/2 over group sizes k."""
    sizes = hashes.groupby("md5").size()
    return int((sizes * (sizes - 1) // 2).sum())


def duplicate_groups(hashes, pairs, T=5):
    """Union-find: images linked by a duplicate pair (distance <= T or exact) share a group_id.
    Used later so all copies of an image go into the same split."""
    parent = list(range(len(hashes)))

    def find(x):                       # follow parents to the group's root
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in zip(within(pairs, T)["a"], within(pairs, T)["b"]):
        parent[find(a)] = find(b)      # join the two groups
    out = hashes[["path", "label", "split"]].copy()
    out["group_id"] = [find(x) for x in range(len(hashes))]
    return out


# ------------------------------------------------------------------------------ plots
def plot_distance_hist(pairs, out_dir, version):
    """Number of train-test pairs at each Hamming distance 0..MAX_DIST."""
    cross = pairs[(pairs["pair_type"] == "train-test") & (pairs["distance"] >= 0)]
    counts = cross["distance"].value_counts().reindex(range(MAX_DIST + 1), fill_value=0)
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(counts.index, counts.values, color="#4C72B0")
    ax.bar_label(bars, fontsize=8)
    for T in THRESHOLDS:
        ax.axvline(T + 0.5, color="red", linestyle="--", linewidth=1)
        ax.text(T + 0.55, ax.get_ylim()[1] * 0.92, f"T={T}", color="red", fontsize=8)
    ax.set_xticks(range(MAX_DIST + 1))
    ax.set_xlabel("pHash Hamming distance (0 = identical hash)")
    ax.set_ylabel("Train-test pairs")
    ax.set_title(f"{version}: train-test pairs by distance")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return _save(fig, out_dir, "distance_hist.png")


def _show_pairs(chosen, out_dir, name, title):
    """Draw each chosen pair as Training (left) | Testing (right)."""
    n = len(chosen)
    fig, axes = plt.subplots(n, 2, figsize=(5, 2.6 * n), squeeze=False)
    for r, row in enumerate(chosen.itertuples()):
        train_path, test_path = ((row.path_a, row.path_b) if row.split_a == "train" else (row.path_b, row.path_a))
        train_lab, test_lab = ((row.label_a, row.label_b) if row.split_a == "train" else (row.label_b, row.label_a))
        for c, (path, lab, split) in enumerate([(train_path, train_lab, "Training"), (test_path, test_lab, "Testing")]):
            grey = load_grey(path)
            axes[r, c].imshow(grey, cmap="gray")
            axes[r, c].set_title(f"{split}: {lab} {grey.shape[1]}x{grey.shape[0]}", fontsize=8)
            axes[r, c].axis("off")
        tag = "exact (MD5)" if row.exact else f"d = {row.distance}"
        axes[r, 0].text(-0.06, 0.5, tag, transform=axes[r, 0].transAxes, rotation=90,
                        ha="right", va="center", fontsize=9, fontweight="bold")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    return _save(fig, out_dir, name)


def plot_near_dup_examples(pairs, out_dir, version, n=6, T=5, seed=SEED):
    """n train/test pairs that are near (not exact) copies, spread over distances 1..T."""
    cross = pairs[(pairs["pair_type"] == "train-test")]
    near = cross[(~cross["exact"]) & (cross["distance"].between(1, T))]
    rng = random.Random(seed)
    chosen = []
    # Round-robin over distances so small and large distances are both shown.
    pools = {dd: near[near["distance"] == dd].index.tolist() for dd in range(1, T + 1)}
    for pool in pools.values():
        rng.shuffle(pool)
    while len(chosen) < n and any(pools.values()):
        for dd in range(1, T + 1):
            if pools[dd] and len(chosen) < n:
                chosen.append(pools[dd].pop())
    if len(chosen) < n:                                  # not enough near pairs: fill with exact ones
        exact = cross[cross["exact"]].index.tolist()
        chosen += rng.sample(exact, min(n - len(chosen), len(exact)))
    sel = pairs.loc[chosen].sort_values("distance")
    return _show_pairs(sel, out_dir, "examples_near_dup.png",
                       f"{version}: near-duplicate Training | Testing pairs")


def plot_examples_by_distance(pairs, out_dir, version, distances=(0, 2, 4, 6, 8), per=2, seed=SEED):
    """per train/test pairs at each chosen distance, to judge by eye where 'same image' stops."""
    cross = pairs[pairs["pair_type"] == "train-test"]
    chosen = []
    for dd in distances:
        pool = cross[cross["distance"] == dd].index.tolist()
        chosen += random.Random(seed).sample(pool, min(per, len(pool)))
    sel = pairs.loc[chosen]
    return _show_pairs(sel, out_dir, "examples_by_distance.png",
                       f"{version}: Training | Testing pairs at distance {', '.join(map(str, distances))}")


def plot_class_pairs(pairs, out_dir, version, label, distances=(2, 4), per=4, seed=SEED):
    """Random non-exact Training | Testing pairs of ONE class at the given distances (visual check)."""
    cross = pairs[(pairs["pair_type"] == "train-test") & (~pairs["exact"])
                  & (pairs["label_a"] == label) & (pairs["label_b"] == label)]
    chosen = []
    for dd in distances:
        pool = cross[cross["distance"] == dd].index.tolist()
        chosen += random.Random(seed).sample(pool, min(per, len(pool)))
    return _show_pairs(pairs.loc[chosen], out_dir, f"examples_{label}.png",
                       f"{version}: random {label} Training | Testing pairs (d = {', '.join(map(str, distances))})")
