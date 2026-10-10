"""Shared data helpers for the brain tumour MRI project.

Everything here only READS ./data/ (it is read-only); plots are written to outputs/.
Used by notebooks/00_data_check.ipynb and later notebooks.
"""
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm

# The 4 classes of the task, in a fixed order used everywhere in the project.
CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
# The two split folders that every dataset version has.
SPLITS = ["Training", "Testing"]
# File endings we treat as images.
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
# One seed for the whole project.
SEED = 42

# Known image totals of the two Kaggle releases (from the dataset page, used only to name them).
EXPECTED_TOTALS = {"v1": 7023, "v2": 7200}


def set_seed(seed=SEED):
    """Make Python and NumPy random choices repeatable."""
    random.seed(seed)
    np.random.seed(seed)


def _norm(name):
    """Normalise a folder name for matching: lowercase, no '_', '-', spaces or the word 'tumor'.
    So 'Glioma', 'glioma_tumor' and 'GLIOMA' all become 'glioma', and 'no_tumor' becomes 'no'."""
    n = name.lower().replace("_", "").replace("-", "").replace(" ", "")
    return n.replace("tumour", "").replace("tumor", "")


def find_subdir(parent, wanted):
    """Return the sub-folder of `parent` whose name matches `wanted` (case-insensitive), or None."""
    target = _norm(wanted)        # e.g. 'notumor' -> 'no', which also matches 'no_tumor'
    for child in Path(parent).iterdir():
        if child.is_dir() and _norm(child.name) == target:
            return child
    return None


def find_versions(data_root="data"):
    """List every version folder in data_root that contains a Training or Testing folder."""
    versions = []
    for child in sorted(Path(data_root).iterdir()):
        if child.is_dir() and any(find_subdir(child, s) for s in SPLITS):
            versions.append(child.name)
    return versions


def list_images(data_root="data"):
    """Walk every version/split/class folder and return one table row per image file."""
    rows = []
    for version in find_versions(data_root):
        for split in SPLITS:
            split_dir = find_subdir(Path(data_root) / version, split)
            if split_dir is None:
                continue
            for label in CLASSES:
                class_dir = find_subdir(split_dir, label)
                if class_dir is None:
                    print(f"WARNING: no '{label}' folder in {split_dir}")
                    continue
                for f in sorted(class_dir.iterdir()):
                    if f.is_file() and f.suffix.lower() in IMG_EXTS:
                        rows.append({"version": version, "split": split, "label": label,
                                     "path": str(f), "ext": f.suffix.lower()})
    return pd.DataFrame(rows)


def count_table(df):
    """Images per (version, split) x class, with a Total column and per-version total rows."""
    table = pd.crosstab([df["version"], df["split"]], df["label"])
    table = table.reindex(columns=CLASSES, fill_value=0)
    table["Total"] = table.sum(axis=1)
    return table


def identify_versions(df):
    """Match each version folder to v1 (~7,023) or v2 (~7,200) by the closest image total."""
    result = {}
    for version, total in df.groupby("version").size().items():
        # Pick the known release whose total is nearest to this folder's total.
        name, expected = min(EXPECTED_TOTALS.items(), key=lambda kv: abs(kv[1] - total))
        diff = total - expected
        result[version] = {"total": total, "looks_like": name, "expected": expected, "difference": diff}
        if abs(diff) > 100:
            print(f"WARNING: {version} has {total} images, far from any known release")
    return pd.DataFrame(result).T


def inspect_images(df):
    """Open every image: record width, height, colour mode, format, and whether it is corrupt."""
    records = []
    for path in tqdm(df["path"], desc="Checking images"):
        rec = {"path": path, "width": None, "height": None, "mode": None,
               "format": None, "ok": True, "error": ""}
        try:
            with Image.open(path) as im:      # 1st open: verify() checks the file structure
                im.verify()
            with Image.open(path) as im:      # 2nd open: load() decodes all pixels (finds truncated files)
                im.load()
                rec.update(width=im.width, height=im.height, mode=im.mode, format=im.format)
        except Exception as e:                # any failure = corrupt / unreadable file
            rec.update(ok=False, error=f"{type(e).__name__}: {e}")
        records.append(rec)
    # Join the new columns back onto the file table using the path.
    return df.merge(pd.DataFrame(records), on="path")


def sample_paths(df, version, n=4, seed=SEED):
    """Pick n random images per class from one version (same picks every run)."""
    picks = {}
    for label in CLASSES:
        paths = df[(df["version"] == version) & (df["label"] == label)]["path"].tolist()
        picks[label] = random.Random(seed).sample(paths, min(n, len(paths)))
    return picks


def _save(fig, out_dir, name):
    """Save a figure as PNG into out_dir (created if missing) and return the path."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)       # free memory and stop the notebook drawing the figure a second time
    return path


def plot_class_counts(df, out_dir):
    """Grouped bar chart: images per class, one panel per version, Training and Testing side by side."""
    versions = sorted(df["version"].unique())
    fig, axes = plt.subplots(1, len(versions), figsize=(6 * len(versions), 4.5), sharey=True)
    axes = np.atleast_1d(axes)
    x = np.arange(len(CLASSES))
    width = 0.38
    for ax, version in zip(axes, versions):
        sub = df[df["version"] == version]
        for i, split in enumerate(SPLITS):
            counts = [((sub["split"] == split) & (sub["label"] == c)).sum() for c in CLASSES]
            bars = ax.bar(x + (i - 0.5) * width, counts, width, label=split)
            ax.bar_label(bars, fontsize=8)          # write the count on top of each bar
        ax.set_xticks(x, CLASSES)
        ax.set_title(f"{version}  (total {len(sub):,} images)")
        ax.set_ylabel("Images")
        ax.legend()
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Images per class")
    fig.tight_layout()
    return _save(fig, out_dir, "class_counts.png")


def plot_sample_grid(df, version, out_dir, n=4, seed=SEED):
    """Grid: one row per class, n sample images per row, titled with size and colour mode."""
    picks = sample_paths(df, version, n, seed)
    fig, axes = plt.subplots(len(CLASSES), n, figsize=(2.6 * n, 2.8 * len(CLASSES)))
    for r, label in enumerate(CLASSES):
        for c in range(n):
            ax = axes[r, c]
            ax.axis("off")
            if c >= len(picks[label]):
                continue
            with Image.open(picks[label][c]) as im:
                ax.imshow(im.convert("RGB"))
                ax.set_title(f"{im.width}x{im.height} {im.mode}", fontsize=8)
            if c == 0:   # class name on the left of each row
                ax.text(-0.08, 0.5, label, transform=ax.transAxes, rotation=90,
                        ha="right", va="center", fontsize=11, fontweight="bold")
    fig.suptitle(f"Sample images per class: {version} (seed {seed})")
    fig.tight_layout()
    return _save(fig, out_dir, f"samples_{version}.png")


def plot_size_hist(info, out_dir):
    """Histogram of image widths and heights per version (shows why we resize to 224)."""
    ok = info[info["ok"]]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, col in zip(axes, ["width", "height"]):
        values = ok[col].astype(int)
        bins = np.linspace(values.min(), values.max(), 41)   # same 40 bins for every version
        for version in sorted(ok["version"].unique()):
            ax.hist(ok[ok["version"] == version][col].astype(int), bins=bins, alpha=0.6, label=version)
        ax.set_xlabel(f"Image {col} (pixels)")
        ax.set_ylabel("Images")
        ax.legend()
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Image size distribution")
    fig.tight_layout()
    return _save(fig, out_dir, "image_sizes.png")
