"""Shared data helpers for the brain tumour MRI project.

Everything here only READS ./data/ (it is read-only); plots are written to outputs/.
Used by notebooks/00_data_check.ipynb and later notebooks.
"""
import random
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.transforms import v2 as T
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


# =====================================================================================
# Week 2: preprocessing (crop + resize), CSV index, cache, transforms, Dataset, timing
# =====================================================================================

# ImageNet colour statistics: our backbones were pre-trained with these, so we normalise the same way.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMG_SIZE = 224


def brain_mask(grey):
    """Binary mask (uint8 0/255) of the head/brain area of a greyscale image.
    Steps: blur (removes noise) -> Otsu threshold (picks the best dark/bright cut automatically)
    -> morphological opening (removes small specks such as text or scanner noise)."""
    blurred = cv2.GaussianBlur(grey, (5, 5), 0)
    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((5, 5), np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)


def crop_brain(grey, margin=0.02, min_area=0.10):
    """Crop the black border around the brain.
    Returns (cropped image, box (x, y, w, h), used_fallback). If no sensible brain region is found,
    the full image is returned unchanged and used_fallback is True."""
    h, w = grey.shape
    contours, _ = cv2.findContours(brain_mask(grey), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return grey, (0, 0, w, h), True
    x, y, bw, bh = cv2.boundingRect(max(contours, key=cv2.contourArea))   # box of the largest blob
    if bw * bh < min_area * w * h:          # blob too small to be the head -> do not trust it
        return grey, (0, 0, w, h), True
    pad = int(margin * max(w, h))           # small safety margin so the skull edge is not cut
    x0, y0 = max(x - pad, 0), max(y - pad, 0)
    x1, y1 = min(x + bw + pad, w), min(y + bh + pad, h)
    return grey[y0:y1, x0:x1], (x0, y0, x1 - x0, y1 - y0), False


def pad_resize(grey, size=IMG_SIZE):
    """Pad the image to a square with black (centred), then resize to size x size.
    Padding instead of stretching keeps the brain's real shape."""
    h, w = grey.shape
    side = max(h, w)
    square = np.zeros((side, side), dtype=grey.dtype)
    top, left = (side - h) // 2, (side - w) // 2
    square[top:top + h, left:left + w] = grey
    return cv2.resize(square, (size, size), interpolation=cv2.INTER_AREA)


def load_grey(path):
    """Open any image file (JPEG/PNG, any colour mode) as a 2-D uint8 greyscale array."""
    with Image.open(path) as im:
        return np.asarray(im.convert("L"))


def preprocess(path, size=IMG_SIZE):
    """Full fixed preprocessing for one file: greyscale -> crop border -> pad to square -> resize.
    Returns (uint8 array size x size, used_fallback)."""
    cropped, _, fallback = crop_brain(load_grey(path))
    return pad_resize(cropped, size), fallback


def make_index_csv(df, version, out_csv):
    """Write the 'original split' index for one version: columns path, label, split (train/test)."""
    sub = df[df["version"] == version]
    split = sub["split"].str.lower().map({"training": "train", "testing": "test"})
    index = pd.DataFrame({"path": sub["path"].values, "label": sub["label"].values, "split": split.values})
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    index.to_csv(out_csv, index=False)
    return index


def build_cache(index_csv, cache_dir, out_csv, size=IMG_SIZE):
    """Preprocess every image in index_csv once and save it as a small greyscale PNG in cache_dir.
    Files already in the cache are skipped (safe to re-run after a crash).
    Writes out_csv with columns path (cached PNG), label, split, source (original file), crop_fallback."""
    index = pd.read_csv(index_csv)
    old = pd.read_csv(out_csv) if Path(out_csv).exists() else None
    old_flags = dict(zip(old["source"], old["crop_fallback"])) if old is not None else {}
    rows = []
    for r in tqdm(index.itertuples(), total=len(index), desc=f"Caching {Path(cache_dir).name}"):
        out = Path(cache_dir) / r.split / r.label / (Path(r.path).stem + ".png")
        if out.exists() and r.path in old_flags:          # already done in an earlier run
            fallback = bool(old_flags[r.path])
        else:
            img, fallback = preprocess(r.path, size)
            out.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(img).save(out)
        rows.append({"path": str(out), "label": r.label, "split": r.split,
                     "source": r.path, "crop_fallback": fallback})
    cached = pd.DataFrame(rows)
    cached.to_csv(out_csv, index=False)
    return cached


def train_transforms(size=IMG_SIZE):
    """Random augmentation for training (applied on the fly, a new random version every epoch)."""
    return T.Compose([
        T.Resize((size, size), antialias=True),        # no-op for cached images, safety for raw ones
        T.RandomHorizontalFlip(p=0.5),                 # left/right brain halves are mirror-like
        T.RandomRotation(degrees=10, fill=0),          # small head tilt, fill corners with black
        T.ColorJitter(brightness=0.1, contrast=0.1),   # small scanner brightness/contrast differences
        T.ToImage(),                                   # PIL -> tensor (C, H, W)
        T.ToDtype(torch.float32, scale=True),          # 0-255 -> 0.0-1.0
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),      # same statistics as ImageNet pre-training
    ])


def eval_transforms(size=IMG_SIZE):
    """Fixed transform for validation/testing: resize + normalise only (no randomness)."""
    return T.Compose([
        T.Resize((size, size), antialias=True),
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def denormalize(tensor):
    """Undo ImageNet normalisation so a (3, H, W) tensor can be shown with imshow (returns H, W, 3 in 0-1)."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (tensor * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()


class MRIDataset(Dataset):
    """PyTorch Dataset reading images listed in a CSV with columns path, label, split.
    crop=True  -> raw files: greyscale + crop + pad-resize happen here (slow, every time).
    crop=False -> cached files: already preprocessed, only opened.
    Kept in src/ (not in a notebook) so Windows DataLoader workers can import it."""

    def __init__(self, csv_path, split, transform=None, crop=False, size=IMG_SIZE):
        table = pd.read_csv(csv_path)
        self.table = table[table["split"] == split].reset_index(drop=True)
        self.transform = transform
        self.crop = crop
        self.size = size
        self.class_to_idx = {c: i for i, c in enumerate(CLASSES)}   # glioma=0 ... pituitary=3

    def __len__(self):
        return len(self.table)

    def __getitem__(self, i):
        row = self.table.iloc[i]
        if self.crop:
            grey, _ = preprocess(row["path"], self.size)
        else:
            grey = load_grey(row["path"])
        image = Image.fromarray(grey).convert("RGB")     # copy grey into 3 identical channels
        if self.transform is not None:
            image = self.transform(image)
        return image, self.class_to_idx[row["label"]]


def time_loader(dataset, num_workers, n_images=500, batch_size=32, seed=SEED):
    """Time how long a DataLoader takes to read n_images (same random subset every run).
    Returns seconds for pass 1 (includes starting the workers) and pass 2 (workers already running)."""
    idx = np.random.default_rng(seed).choice(len(dataset), size=min(n_images, len(dataset)), replace=False)
    loader = DataLoader(Subset(dataset, idx.tolist()), batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
                        persistent_workers=num_workers > 0)
    times = []
    for _ in range(2):
        start = time.perf_counter()
        n = sum(images.shape[0] for images, _ in loader)
        times.append(time.perf_counter() - start)
    del loader                                            # shuts the worker processes down
    return {"num_workers": num_workers, "images": n,
            "pass1_s": round(times[0], 2), "pass2_s": round(times[1], 2),
            "pass2_img_per_s": round(n / times[1], 1)}


def plot_crop_examples(index_csv, out_dir, n_per_class=2, seed=SEED):
    """Top row: original image with the crop box in red. Bottom row: the final 224x224 result."""
    index = pd.read_csv(index_csv)
    picks = []
    for c in CLASSES:
        paths = index[index["label"] == c]["path"].tolist()
        picks += random.Random(seed).sample(paths, n_per_class)
    fig, axes = plt.subplots(2, len(picks), figsize=(2.2 * len(picks), 5.6),
                             gridspec_kw={"hspace": 0.35})
    for j, path in enumerate(picks):
        grey = load_grey(path)
        cropped, (x, y, w, h), fallback = crop_brain(grey)
        axes[0, j].imshow(grey, cmap="gray")
        axes[0, j].add_patch(plt.Rectangle((x, y), w, h, fill=False, edgecolor="red", linewidth=1.5))
        axes[0, j].set_title(f"{CLASSES[j // n_per_class]}\n{grey.shape[1]}x{grey.shape[0]}", fontsize=8)
        axes[1, j].imshow(pad_resize(cropped), cmap="gray")
        axes[1, j].set_title("224x224" + (" (fallback)" if fallback else ""), fontsize=8)
        for ax in axes[:, j]:
            ax.axis("off")
    fig.suptitle("Before (red box = detected brain) and after crop + pad + resize")
    fig.tight_layout()
    return _save(fig, out_dir, "crop_examples.png")


def plot_augment_examples(cache_csv, out_dir, n_aug=5, seed=SEED):
    """Per class: the cached image, the eval transform, and n_aug random train augmentations."""
    torch.manual_seed(seed)
    table = pd.read_csv(cache_csv)
    train_t, eval_t = train_transforms(), eval_transforms()
    fig, axes = plt.subplots(len(CLASSES), n_aug + 2, figsize=(2.1 * (n_aug + 2), 2.3 * len(CLASSES)))
    for r, label in enumerate(CLASSES):
        paths = table[(table["label"] == label) & (table["split"] == "train")]["path"].tolist()
        image = Image.open(random.Random(seed).choice(paths)).convert("RGB")
        panels = [("cached", np.asarray(image)), ("eval", denormalize(eval_t(image)))]
        panels += [(f"train aug {k + 1}", denormalize(train_t(image))) for k in range(n_aug)]
        for c, (title, picture) in enumerate(panels):
            axes[r, c].imshow(picture)
            axes[r, c].set_title(title, fontsize=8)
            axes[r, c].axis("off")
        axes[r, 0].text(-0.08, 0.5, label, transform=axes[r, 0].transAxes, rotation=90,
                        ha="right", va="center", fontsize=11, fontweight="bold")
    fig.suptitle("Eval transform vs random training augmentations (flip, rotate +-10 deg, brightness/contrast)")
    fig.tight_layout()
    return _save(fig, out_dir, "augment_examples.png")


def plot_loader_timing(timing, out_dir):
    """Bar chart of images/second (pass 2) for each data source and num_workers setting."""
    fig, ax = plt.subplots(figsize=(6.5, 4))
    labels = [f"{s}\nworkers={w}" for s, w in zip(timing["source"], timing["num_workers"])]
    colours = ["#4C72B0" if s == "cached" else "#DD8452" for s in timing["source"]]
    bars = ax.bar(labels, timing["pass2_img_per_s"], color=colours)
    ax.bar_label(bars, fmt="%.0f", fontsize=9)
    ax.set_ylabel("Images per second (pass 2)")
    ax.set_title(f"DataLoader speed, {timing['images'].iloc[0]} images")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return _save(fig, out_dir, "loader_timing.png")
