"""Week 9: Grad-CAM and the "inside-brain ratio".

For every test image of a run, Grad-CAM shows where the network looked when it chose its predicted class.
The inside-brain ratio is the share of that heat that lies inside the brain:
    ratio = (sum of heat inside the brain mask) / (sum of all heat)      0 = all outside, 1 = all inside
A uniform heat map would score exactly mask_frac (the share of the picture covered by the brain), so the
ratio is also compared with that "chance" level (ratio_over_chance = ratio / mask_frac).
Brain mask = Otsu threshold -> keep the largest connected component -> fill holes.
Used by notebooks/07_gradcam.ipynb.
"""
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from scipy.ndimage import binary_fill_holes
from torch.utils.data import DataLoader, Subset

from src.data import CLASSES, SEED, _save, brain_mask, denormalize
from src.evaluate import load_run, make_dataset, softmax
from src.train import NUM_WORKERS, _atomic, set_all_seeds

# The last layer that still has a spatial map (7 x 7 at 224 input) for every architecture.
# mobilenet: NOT conv_head, it sits after global pooling (its map is 1 x 1 and the CAM would be all zero).
TARGET_LAYER = {
    "mobilenetv3_small_100": lambda m: [m.blocks[-1]],
    "efficientnet_b0": lambda m: [m.bn2],
    "resnet50": lambda m: [m.layer4[-1]],
    "vgg16": lambda m: [m.features[-1]],
}
BATCH = {"vgg16": 8, "resnet50": 16}      # Grad-CAM keeps gradients, so batches are smaller than at evaluation
DEFAULT_BATCH = 32
TINY_MASK = 0.25                          # a brain mask covering less than 25% of the picture is flagged as suspicious


# ------------------------------------------------------------------------------------ mask + ratio
def inside_mask(grey):
    """Brain mask of a 2-D uint8 image: Otsu (data.brain_mask) -> largest connected blob -> holes filled.
    Returns a boolean array of the same shape."""
    raw = brain_mask(grey) > 0
    n, labels, stats, _ = cv2.connectedComponentsWithStats(raw.astype(np.uint8), connectivity=8)
    if n <= 1:                                               # label 0 is the background: nothing was found
        return np.zeros(grey.shape, dtype=bool)
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))   # skip the background row, take the largest blob
    return binary_fill_holes(labels == biggest)             # ventricles / dark tumour parts inside the brain count as brain


def inside_ratio(cam, mask):
    """Share of the heat map (values 0..1, shape H x W) that lies inside the boolean mask.
    Returns (ratio, empty); empty=True means the whole heat map is zero (ratio is then set to 0)."""
    total = float(cam.sum())
    if total <= 0:
        return 0.0, True
    return float(cam[mask].sum() / total), False


def load_grey(path):
    """The cached 224 x 224 greyscale PNG as a uint8 array (the same pixels the network sees)."""
    return np.array(Image.open(str(path).replace("\\", "/")).convert("L"))


# ----------------------------------------------------------------------------------- Grad-CAM runs
def make_cam(model, cfg):
    """GradCAM object for the model's last spatial layer (use it in a `with` block so the hooks are removed)."""
    return GradCAM(model=model, target_layers=TARGET_LAYER[cfg["model"]](model))


def compute_cams(cam, images, targets):
    """Heat maps (B, 224, 224) in 0..1 for the given class of each image, plus the logits of that forward pass."""
    maps = cam(input_tensor=images, targets=[ClassifierOutputTarget(int(t)) for t in targets])
    return maps, cam.outputs.detach().cpu().numpy()


def _pass(model, cfg, ds, table, labels, preds, conf, batch, device, run_name):
    """One pass over the dataset with a given batch size. Returns the list of per-image rows."""
    loader = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    rows = []
    with make_cam(model, cfg) as cam:
        for i, (images, y) in enumerate(loader):
            start = i * batch                                         # shuffle=False: batch i holds rows start, start+1, ...
            idx = np.arange(start, start + len(y))
            assert (y.numpy() == labels[idx]).all(), "row order differs from test_outputs.npz"
            images = images.to(device).to(memory_format=torch.channels_last)
            maps, logits = compute_cams(cam, images, preds[idx])
            for j, r in enumerate(idx):
                grey = load_grey(table.loc[r, "path"])      # table = the test rows, same order as ds
                mask = inside_mask(grey)
                ratio, empty = inside_ratio(maps[j], mask)
                frac = float(mask.mean())
                rows.append({"run": run_name, "row": int(r), "label": int(labels[r]), "pred": int(preds[r]),
                             "correct": bool(labels[r] == preds[r]), "confidence": float(conf[r]),
                             "inside_ratio": ratio, "mask_frac": frac,
                             "ratio_over_chance": ratio / frac if frac > 0 else np.nan,
                             "tiny_mask": frac < TINY_MASK, "empty_cam": empty,
                             "same_pred_as_week8": bool(logits[j].argmax() == preds[r])})
    return rows


def gradcam_run(run_dir, limit=None, device="cuda"):
    """Grad-CAM for the PREDICTED class of every test image of one run -> DataFrame (one row per image).
    Predictions and confidence come from the Week-8 test_outputs.npz (temperature-scaled confidence).
    limit=N only uses the first N test images (smoke test). On CUDA out-of-memory the batch is halved and the pass restarted."""
    run_dir = Path(run_dir)
    set_all_seeds()
    model, cfg = load_run(run_dir, device)
    model.eval()
    name = f"{run_dir.parent.name}_{run_dir.name}"
    saved = np.load(run_dir / "test_outputs.npz")
    labels, preds = saved["labels"], saved["logits"].argmax(1)
    conf = softmax(saved["logits"], cfg["temperature"]).max(1)       # same confidence as the referral rule
    ds = make_dataset(cfg, "test")
    assert len(ds) == len(labels), "test set size differs from test_outputs.npz"
    table = ds.table
    if limit:
        ds = Subset(ds, range(min(limit, len(ds))))
    batch = BATCH.get(cfg["model"], DEFAULT_BATCH)
    while True:
        try:
            rows = _pass(model, cfg, ds, table, labels, preds, conf, batch, device, name)
            break
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if batch == 1:
                raise
            batch //= 2
            print(f"{name}: out of memory, retrying with batch {batch}", flush=True)
    del model
    torch.cuda.empty_cache()
    df = pd.DataFrame(rows)
    df.insert(1, "model", cfg["model"])
    return df


def gradcam_all(runs_root="outputs/runs", out_csv="outputs/results/week9_gradcam.csv", force=False):
    """Grad-CAM statistics for every finished run (one at a time). Each run is saved in
    outputs/gradcam/<run>/inside_ratio.csv; runs whose file is newer than best.pt are skipped unless force=True."""
    from src.evaluate import finished_runs
    frames = []
    for run_dir in finished_runs(runs_root):
        name = f"{run_dir.parent.name}_{run_dir.name}"
        path = Path("outputs/gradcam") / name / "inside_ratio.csv"
        if path.exists() and not force and path.stat().st_mtime > (run_dir / "best.pt").stat().st_mtime:
            print(f"{name:40s} already done", flush=True)
        else:
            t0 = time.time()
            df = gradcam_run(run_dir)
            path.parent.mkdir(parents=True, exist_ok=True)
            _atomic(path, lambda p: df.to_csv(p, index=False))
            print(f"{name:40s} {len(df)} images, mean ratio {df['inside_ratio'].mean():.3f} ({time.time() - t0:.0f} s)", flush=True)
        frames.append(pd.read_csv(path))
    allruns = pd.concat(frames, ignore_index=True)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    allruns.to_csv(out_csv, index=False)
    return allruns


# ---------------------------------------------------------------------------------------- summaries
def _stats(g):
    """n, mean and median of the ratio and of ratio_over_chance for a group of images."""
    return pd.Series({"n": len(g), "ratio_mean": g["inside_ratio"].mean(), "ratio_median": g["inside_ratio"].median(),
                      "chance": g["mask_frac"].mean(), "over_chance_median": g["ratio_over_chance"].median()})


def summarise(df):
    """Tables: per run, per true class, and correct vs wrong predictions (n is always shown: wrong groups are small)."""
    df = df.copy()
    df["class"] = [CLASSES[k] for k in df["label"]]
    df["outcome"] = np.where(df["correct"], "correct", "wrong")
    per_run = df.groupby("run").apply(_stats, include_groups=False)
    per_run["tiny_masks"] = df.groupby("run")["tiny_mask"].sum()
    per_run["empty_cams"] = df.groupby("run")["empty_cam"].sum()
    return {"per_run": per_run,
            "per_class": df.groupby(["run", "class"]).apply(_stats, include_groups=False),
            "correct_vs_wrong": df.groupby(["run", "outcome"]).apply(_stats, include_groups=False)}


# --------------------------------------------------------------------------------------- gallery
def pick_gallery(df, n_wrong=6, n_total=12, seed=SEED):
    """Row numbers of a random (seeded, not hand-picked) gallery: up to n_wrong wrong predictions, the rest correct."""
    rng = np.random.default_rng(seed)
    wrong = df.index[~df["correct"]].to_numpy()
    right = df.index[df["correct"]].to_numpy()
    pick_wrong = rng.choice(wrong, size=min(n_wrong, len(wrong)), replace=False)
    pick_right = rng.choice(right, size=n_total - len(pick_wrong), replace=False)
    return df.loc[np.concatenate([pick_wrong, pick_right])]


def plot_gallery(run_dir, picks, out_dir="outputs/plots/week9", device="cuda"):
    """Tiles: MRI + Grad-CAM overlay (inferno) + brain-mask outline (cyan). Title = true / predicted, confidence, ratio.
    Wrong predictions have a thick red frame AND the word WRONG; correct ones a green frame."""
    run_dir = Path(run_dir)
    model, cfg = load_run(run_dir, device)
    model.eval()
    ds = make_dataset(cfg, "test")
    images = torch.stack([ds[int(r)][0] for r in picks["row"]]).to(device).to(memory_format=torch.channels_last)
    with make_cam(model, cfg) as cam:
        maps, _ = compute_cams(cam, images, picks["pred"].to_numpy())
    cols = 4
    fig, axes = plt.subplots(int(np.ceil(len(picks) / cols)), cols, figsize=(3.4 * cols, 3.9 * int(np.ceil(len(picks) / cols))))
    for ax, (_, p), m in zip(axes.ravel(), picks.iterrows(), maps):
        grey = load_grey(ds.table.loc[int(p["row"]), "path"])
        ax.imshow(grey, cmap="gray")
        ax.imshow(m, cmap="inferno", alpha=0.45)
        ax.contour(inside_mask(grey), levels=[0.5], colors="cyan", linewidths=1.0)
        ax.set_xticks([])
        ax.set_yticks([])
        colour = "#009E73" if p["correct"] else "#D55E00"
        for spine in ax.spines.values():
            spine.set_edgecolor(colour)
            spine.set_linewidth(4 if not p["correct"] else 2)
        ax.set_title(f"{'OK' if p['correct'] else 'WRONG'}: true {CLASSES[int(p['label'])]}\npred {CLASSES[int(p['pred'])]} "
                     f"({p['confidence']:.2f}), inside {p['inside_ratio']:.2f}", fontsize=8.5)
    for ax in axes.ravel()[len(picks):]:
        ax.axis("off")
    fig.suptitle(f"{run_dir.parent.name} / {run_dir.name}: Grad-CAM of the predicted class (cyan = brain mask)", y=1.0)
    del model
    torch.cuda.empty_cache()
    return _save(fig, out_dir, f"gallery_{run_dir.parent.name}_{run_dir.name}.png")


def plot_correct_vs_wrong(df, out_dir="outputs/plots/week9"):
    """Boxplot of the inside-brain ratio per run, correct vs wrong predictions side by side (n written under each box)."""
    runs = sorted(df["run"].unique())
    fig, ax = plt.subplots(figsize=(14, 4.8))
    for k, (outcome, colour) in enumerate((("correct", "#0072B2"), ("wrong", "#D55E00"))):
        data = [df[(df["run"] == r) & (df["correct"] == (outcome == "correct"))]["inside_ratio"].to_numpy() for r in runs]
        pos = np.arange(len(runs)) + (k - 0.5) * 0.35
        ax.boxplot([d if len(d) else [np.nan] for d in data], positions=pos, widths=0.3, patch_artist=True,
                   boxprops=dict(facecolor=colour, alpha=0.6), medianprops=dict(color="black"), showfliers=False)
        for x, d in zip(pos, data):
            ax.text(x, 0.01, f"n={len(d)}", ha="center", fontsize=6.5, rotation=90)
        ax.plot([], [], "s", color=colour, alpha=0.6, label=outcome)
    ax.set_xticks(range(len(runs)), [r.replace("mobilenetv3_small_100", "mobilenet").replace("efficientnet_b0", "effnet") for r in runs],
                  rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("inside-brain ratio")
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right")
    ax.set_title("Inside-brain ratio, correct vs wrong predictions (test sets)")
    return _save(fig, out_dir, "ratio_correct_vs_wrong.png")
