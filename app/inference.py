"""Inference helpers for the Streamlit demo (no Streamlit code here, so they can be tested on their own).

Everything is reused from src/ so the app behaves exactly like the evaluation in Weeks 8-9:
  preprocessing  : src.data.crop_brain + pad_resize (same steps as the training cache) + eval_transforms
  model          : src.evaluate.load_run (rebuilds the timm model from config.json + best.pt)
  confidence     : src.evaluate.softmax with the temperature stored in config.json
  referral rule  : src.evaluate.REFER_BELOW (0.70)
  Grad-CAM       : src.gradcam.make_cam / compute_cams, inside-brain ratio from inside_mask / inside_ratio
  MC dropout     : same seeded dropout on the classifier head as src.evaluate.mc_dropout
Everything runs on the CPU, so the app works on any computer.
"""
import json
import sys
import threading
from pathlib import Path

import cv2
import numpy as np
import torch
from matplotlib import colormaps
from PIL import Image

# `streamlit run app/app.py` only puts app/ on the import path, so add the project root to find src/
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data import SEED, eval_transforms, crop_brain, pad_resize, set_seed   # noqa: E402
from src.evaluate import MC_P, MC_PASSES, REFER_BELOW, load_run, softmax       # noqa: E402
from src.gradcam import compute_cams, inside_mask, inside_ratio, make_cam      # noqa: E402

MODEL_DIR = Path(__file__).resolve().parent / "model"      # best.pt, config.json, metrics.json, test_logits.npz (made by make_samples.py)
HEAT_ALPHA = 0.45                                          # how strongly the heat map covers the MRI (same as the Week-9 gallery)
FOCUS_LEVEL = 0.5                                          # focus region = heat above 50% of its maximum
_cam_lock = threading.Lock()                               # Grad-CAM hooks live on the shared model: one request at a time


def load_bundle(model_dir=MODEL_DIR):
    """Load the model once. Returns a dict with the model, its config, the eval transform, metrics.json (or None)
    and the clean-test-set logits/labels (or None) that the model-performance plots need."""
    set_seed(SEED)
    torch.set_num_threads(2)                               # a small, polite CPU footprint
    model, cfg = load_run(model_dir, device="cpu")          # model is in eval mode, float32
    metrics_file = Path(model_dir) / "metrics.json"
    metrics = json.loads(metrics_file.read_text()) if metrics_file.exists() else None
    test_file = Path(model_dir) / "test_logits.npz"
    test = dict(np.load(test_file)) if test_file.exists() else None
    return {"model": model, "cfg": cfg, "transform": eval_transforms(cfg["img_size"]), "metrics": metrics, "test": test}


# ------------------------------------------------------------------------------------ image details
def preprocess_steps(image, size):
    """PIL image (any mode) -> dict of the preprocessing steps:
    grey (original greyscale), box (x, y, w, h of the crop), cropped, final (size x size uint8) and fallback.
    These are the same three steps src.data.preprocess() applied to every training image."""
    grey = np.asarray(image.convert("L"))
    cropped, box, fallback = crop_brain(grey)
    return {"grey": grey, "box": box, "cropped": cropped, "final": pad_resize(cropped, size), "fallback": fallback}


def preprocess(image, size):
    """PIL image -> (2-D uint8 array size x size, True if no head region was found and the full image was used)."""
    steps = preprocess_steps(image, size)
    return steps["final"], steps["fallback"]


def image_facts(image, n_bytes):
    """Simple facts about the file as it was uploaded."""
    return {"Format": image.format or "unknown", "Colour mode": image.mode,
            "Width x height (px)": f"{image.width} x {image.height}", "File size (KB)": f"{n_bytes / 1024:.1f}"}


def intensity_stats(grey, mask):
    """Brightness statistics of the 224 x 224 picture: whole image and inside the brain mask, plus the brain's share of the picture."""
    inside = grey[mask] if mask.any() else grey.ravel()    # an empty mask would give NaN statistics
    return {"brain_share": float(mask.mean()), "mean": float(inside.mean()), "std": float(inside.std()),
            "min": int(inside.min()), "max": int(inside.max())}


# -------------------------------------------------------------------------------------- prediction
def _to_tensor(bundle, grey):
    """Greyscale array -> normalised (1, 3, H, W) tensor, the same way MRIDataset feeds the network."""
    rgb = Image.fromarray(grey).convert("RGB")             # copy the grey channel into 3 identical channels
    x = bundle["transform"](rgb).unsqueeze(0)
    return x.to(memory_format=torch.channels_last)


def predict(bundle, grey):
    """Class probabilities for one preprocessed image, before and after temperature scaling.
    Returns a dict: probs (calibrated, 4), raw_probs (T = 1, 4), pred (class index), conf (highest calibrated probability),
    refer (True if conf < 0.70), margin (top-1 minus top-2 probability), entropy (of the calibrated probabilities)."""
    with torch.no_grad():
        logits = bundle["model"](_to_tensor(bundle, grey)).float().numpy()
    probs = softmax(logits, bundle["cfg"]["temperature"])[0]       # temperature scaling fitted on the validation set
    top2 = np.sort(probs)[::-1][:2]
    conf = float(probs.max())
    return {"probs": probs, "raw_probs": softmax(logits)[0], "pred": int(probs.argmax()), "conf": conf,
            "refer": conf < REFER_BELOW, "margin": float(top2[0] - top2[1]),
            "entropy": float(-(probs * np.log(probs + 1e-12)).sum())}


def mc_passes(bundle, grey):
    """MC dropout on the classifier head: dropout (p = 0.3) on the feature vector, 30 passes through the last Linear layer.
    Same seed and mask recipe as src.evaluate.mc_dropout, but every pass is kept so the spread can be plotted.
    Returns an array (30, 4) of calibrated probabilities."""
    model = bundle["model"]
    temperature = bundle["cfg"]["temperature"]
    classifier = model.get_classifier()
    with torch.no_grad():
        feats = model.forward_head(model.forward_features(_to_tensor(bundle, grey)), pre_logits=True)   # everything except the last Linear
        gen = torch.Generator(device="cpu").manual_seed(SEED)                                          # fixed seed -> same masks every time
        passes = []
        for _ in range(MC_PASSES):
            keep = (torch.rand(feats.shape, generator=gen) > MC_P).float() / (1 - MC_P)                 # inverted dropout mask
            passes.append(torch.softmax(classifier(feats * keep).float() / temperature, dim=1)[0])
    return torch.stack(passes).numpy()


def test_confidences(bundle):
    """Calibrated confidence and correctness of every clean-test image (from test_logits.npz), or None if the file is missing."""
    if bundle["test"] is None:
        return None
    probs = softmax(bundle["test"]["logits"], bundle["cfg"]["temperature"])
    return {"conf": probs.max(1), "correct": probs.argmax(1) == bundle["test"]["labels"]}


# ---------------------------------------------------------------------------------------- Grad-CAM
def gradcam_maps(bundle, grey, class_idx):
    """Grad-CAM of one class on one preprocessed image.
    Returns a dict: heat (224 x 224, 0..1), mask (boolean brain mask), ratio (inside-brain ratio), empty (heat all zero),
    mask_frac (share of the picture the mask covers = the ratio a uniform heat map would get)."""
    x = _to_tensor(bundle, grey)                           # Grad-CAM needs gradients, so no torch.no_grad() here
    with _cam_lock, make_cam(bundle["model"], bundle["cfg"]) as cam:
        heat = compute_cams(cam, x, [class_idx])[0][0]     # (224, 224) values in 0..1
    mask = inside_mask(grey)
    ratio, empty = inside_ratio(heat, mask)
    return {"heat": heat, "mask": mask, "ratio": ratio, "empty": empty, "mask_frac": float(mask.mean())}


def focus_circle(heat, level=FOCUS_LEVEL):
    """Circle around the strongest heat blob: keep pixels above level * max, take the largest connected blob and
    return its smallest enclosing circle (x, y, radius) in pixels, or None if the heat map is empty.
    This marks where the MODEL looked (a 7 x 7 map stretched to 224 x 224), it is not a tumour outline."""
    if heat.max() <= 0:
        return None
    hot = (heat >= level * heat.max()).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(hot, connectivity=8)
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))      # label 0 is the background
    ys, xs = np.nonzero(labels == biggest)
    (x, y), r = cv2.minEnclosingCircle(np.stack([xs, ys], axis=1).astype(np.float32))
    return float(x), float(y), float(r)


def make_overlay(grey, heat, mask, alpha=HEAT_ALPHA, circle=None):
    """RGB uint8 picture: MRI + inferno heat map + cyan brain-mask outline (as in the Week-9 gallery), optional white focus circle."""
    colours = (colormaps["inferno"](heat)[..., :3] * 255).astype(np.uint8)       # heat -> RGB
    base = np.stack([grey] * 3, axis=-1)
    overlay = ((1 - alpha) * base + alpha * colours).astype(np.uint8)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 255), 1)
    if circle is not None:
        cv2.circle(overlay, (round(circle[0]), round(circle[1])), max(round(circle[2]), 3), (255, 255, 255), 2)
    return overlay
