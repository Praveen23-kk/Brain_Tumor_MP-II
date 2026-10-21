"""Inference helpers for the Streamlit demo (no Streamlit code here, so they can be tested on their own).

Everything is reused from src/ so the app behaves exactly like the evaluation in Weeks 8-9:
  preprocessing  : src.data.crop_brain + pad_resize (same steps as the training cache) + eval_transforms
  model          : src.evaluate.load_run (rebuilds the timm model from config.json + best.pt)
  confidence     : src.evaluate.softmax with the temperature stored in config.json
  referral rule  : src.evaluate.REFER_BELOW (0.70)
  Grad-CAM       : src.gradcam.make_cam / compute_cams, inside-brain ratio from inside_mask / inside_ratio
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
from src.evaluate import REFER_BELOW, load_run, softmax                        # noqa: E402
from src.gradcam import compute_cams, inside_mask, inside_ratio, make_cam      # noqa: E402

MODEL_DIR = Path(__file__).resolve().parent / "model"      # best.pt + config.json (+ metrics.json) copied by make_samples.py
HEAT_ALPHA = 0.45                                          # how strongly the heat map covers the MRI (same as the Week-9 gallery)
_cam_lock = threading.Lock()                               # Grad-CAM hooks live on the shared model: one request at a time


def load_bundle(model_dir=MODEL_DIR):
    """Load the model once. Returns a dict with the model, its config, the eval transform and metrics.json (or None)."""
    set_seed(SEED)
    torch.set_num_threads(2)                               # a small, polite CPU footprint
    model, cfg = load_run(model_dir, device="cpu")          # model is in eval mode, float32
    metrics_file = Path(model_dir) / "metrics.json"
    metrics = json.loads(metrics_file.read_text()) if metrics_file.exists() else None
    return {"model": model, "cfg": cfg, "transform": eval_transforms(cfg["img_size"]), "metrics": metrics}


def preprocess(image, size):
    """PIL image (any mode) -> 2-D uint8 array size x size: greyscale, crop the black border, pad to a square, resize.
    These are the same three steps src.data.preprocess() applied to every training image."""
    grey = np.asarray(image.convert("L"))
    cropped, _, used_fallback = crop_brain(grey)
    return pad_resize(cropped, size), used_fallback


def _to_tensor(bundle, grey):
    """Greyscale array -> normalised (1, 3, H, W) tensor, the same way MRIDataset feeds the network."""
    rgb = Image.fromarray(grey).convert("RGB")             # copy the grey channel into 3 identical channels
    x = bundle["transform"](rgb).unsqueeze(0)
    return x.to(memory_format=torch.channels_last)


def predict(bundle, grey):
    """Calibrated class probabilities for one preprocessed image.
    Returns a dict: probs (4,), pred (class index), conf (highest probability), refer (True if conf < 0.70)."""
    with torch.no_grad():
        logits = bundle["model"](_to_tensor(bundle, grey)).float().numpy()
    probs = softmax(logits, bundle["cfg"]["temperature"])[0]       # temperature scaling fitted on the validation set
    conf = float(probs.max())
    return {"probs": probs, "pred": int(probs.argmax()), "conf": conf, "refer": conf < REFER_BELOW}


def gradcam_overlay(bundle, grey, class_idx):
    """Grad-CAM of one class on one preprocessed image.
    Returns (RGB uint8 overlay with the brain-mask outline in cyan, inside-brain ratio, True if the heat map is all zero)."""
    x = _to_tensor(bundle, grey)                           # Grad-CAM needs gradients, so no torch.no_grad() here
    with _cam_lock, make_cam(bundle["model"], bundle["cfg"]) as cam:
        heat = compute_cams(cam, x, [class_idx])[0][0]     # (224, 224) values in 0..1
    mask = inside_mask(grey)
    ratio, empty = inside_ratio(heat, mask)
    colours = (colormaps["inferno"](heat)[..., :3] * 255).astype(np.uint8)       # heat -> RGB
    base = np.stack([grey] * 3, axis=-1)
    overlay = ((1 - HEAT_ALPHA) * base + HEAT_ALPHA * colours).astype(np.uint8)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 255), 1)                    # cyan = brain mask, as in the Week-9 gallery
    return overlay, ratio, empty
