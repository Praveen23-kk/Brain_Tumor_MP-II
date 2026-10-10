"""Week 5: one training pipeline for every model and every split.

run() trains one model on one split CSV and writes outputs/runs/<split_name>/<model>/:
  best.pt      weights of the epoch with the best validation macro-F1
  history.csv  one row per epoch (loss, accuracy, macro-F1, learning rate, time)
  config.json  everything needed to reload and report the run; "finished": true marks a completed run
The TEST part of the split is never used here (it is kept for the final evaluation).
Used by notebooks/04_training.ipynb.
"""
import json
import math
import os
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Subset

from src.data import (CLASSES, IMAGENET_MEAN, IMAGENET_STD, SEED, MRIDataset, _save, eval_transforms,
                      set_seed, train_transforms)
# src.models sets the weight-cache folder (HF_HOME) before timm is imported, so timm is NOT imported here.
from src.models import (FREEZE_EPOCHS, MODELS, TIMM_VERSION, build_model, count_params,
                        set_backbone_trainable)

LABEL_SMOOTHING = 0.1     # targets become 0.925 / 0.025 instead of 1 / 0: the model is less over-confident
WEIGHT_DECAY = 0.05       # AdamW weight decay (keeps weights small, less over-fitting)
LR_LIGHT = 1e-3           # learning rate for the light models
LR_HEAVY = 3e-4           # smaller learning rate for the big baselines (they over-fit / diverge more easily)
NUM_WORKERS = 2           # DataLoader worker processes (chosen in Week 2)


def set_all_seeds(seed=SEED):
    """Seed Python, NumPy and PyTorch (CPU + GPU) so runs are repeatable."""
    set_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # benchmark=True lets cuDNN pick the fastest convolution algorithm for our fixed image size.
    # (It makes runs very slightly non-deterministic on the GPU; the speed gain is worth it.)
    torch.backends.cudnn.benchmark = True


# ------------------------------------------------------------------------------- data
def _pick(n_total, n, seed=SEED):
    """n fixed random row numbers out of n_total (the same ones every run)."""
    return sorted(random.Random(seed).sample(range(n_total), min(n, n_total)))


def make_loaders(split_csv, img_size, batch, subset=None, seed=SEED):
    """Train and validation DataLoaders for one split CSV (cached 224 images).
    subset=N uses only N random train and N random val images (for the smoke test).
    Returns (train_loader, val_loader, train label counts per class)."""
    train_set = MRIDataset(split_csv, "train", transform=train_transforms(img_size))
    val_set = MRIDataset(split_csv, "val", transform=eval_transforms(img_size))
    train_rows = list(range(len(train_set)))
    if subset is not None:
        train_rows = _pick(len(train_set), subset, seed)
        train_set = Subset(train_set, train_rows)
        val_set = Subset(val_set, _pick(len(val_set), subset, seed))
    labels = (train_set.dataset if subset is not None else train_set).table["label"].iloc[train_rows]
    counts = [int((labels == c).sum()) for c in CLASSES]

    g = torch.Generator()
    g.manual_seed(seed)                         # same shuffling order every run
    common = dict(num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available(), persistent_workers=True)
    # drop_last: skip the smaller final batch. Each new batch shape costs cuDNN benchmark ~30-90 s of
    # autotuning on this GPU (measured in Week 5); shuffling means different images are skipped each epoch.
    drop = len(train_set) > batch                # but never drop the only batch (tiny smoke-test subsets)
    train_loader = DataLoader(train_set, batch_size=batch, shuffle=True, generator=g, drop_last=drop, **common)
    val_loader = DataLoader(val_set, batch_size=batch, shuffle=False, **common)
    return train_loader, val_loader, counts


def class_weights(counts):
    """Inverse-frequency class weights: N / (n_classes * count_c).
    A class with half the average number of images gets weight 2, so every class matters equally in the loss."""
    counts = np.maximum(np.array(counts, dtype=float), 1)      # max(.,1) avoids division by zero
    return (counts.sum() / (len(counts) * counts)).tolist()


# ---------------------------------------------------------------------- LR schedule
def warmup_cosine(steps_per_epoch, epochs):
    """Learning-rate factor per optimiser step: linear warm-up during epoch 1, then cosine decay to 0.
    Warm-up avoids big, random updates while the new classifier is still random."""
    warm = steps_per_epoch
    total = steps_per_epoch * epochs

    def factor(step):
        if step < warm:
            return (step + 1) / warm
        progress = (step - warm) / max(1, total - warm)            # 0 -> 1 over the remaining epochs
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))
    return factor


# --------------------------------------------------------------------- train / eval
class AmpState:
    """Remembers whether mixed precision (float16) is still on.
    On this GTX 1650 Ti, float16 sometimes gives NaN (Week 4). Then we switch to float32 for the rest of the run."""

    def __init__(self, enabled):
        self.enabled = enabled
        self.disabled_at_epoch = None
        self.scaler = torch.amp.GradScaler("cuda", enabled=enabled)

    def disable(self, epoch):
        print(f"  NaN/Inf with float16 at epoch {epoch} -> switching to float32 for the rest of the run")
        self.enabled = False
        self.disabled_at_epoch = epoch
        self.scaler = torch.amp.GradScaler("cuda", enabled=False)


def _forward(model, images, amp_on):
    """Forward pass, in float16 (autocast) if amp_on, else in float32."""
    with torch.autocast("cuda", dtype=torch.float16, enabled=amp_on):
        return model(images)


def train_one_epoch(model, loader, loss_fn, optimizer, scheduler, amp, accum, epoch, device):
    """One pass over the training data. Gradients are accumulated over `accum` batches before each update,
    so a halved batch with accum=2 behaves like the original batch. Returns (mean loss, accuracy)."""
    model.train()
    total_loss, correct, seen = 0.0, 0, 0
    optimizer.zero_grad(set_to_none=True)
    n_batches = len(loader)
    for i, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        labels = labels.to(device, non_blocking=True)
        logits = _forward(model, images, amp.enabled)
        loss = loss_fn(logits.float(), labels)
        if not torch.isfinite(loss):
            if not amp.enabled:
                raise RuntimeError(f"loss is {loss.item()} even in float32 (epoch {epoch}, batch {i})")
            amp.disable(epoch)
            optimizer.zero_grad(set_to_none=True)            # drop half-accumulated float16 gradients
            logits = _forward(model, images, False)            # redo this batch in float32
            loss = loss_fn(logits.float(), labels)
        amp.scaler.scale(loss / accum).backward()            # /accum: average over the accumulated batches
        if (i + 1) % accum == 0 or i + 1 == n_batches:       # time to update the weights
            amp.scaler.step(optimizer)
            amp.scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()                                  # learning rate changes after every update
        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        seen += labels.size(0)
    return total_loss / seen, correct / seen


@torch.no_grad()
def evaluate(model, loader, loss_fn, amp, epoch, device):
    """Loss, accuracy and macro-F1 on a loader (no augmentation, no gradient).
    Macro-F1 = average of the 4 per-class F1 scores, so the small notumor class counts as much as the others."""
    model.eval()
    total_loss, preds, targets = 0.0, [], []
    for images, labels in loader:
        images = images.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        labels = labels.to(device, non_blocking=True)
        logits = _forward(model, images, amp.enabled)
        if not torch.isfinite(logits).all():
            if amp.enabled:
                amp.disable(epoch)
            logits = _forward(model, images, False)
        total_loss += loss_fn(logits.float(), labels).item() * labels.size(0)
        preds.append(logits.argmax(1).cpu())
        targets.append(labels.cpu())
    preds, targets = torch.cat(preds).numpy(), torch.cat(targets).numpy()
    return (total_loss / len(targets), float((preds == targets).mean()),
            float(f1_score(targets, preds, average="macro", labels=list(range(len(CLASSES))), zero_division=0)))


# -------------------------------------------------------------------------------- run
def _atomic(path, write):
    """Write a file safely: write(tmp_path) first, then rename it to path in one step.
    If the laptop crashes mid-write, the old file stays intact instead of a half-written one."""
    tmp = Path(str(path) + ".tmp")
    write(tmp)
    os.replace(tmp, path)


def _is_finished(run_dir):
    cfg = Path(run_dir) / "config.json"
    return cfg.exists() and json.loads(cfg.read_text()).get("finished", False)


def _train(model_name, split_csv, run_dir, epochs, patience, subset, batch, accum, device):
    """One complete training attempt with a given batch / accumulation. Returns the config dict.
    Raises torch.cuda.OutOfMemoryError if the GPU is too small for this batch (run() then retries)."""
    spec = MODELS[model_name]
    set_all_seeds()
    train_loader, val_loader, counts = make_loaders(split_csv, spec["img_size"], batch, subset)
    weights = class_weights(counts)
    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor(weights, device=device),
                                        label_smoothing=LABEL_SMOOTHING)
    model = build_model(model_name, num_classes=len(CLASSES)).to(device)
    lr = LR_HEAVY if spec["heavy"] else LR_LIGHT
    # All parameters are given to AdamW; frozen ones get no gradient, so AdamW simply skips them.
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    steps_per_epoch = math.ceil(len(train_loader) / accum)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_cosine(steps_per_epoch, epochs))
    amp = AmpState(enabled=device == "cuda")
    if spec["heavy"]:
        set_backbone_trainable(model, False)

    history, best_f1, best_epoch, bad_epochs = [], -1.0, 0, 0
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        if spec["heavy"] and epoch == FREEZE_EPOCHS + 1:
            set_backbone_trainable(model, True)
            print("  backbone unfrozen")
        t0 = time.perf_counter()
        lr_now = optimizer.param_groups[0]["lr"]
        tr_loss, tr_acc = train_one_epoch(model, train_loader, loss_fn, optimizer, scheduler, amp, accum,
                                          epoch, device)
        va_loss, va_acc, va_f1 = evaluate(model, val_loader, loss_fn, amp, epoch, device)
        history.append({"epoch": epoch, "lr": lr_now, "train_loss": tr_loss, "train_acc": tr_acc,
                        "val_loss": va_loss, "val_acc": va_acc, "val_macro_f1": va_f1,
                        "seconds": round(time.perf_counter() - t0, 1), "batch": batch, "amp": amp.enabled})
        _atomic(run_dir / "history.csv", lambda f: pd.DataFrame(history).to_csv(f, index=False))  # every epoch
        mark = ""
        if va_f1 > best_f1:                                  # new best -> save the weights
            best_f1, best_epoch, bad_epochs = va_f1, epoch, 0
            best = {"model": model.state_dict(), "epoch": epoch, "val_macro_f1": va_f1}
            _atomic(run_dir / "best.pt", lambda f: torch.save(best, f))
            mark = "  *best*"
        else:
            bad_epochs += 1
        print(f"  epoch {epoch:2d}  train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
              f"val loss {va_loss:.4f} acc {va_acc:.4f} F1 {va_f1:.4f} | {history[-1]['seconds']}s{mark}")
        if bad_epochs >= patience:
            print(f"  early stop: no val macro-F1 gain for {patience} epochs")
            break

    config = {
        "model": model_name, "split_csv": str(split_csv), "subset": subset, "classes": CLASSES,
        "img_size": spec["img_size"], "normalize": {"mean": IMAGENET_MEAN, "std": IMAGENET_STD},
        "batch": batch, "accum_steps": accum, "effective_batch": batch * accum,
        "optimizer": "AdamW", "lr": lr, "weight_decay": WEIGHT_DECAY, "schedule": "1 warm-up epoch + cosine",
        "label_smoothing": LABEL_SMOOTHING, "train_counts": counts, "class_weights": weights,
        "freeze_epochs": FREEZE_EPOCHS if spec["heavy"] else 0,
        "max_epochs": epochs, "patience": patience, "epochs_run": len(history),
        "best_epoch": best_epoch, "best_val_macro_f1": best_f1,
        "amp_disabled_at_epoch": amp.disabled_at_epoch,
        "peak_gpu_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1) if device == "cuda" else None,
        "train_time_s": round(time.perf_counter() - start, 1), "params": count_params(model),
        "seed": SEED, "torch": torch.__version__, "timm": TIMM_VERSION, "finished": True,
    }
    del model, optimizer, train_loader, val_loader               # free GPU memory and worker processes
    return config


def run(model_name, split_name, split_csv, epochs=20, patience=5, subset=None, out_root="outputs/runs"):
    """Train model_name on split_csv; results in out_root/split_name/model_name/.
    Skips the run if it already finished. On CUDA out-of-memory: halve the batch, double the gradient
    accumulation (same effective batch) and start the run again."""
    run_dir = Path(out_root) / split_name / model_name
    if _is_finished(run_dir):
        print(f"skipped {split_name}/{model_name}: already finished")
        return json.loads((run_dir / "config.json").read_text())
    run_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch, accum = MODELS[model_name]["batch"], 1
    print(f"training {split_name}/{model_name} (batch {batch}, device {device})")
    while True:
        try:
            config = _train(model_name, split_csv, run_dir, epochs, patience, subset, batch, accum, device)
            break
        except torch.cuda.OutOfMemoryError:
            if batch == 1:
                raise
            batch, accum = batch // 2, accum * 2
            print(f"  CUDA out of memory -> restarting with batch {batch} x {accum} accumulation steps")
        torch.cuda.empty_cache()                             # outside 'except' so the failed tensors are freed
    _atomic(run_dir / "config.json", lambda f: f.write_text(json.dumps(config, indent=2)))
    torch.cuda.empty_cache()
    return config


# ------------------------------------------------------------------------------- plots
def plot_history(run_dir, out_dir=None):
    """Loss (left) and accuracy + macro-F1 (right) per epoch, train vs val; best epoch marked.
    Saved as curves.png in run_dir (and also in out_dir, if given)."""
    run_dir = Path(run_dir)
    h = pd.read_csv(run_dir / "history.csv")
    best = int(h.loc[h["val_macro_f1"].idxmax(), "epoch"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(h["epoch"], h["train_loss"], "o-", label="train")
    axes[0].plot(h["epoch"], h["val_loss"], "o-", label="val")
    axes[0].set_ylabel("Loss (label-smoothed, class-weighted)")
    axes[1].plot(h["epoch"], h["train_acc"], "o-", label="train accuracy")
    axes[1].plot(h["epoch"], h["val_acc"], "o-", label="val accuracy")
    axes[1].plot(h["epoch"], h["val_macro_f1"], "s--", label="val macro-F1")
    axes[1].set_ylabel("Score")
    for ax in axes:
        ax.axvline(best, color="grey", linestyle=":", linewidth=1)
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"{run_dir.parent.name} / {run_dir.name}  (best val macro-F1 at epoch {best})")
    fig.tight_layout()
    if out_dir is not None:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        fig.savefig(Path(out_dir) / f"curves_{run_dir.parent.name}_{run_dir.name}.png", dpi=150, bbox_inches="tight")
    return _save(fig, run_dir, "curves.png")
