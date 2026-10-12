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


def _save_last(path, model, optimizer, scheduler, amp, train_loader, epoch, history, best_f1, best_epoch,
               bad_epochs, batch, accum, used_s, peak_mb, resumed_at, stopped_by):
    """Week 7: everything needed to continue the run after `epoch` if the laptop stops (crash, power, Ctrl+C).
    stopped_by is set if an early-stop rule fired at this epoch (then a resume must not train further).
    Deleted by run() when the run finishes, so in the end only best.pt is kept."""
    state = {
        "epoch": epoch, "stopped_by": stopped_by,
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(), "scaler": amp.scaler.state_dict(),
        "amp_enabled": amp.enabled, "amp_disabled_at_epoch": amp.disabled_at_epoch,
        "history": history, "best_f1": best_f1, "best_epoch": best_epoch, "bad_epochs": bad_epochs,
        "batch": batch, "accum": accum, "used_s": used_s, "peak_mb": peak_mb, "resumed_at": resumed_at,
        # random-number states, so shuffling and dropout continue as if nothing had happened
        "generator": train_loader.generator.get_state(), "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(), "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    _atomic(path, lambda f: torch.save(state, f))


def _train(model_name, split_csv, run_dir, epochs, patience, subset, batch, accum, device, target_acc=None):
    """One complete training attempt with a given batch / accumulation. Returns the config dict.
    Raises torch.cuda.OutOfMemoryError if the GPU is too small for this batch (run() then retries).
    Week 7: if run_dir/last.pt exists (an interrupted run with the same batch), training continues after its epoch."""
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
    amp = AmpState(enabled=device == "cuda" and spec["amp"])     # float16 only if the model allows it
    if spec["heavy"]:
        set_backbone_trainable(model, False)

    history, best_f1, best_epoch, bad_epochs, stopped_by = [], -1.0, 0, 0, "max_epochs"
    first_epoch, used_s, peak_before, resumed_at = 1, 0.0, 0.0, []
    last_path = run_dir / "last.pt"
    if last_path.exists():
        # weights_only=False: the file also holds RNG states (not only tensors). Safe: we wrote it ourselves.
        ck = torch.load(last_path, map_location="cpu", weights_only=False)
        if ck["batch"] == batch and ck["accum"] == accum:
            model.load_state_dict(ck["model"])
            optimizer.load_state_dict(ck["optimizer"])
            scheduler.load_state_dict(ck["scheduler"])
            amp.enabled, amp.disabled_at_epoch = ck["amp_enabled"], ck["amp_disabled_at_epoch"]
            amp.scaler = torch.amp.GradScaler("cuda", enabled=amp.enabled)
            if amp.enabled:
                amp.scaler.load_state_dict(ck["scaler"])
            history, best_f1, best_epoch = ck["history"], ck["best_f1"], ck["best_epoch"]
            bad_epochs, used_s, peak_before = ck["bad_epochs"], ck["used_s"], ck["peak_mb"]
            # best.pt is the truth for the best epoch: if the laptop stopped between saving best.pt and
            # last.pt, best.pt is one epoch newer than last.pt.
            best_file = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
            best_f1, best_epoch = best_file["val_macro_f1"], best_file["epoch"]
            del best_file
            train_loader.generator.set_state(ck["generator"])
            random.setstate(ck["python_rng"])
            np.random.set_state(ck["numpy_rng"])
            torch.set_rng_state(ck["torch_rng"])
            if ck["cuda_rng"] is not None:
                torch.cuda.set_rng_state_all(ck["cuda_rng"])
            first_epoch = ck["epoch"] + 1
            resumed_at = ck["resumed_at"] + [first_epoch]
            if ck["stopped_by"] is not None:          # it had already stopped early: do not train further
                stopped_by, first_epoch = ck["stopped_by"], epochs + 1
            # Not bit-identical to an uninterrupted run: the DataLoader workers start again with new
            # augmentation seeds. Everything else (weights, optimizer, LR, shuffling) continues exactly.
            print(f"  resuming from epoch {first_epoch} (last.pt found)")
            if spec["heavy"] and first_epoch > FREEZE_EPOCHS:
                set_backbone_trainable(model, True)
                print("  backbone unfrozen")
        del ck

    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for epoch in range(first_epoch, epochs + 1):
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
                        "seconds": round(time.perf_counter() - t0, 1), "batch": batch, "accum": accum,
                        "amp": amp.enabled})
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
              f"val loss {va_loss:.4f} acc {va_acc:.4f} F1 {va_f1:.4f} | {history[-1]['seconds']}s{mark}",
              flush=True)
        stop = None
        if bad_epochs >= patience:
            print(f"  early stop: no val macro-F1 gain for {patience} epochs")
            stop = "patience"
        elif target_acc is not None and va_acc >= target_acc:
            # Target reached: stop to save time. best.pt still holds the best-macro-F1 epoch so far.
            print(f"  target reached: val acc {va_acc:.4f} >= {target_acc}")
            stop = "target_acc"
        peak_mb = max(peak_before, torch.cuda.max_memory_allocated() / 2**20 if device == "cuda" else 0.0)
        _save_last(last_path, model, optimizer, scheduler, amp, train_loader, epoch, history, best_f1,
                   best_epoch, bad_epochs, batch, accum, used_s + time.perf_counter() - start, peak_mb, resumed_at,
                   stop)
        if stop is not None:
            stopped_by = stop
            break

    config = {
        "model": model_name, "split_csv": str(split_csv), "subset": subset, "classes": CLASSES,
        "img_size": spec["img_size"], "normalize": {"mean": IMAGENET_MEAN, "std": IMAGENET_STD},
        "batch": batch, "accum_steps": accum, "effective_batch": batch * accum,
        "optimizer": "AdamW", "lr": lr, "weight_decay": WEIGHT_DECAY, "schedule": "1 warm-up epoch + cosine",
        "label_smoothing": LABEL_SMOOTHING, "train_counts": counts, "class_weights": weights,
        "freeze_epochs": FREEZE_EPOCHS if spec["heavy"] else 0,
        "max_epochs": epochs, "patience": patience, "target_acc": target_acc, "stopped_by": stopped_by,
        "epochs_run": len(history),
        "best_epoch": best_epoch, "best_val_macro_f1": best_f1,
        "amp_at_start": device == "cuda" and spec["amp"], "amp_disabled_at_epoch": amp.disabled_at_epoch,
        "peak_gpu_mb": (round(max(peak_before, torch.cuda.max_memory_allocated() / 2**20), 1)
                        if device == "cuda" else None),
        "train_time_s": round(used_s + time.perf_counter() - start, 1), "params": count_params(model),
        "resumed_at_epochs": resumed_at,
        "seed": SEED, "torch": torch.__version__, "timm": TIMM_VERSION, "finished": True,
    }
    del model, optimizer, train_loader, val_loader               # free GPU memory and worker processes
    return config


def run(model_name, split_name, split_csv, epochs=20, patience=5, subset=None, out_root="outputs/runs",
        target_acc=None):
    """Train model_name on split_csv; results in out_root/split_name/model_name/.
    target_acc (e.g. 0.92): also stop as soon as val accuracy reaches it (saves time; None = off).
    Skips the run if it already finished. On CUDA out-of-memory: halve the batch, double the gradient
    accumulation (same effective batch) and start the run again."""
    run_dir = Path(out_root) / split_name / model_name
    if _is_finished(run_dir):
        print(f"skipped {split_name}/{model_name}: already finished")
        return json.loads((run_dir / "config.json").read_text())
    run_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch, accum = MODELS[model_name]["batch"], 1
    last_path = run_dir / "last.pt"
    if last_path.exists() and (run_dir / "history.csv").exists():
        # Interrupted run: continue with the batch it was using (it may have been halved after an OOM).
        last_row = pd.read_csv(run_dir / "history.csv").iloc[-1]
        if "accum" in last_row:
            batch, accum = int(last_row["batch"]), int(last_row["accum"])
    print(f"training {split_name}/{model_name} (batch {batch} x {accum}, device {device})", flush=True)
    while True:
        try:
            config = _train(model_name, split_csv, run_dir, epochs, patience, subset, batch, accum, device,
                            target_acc)
            break
        except torch.cuda.OutOfMemoryError:
            if batch == 1:
                raise
            batch, accum = batch // 2, accum * 2
            print(f"  CUDA out of memory -> restarting with batch {batch} x {accum} accumulation steps")
            last_path.unlink(missing_ok=True)                # its batch no longer matches: start from epoch 1
        torch.cuda.empty_cache()                             # outside 'except' so the failed tensors are freed
    _atomic(run_dir / "config.json", lambda f: f.write_text(json.dumps(config, indent=2)))
    last_path.unlink(missing_ok=True)                        # finished: keep only best.pt (CLAUDE.md rule)
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


# ------------------------------------------------------------------- Week 6: run reports
def run_summary(run_dir):
    """One-row summary of a finished run: validation scores AT THE BEST EPOCH (from history.csv)
    plus epochs, time and peak GPU memory (from config.json)."""
    run_dir = Path(run_dir)
    h = pd.read_csv(run_dir / "history.csv")
    cfg = json.loads((run_dir / "config.json").read_text())
    best = h.loc[h["epoch"] == cfg["best_epoch"]].iloc[0]
    return {"split": run_dir.parent.name, "model": run_dir.name,
            "val_acc": round(float(best["val_acc"]), 4), "val_macro_f1": round(float(best["val_macro_f1"]), 4),
            "epochs_run": cfg["epochs_run"], "best_epoch": cfg["best_epoch"],
            "train_time_min": round(cfg["train_time_s"] / 60, 1), "peak_gpu_mb": cfg["peak_gpu_mb"]}


def health_check(run_dir, max_gap=0.10, max_loss_rise=0.25):
    """Warnings for a finished run (empty list = healthy).
    NaN:          float16 had to be switched off, or the history contains NaN/Inf.
    Overfitting:  at the best epoch train accuracy is more than max_gap above val accuracy, OR
                  the last val loss is more than max_loss_rise (25%) above its minimum while the
                  train loss kept falling (the model memorises training images, val gets worse)."""
    run_dir = Path(run_dir)
    h = pd.read_csv(run_dir / "history.csv")
    cfg = json.loads((run_dir / "config.json").read_text())
    warnings = []
    if cfg.get("amp_disabled_at_epoch") is not None:
        warnings.append(f"NaN/Inf with float16 at epoch {cfg['amp_disabled_at_epoch']} (switched to float32)")
    numbers = h[["train_loss", "val_loss", "val_acc", "val_macro_f1"]].to_numpy(dtype=float)
    if not np.isfinite(numbers).all():
        warnings.append("history contains NaN/Inf values")
    best = h.loc[h["epoch"] == cfg["best_epoch"]].iloc[0]
    gap = best["train_acc"] - best["val_acc"]
    if gap > max_gap:
        warnings.append(f"overfitting: train acc - val acc = {gap:.3f} at the best epoch")
    i_min = int(h["val_loss"].idxmin())
    rise = h["val_loss"].iloc[-1] / h["val_loss"].iloc[i_min] - 1
    if rise > max_loss_rise and h["train_loss"].iloc[-1] < h["train_loss"].iloc[i_min]:
        warnings.append(f"overfitting: final val loss {100 * rise:.0f}% above its minimum (epoch "
                        f"{int(h['epoch'].iloc[i_min])}) while train loss kept falling")
    return warnings


def plot_all_histories(run_dirs, out_path):
    """All runs in ONE figure: one column per run.
    Top row: train and val loss. Bottom row: train accuracy, val accuracy, val macro-F1.
    Rows share their y-axis so runs can be compared; the dotted line marks the best epoch."""
    run_dirs = [Path(d) for d in run_dirs]
    n = len(run_dirs)
    fig, axes = plt.subplots(2, n, figsize=(3.2 * n, 6), sharey="row", squeeze=False)
    for j, run_dir in enumerate(run_dirs):
        h = pd.read_csv(run_dir / "history.csv")
        best = json.loads((run_dir / "config.json").read_text())["best_epoch"]
        top, bottom = axes[0, j], axes[1, j]
        top.plot(h["epoch"], h["train_loss"], "-", label="train loss")
        top.plot(h["epoch"], h["val_loss"], "-", label="val loss")
        bottom.plot(h["epoch"], h["train_acc"], "-", label="train acc")
        bottom.plot(h["epoch"], h["val_acc"], "-", label="val acc")
        bottom.plot(h["epoch"], h["val_macro_f1"], "--", label="val macro-F1")
        top.set_title(f"{run_dir.parent.name}\n{run_dir.name}", fontsize=9)
        top.set_ylim(0, 2.5)            # epoch-1 val loss (BatchNorm lag) can be ~7; clip so the rest is readable
        bottom.set_ylim(0, 1.02)
        bottom.set_xlabel("Epoch")
        for ax in (top, bottom):
            ax.axvline(best, color="grey", linestyle=":", linewidth=1)
            ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].set_ylabel("Loss (y clipped at 2.5)")
    axes[1, 0].set_ylabel("Score")
    axes[0, 0].legend(fontsize=8)
    axes[1, 0].legend(fontsize=8, loc="lower right")
    fig.suptitle("Training curves of all runs (dotted line = best val macro-F1 epoch)")
    fig.tight_layout()
    out_path = Path(out_path)
    return _save(fig, out_path.parent, out_path.name)


# ---------------------------------------------------------------- Week 7: heavy models
def probe(model_name, batch, steps=10, amp=None):
    """Measure GPU memory and speed of one model BEFORE a long run (random images, not the dataset).
    The backbone is UNFROZEN (worst case: gradients + AdamW state for every weight), same optimizer and
    float16 setting as training. Returns a dict with peak GPU MB and seconds per batch for: full training,
    frozen-backbone training (epochs 1-2 of heavy models) and evaluation; or {"oom": True} if it does not fit.
    amp=None uses the model's own float16 setting; True/False overrides it (to compare float16 with float32)."""
    spec = dict(MODELS[model_name])
    if amp is not None:
        spec["amp"] = amp
    set_all_seeds()
    device = "cuda"
    result = {"model": model_name, "batch": batch, "img_size": spec["img_size"], "amp": spec["amp"]}
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        model = build_model(model_name, num_classes=len(CLASSES)).to(device)
        lr = LR_HEAVY if spec["heavy"] else LR_LIGHT
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
        scaler = torch.amp.GradScaler("cuda", enabled=spec["amp"])
        loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
        size = spec["img_size"]
        images = torch.randn(batch, 3, size, size, device=device).to(memory_format=torch.channels_last)
        labels = torch.randint(0, len(CLASSES), (batch,), device=device)

        def train_step():
            loss = loss_fn(_forward(model, images, spec["amp"]).float(), labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        def timed(step):
            """Seconds per call of step(): 2 warm-up calls (cuDNN autotuning), then the mean of `steps` calls."""
            for _ in range(2):
                step()
            torch.cuda.synchronize()                    # the GPU works asynchronously: wait before timing
            t0 = time.perf_counter()
            for _ in range(steps):
                step()
            torch.cuda.synchronize()
            return (time.perf_counter() - t0) / steps

        model.train()
        result["train_s_per_batch"] = round(timed(train_step), 4)          # full fine-tuning (worst memory)
        result["peak_gpu_mb"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
        result["reserved_gpu_mb"] = round(torch.cuda.max_memory_reserved() / 2**20, 1)
        set_backbone_trainable(model, False)
        result["frozen_s_per_batch"] = round(timed(train_step), 4)         # epochs 1-2 of heavy models
        model.eval()

        def eval_step():
            with torch.no_grad():
                _forward(model, images, spec["amp"])
        result["eval_s_per_batch"] = round(timed(eval_step), 4)
        result["oom"] = False
    except torch.cuda.OutOfMemoryError:
        result["oom"] = True
    model = optimizer = images = None                    # free everything before the next probe
    torch.cuda.empty_cache()
    return result


def estimate_minutes(p, n_train, n_val, epochs, frozen_epochs=FREEZE_EPOCHS, accum=1):
    """Estimated training minutes for `epochs` epochs from a probe() result p.
    Per epoch: train batches x seconds per batch + val batches x seconds per eval batch.
    The first frozen_epochs epochs use the faster frozen-backbone time. Data loading is assumed to keep up
    (it did for the GPU-bound efficientnet in Week 6); the one-off first-epoch overhead is NOT included."""
    train_batches = n_train // p["batch"]                 # drop_last=True in make_loaders
    val_batches = math.ceil(n_val / p["batch"])
    val_s = val_batches * p["eval_s_per_batch"]
    frozen = min(epochs, frozen_epochs)
    seconds = (frozen * (train_batches * p["frozen_s_per_batch"] + val_s)
               + (epochs - frozen) * (train_batches * p["train_s_per_batch"] + val_s))
    return seconds / 60


def runs_table(runs_root="outputs/runs"):
    """One row per finished full run (smoke tests and subsets left out), read from config.json + history.csv."""
    rows = []
    for cfg_path in sorted(Path(runs_root).glob("*/*/config.json")):
        cfg = json.loads(cfg_path.read_text())
        if cfg.get("subset") is not None or not cfg.get("finished"):
            continue
        s = run_summary(cfg_path.parent)
        if not cfg.get("amp_at_start", True):
            fp16 = "no (float32)"
        elif cfg.get("amp_disabled_at_epoch") is not None:
            fp16 = f"NaN at epoch {cfg['amp_disabled_at_epoch']} -> float32"
        else:
            fp16 = "yes"
        rows.append({"split": s["split"], "model": s["model"], "params_M": round(cfg["params"] / 1e6, 2),
                     "val_acc": s["val_acc"], "val_macro_f1": s["val_macro_f1"],
                     "epochs (best)": f"{s['epochs_run']} ({s['best_epoch']})", "max_epochs": cfg["max_epochs"],
                     # Week 5 configs have no "stopped_by"; they had no target, so it was patience or max epochs
                     "stopped_by": cfg.get("stopped_by") or ("max_epochs" if cfg["epochs_run"] == cfg["max_epochs"]
                                                             else "patience"),
                     "target_acc": cfg.get("target_acc") or "-",
                     "batch x accum": f"{cfg['batch']} x {cfg['accum_steps']}",
                     "time_min": s["train_time_min"], "peak_gpu_mb": s["peak_gpu_mb"], "float16": fp16,
                     "resumed": len(cfg.get("resumed_at_epochs", [])) > 0})
    return pd.DataFrame(rows)


def to_markdown(df):
    """A DataFrame as a Markdown table (for PROGRESS.md; avoids needing the 'tabulate' package)."""
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


if __name__ == "__main__":
    # Week 7: train one run from the command line (so long runs can go in the background), e.g.
    #   python -u -m src.train vgg16 v1_clean --epochs 12 --target-acc 0.92
    # Run it again after a crash: it continues from last.pt.
    import argparse
    parser = argparse.ArgumentParser(description="Train one model on one split.")
    parser.add_argument("model", choices=list(MODELS))
    parser.add_argument("split", help="e.g. v1_clean -> outputs/splits/v1_split_clean.csv")
    parser.add_argument("--split-csv", default=None, help="explicit split CSV (needed for 'smoke')")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--target-acc", type=float, default=None)
    parser.add_argument("--subset", type=int, default=None)
    args = parser.parse_args()
    version, split_type = (args.split.split("_", 1) + [""])[:2]
    csv = args.split_csv or f"outputs/splits/{version}_split_{split_type}.csv"
    cfg = run(args.model, args.split, csv, epochs=args.epochs, patience=args.patience, subset=args.subset,
              target_acc=args.target_acc)
    print(json.dumps({k: cfg[k] for k in ("best_epoch", "best_val_macro_f1", "stopped_by", "epochs_run",
                                          "train_time_s", "peak_gpu_mb")}))
