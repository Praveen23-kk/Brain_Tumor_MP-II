"""Week 8: test-set evaluation of every trained run.

evaluate_run() reloads best.pt of one run and measures, on the run's own test set:
  - accuracy, balanced accuracy, macro-F1, per-class sensitivity / precision, one-vs-rest AUC, confusion matrix
  - temperature scaling (T fitted on the VALIDATION logits only), ECE with 15 bins, reliability diagrams
  - MC dropout on the classifier head (30 passes), predictive entropy, accuracy-vs-coverage curve
  - "refer to radiologist" rate at confidence < 0.70 and accuracy on the scans that are NOT referred
  - model size (MB), parameters, CPU and GPU latency per image
and writes  metrics.json  into the run folder and  "temperature"  into its config.json.
evaluate_all() does this for every finished run and builds one results table (CSV).
Used by notebooks/06_evaluation.ipynb.
"""
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (balanced_accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support,
                             roc_auc_score)
from torch.utils.data import DataLoader

from src.data import CLASSES, SEED, MRIDataset, _save, eval_transforms
from src.models import build_model, count_params
from src.train import NUM_WORKERS, _atomic, set_all_seeds

N_BINS = 15            # reliability bins for ECE
MC_PASSES = 30         # MC dropout forward passes through the classifier head
MC_P = 0.3             # dropout probability used at inference (the models were trained WITHOUT dropout)
REFER_BELOW = 0.70     # scans with confidence below this are "referred to a radiologist"
EVAL_BATCH = 32        # small batch so vgg16 fits in 4 GB
TIMING_WARMUP = 20     # first images are not timed (GPU / CPU warm-up)
TIMING_RUNS = 100      # images timed one by one (batch size 1)
LABELS = list(range(len(CLASSES)))


def _num(x):
    """float for JSON; NaN becomes None (JSON has no NaN)."""
    x = float(x)
    return None if np.isnan(x) else x


# --------------------------------------------------------------------------- load + predict
def load_run(run_dir, device="cuda"):
    """Rebuild the model of a finished run from config.json + best.pt (weights of the best epoch)."""
    run_dir = Path(run_dir)
    cfg = json.loads((run_dir / "config.json").read_text())
    model = build_model(cfg["model"], num_classes=len(CLASSES), pretrained=False)   # no ImageNet download needed
    ckpt = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), cfg


def make_dataset(cfg, split):
    """The val or test part of the run's split CSV, with the fixed (non-random) transform."""
    split_csv = Path(str(cfg["split_csv"]).replace("\\", "/"))   # older configs were written with backslashes
    return MRIDataset(split_csv, split, transform=eval_transforms(cfg["img_size"]))


@torch.no_grad()
def extract(model, cfg, split, device="cuda"):
    """Run a split through the model once (float32, no augmentation, no shuffling).
    Returns logits (N, 4), labels (N,) and feats (N, D) = the vector that goes into the final Linear layer.
    The feature vector is kept because MC dropout only needs the head, not the whole network again."""
    loader = DataLoader(make_dataset(cfg, split), batch_size=EVAL_BATCH, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=True)
    classifier = model.get_classifier()          # the last Linear layer (4 outputs) in every architecture used here
    logits, labels, feats = [], [], []
    for i, (images, y) in enumerate(loader):
        images = images.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        f = model.forward_head(model.forward_features(images), pre_logits=True)   # everything except the last Linear
        z = classifier(f)
        if i == 0:                               # safety check: splitting the forward pass must not change the output
            if not torch.allclose(z, model(images), atol=1e-3, rtol=1e-3):
                raise ValueError(f"{cfg['model']}: head split does not reproduce model(images)")
        logits.append(z.float().cpu())
        feats.append(f.float().cpu())
        labels.append(y)
    return torch.cat(logits).numpy(), torch.cat(labels).numpy(), torch.cat(feats).numpy()


def extract_cached(run_dir, model, cfg, split, device="cuda", force=False):
    """extract() with a cache file <split>_outputs.npz in the run folder (ignored if best.pt is newer)."""
    run_dir = Path(run_dir)
    cache = run_dir / f"{split}_outputs.npz"
    if cache.exists() and not force and cache.stat().st_mtime > (run_dir / "best.pt").stat().st_mtime:
        d = np.load(cache)
        return d["logits"], d["labels"], d["feats"]
    logits, labels, feats = extract(model, cfg, split, device)
    # feats are only needed for MC dropout on the test set; skip them for val to keep the file small
    def write(p):
        with open(p, "wb") as fh:                # a file handle stops numpy from appending ".npz" to the temp name
            np.savez(fh, logits=logits, labels=labels, feats=feats if split == "test" else np.zeros((0,), np.float32))
    _atomic(cache, write)
    return logits, labels, feats


def softmax(logits, temperature=1.0):
    """Class probabilities from logits; temperature > 1 makes them softer, < 1 sharper."""
    z = logits / temperature
    z = z - z.max(axis=1, keepdims=True)         # subtract the max so exp() cannot overflow
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


# ------------------------------------------------------------------------------ classification
def classification_metrics(probs, labels):
    """Accuracy, balanced accuracy, macro-F1, per-class sensitivity / precision / AUC, confusion matrix.
    Sensitivity = recall of the class (share of its scans that are found). AUC is one-vs-rest."""
    preds = probs.argmax(1)
    prec, rec, f1, _ = precision_recall_fscore_support(labels, preds, labels=LABELS, zero_division=0)
    per_class = {}
    for k, name in enumerate(CLASSES):
        per_class[name] = {"sensitivity": float(rec[k]), "precision": float(prec[k]), "f1": float(f1[k]),
                           "auc": float(roc_auc_score(labels == k, probs[:, k]))}
    return {"accuracy": float((preds == labels).mean()),
            "balanced_accuracy": float(balanced_accuracy_score(labels, preds)),
            "macro_f1": float(f1_score(labels, preds, average="macro", labels=LABELS, zero_division=0)),
            "macro_auc": float(roc_auc_score(labels, probs, multi_class="ovr", average="macro", labels=LABELS)),
            "per_class": per_class,
            "confusion_matrix": confusion_matrix(labels, preds, labels=LABELS).tolist()}


# -------------------------------------------------------------------------------- calibration
def fit_temperature(val_logits, val_labels):
    """Temperature scaling: find one number T > 0 so that softmax(logits / T) has the lowest
    negative log-likelihood on the VALIDATION set. T does not change the predicted class, only the confidence."""
    z = torch.tensor(val_logits, dtype=torch.float64)
    y = torch.tensor(val_labels)
    log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)   # optimise log T so that T stays positive
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(z / log_t.exp(), y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.detach().exp().clamp(0.05, 20.0))     # clamp = guard against a runaway fit


def nll(probs, labels):
    """Negative log-likelihood: mean of -log(probability given to the true class)."""
    return float(-np.log(probs[np.arange(len(labels)), labels] + 1e-12).mean())


def brier(probs, labels):
    """Brier score: mean squared distance between the probability vector and the one-hot true label."""
    onehot = np.eye(len(CLASSES))[labels]
    return float(((probs - onehot) ** 2).sum(1).mean())


def ece(probs, labels, n_bins=N_BINS):
    """Expected Calibration Error: split scans into equal-width confidence bins, then average
    |accuracy - mean confidence| over the bins, weighted by how many scans each bin holds.
    Returns the ECE and the per-bin numbers (None for empty bins) for the reliability diagram."""
    conf = probs.max(1)
    correct = (probs.argmax(1) == labels).astype(float)
    bin_id = np.minimum((conf * n_bins).astype(int), n_bins - 1)    # conf = 1.0 goes into the last bin
    total, bins = 0.0, {"confidence": [], "accuracy": [], "count": []}
    for b in range(n_bins):
        in_bin = bin_id == b
        n = int(in_bin.sum())
        bins["count"].append(n)
        if n == 0:
            bins["confidence"].append(None)
            bins["accuracy"].append(None)
            continue
        bins["confidence"].append(float(conf[in_bin].mean()))
        bins["accuracy"].append(float(correct[in_bin].mean()))
        total += n / len(labels) * abs(correct[in_bin].mean() - conf[in_bin].mean())
    return float(total), bins


def calibration_block(val_logits, val_labels, test_logits, test_labels):
    """Fit T on validation, then compare test calibration before (T = 1) and after scaling."""
    t = fit_temperature(val_logits, val_labels)
    out = {"temperature": t,
           "val_nll_before": nll(softmax(val_logits), val_labels),
           "val_nll_after": nll(softmax(val_logits, t), val_labels)}
    for name, temp in (("before", 1.0), ("after", t)):
        p = softmax(test_logits, temp)
        e, bins = ece(p, test_labels)
        out[name] = {"ece": e, "nll": nll(p, test_labels), "brier": brier(p, test_labels),
                     "mean_confidence": float(p.max(1).mean()), "bins": bins}
    return out


# ----------------------------------------------------------------------- MC dropout + referral
@torch.no_grad()
def mc_dropout(model, feats, temperature=1.0, device="cuda", passes=MC_PASSES, p=MC_P):
    """Monte-Carlo dropout on the classifier head: switch dropout ON for the feature vector, push it through
    the last Linear layer `passes` times and average the softmax outputs. Each pass divides its logits by the
    fitted temperature first, so the averaged probabilities are on the same calibrated scale as the deterministic ones.
    Returns the mean probabilities (N, 4) and the predictive entropy (N,) = entropy of that mean
    (0 = all passes agree on one class, log 4 = 1.39 = completely unsure)."""
    classifier = model.get_classifier()
    f = torch.tensor(feats, device=device)
    gen = torch.Generator(device=device).manual_seed(SEED)           # fixed seed -> same masks every run
    probs = torch.zeros(len(f), len(CLASSES), device=device)
    for _ in range(passes):
        keep = (torch.rand(f.shape, generator=gen, device=device) > p).float() / (1 - p)   # inverted dropout mask
        probs += torch.softmax(classifier(f * keep).float() / temperature, dim=1)
    probs = (probs / passes).cpu().numpy().astype(np.float64)
    entropy = -(probs * np.log(probs + 1e-12)).sum(1)
    return probs, entropy


def coverage_curve(uncertainty, correct, coverages=tuple(np.arange(0.10, 1.0001, 0.05))):
    """Accuracy-vs-coverage: keep only the most certain x % of scans (lowest uncertainty) and measure accuracy there."""
    order = np.argsort(uncertainty, kind="stable")                   # most certain first
    cum_acc = np.cumsum(correct[order]) / np.arange(1, len(correct) + 1)
    ks = [max(1, int(round(c * len(correct)))) for c in coverages]    # number of scans kept at each coverage
    return {"coverage": [float(c) for c in coverages], "accuracy": [float(cum_acc[k - 1]) for k in ks]}


def referral_stats(probs, labels, threshold=REFER_BELOW):
    """'Refer to radiologist' if the top probability is below the threshold.
    Reports the referral rate and the accuracy on the scans that are kept (not referred)."""
    refer = probs.max(1) < threshold
    correct = probs.argmax(1) == labels
    return {"threshold": threshold, "rate": float(refer.mean()), "n_referred": int(refer.sum()),
            "n_kept": int((~refer).sum()),
            "acc_kept": _num(correct[~refer].mean()) if (~refer).any() else None,
            "acc_referred": _num(correct[refer].mean()) if refer.any() else None}


# ------------------------------------------------------------------------------------ efficiency
def _time_ms(model, images, device):
    """Milliseconds per image, batch size 1: forward pass only (the input is already on the device).
    The first TIMING_WARMUP images are run but not counted."""
    times = []
    with torch.no_grad():
        for i, x in enumerate(images):
            x = x.to(device).to(memory_format=torch.channels_last)
            if device == "cuda":
                torch.cuda.synchronize()                 # GPU runs asynchronously: wait before reading the clock
            t0 = time.perf_counter()
            model(x)
            if device == "cuda":
                torch.cuda.synchronize()
            if i >= TIMING_WARMUP:
                times.append((time.perf_counter() - t0) * 1000)
    return float(np.mean(times)), float(np.std(times))


def efficiency(run_dir, model, cfg):
    """Model size (MB of best.pt), parameters, GPU and CPU latency per image. Moves the model to the CPU at the end."""
    ds = make_dataset(cfg, "test")
    images = [ds[i][0].unsqueeze(0) for i in range(TIMING_WARMUP + TIMING_RUNS)]   # the first real test images
    gpu_ms, gpu_std = _time_ms(model, images, "cuda") if torch.cuda.is_available() else (None, None)
    model = model.cpu()
    cpu_ms, cpu_std = _time_ms(model, images, "cpu")
    return {"params": count_params(model), "size_mb": (Path(run_dir) / "best.pt").stat().st_size / 1024 ** 2,
            "cpu_ms": cpu_ms, "cpu_ms_std": cpu_std, "gpu_ms": gpu_ms, "gpu_ms_std": gpu_std,
            "cpu_threads": torch.get_num_threads(), "timed_images": TIMING_RUNS}


# ----------------------------------------------------------------------------------------- plots
def plot_run(metrics, out_dir):
    """Confusion matrix, reliability diagrams (before / after) and accuracy-vs-coverage for one run."""
    out_dir = Path(out_dir)
    name = metrics["run"]
    paths = {}

    # 1. confusion matrix: counts, and the same numbers as a share of each true class (row)
    cm = np.array(metrics["test"]["confusion_matrix"])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, data, title, fmt in ((axes[0], cm, "counts", "d"), (axes[1], cm / cm.sum(1, keepdims=True), "row share", ".2f")):
        ax.imshow(data, cmap="Blues")
        ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=30)
        ax.set_yticks(range(len(CLASSES)), CLASSES)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_title(f"{name}: {title}")
        for i in range(len(CLASSES)):
            for j in range(len(CLASSES)):
                ax.text(j, i, format(data[i, j], fmt), ha="center", va="center",
                        color="white" if data[i, j] > data.max() / 2 else "black")
    paths["confusion"] = _save(fig, out_dir, "confusion.png")

    # 2. reliability diagrams: bar height = accuracy in the bin; perfect calibration = the diagonal
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    centres = (np.arange(N_BINS) + 0.5) / N_BINS
    for ax, key, title in ((axes[0], "before", "before (T = 1)"),
                           (axes[1], "after", f"after (T = {metrics['calibration']['temperature']:.2f})")):
        block = metrics["calibration"][key]
        acc = np.array([np.nan if a is None else a for a in block["bins"]["accuracy"]])
        ax.bar(centres, np.nan_to_num(acc), width=1 / N_BINS, edgecolor="black", alpha=0.7, label="accuracy in bin")
        ax.plot([0, 1], [0, 1], "r--", label="perfect")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("confidence")
        ax.set_title(f"{name} {title}\nECE = {block['ece']:.3f}")
    axes[0].set_ylabel("accuracy")
    axes[0].legend(loc="upper left")
    paths["reliability"] = _save(fig, out_dir, "reliability.png")

    # 3. accuracy vs coverage (scans ranked by MC-dropout entropy)
    cov = metrics["mc_dropout"]["coverage_curve"]
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    ax.plot(cov["coverage"], cov["accuracy"], "o-")
    ax.axhline(metrics["mc_dropout"]["accuracy"], color="grey", ls="--", label="all scans")
    ax.set_xlabel("coverage (share of scans kept, most certain first)")
    ax.set_ylabel("accuracy on kept scans")
    ax.set_title(f"{name}: accuracy vs coverage")
    ax.legend()
    paths["coverage"] = _save(fig, out_dir, "coverage.png")
    return paths


# ----------------------------------------------------------------------------------- one run
def evaluate_run(run_dir, plots_root="outputs/plots/week8", force=False, device="cuda"):
    """Full Week-8 evaluation of one run. Writes metrics.json, adds temperature to config.json, saves plots."""
    run_dir = Path(run_dir)
    set_all_seeds()
    model, cfg = load_run(run_dir, device)
    name = f"{run_dir.parent.name}_{run_dir.name}"

    val_logits, val_labels, _ = extract_cached(run_dir, model, cfg, "val", device, force)
    test_logits, test_labels, test_feats = extract_cached(run_dir, model, cfg, "test", device, force)

    metrics = {"run": name, "model": cfg["model"], "split": run_dir.parent.name,
               "n_val": int(len(val_labels)), "n_test": int(len(test_labels)), "classes": list(CLASSES)}
    metrics["test"] = classification_metrics(softmax(test_logits), test_labels)
    metrics["calibration"] = calibration_block(val_logits, val_labels, test_logits, test_labels)
    t = metrics["calibration"]["temperature"]

    # MC dropout (each pass temperature-scaled) and the referral rule on the temperature-scaled confidence
    mc_probs, mc_entropy = mc_dropout(model, test_feats, t, device)
    correct = mc_probs.argmax(1) == test_labels
    metrics["mc_dropout"] = {
        "passes": MC_PASSES, "p": MC_P, "accuracy": float(correct.mean()),
        "macro_f1": float(f1_score(test_labels, mc_probs.argmax(1), average="macro", labels=LABELS, zero_division=0)),
        "mean_entropy": float(mc_entropy.mean()),
        "mean_entropy_correct": float(mc_entropy[correct].mean()),
        "mean_entropy_wrong": _num(mc_entropy[~correct].mean()) if (~correct).any() else None,
        "coverage_curve": coverage_curve(mc_entropy, correct)}
    metrics["referral"] = {"rule": "max temperature-scaled probability < threshold",
                           **referral_stats(softmax(test_logits, t), test_labels),
                           "mc": referral_stats(mc_probs, test_labels)}

    # sanity numbers: the validation macro-F1 recomputed here vs the one logged during training
    val_f1 = float(f1_score(val_labels, val_logits.argmax(1), average="macro", labels=LABELS, zero_division=0))
    metrics["checks"] = {"val_macro_f1_recomputed": val_f1, "val_macro_f1_training": cfg.get("best_val_macro_f1"),
                         "probs_sum_to_one": bool(np.allclose(softmax(test_logits, t).sum(1), 1.0)),
                         "same_predictions_after_scaling": bool((test_logits.argmax(1) == softmax(test_logits, t).argmax(1)).all())}

    metrics["efficiency"] = efficiency(run_dir, model, cfg)      # last: it moves the model to the CPU

    # write results: metrics.json (new) and config.json (add the temperature, keep every other key)
    _atomic(run_dir / "metrics.json", lambda p: p.write_text(json.dumps(metrics, indent=2)))
    cfg["temperature"] = t
    cfg["mc_dropout"] = {"passes": MC_PASSES, "p": MC_P}
    _atomic(run_dir / "config.json", lambda p: p.write_text(json.dumps(cfg, indent=2)))
    plot_run(metrics, Path(plots_root) / name)

    del model
    torch.cuda.empty_cache()
    return metrics


# --------------------------------------------------------------------------------- all runs
def finished_runs(runs_root="outputs/runs"):
    """Run folders of finished full runs (smoke tests, subsets and failed runs are left out)."""
    runs = []
    for cfg_path in sorted(Path(runs_root).glob("*/*/config.json")):
        cfg = json.loads(cfg_path.read_text())
        if cfg.get("subset") is None and cfg.get("finished") and (cfg_path.parent / "best.pt").exists():
            runs.append(cfg_path.parent)
    return runs


def results_row(metrics, cfg):
    """Flatten one run's metrics.json into one table row."""
    version, split = metrics["split"].split("_")
    t, c, r, mc, e = metrics["test"], metrics["calibration"], metrics["referral"], metrics["mc_dropout"], metrics["efficiency"]
    row = {"version": version, "split": split, "model": metrics["model"], "n_test": metrics["n_test"],
           "accuracy": t["accuracy"], "balanced_acc": t["balanced_accuracy"], "macro_f1": t["macro_f1"],
           "macro_auc": t["macro_auc"]}
    for name in CLASSES:
        row[f"sens_{name}"] = t["per_class"][name]["sensitivity"]
    for name in CLASSES:
        row[f"prec_{name}"] = t["per_class"][name]["precision"]
    row.update({"ece_before": c["before"]["ece"], "ece_after": c["after"]["ece"], "temperature": c["temperature"],
                "nll_before": c["before"]["nll"], "nll_after": c["after"]["nll"],
                "brier_before": c["before"]["brier"], "brier_after": c["after"]["brier"],
                "mc_acc": mc["accuracy"], "mc_mean_entropy": mc["mean_entropy"],
                "entropy_correct": mc["mean_entropy_correct"], "entropy_wrong": mc["mean_entropy_wrong"],
                "refer_rate": r["rate"], "acc_non_referred": r["acc_kept"], "refer_rate_mc": r["mc"]["rate"],
                "acc_non_referred_mc": r["mc"]["acc_kept"],
                "params_M": e["params"] / 1e6, "size_mb": e["size_mb"], "cpu_ms": e["cpu_ms"], "gpu_ms": e["gpu_ms"],
                "epochs_run": cfg.get("epochs_run"),
                # Week 5-6 configs have no "stopped_by": without a target it was patience or the epoch limit
                "stopped_by": cfg.get("stopped_by") or ("max_epochs" if cfg.get("epochs_run") == cfg.get("max_epochs")
                                                        else "patience")})
    return row


def results_table(runs_root="outputs/runs", out_csv="outputs/results/week8_results.csv"):
    """One row per run, read from the metrics.json files (no GPU needed), saved as a CSV."""
    rows = []
    for run_dir in finished_runs(runs_root):
        if (run_dir / "metrics.json").exists():
            metrics = json.loads((run_dir / "metrics.json").read_text())
            rows.append(results_row(metrics, json.loads((run_dir / "config.json").read_text())))
    df = pd.DataFrame(rows)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df


def evaluate_all(runs_root="outputs/runs", force=False):
    """Evaluate every finished run, one at a time (4 GB GPU), then build the results table.
    Runs whose metrics.json is newer than best.pt are skipped unless force=True."""
    for run_dir in finished_runs(runs_root):
        done = run_dir / "metrics.json"
        if done.exists() and not force and done.stat().st_mtime > (run_dir / "best.pt").stat().st_mtime:
            print(f"{run_dir.parent.name}_{run_dir.name:30s} already evaluated (metrics.json is newer than best.pt)")
            continue
        t0 = time.time()
        m = evaluate_run(run_dir, force=force)
        print(f"{m['run']:40s} acc {m['test']['accuracy']:.4f}  T {m['calibration']['temperature']:.2f}  "
              f"({time.time() - t0:.0f} s)", flush=True)
    return results_table(runs_root)
