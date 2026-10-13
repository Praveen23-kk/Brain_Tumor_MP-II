"""Week 9: the six final report figures (one function per figure).

Inputs are only result tables that already exist (nothing is recomputed here):
  outputs/results/week8_results.csv   test metrics, calibration, size, latency (src/evaluate.py)
  outputs/results/week9_gradcam.csv   one row per test image with the inside-brain ratio (src/gradcam.py)
  outputs/audit/leak_summary.csv      share of test images with a twin in training (Week 3-4)
Colours are the Okabe-Ito colour-blind-safe palette. Colour is never the only cue: models also differ by
marker shape, and original vs clean differs by hollow vs filled markers. All figures are saved at 150 dpi
into outputs/plots/final/ through src.data._save.
Used by notebooks/08_results_plots.ipynb.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data import CLASSES, _save

FINAL_DIR = "outputs/plots/final"
ORDER = ["mobilenetv3_small_100", "efficientnet_b0", "resnet50", "vgg16"]          # light first, then heavy
SHORT = {"mobilenetv3_small_100": "MobileNetV3-S", "efficientnet_b0": "EfficientNet-B0",
         "resnet50": "ResNet-50", "vgg16": "VGG16"}
COLOUR = {"mobilenetv3_small_100": "#0072B2", "efficientnet_b0": "#E69F00",          # Okabe-Ito blue, orange,
          "resnet50": "#009E73", "vgg16": "#D55E00"}                                 # bluish green, vermillion
MARKER = {"mobilenetv3_small_100": "o", "efficientnet_b0": "s", "resnet50": "^", "vgg16": "D"}
EARLY = ("target_acc", "user_stop")        # runs that stopped before patience / epoch limit: unequal training budget
STYLE = {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
         "grid.alpha": 0.25, "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white"}


def _runs(results):
    """Add a 'group' label like 'v1 MobileNetV3-S' and keep a fixed order (v1 models first, then v2)."""
    df = results.copy()
    df["short"] = df["model"].map(SHORT)
    df["early"] = df["stopped_by"].isin(EARLY)
    return df


# ----------------------------------------------------------------------- 1. original vs clean (main result)
def fig1_original_vs_clean(results, out_dir=FINAL_DIR):
    """Paired dots: for every model and dataset version the test score on the original split (hollow) and the clean
    split (filled), joined by a line. A star marks a run that stopped early (unequal training budget)."""
    df = _runs(results)
    groups = [(v, m) for v in sorted(df["version"].unique()) for m in ORDER if ((df["version"] == v) & (df["model"] == m)).any()]
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
        for ax, metric, title in ((axes[0], "accuracy", "Accuracy"), (axes[1], "macro_f1", "Macro-F1")):
            for x, (v, m) in enumerate(groups):
                pair = {}
                for split in ("original", "clean"):
                    row = df[(df["version"] == v) & (df["model"] == m) & (df["split"] == split)]
                    if len(row):
                        pair[split] = row.iloc[0]
                ys = [pair[s][metric] for s in ("original", "clean") if s in pair]
                ax.plot([x, x], [min(ys), max(ys)], color=COLOUR[m], lw=2, alpha=0.6, zorder=1)
                for split, r in pair.items():
                    ax.scatter(x, r[metric], s=90, marker=MARKER[m], zorder=3, edgecolor=COLOUR[m], linewidth=2,
                               facecolor="white" if split == "original" else COLOUR[m])
                    if r["early"]:
                        ax.annotate("*", (x, r[metric]), textcoords="offset points", xytext=(9, -3), fontsize=15, color="black")
            ax.set_xticks(range(len(groups)), [f"{v} {SHORT[m]}" for v, m in groups], rotation=30, ha="right", fontsize=8.5)
            ax.set_ylabel(f"test {title.lower()}")
            ax.set_title(title)
        h1 = axes[0].scatter([], [], s=70, marker="o", facecolor="white", edgecolor="black", linewidth=2, label="original split (leaky test set)")
        h2 = axes[0].scatter([], [], s=70, marker="o", facecolor="black", edgecolor="black", linewidth=2, label="clean split (leak-free)")
        fig.legend(handles=[h1, h2], loc="upper right", bbox_to_anchor=(0.98, 0.99), ncol=1, fontsize=9, frameon=False)
        fig.suptitle("Original vs clean split: test accuracy and macro-F1 per model", y=1.02, fontsize=12)
        fig.text(0.5, -0.12, "* run stopped early (validation target reached or stopped by hand): its training budget differs from the other run of the pair, "
                 "so the gap is not purely leakage.\nTest sets differ (original 1311 / 1600 images, clean 520 / 538). One seed.",
                 ha="center", fontsize=8.5)
        return _save(fig, out_dir, "fig1_original_vs_clean.png")


# --------------------------------------------------------------------------------- 2. size vs macro-F1
def fig2_size_vs_f1(results, out_dir=FINAL_DIR):
    """Model size (MB, log axis) vs test macro-F1. Filled = clean split, hollow = original split."""
    df = _runs(results)
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(7.5, 5))
        for m in ORDER:
            for split in ("original", "clean"):
                sub = df[(df["model"] == m) & (df["split"] == split)]
                ax.scatter(sub["size_mb"], sub["macro_f1"], s=90, marker=MARKER[m], edgecolor=COLOUR[m], linewidth=2,
                           facecolor="white" if split == "original" else COLOUR[m], zorder=3)
            ax.annotate(SHORT[m], (df[df["model"] == m]["size_mb"].iloc[0], df[df["model"] == m]["macro_f1"].max()),
                        textcoords="offset points", xytext=(0, 10), ha="center", fontsize=9, color=COLOUR[m])
        ax.set_xscale("log")
        ax.set_xticks([5, 10, 20, 50, 100, 200, 500], ["5", "10", "20", "50", "100", "200", "500"])   # plain numbers, not 10^x
        ax.minorticks_off()
        ax.set_xlabel("model size on disk (MB, log scale)")
        ax.set_ylabel("test macro-F1")
        ax.set_title("Model size vs macro-F1 (each marker = one run, v1 and v2)", fontsize=10)
        h1 = ax.scatter([], [], s=70, marker="o", facecolor="white", edgecolor="black", linewidth=2, label="original split")
        h2 = ax.scatter([], [], s=70, marker="o", facecolor="black", edgecolor="black", linewidth=2, label="clean split")
        ax.legend(handles=[h1, h2], loc="lower right")
        return _save(fig, out_dir, "fig2_size_vs_f1.png")


# ------------------------------------------------------------------------------ 3. per-class sensitivity
def fig3_sensitivity(results, version="v1", out_dir=FINAL_DIR):
    """Sensitivity (recall) of each class, one marker per model, clean split and original split side by side."""
    df = _runs(results)
    df = df[df["version"] == version]
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
        for ax, split in zip(axes, ("clean", "original")):
            for k, m in enumerate(ORDER):
                row = df[(df["model"] == m) & (df["split"] == split)]
                if row.empty:
                    continue
                ys = [row.iloc[0][f"sens_{c}"] for c in CLASSES]
                xs = np.arange(len(CLASSES)) + (k - 1.5) * 0.14           # small horizontal offsets so markers do not overlap
                ax.scatter(xs, ys, s=80, marker=MARKER[m], color=COLOUR[m], label=SHORT[m], zorder=3)
            ax.set_xticks(range(len(CLASSES)), CLASSES)
            ax.set_title(f"{version} {split} split")
        axes[0].set_ylabel("sensitivity (share of the class found)")
        axes[0].legend(loc="lower left", fontsize=8.5)
        fig.suptitle("Per-class sensitivity on the test set", y=1.02, fontsize=12)
        return _save(fig, out_dir, "fig3_sensitivity.png")


# ------------------------------------------------------------------------------------------ 4. ECE
def fig4_ece(results, out_dir=FINAL_DIR):
    """ECE before (hollow) and after (filled) temperature scaling for every run, joined by a line; T written at the right."""
    df = _runs(results).sort_values(["version", "split", "model"]).reset_index(drop=True)
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(8, 6))
        for y, r in df.iterrows():
            ax.plot([r["ece_before"], r["ece_after"]], [y, y], color=COLOUR[r["model"]], lw=2, alpha=0.6, zorder=1)
            ax.scatter(r["ece_before"], y, s=70, marker=MARKER[r["model"]], facecolor="white", edgecolor=COLOUR[r["model"]], linewidth=2, zorder=3)
            ax.scatter(r["ece_after"], y, s=70, marker=MARKER[r["model"]], color=COLOUR[r["model"]], zorder=3)
            ax.text(df["ece_before"].max() * 1.08, y, f"T = {r['temperature']:.2f}", va="center", fontsize=8)
        ax.set_yticks(range(len(df)), [f"{r['version']} {r['split']} {r['short']}" for _, r in df.iterrows()], fontsize=8.5)
        ax.invert_yaxis()
        ax.set_xlim(0, df["ece_before"].max() * 1.3)
        ax.set_xlabel("expected calibration error, 15 bins (lower is better)")
        ax.set_title("Calibration before (hollow) and after (filled) temperature scaling")
        return _save(fig, out_dir, "fig4_ece.png")


# ------------------------------------------------------------------------------ 5. inside-brain ratio
def fig5_inside_ratio(gradcam, out_dir=FINAL_DIR):
    """Boxplots of the inside-brain ratio per model (all runs of a model pooled), original vs clean test sets.
    The dashed line is the average share of the picture covered by the brain: a perfectly uniform heat map would score that."""
    df = gradcam.copy()
    df["split"] = df["run"].str.split("_").str[1]
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(9, 5))
        for k, split in enumerate(("original", "clean")):
            data = [df[(df["model"] == m) & (df["split"] == split)]["inside_ratio"].to_numpy() for m in ORDER]
            pos = np.arange(len(ORDER)) + (k - 0.5) * 0.36
            bp = ax.boxplot(data, positions=pos, widths=0.3, patch_artist=True, showfliers=False, medianprops=dict(color="black", lw=2))
            for patch, m in zip(bp["boxes"], ORDER):
                patch.set_facecolor(COLOUR[m])
                patch.set_alpha(0.35 if split == "original" else 0.9)
                patch.set_hatch("//" if split == "original" else "")
            for x, d in zip(pos, data):
                ax.text(x, 1.04, f"n={len(d)}", ha="center", fontsize=7.5)      # above the boxes, clear of the whiskers
        ax.axhline(df["mask_frac"].mean(), color="black", ls="--", lw=1, label="uniform heat map (chance level)")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.07), frameon=False)
        ax.set_xticks(range(len(ORDER)), [SHORT[m] for m in ORDER])
        ax.set_ylim(0, 1.1)
        ax.set_ylabel("inside-brain ratio")
        ax.set_title("Grad-CAM inside-brain ratio per model\n(hatched = original split, solid = clean split; n = test images pooled over v1 and v2)", fontsize=10)
        return _save(fig, out_dir, "fig5_inside_ratio.png")


# ------------------------------------------------------------------------------------------ 6. leakage
def fig6_leakage(leak, out_dir=FINAL_DIR):
    """Share of the ORIGINAL test images that have a twin in training, per dataset version and duplicate type,
    and the same share for the clean split (0 by construction). `leak` = outputs/audit/leak_summary.csv with index_col=0."""
    kinds = list(leak.columns)
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(10, 4.8))
        width = 0.38
        for k, (version, colour, hatch) in enumerate(zip(leak.index, ("#0072B2", "#E69F00"), ("", "//"))):
            bars = ax.bar(np.arange(len(kinds)) + (k - 0.5) * width, leak.loc[version].to_numpy(), width, color=colour,
                          hatch=hatch, edgecolor="black", label=f"dataset {version}")
            ax.bar_label(bars, fmt="%.1f", fontsize=8.5)
        ax.set_xticks(range(len(kinds)), [c.replace(" (", "\n(") for c in kinds], fontsize=8.5)
        ax.set_ylabel("% of test images with a twin in training")
        ax.set_ylim(0, 80)
        ax.legend()
        ax.set_title("Train-test leakage in the original split, and after the clean split")
        return _save(fig, out_dir, "fig6_leakage.png")
