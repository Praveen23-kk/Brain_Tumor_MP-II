"""Plots for the Streamlit demo (matplotlib only, no Streamlit code). Every function returns a Figure.
Figure objects are created directly (not with pyplot), so nothing piles up in memory between button clicks.
Colours are the colour-blind-safe Okabe-Ito palette, as in src/report.py; colour is never the only cue (labels and numbers are written too).
"""
import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import Circle, Rectangle

BLUE, ORANGE, GREEN, PINK, RED, GREY = "#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00", "#7F7F7F"
CLASS_COLOURS = [BLUE, ORANGE, GREEN, PINK]            # glioma, meningioma, notumor, pituitary (config.json order)


def _blank(ax):
    """Image axes without ticks."""
    ax.set_xticks([])
    ax.set_yticks([])


# ------------------------------------------------------------------------------------ image details
def plot_steps(steps, mask):
    """Preprocessing strip: original greyscale with the crop box -> cropped -> padded and resized (224) -> brain mask outline."""
    fig = Figure(figsize=(11, 3.2), layout="constrained")
    axes = fig.subplots(1, 4)
    x, y, w, h = steps["box"]
    axes[0].imshow(steps["grey"], cmap="gray")
    axes[0].add_patch(Rectangle((x, y), w, h, fill=False, edgecolor=ORANGE, linewidth=2))
    axes[0].set_title(f"1. Greyscale, {steps['grey'].shape[1]} x {steps['grey'].shape[0]}\n(orange = crop box)", fontsize=9)
    axes[1].imshow(steps["cropped"], cmap="gray")
    axes[1].set_title(f"2. Cropped, {steps['cropped'].shape[1]} x {steps['cropped'].shape[0]}", fontsize=9)
    axes[2].imshow(steps["final"], cmap="gray")
    axes[2].set_title(f"3. Padded square + resized,\n{steps['final'].shape[1]} x {steps['final'].shape[0]} (network input)", fontsize=9)
    axes[3].imshow(steps["final"], cmap="gray")
    axes[3].contour(mask, levels=[0.5], colors="cyan", linewidths=1.2)
    axes[3].set_title("4. Brain mask (Otsu, cyan)", fontsize=9)
    for ax in axes:
        _blank(ax)
    return fig


def plot_histogram(grey, mask):
    """Brightness histogram of the 224 x 224 picture: all pixels vs pixels inside the brain mask."""
    fig = Figure(figsize=(6, 3), layout="constrained")
    ax = fig.subplots()
    bins = np.linspace(0, 255, 52)
    ax.hist(grey.ravel(), bins=bins, color=GREY, alpha=0.6, label="all pixels")
    if mask.any():
        ax.hist(grey[mask], bins=bins, color=BLUE, alpha=0.7, label="inside brain mask")
    ax.set_yscale("log")                       # the black background would otherwise hide everything else
    ax.set_xlabel("pixel brightness (0 = black, 255 = white)")
    ax.set_ylabel("pixels (log scale)")
    ax.legend(fontsize=8)
    return fig


# ------------------------------------------------------------------------------ probabilities / MC
def plot_probs_compare(raw, calibrated, classes, threshold):
    """Probabilities before (T = 1) and after temperature scaling, side by side, with the referral threshold."""
    fig = Figure(figsize=(6, 3.2), layout="constrained")
    ax = fig.subplots()
    pos = np.arange(len(classes))
    ax.bar(pos - 0.2, raw, width=0.4, color=GREY, label="before scaling (T = 1)")
    ax.bar(pos + 0.2, calibrated, width=0.4, color=BLUE, label="after temperature scaling")
    ax.axhline(threshold, color=RED, linestyle="--", linewidth=1.2)
    ax.text(len(classes) - 0.5, threshold + 0.02, f"referral threshold {threshold:.2f}", color=RED, ha="right", fontsize=8)
    ax.set_xticks(pos, classes)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("probability")
    ax.legend(fontsize=8, loc="upper left")
    return fig


def plot_mc(passes, plain, classes):
    """MC dropout: mean probability of the 30 passes with +- 1 standard deviation, every single pass as a dot,
    and the plain (no dropout) calibrated probability as a black diamond."""
    fig = Figure(figsize=(6, 3.4), layout="constrained")
    ax = fig.subplots()
    pos = np.arange(len(classes))
    ax.bar(pos, passes.mean(0), yerr=passes.std(0), capsize=5, color=CLASS_COLOURS, alpha=0.75)
    jitter = np.random.default_rng(42).uniform(-0.18, 0.18, size=passes.shape)    # spread the dots sideways so they do not overlap
    ax.scatter(pos + jitter, passes, s=7, color="black", alpha=0.35)
    ax.scatter(pos, plain, marker="D", s=40, color="white", edgecolor="black", zorder=3, label="plain prediction (no dropout)")
    ax.set_xticks(pos, classes)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("probability")
    ax.set_title(f"{len(passes)} MC-dropout passes: bar = mean, whisker = std, dots = passes", fontsize=9)
    ax.legend(fontsize=8, loc="upper left")
    return fig


# ------------------------------------------------------------------------------------------ Grad-CAM
def plot_gradcam_panels(grey, gm, circle, alpha, title):
    """Three panels: the MRI, the heat map alone (with colour bar), and the overlay with the brain-mask outline and the optional focus circle."""
    fig = Figure(figsize=(11, 3.8), layout="constrained")
    axes = fig.subplots(1, 3)
    axes[0].imshow(grey, cmap="gray")
    axes[0].set_title("MRI (network input)", fontsize=9)
    shown = axes[1].imshow(gm["heat"], cmap="inferno", vmin=0, vmax=1)
    axes[1].set_title("Grad-CAM heat map alone", fontsize=9)
    fig.colorbar(shown, ax=axes[1], fraction=0.046, label="relative heat")
    axes[2].imshow(grey, cmap="gray")
    axes[2].imshow(gm["heat"], cmap="inferno", alpha=alpha, vmin=0, vmax=1)
    axes[2].contour(gm["mask"], levels=[0.5], colors="cyan", linewidths=1.0)
    if circle is not None:
        axes[2].add_patch(Circle((circle[0], circle[1]), circle[2], fill=False, edgecolor="white", linewidth=2))
    axes[2].set_xlim(0, grey.shape[1] - 1)             # a circle at the border must not enlarge the axes
    axes[2].set_ylim(grey.shape[0] - 1, 0)
    axes[2].set_title(title, fontsize=9)
    for ax in axes:
        _blank(ax)
    return fig


# ------------------------------------------------------------------------------------ test-set plots
def plot_confusion(cm, classes):
    """Confusion matrix of the clean test set: rows = true class, columns = predicted class, numbers = images."""
    cm = np.array(cm)
    fig = Figure(figsize=(4.6, 4), layout="constrained")
    ax = fig.subplots()
    ax.imshow(cm, cmap="Blues")
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks(range(len(classes)), classes, rotation=30, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    return fig


def plot_per_class(per_class, classes):
    """Sensitivity (recall) and precision of every class on the clean test set."""
    fig = Figure(figsize=(5.6, 3.6), layout="constrained")
    ax = fig.subplots()
    pos = np.arange(len(classes))
    sens = [per_class[c]["sensitivity"] for c in classes]
    prec = [per_class[c]["precision"] for c in classes]
    ax.bar(pos - 0.2, sens, width=0.4, color=BLUE, label="sensitivity")
    ax.bar(pos + 0.2, prec, width=0.4, color=ORANGE, label="precision")
    for p, s, q in zip(pos, sens, prec):
        ax.text(p - 0.2, s + 0.005, f"{s:.2f}", ha="center", fontsize=7)
        ax.text(p + 0.2, q + 0.005, f"{q:.2f}", ha="center", fontsize=7)
    ax.set_xticks(pos, classes)
    ax.set_ylim(0.7, 1.04)                     # starts at 0.7 so the differences are visible; written in the label
    ax.set_ylabel("score (axis starts at 0.7)")
    ax.legend(fontsize=8, loc="lower right")
    return fig


def plot_reliability(calibration):
    """Reliability diagram on the test set before and after temperature scaling: points on the diagonal = confidence equals accuracy.
    Marker area grows with the number of images in the confidence bin."""
    fig = Figure(figsize=(5, 4), layout="constrained")
    ax = fig.subplots()
    ax.plot([0, 1], [0, 1], color=GREY, linestyle="--", label="perfectly calibrated")
    for key, colour, name in (("before", ORANGE, "before scaling"), ("after", BLUE, "after scaling")):
        bins = calibration[key]["bins"]
        keep = [i for i, c in enumerate(bins["confidence"]) if c is not None]
        conf = [bins["confidence"][i] for i in keep]
        acc = [bins["accuracy"][i] for i in keep]
        size = [10 + 3 * bins["count"][i] ** 0.8 for i in keep]
        ax.plot(conf, acc, color=colour, alpha=0.6)
        ax.scatter(conf, acc, s=size, color=colour, label=f"{name} (ECE {calibration[key]['ece']:.3f})", alpha=0.85)
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("confidence")
    ax.set_ylabel("accuracy in the bin")
    ax.legend(fontsize=8, loc="upper left")
    return fig


def plot_coverage(curve):
    """Accuracy when only the most certain share of the test images is kept (MC-dropout entropy order)."""
    fig = Figure(figsize=(5, 3.6), layout="constrained")
    ax = fig.subplots()
    ax.plot(np.array(curve["coverage"]) * 100, np.array(curve["accuracy"]) * 100, color=BLUE, marker="o", markersize=3)
    ax.set_xlabel("share of test images kept (%), most certain first")
    ax.set_ylabel("accuracy on the kept images (%)")
    ax.grid(alpha=0.3)
    return fig


def plot_confidence_hist(test, this_conf, threshold):
    """Histogram of the calibrated confidence of every clean-test image (correct vs wrong), with this image's confidence marked."""
    fig = Figure(figsize=(6, 3.4), layout="constrained")
    ax = fig.subplots()
    bins = np.linspace(0.25, 1, 31)
    ax.hist(test["conf"][test["correct"]], bins=bins, color=BLUE, alpha=0.75, label="test images predicted correctly")
    ax.hist(test["conf"][~test["correct"]], bins=bins, color=RED, alpha=0.85, label="test images predicted wrongly")
    ax.axvline(threshold, color=GREY, linestyle="--")
    ax.text(threshold - 0.005, ax.get_ylim()[1] * 0.55, "referral\nthreshold", ha="right", fontsize=8, color=GREY)
    ax.axvline(this_conf, color="black", linewidth=2.5, label=f"this image ({this_conf:.2f})")
    ax.set_yscale("log")
    ax.set_xlabel("calibrated confidence")
    ax.set_ylabel("images (log scale)")
    ax.legend(fontsize=8, loc="upper left")
    return fig
