"""Streamlit demo: brain tumour MRI classification with calibrated confidence, Grad-CAM and a referral flag.
Run from the project root:  streamlit run app/app.py
All model code lives in inference.py (which reuses src/) and all figures in plots.py; this file only draws the page.
"""
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, UnidentifiedImageError

import inference
import plots
from inference import REFER_BELOW

SAMPLE_DIR = Path(__file__).resolve().parent / "samples"
DISCLAIMER = ("**Research prototype - not a medical device.** This tool was built for a university project. "
              "It must not be used to diagnose, treat or make any clinical decision. Always ask a qualified radiologist.")
# General background text (typed by us, NOT a result of the model). Keyed by the class names in config.json.
CLASS_INFO = {
    "glioma": "Gliomas are tumours that start in glial cells, the supporting cells of the brain and spinal cord. "
              "They range from slow-growing to fast-growing, and the treatment depends on the type and grade.",
    "meningioma": "Meningiomas grow from the meninges, the membranes that cover the brain and spinal cord. "
                  "Many grow slowly and are benign, but they can press on the brain tissue next to them.",
    "pituitary": "Pituitary tumours start in the pituitary gland at the base of the brain. Most are benign; "
                 "they can disturb hormone levels or press on the nerves that carry vision.",
    "notumor": "The model found no tumour pattern in this slice. One slice cannot rule out disease elsewhere in the scan, "
               "and this research model is not a diagnostic test.",
}
FOCUS_CAPTION = ("White circle = the strongest region of the Grad-CAM heat map (heat above 50% of its maximum). It shows where the "
                 "**model** looked. It is **not** a tumour outline: the model was never trained with tumour outlines, and the map is only "
                 "7 x 7 cells stretched to 224 x 224.")

st.set_page_config(page_title="Brain tumour MRI classifier", layout="wide")


@st.cache_resource                      # load the model once per server, not on every click
def get_bundle():
    return inference.load_bundle()


bundle = get_bundle()
cfg, metrics = bundle["cfg"], bundle["metrics"]
classes = cfg["classes"]

# ---------------------------------------------------------------------------------------------- header
st.title("Brain tumour MRI classifier")
st.caption("Glioma, meningioma, pituitary tumour or no tumour - with calibrated confidence, Grad-CAM and a referral flag.")
st.warning(DISCLAIMER, icon="⚠️")

with st.sidebar:
    st.header("About this model")
    st.write(f"**{cfg['model']}** trained on the leakage-free (clean) split, run on the CPU.")
    st.write(f"Temperature scaling: T = {cfg['temperature']:.3f}")
    st.write(f"Scans with confidence below **{REFER_BELOW:.0%}** are flagged for a radiologist.")
    if metrics:                         # numbers come from metrics.json written by src/evaluate.py, nothing is typed in by hand
        st.write(f"Test accuracy on {metrics['n_test']} held-out images: {metrics['test']['accuracy']:.1%}")
        st.write(f"Parameters: {metrics['efficiency']['params'] / 1e6:.2f} M, file size {metrics['efficiency']['size_mb']:.1f} MB")
    st.divider()
    st.caption(DISCLAIMER)
    st.caption("Limits: the model always answers with one of the four classes, even for a picture that is not a brain MRI. "
               "Grad-CAM shows where the network looked, not proof of what it understood.")

# --------------------------------------------------------------------------------------------- input
if "chosen" not in st.session_state:
    st.session_state.chosen = None      # ("sample", class name) or ("upload", None)


def choose_sample(name):
    st.session_state.chosen = ("sample", name)


def choose_upload():
    if st.session_state.get("upload") is not None:
        st.session_state.chosen = ("upload", None)


st.subheader("1. Choose an MRI")
st.file_uploader("Upload a brain MRI slice (JPG or PNG)", type=["jpg", "jpeg", "png"], key="upload", on_change=choose_upload)
st.write("Or click a sample (held-out test images that the model never saw in training):")
for col, name in zip(st.columns(len(classes)), classes):
    with col:
        st.image(str(SAMPLE_DIR / f"{name}.jpg"), caption=f"Sample {classes.index(name) + 1}", width="stretch")
        st.button("Try this sample", key=f"try_{name}", on_click=choose_sample, args=(name,), width="stretch")

chosen = st.session_state.chosen
if chosen is None:
    st.info("Upload an image or click a sample to start.")
    st.stop()

try:
    if chosen[0] == "sample":
        path = SAMPLE_DIR / f"{chosen[1]}.jpg"
        image, known, n_bytes = Image.open(path), chosen[1], path.stat().st_size
    else:
        image, known, n_bytes = Image.open(st.session_state["upload"]), None, st.session_state["upload"].size
    image.load()                        # force decoding now so a broken file fails here
except (UnidentifiedImageError, OSError):
    st.error("This file could not be read as an image. Please upload a JPG or PNG.")
    st.stop()

# ------------------------------------------------------------------------------------ all computations
steps = inference.preprocess_steps(image, cfg["img_size"])              # same preprocessing as in training
grey, no_head_found = steps["final"], steps["fallback"]
result = inference.predict(bundle, grey)
pred_name = classes[result["pred"]]
gm = inference.gradcam_maps(bundle, grey, result["pred"])               # Grad-CAM of the predicted class

st.subheader("2. Result")
if result["refer"]:
    st.error(f"### 🚨 Refer to radiologist\nConfidence {result['conf']:.1%} is below {REFER_BELOW:.0%}. "
             "The model is not sure enough about this scan.")
else:
    st.info(f"Confidence {result['conf']:.1%} is at or above the {REFER_BELOW:.0%} referral threshold. "
            "This is still only a research model, not a diagnosis.")
if no_head_found:
    st.warning("No clear head region was found in this image, so it was used as it is. "
               "It may not be a brain MRI, and the prediction is then meaningless.")

show_circle = st.checkbox("Mark the model's focus region with a circle", value=(pred_name != "notumor"),
                          help="Off by default when the prediction is 'notumor', because there is no tumour to mark.")

tab_sum, tab_img, tab_cam, tab_unc, tab_perf, tab_info = st.tabs(
    ["Summary", "Image details", "Grad-CAM explorer", "Uncertainty", "Model performance", "About the class"])

# ------------------------------------------------------------------------------------------ tab: summary
with tab_sum:
    circle = inference.focus_circle(gm["heat"]) if show_circle else None
    left, middle, right = st.columns([1, 1.2, 1.2])
    with left:
        st.metric("Predicted class", pred_name)
        st.metric("Calibrated confidence", f"{result['conf']:.1%}")
        if known is not None:
            st.write(f"Known label of this sample: **{known}** - {'correct' if known == pred_name else 'WRONG prediction'}")
        st.write("**Confidence for every class**")
        for name, p in zip(classes, result["probs"]):
            st.progress(float(p), text=f"{name}: {p:.1%}" + ("  (predicted)" if name == pred_name else ""))
    with middle:
        st.image(grey, caption="What the network sees (cropped, padded, 224 x 224)", width="stretch", clamp=True)
    with right:
        st.image(inference.make_overlay(grey, gm["heat"], gm["mask"], circle=circle), width="stretch",
                 caption=f"Grad-CAM for '{pred_name}' (cyan line = brain mask)")
        if gm["empty"]:
            st.caption("The heat map is empty for this image.")
        else:
            st.caption(f"Inside-brain ratio: {gm['ratio']:.2f} = share of the heat inside the cyan mask "
                       f"(a uniform heat map would give {gm['mask_frac']:.2f}). The mask covers the whole head, "
                       "so a high ratio does not prove the model looks at the tumour.")
        if circle is not None:
            st.caption(FOCUS_CAPTION)

# -------------------------------------------------------------------------------------- tab: image details
with tab_img:
    stats = inference.intensity_stats(grey, gm["mask"])
    st.write("**The uploaded file**")
    facts = inference.image_facts(image, n_bytes)
    for col, (key, value) in zip(st.columns(len(facts)), facts.items()):
        col.metric(key, value)
    st.write("**Preprocessing, step by step** (the same steps were applied to every training image)")
    st.pyplot(plots.plot_steps(steps, gm["mask"]), width="stretch")
    if no_head_found:
        st.warning("The crop step found no clear head region, so the whole image was used (box = full picture).")
    st.write("**Brightness of the 224 x 224 picture the network sees**")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Brain mask covers", f"{stats['brain_share']:.1%}", help="Share of the picture inside the Otsu brain mask (it includes skull and scalp).")
    c2.metric("Mean (in mask)", f"{stats['mean']:.1f}")
    c3.metric("Std (in mask)", f"{stats['std']:.1f}")
    c4.metric("Darkest (in mask)", stats["min"])
    c5.metric("Brightest (in mask)", stats["max"])
    st.pyplot(plots.plot_histogram(grey, gm["mask"]), width=600)

# ------------------------------------------------------------------------------------ tab: Grad-CAM explorer
with tab_cam:
    c_left, c_right = st.columns(2)
    explain = c_left.selectbox("Class to explain", classes, index=result["pred"],
                               help="Grad-CAM can explain any class: it shows which regions pushed the score of that class up.")
    alpha = c_right.slider("Heat map opacity in the overlay", 0.0, 1.0, inference.HEAT_ALPHA, 0.05)
    idx = classes.index(explain)
    gm_sel = gm if idx == result["pred"] else inference.gradcam_maps(bundle, grey, idx)
    circle_sel = inference.focus_circle(gm_sel["heat"]) if show_circle else None
    title = f"overlay for '{explain}'" + (" (predicted)" if idx == result["pred"] else " (not the predicted class)")
    st.pyplot(plots.plot_gradcam_panels(grey, gm_sel, circle_sel, alpha, title), width="stretch")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric(f"Probability of '{explain}'", f"{result['probs'][idx]:.1%}")
    m2.metric("Inside-brain ratio", f"{gm_sel['ratio']:.2f}")
    m3.metric("Chance level (uniform heat)", f"{gm_sel['mask_frac']:.2f}")
    m4.metric("Ratio / chance", f"{gm_sel['ratio'] / gm_sel['mask_frac']:.2f}" if gm_sel["mask_frac"] > 0 else "n/a")
    if circle_sel is not None:
        x, y, r = circle_sel
        st.write(f"Focus region: centre at pixel (x = {x:.0f}, y = {y:.0f}), radius {r:.0f} px (of 224 x 224).")
        if not gm_sel["mask"][min(int(y), 223), min(int(x), 223)]:
            st.warning("The centre of the focus region lies **outside** the brain mask. For this image the model's strongest heat "
                       "is on the background or the picture border, so do not read the circle as a tumour location.")
        st.caption(FOCUS_CAPTION)
    elif show_circle:
        st.caption("The heat map is empty, so there is no focus region.")

# --------------------------------------------------------------------------------------- tab: uncertainty
with tab_unc:
    passes = inference.mc_passes(bundle, grey)
    mc_mean = passes.mean(0)
    mc_entropy = float(-(mc_mean * np.log(mc_mean + 1e-12)).sum())
    u_left, u_right = st.columns(2)
    with u_left:
        st.write("**Temperature scaling**")
        st.pyplot(plots.plot_probs_compare(result["raw_probs"], result["probs"], classes, REFER_BELOW), width="stretch")
        a, b_, c = st.columns(3)
        a.metric("Margin top-1 vs top-2", f"{result['margin']:.2f}")
        b_.metric("Entropy", f"{result['entropy']:.2f}", help=f"0 = sure, {np.log(len(classes)):.2f} = completely unsure (4 classes).")
        c.metric("Refer to radiologist?", "yes" if result["refer"] else "no")
        direction = "sharpens" if cfg["temperature"] < 1 else "softens"
        st.caption(f"The fitted temperature T = {cfg['temperature']:.3f} {direction} the probabilities (fitted on the validation set). "
                   "It never changes which class is predicted.")
    with u_right:
        st.write(f"**MC dropout** ({len(passes)} passes, dropout {inference.MC_P} on the last layer)")
        st.pyplot(plots.plot_mc(passes, result["probs"], classes), width="stretch")
        a, b_, c = st.columns(3)
        a.metric("MC prediction", classes[int(mc_mean.argmax())])
        b_.metric("MC entropy", f"{mc_entropy:.2f}")
        c.metric("Same as plain?", "yes" if int(mc_mean.argmax()) == result["pred"] else "NO",
                 help="Does the MC-dropout mean predict the same class as the plain (no dropout) prediction?")
        st.caption("The model was trained without dropout, so single passes vary a lot and the MC mean is lower than the plain probability. "
                   "Look at the entropy and at whether the prediction stays the same, not at the exact numbers.")

# ----------------------------------------------------------------------------------- tab: model performance
with tab_perf:
    if metrics is None:
        st.info("metrics.json is missing, so the test-set plots cannot be drawn.")
    else:
        st.write(f"**Results of this model on its {metrics['n_test']} held-out clean-test images** (not on the image you uploaded; "
                 "read from `metrics.json`, produced by `src/evaluate.py` in Week 8).")
        t = metrics["test"]
        cal, ref, eff = metrics["calibration"], metrics["referral"], metrics["efficiency"]
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Accuracy", f"{t['accuracy']:.1%}")
        k2.metric("Macro-F1", f"{t['macro_f1']:.3f}")
        k3.metric("ECE after scaling", f"{cal['after']['ece']:.3f}", delta=f"{cal['after']['ece'] - cal['before']['ece']:.3f} vs before",
                  delta_color="inverse", help="Expected calibration error: lower is better.")
        k4.metric("Referred (conf < 0.70)", f"{ref['rate']:.1%}", help=f"{ref['n_referred']} of {metrics['n_test']} images.")
        k5.metric("CPU time per image", f"{eff['cpu_ms']:.1f} ms")
        st.caption(f"Accuracy on the images that were not referred: {ref['acc_kept']:.1%}; on the referred ones: {ref['acc_referred']:.1%}.")
        p1, p2 = st.columns(2)
        p1.pyplot(plots.plot_confusion(t["confusion_matrix"], classes), width="stretch")
        p2.pyplot(plots.plot_per_class(t["per_class"], classes), width="stretch")
        p3, p4 = st.columns(2)
        p3.pyplot(plots.plot_reliability(cal), width="stretch")
        p4.pyplot(plots.plot_coverage(metrics["mc_dropout"]["coverage_curve"]), width="stretch")
        test_conf = inference.test_confidences(bundle)
        if test_conf is not None:
            st.write("**Where does this image's confidence fall among the test images?**")
            st.pyplot(plots.plot_confidence_hist(test_conf, result["conf"], REFER_BELOW), width=700)
            below = float((test_conf["conf"] < result["conf"]).mean())
            st.caption(f"{below:.0%} of the test images got a lower confidence than this image. "
                       "Wrong predictions (orange) concentrate at lower confidence, which is why the 0.70 referral rule helps.")

# ------------------------------------------------------------------------------------------ tab: about
with tab_info:
    st.subheader(f"Predicted class: {pred_name}")
    st.info(CLASS_INFO[pred_name])
    st.caption("General background information, written by the project team. It is not produced by the model and is not a diagnosis.")
    with st.expander("The four classes"):
        for name in classes:
            st.write(f"**{name}** - {CLASS_INFO[name]}")
