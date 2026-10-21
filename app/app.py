"""Streamlit demo: brain tumour MRI classification with calibrated confidence, Grad-CAM and a referral flag.
Run from the project root:  streamlit run app/app.py
All model code lives in inference.py (which reuses src/); this file only draws the page.
"""
from pathlib import Path

import streamlit as st
from PIL import Image, UnidentifiedImageError

import inference
from inference import REFER_BELOW

SAMPLE_DIR = Path(__file__).resolve().parent / "samples"
DISCLAIMER = ("**Research prototype - not a medical device.** This tool was built for a university project. "
              "It must not be used to diagnose, treat or make any clinical decision. Always ask a qualified radiologist.")

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
        image, known = Image.open(SAMPLE_DIR / f"{chosen[1]}.jpg"), chosen[1]
    else:
        image, known = Image.open(st.session_state["upload"]), None
    image.load()                        # force decoding now so a broken file fails here
except (UnidentifiedImageError, OSError):
    st.error("This file could not be read as an image. Please upload a JPG or PNG.")
    st.stop()

# ------------------------------------------------------------------------------------------- results
grey, no_head_found = inference.preprocess(image, cfg["img_size"])      # same preprocessing as in training
result = inference.predict(bundle, grey)
overlay, ratio, empty_cam = inference.gradcam_overlay(bundle, grey, result["pred"])
pred_name = classes[result["pred"]]

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
    st.image(overlay, caption=f"Grad-CAM for '{pred_name}' (cyan line = brain mask)", width="stretch")
    if empty_cam:
        st.caption("The heat map is empty for this image.")
    else:
        st.caption(f"Inside-brain ratio: {ratio:.2f} = share of the heat inside the cyan mask. "
                   "The mask covers the whole head, so a high ratio does not prove the model looks at the tumour.")
