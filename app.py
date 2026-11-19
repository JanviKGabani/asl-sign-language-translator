"""Streamlit demo: read an ASL alphabet letter from a photo of your hand.

MediaPipe finds the 21 hand landmarks, the same normalisation used in training makes them
translation/scale/handedness invariant, and a small MLP (exported to NumPy) classifies the
letter. No TensorFlow needed at serve time.
"""
import sys
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, ImageOps

ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inference.numpy_predictor import NumpyASLModel
from utils.mediapipe_utils import (
    create_hands, extract_hands, select_primary_hand, landmarks_to_feature,
    draw_hand_landmarks,
)

MODEL_PATH = ROOT / "models" / "asl_mlp_numpy.npz"

st.set_page_config(page_title="ASL Sign Language Translator", page_icon="🤟", layout="wide")


@st.cache_resource
def load_model():
    return NumpyASLModel(str(MODEL_PATH))


@st.cache_resource
def load_hands():
    # static_image_mode=True: treat each photo independently rather than as a video stream
    return create_hands(static_image_mode=True, max_num_hands=2,
                        min_detection_confidence=0.5, min_tracking_confidence=0.5)


def analyse(pil_img):
    """Return (annotated image, letter, top-3, handedness) or (None, ...) if no hand found."""
    img = ImageOps.exif_transpose(pil_img).convert("RGB")
    rgb = np.array(img)

    results = load_hands().process(rgb)
    hands = extract_hands(results)
    hand = select_primary_hand(hands)
    if hand is None:
        return None, None, None, None

    annotated = rgb.copy()
    draw_hand_landmarks(annotated, hand["raw"])

    feature = landmarks_to_feature(hand["landmarks"], hand["handedness"])[None, :]
    top = load_model().predict_top(feature, k=3)
    return annotated, top[0][0], top, hand["handedness"]


# ---------------- UI ----------------
st.title("🤟 ASL Sign Language Translator")
st.caption("Hold up an American Sign Language letter (A–Z) and the model reads it from the "
           "shape of your hand.")

with st.sidebar:
    st.header("How it works")
    st.markdown(
        "1. **MediaPipe Hands** finds 21 landmarks on the hand.\n"
        "2. Landmarks are **normalised** — wrist moved to the origin, scaled to unit size, "
        "left hands mirrored — so position, distance from the camera and which hand you use "
        "stop mattering.\n"
        "3. A small **MLP** (256→128→64→26) classifies the 63 numbers into a letter.\n"
        "4. For this demo the network is exported to NumPy, so no deep-learning runtime "
        "is loaded at serve time."
    )
    st.header("Honest accuracy note")
    st.markdown(
        "The model scores **100% on held-out frames**, including a split that holds out whole "
        "contiguous runs rather than shuffling near-identical frames. That sounds better than "
        "it is: all 2,081 training frames came from **two people in one afternoon**, so the "
        "test frames look a lot like the training ones.\n\n"
        "Expect it to do **worse on you** — different hand shape, lighting and camera. "
        "Letters that differ only subtly (**M / N / S / T**, and **R / U / V**) are the first "
        "to break."
    )
    st.markdown("[Source code](https://github.com/JanviKGabani/asl-sign-language-translator) · "
                "[Portfolio](https://janvikgabani.github.io)")

st.info("**J and Z involve motion in real ASL.** This model sees a single frame, so it reads "
        "their static hand shape only.", icon="ℹ️")

tab_cam, tab_up = st.tabs(["📷 Use your camera", "📁 Upload a photo"])
with tab_cam:
    shot = st.camera_input("Make a letter with one hand, then take a photo")
with tab_up:
    upload = st.file_uploader("Photo of a hand making an ASL letter",
                              type=["jpg", "jpeg", "png", "webp"])

source = shot or upload
if source is None:
    st.stop()

with st.spinner("Looking for a hand…"):
    annotated, letter, top, handedness = analyse(Image.open(source))

if annotated is None:
    st.error("No hand found in that image. Try better lighting, and keep your whole hand "
             "inside the frame against a plainer background.")
    st.stop()

left, right = st.columns([1, 1])
with left:
    st.subheader("Detected hand")
    st.image(annotated, caption=f"{handedness} hand, 21 landmarks", use_container_width=True)
with right:
    st.subheader("Prediction")
    st.markdown(f"# {letter}")
    st.caption(f"Confidence {top[0][1]:.1%}")
    if top[0][1] < 0.60:
        st.warning("Low confidence — the hand shape is ambiguous to the model here.")
    st.markdown("**Next most likely**")
    for lab, p in top[1:]:
        st.write(f"{lab} — {p:.1%}")
        st.progress(min(p, 1.0))
