"""
Waste Image Classification Demo
Phase 11 - Streamlit web app
Group G20 / Project P04
"""

from pathlib import Path

import numpy as np
import streamlit as st
import torch
import torchvision.transforms.functional as TF
from PIL import Image

# ── constants ──────────────────────────────────────────────────────────────────

MODEL_PATH = Path("models/mobilenet_v2_v1.pt")

CLASS_NOTES = {
    "battery":     "Battery waste. Handle separately — do not throw in general bin.",
    "biological":  "Organic/food waste. Can be composted.",
    "brown-glass": "Brown glass. Usually recyclable — rinse before disposal.",
    "cardboard":   "Cardboard waste. Recyclable if clean and dry.",
    "clothes":     "Textile waste. Donate if reusable, else textile recycling.",
    "green-glass": "Green glass. Usually recyclable — rinse before disposal.",
    "metal":       "Metal waste. Usually recyclable at a scrap or recycling centre.",
    "paper":       "Paper waste. Recyclable if dry and not heavily soiled.",
    "plastic":     "Plastic waste. Recyclable if clean — check local guidelines.",
    "shoes":       "Footwear waste. Donate if reusable, else special disposal.",
    "trash":       "Mixed/general waste. Dispose in general waste bin.",
    "white-glass": "White/clear glass. Usually recyclable — rinse before disposal.",
}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ── model helpers ──────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading model…")
def load_model(model_path: Path):
    """Load MobileNetV2 checkpoint. Returns (model, index_to_class, image_size) or None."""
    if not model_path.exists():
        return None

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    class_to_index = checkpoint["class_to_index"]
    image_size = checkpoint.get("image_size", 224)
    num_classes = len(class_to_index)

    from torchvision import models
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    index_to_class = {v: k for k, v in class_to_index.items()}
    return model, index_to_class, image_size


def preprocess(image: Image.Image, image_size: int) -> torch.Tensor:
    """Preprocess a PIL image to match Phase 6 training pipeline."""
    image = image.convert("RGB").resize((image_size, image_size))
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)            # H x W x C -> C x H x W
    tensor = TF.normalize(tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD)
    return tensor.unsqueeze(0)                                    # add batch dim


def predict(model, index_to_class, image_size, image: Image.Image):
    """Return list of (class_name, probability) sorted by probability descending."""
    tensor = preprocess(image, image_size)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze()

    ranked = sorted(
        [(index_to_class[i], float(probs[i])) for i in range(len(index_to_class))],
        key=lambda x: x[1],
        reverse=True,
    )
    return ranked

# ── page config ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Waste Image Classification Demo",
    page_icon="♻️",
    layout="centered",
)

# ── title & description ────────────────────────────────────────────────────────

st.title("♻️ Waste Image Classification Demo")
st.caption("CO5430 Computer Vision · Group G20 · Project P04")

st.markdown(
    """
    Upload a photo of a waste item and the app will predict which of the
    **12 waste categories** it belongs to.  
    The model is **MobileNetV2** fine-tuned with transfer learning
    (test accuracy 90.4 %, macro F1 0.865).
    """
)

# ── model loading ──────────────────────────────────────────────────────────────

result = load_model(MODEL_PATH)
model_ready = result is not None

if not model_ready:
    st.warning(
        "⚠️  **Model checkpoint not found.**  "
        "Place `mobilenet_v2_v1.pt` inside the `models/` folder to run predictions.",
        icon="📁",
    )
else:
    model, index_to_class, image_size = result
    st.success("✅ Model loaded — ready to predict.", icon="🤖")

st.divider()

# ── image upload ───────────────────────────────────────────────────────────────

uploaded_file = st.file_uploader(
    "Upload a waste image",
    type=["jpg", "jpeg", "png"],
    label_visibility="visible",
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    col_img, col_info = st.columns([1, 1], gap="large")

    with col_img:
        st.image(image, caption="Uploaded image", use_container_width=True)

    with col_info:
        if not model_ready:
            st.info("Upload is shown above. Add the model checkpoint to get predictions.")
        else:
            with st.spinner("Predicting…"):
                ranked = predict(model, index_to_class, image_size, image)

            top_class, top_prob = ranked[0]

            # ── top prediction ──
            st.markdown("#### Prediction")
            st.markdown(f"**{top_class.upper()}**")
            st.progress(top_prob, text=f"Confidence: {top_prob * 100:.1f} %")

            # ── disposal note ──
            note = CLASS_NOTES.get(top_class, "Check local disposal guidelines.")
            st.info(f"🗑️ {note}", icon=None)

            st.markdown("---")

            # ── top 3 ──
            st.markdown("#### Top 3 predictions")
            for rank, (class_name, prob) in enumerate(ranked[:3], start=1):
                st.markdown(f"**{rank}. {class_name}**")
                st.progress(prob, text=f"{prob * 100:.1f} %")

# ── class information expander ─────────────────────────────────────────────────

st.divider()

with st.expander("📋 All 12 waste categories"):
    for class_name, note in CLASS_NOTES.items():
        st.markdown(f"- **{class_name}** — {note}")

# ── footer ─────────────────────────────────────────────────────────────────────

st.markdown(
    """
    <hr style="margin-top:2rem;margin-bottom:0.5rem">
    <p style="font-size:0.8rem;color:grey;text-align:center">
    E/22/051 · E/22/385 · E/22/227 · E/22/004 &nbsp;|&nbsp;
    Dataset: <a href="https://www.kaggle.com/datasets/mostafaabla/garbage-classification" target="_blank">Kaggle Garbage Classification</a>
    </p>
    """,
    unsafe_allow_html=True,
)
