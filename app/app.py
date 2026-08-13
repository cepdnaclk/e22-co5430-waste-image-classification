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
    page_title="Waste Image Classification",
    page_icon="♻️",
    layout="centered",
)

# ── custom styles ──────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── page background ── */
.stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #1a1a2e 40%, #16213e 100%);
    min-height: 100vh;
}

/* ── hero header block ── */
.hero-block {
    background: linear-gradient(135deg, #1a6b3a 0%, #0d9488 50%, #0ea5e9 100%);
    border-radius: 18px;
    padding: 2.2rem 2rem 1.8rem 2rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 8px 32px rgba(14,165,233,0.25);
    text-align: center;
}
.hero-block h1 {
    color: #ffffff;
    font-size: 2.1rem;
    font-weight: 700;
    margin: 0 0 0.3rem 0;
    letter-spacing: -0.5px;
    text-shadow: 0 2px 8px rgba(0,0,0,0.3);
}
.hero-block .sub {
    color: rgba(255,255,255,0.78);
    font-size: 0.85rem;
    font-weight: 400;
    letter-spacing: 0.5px;
    margin-bottom: 0.9rem;
}
.hero-block .desc {
    color: rgba(255,255,255,0.9);
    font-size: 0.97rem;
    line-height: 1.65;
    max-width: 560px;
    margin: 0 auto;
}
.hero-block .desc b {
    color: #a7f3d0;
}

/* ── stat pills row ── */
.stat-row {
    display: flex;
    justify-content: center;
    gap: 0.8rem;
    margin-top: 1.1rem;
    flex-wrap: wrap;
}
.stat-pill {
    background: rgba(255,255,255,0.15);
    border: 1px solid rgba(255,255,255,0.25);
    border-radius: 30px;
    padding: 0.28rem 0.9rem;
    font-size: 0.78rem;
    color: #fff;
    font-weight: 500;
    backdrop-filter: blur(4px);
}

/* ── warning / success banners ── */
.stAlert {
    border-radius: 12px !important;
    border: none !important;
}
div[data-baseweb="notification"] {
    border-radius: 12px;
}

/* ── file uploader zone ── */
[data-testid="stFileUploadDropzone"] {
    background: linear-gradient(135deg, rgba(14,165,233,0.07), rgba(16,185,129,0.07)) !important;
    border: 2px dashed rgba(14,165,233,0.45) !important;
    border-radius: 14px !important;
    transition: border-color 0.25s, background 0.25s;
}
[data-testid="stFileUploadDropzone"]:hover {
    border-color: #0ea5e9 !important;
    background: linear-gradient(135deg, rgba(14,165,233,0.13), rgba(16,185,129,0.13)) !important;
}

/* ── section label above uploader ── */
[data-testid="stFileUploaderLabel"] {
    font-size: 1rem !important;
    font-weight: 600 !important;
    color: #e2e8f0 !important;
    margin-bottom: 0.4rem;
}

/* ── prediction label headers ── */
h4 {
    color: #a7f3d0 !important;
    font-weight: 600 !important;
    letter-spacing: 0.3px;
}

/* ── class badge inside prediction ── */
.class-badge {
    display: inline-block;
    background: linear-gradient(90deg, #0d9488, #0ea5e9);
    color: #fff;
    font-weight: 700;
    font-size: 1.25rem;
    letter-spacing: 1px;
    border-radius: 10px;
    padding: 0.35rem 1.1rem;
    margin: 0.3rem 0 0.6rem 0;
    box-shadow: 0 4px 14px rgba(14,165,233,0.3);
}

/* ── progress bars ── */
[role="progressbar"] > div {
    background: linear-gradient(90deg, #0d9488, #0ea5e9) !important;
    border-radius: 6px !important;
}

/* ── expander ── */
[data-testid="stExpander"] {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 12px !important;
}
[data-testid="stExpander"] summary {
    font-weight: 600;
    color: #cbd5e1 !important;
}

/* ── image rounded ── */
[data-testid="stImage"] img {
    border-radius: 12px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.4);
}

/* ── divider ── */
[data-testid="stDivider"] hr {
    border-color: rgba(255,255,255,0.08) !important;
}

/* ── general text ── */
p, li, label {
    color: #cbd5e1;
}
</style>
""", unsafe_allow_html=True)

# ── title & description ────────────────────────────────────────────────────────

st.markdown("""
<div class="hero-block">
    <h1>♻️ Waste Image Classification</h1>
    <p class="sub">CO5430 Computer Vision &nbsp;·&nbsp; Group G20 &nbsp;·&nbsp; Project P04</p>
    <p class="desc">
        Upload a photo of a waste item and the app will predict which of the
        <b>12 waste categories</b> it belongs to.<br>
        Powered by <b>MobileNetV2</b> fine-tuned with transfer learning.
    </p>
    <div class="stat-row">
        <span class="stat-pill">🎯 Test Accuracy 90.4%</span>
        <span class="stat-pill">📊 Macro F1 0.865</span>
        <span class="stat-pill">🗂️ 12 Classes</span>
        <span class="stat-pill">🖼️ 15,515 Images</span>
    </div>
</div>
""", unsafe_allow_html=True)

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
            st.warning(
                "⚠️ **Model checkpoint not found.**  \n"
                "Place `mobilenet_v2_v1.pt` inside the `models/` folder to run predictions.",
                icon="📁",
            )
        else:
            with st.spinner("Predicting…"):
                ranked = predict(model, index_to_class, image_size, image)

            top_class, top_prob = ranked[0]

            # ── top prediction ──
            st.markdown("#### Prediction")
            st.markdown(f'<div class="class-badge">{top_class.upper()}</div>', unsafe_allow_html=True)
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
