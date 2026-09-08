"""Streamlit demo for waste image classification."""

from pathlib import Path

import numpy as np
import streamlit as st
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision import models


MODEL_PATH = Path("models/mobilenet_v2_v1.pt")
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

CLASS_NOTES = {
    "battery": "Battery waste. Keep it separate from general waste.",
    "biological": "Organic waste. It can usually go for composting.",
    "brown-glass": "Brown glass waste. Rinse before recycling when possible.",
    "cardboard": "Cardboard waste. Recycle it if it is clean and dry.",
    "clothes": "Textile waste. Donate if reusable.",
    "green-glass": "Green glass waste. Rinse before recycling when possible.",
    "metal": "Metal waste. Usually recyclable.",
    "paper": "Paper waste. Recycle it if it is dry and clean.",
    "plastic": "Plastic waste. Recycle it if it is clean and accepted locally.",
    "shoes": "Footwear waste. Donate if reusable.",
    "trash": "Mixed trash. Dispose in the general waste bin.",
    "white-glass": "Clear glass waste. Rinse before recycling when possible.",
}


@st.cache_resource(show_spinner="Loading model...")
def load_model(model_path: Path):
    """Load the MobileNetV2 checkpoint if it is available."""
    if not model_path.exists():
        return None

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    class_to_index = checkpoint["class_to_index"]
    image_size = checkpoint.get("image_size", 224)

    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = torch.nn.Linear(
        model.classifier[1].in_features,
        len(class_to_index),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    index_to_class = {index: label for label, index in class_to_index.items()}
    return model, index_to_class, image_size


def preprocess(image: Image.Image, image_size: int) -> torch.Tensor:
    """Prepare one image in the same way as the training script."""
    image = image.convert("RGB").resize((image_size, image_size))
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    tensor = TF.normalize(tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD)
    return tensor.unsqueeze(0)


def predict(model, index_to_class, image_size, image: Image.Image):
    """Return class probabilities sorted from high to low."""
    tensor = preprocess(image, image_size)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze()

    results = []
    for index in range(len(index_to_class)):
        results.append((index_to_class[index], float(probs[index])))
    return sorted(results, key=lambda item: item[1], reverse=True)


st.set_page_config(page_title="Waste Image Classification", layout="centered")

st.title("Waste Image Classification Demo")
st.caption("CO5430 Computer Vision - Group G20 - Project P04")

st.write(
    "Upload a waste item image. The app predicts one of the 12 waste classes "
    "using the MobileNetV2 model from the project."
)

model_data = load_model(MODEL_PATH)
if model_data is None:
    st.warning(
        "Model checkpoint not found. Place `mobilenet_v2_v1.pt` inside "
        "the `models/` folder to run predictions."
    )
else:
    st.success("Model loaded successfully.")

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded image", use_container_width=True)

    if model_data is not None:
        model, index_to_class, image_size = model_data
        ranked = predict(model, index_to_class, image_size, image)
        predicted_class, confidence = ranked[0]

        st.subheader("Prediction")
        st.metric("Class", predicted_class)
        st.metric("Confidence", f"{confidence * 100:.2f}%")
        st.info(CLASS_NOTES.get(predicted_class, "No disposal note available."))

        st.subheader("Top 3 Predictions")
        for class_name, probability in ranked[:3]:
            st.write(f"{class_name}: {probability * 100:.2f}%")
            st.progress(probability)
    else:
        st.info("The image preview works. Add the model checkpoint to enable prediction.")
else:
    st.info("Upload a JPG, JPEG, or PNG image to start.")
