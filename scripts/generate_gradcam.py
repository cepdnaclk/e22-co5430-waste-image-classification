"""Phase 9: Generate Grad-CAM examples for the final transfer model.

Grad-CAM highlights the image areas that strongly affected a CNN decision.
This script saves examples for correct and wrong predictions.
"""

import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw, ImageFont, ImageOps
from torch import nn
from torchvision import models


def get_font(size=14):
    font_paths = [
        "C:/Windows/Fonts/arial.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Arial.ttf",
    ]
    for font_path in font_paths:
        if Path(font_path).exists():
            try:
                return ImageFont.truetype(font_path, size)
            except OSError:
                pass
    return ImageFont.load_default()


def read_manifest(path):
    with path.open() as file:
        return list(csv.DictReader(file))


def build_model(model_name, num_classes):
    if model_name == "mobilenet_v2":
        model = models.mobilenet_v2(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return model


def get_target_layer(model, model_name):
    if model_name == "mobilenet_v2":
        return model.features[-1]
    if model_name == "efficientnet_b0":
        return model.features[-1]
    raise ValueError(f"Unknown model: {model_name}")


def load_image_tensor(path, image_size):
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        original = image.resize((image_size, image_size))
        array = np.asarray(original, dtype=np.float32) / 255.0

    tensor = torch.from_numpy(array).permute(2, 0, 1)
    tensor = TF.normalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return original, tensor.unsqueeze(0)


def make_heatmap(cam, size):
    cam = cam - cam.min()
    if cam.max() > 0:
        cam = cam / cam.max()
    cam_image = Image.fromarray(np.uint8(cam * 255)).resize(size, Image.Resampling.BILINEAR)
    heat = Image.new("RGBA", size)
    pixels = cam_image.load()
    heat_pixels = heat.load()
    for y in range(size[1]):
        for x in range(size[0]):
            value = pixels[x, y]
            heat_pixels[x, y] = (255, max(0, 180 - value // 2), 0, int(value * 0.55))
    return heat


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.forward_handle = target_layer.register_forward_hook(self.save_activation)
        self.backward_handle = target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, _module, _inputs, output):
        self.activations = output.detach()

    def save_gradient(self, _module, _grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def close(self):
        self.forward_handle.remove()
        self.backward_handle.remove()

    def __call__(self, image_tensor, class_index):
        self.model.zero_grad()
        output = self.model(image_tensor)
        score = output[0, class_index]
        score.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1).squeeze(0)
        cam = torch.relu(cam).cpu().numpy()
        return cam, output.detach()


def save_overlay(path, original, heatmap, true_label, predicted_label, confidence):
    path.parent.mkdir(parents=True, exist_ok=True)
    overlay = original.convert("RGBA")
    overlay.alpha_composite(heatmap)

    width, height = overlay.size
    canvas = Image.new("RGB", (width, height + 54), "white")
    canvas.paste(overlay.convert("RGB"), (0, 0))
    draw = ImageDraw.Draw(canvas)
    font = get_font(13)
    draw.text((8, height + 8), f"True: {true_label}", fill="#166534", font=font)
    draw.text((8, height + 28), f"Pred: {predicted_label} ({confidence:.2f})", fill="#991b1b", font=font)
    canvas.save(path)


def save_grid(path, images, title):
    if not images:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    loaded = [Image.open(image_path).convert("RGB") for image_path in images]
    cols = min(4, len(loaded))
    rows = (len(loaded) + cols - 1) // cols
    cell_w, cell_h = loaded[0].size
    title_h = 45
    grid = Image.new("RGB", (cols * cell_w, rows * cell_h + title_h), "white")
    draw = ImageDraw.Draw(grid)
    draw.text((10, 12), title, fill="#111827", font=get_font(18))
    for index, image in enumerate(loaded):
        x = (index % cols) * cell_w
        y = title_h + (index // cols) * cell_h
        grid.paste(image, (x, y))
    grid.save(path)


def write_report(path, correct_count, wrong_count):
    path.parent.mkdir(parents=True, exist_ok=True)
    report = f"""# Phase 9: Grad-CAM Explainability Report

## What this phase does

This phase uses Grad-CAM to explain the final MobileNetV2 predictions. Grad-CAM creates a heatmap over the image. The warmer areas show the parts that affected the model decision more strongly.

## Outputs created

- Correct prediction examples: {correct_count}
- Wrong prediction examples: {wrong_count}
- Correct prediction grid: `images/gradcam_correct_grid_v1.jpg`
- Wrong prediction grid: `images/gradcam_wrong_grid_v1.jpg`

## How to read the images

If the warm area is on the waste item, the model is using useful visual evidence. If the warm area is on the background, label text, or another object, the prediction is less reliable.

## What we expect to learn

- Correct predictions should mostly focus on the object shape, color, and texture.
- Wrong predictions may focus on shared visual parts, such as shiny surfaces in plastic, glass, and metal.
- These examples help explain why the final model performs well overall but still struggles on plastic, white-glass, and metal.
"""
    path.write_text(report, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Generate Grad-CAM examples for Phase 9.")
    parser.add_argument("--split-dir", type=Path, default=Path("data/processed/splits"))
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-name", choices=["mobilenet_v2", "efficientnet_b0"], default="mobilenet_v2")
    parser.add_argument("--output-dir", type=Path, default=Path("docs/phase9"))
    parser.add_argument("--examples-per-group", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    class_to_index = checkpoint["class_to_index"]
    image_size = checkpoint.get("image_size", 224)
    index_to_class = {index: label for label, index in class_to_index.items()}

    model = build_model(args.model_name, len(class_to_index))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    target_layer = get_target_layer(model, args.model_name)
    gradcam = GradCAM(model, target_layer)

    rows = read_manifest(args.split_dir / "test_v1.csv")
    random.Random(args.seed).shuffle(rows)

    correct_images = []
    wrong_images = []
    image_dir = args.output_dir / "images"

    for row in rows:
        if len(correct_images) >= args.examples_per_group and len(wrong_images) >= args.examples_per_group:
            break

        true_label = row["class_name"]
        image_path = Path(row["image_path"])
        original, tensor = load_image_tensor(image_path, image_size)
        tensor = tensor.to(device)

        with torch.no_grad():
            output = model(tensor)
            probs = torch.softmax(output, dim=1)
            pred_index = int(probs.argmax(dim=1).item())
            confidence = float(probs[0, pred_index].item())

        cam, _ = gradcam(tensor, pred_index)
        predicted_label = index_to_class[pred_index]
        heatmap = make_heatmap(cam, original.size)

        group = "correct" if predicted_label == true_label else "wrong"
        if group == "correct" and len(correct_images) < args.examples_per_group:
            out_path = image_dir / f"gradcam_correct_{len(correct_images) + 1}_v1.jpg"
            save_overlay(out_path, original, heatmap, true_label, predicted_label, confidence)
            correct_images.append(out_path)
        elif group == "wrong" and len(wrong_images) < args.examples_per_group:
            out_path = image_dir / f"gradcam_wrong_{len(wrong_images) + 1}_v1.jpg"
            save_overlay(out_path, original, heatmap, true_label, predicted_label, confidence)
            wrong_images.append(out_path)

    gradcam.close()

    save_grid(image_dir / "gradcam_correct_grid_v1.jpg", correct_images, "Grad-CAM: correct predictions")
    save_grid(image_dir / "gradcam_wrong_grid_v1.jpg", wrong_images, "Grad-CAM: wrong predictions")
    write_report(args.output_dir / "phase9_gradcam_report_v1.md", len(correct_images), len(wrong_images))

    print(f"Phase 9 outputs saved to {args.output_dir}")


if __name__ == "__main__":
    main()
