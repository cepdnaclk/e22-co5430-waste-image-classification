"""Phase 7: Final evaluation on the test set.

This script loads the best model from Phase 6 and evaluates it once
on the held-out test set. No training or hyperparameter tuning is done.

Usage:
    python scripts/evaluate_final.py \
        --split-dir data/processed/splits \
        --model-path models/mobilenet_v2_v1.pt \
        --model-name mobilenet_v2 \
        --output-dir docs/phase7
"""

import argparse
import csv
import datetime
import random
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF
from torchvision import models
from PIL import Image, ImageDraw, ImageFont, ImageOps
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


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
    """Read a split CSV file. Returns list of dicts with image_path and class_name."""
    with path.open() as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------
# Dataset — same preprocessing as Phase 6
# ---------------------------------------------------------------------------

class WasteImageDataset(Dataset):
    """Load images with same resize and ImageNet normalization as Phase 6."""

    def __init__(self, rows, class_to_index, image_size):
        self.rows = rows
        self.class_to_index = class_to_index
        self.image_size = image_size

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        image_path = row["image_path"]
        class_name = row["class_name"]

        with Image.open(image_path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            image = image.resize((self.image_size, self.image_size))
            array = np.asarray(image, dtype=np.float32) / 255.0

        tensor = torch.from_numpy(array).permute(2, 0, 1)
        # ImageNet normalization — exactly as in Phase 6
        tensor = TF.normalize(
            tensor,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
        label = self.class_to_index[class_name]
        return tensor, label, image_path


# ---------------------------------------------------------------------------
# Model builder — same architecture as Phase 6
# ---------------------------------------------------------------------------

def build_model(model_name, num_classes):
    """Rebuild the same transfer model architecture used in Phase 6."""
    if model_name == "mobilenet_v2":
        model = models.mobilenet_v2(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return model


# ---------------------------------------------------------------------------
# CSV / report writers
# ---------------------------------------------------------------------------

def write_metrics_csv(path, metrics):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "value"])
        writer.writeheader()
        for k, v in metrics.items():
            writer.writerow({"metric": k, "value": v})


def write_class_report_csv(path, labels, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["class_name", "precision", "recall", "f1_score", "support"]
        )
        writer.writeheader()
        for label in labels:
            vals = report[label]
            writer.writerow({
                "class_name": label,
                "precision": round(vals["precision"], 4),
                "recall": round(vals["recall"], 4),
                "f1_score": round(vals["f1-score"], 4),
                "support": int(vals["support"]),
            })


def write_confusion_matrix_csv(path, labels, matrix):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["actual/predicted", *labels])
        for label, row in zip(labels, matrix):
            writer.writerow([label, *row.tolist()])


def write_model_summary_csv(path, model_name, val_acc, val_f1, test_acc, test_f1):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "model", "val_accuracy", "val_macro_f1", "test_accuracy", "test_macro_f1",
        ])
        writer.writeheader()
        writer.writerow({
            "model": model_name,
            "val_accuracy": round(val_acc, 4),
            "val_macro_f1": round(val_f1, 4),
            "test_accuracy": round(test_acc, 4),
            "test_macro_f1": round(test_f1, 4),
        })


# ---------------------------------------------------------------------------
# Visual outputs
# ---------------------------------------------------------------------------

def save_confusion_matrix_image(path, labels, matrix, title):
    path.parent.mkdir(parents=True, exist_ok=True)
    cell = 48
    left_margin = 150
    top_margin = 130
    width = left_margin + cell * len(labels) + 40
    height = top_margin + cell * len(labels) + 40
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = get_font(20)
    font = get_font(12)
    small_font = get_font(10)

    draw.text((left_margin, 30), title, fill="#111827", font=title_font)
    draw.text((left_margin, 70), "Predicted class", fill="#111827", font=font)
    draw.text((20, top_margin - 30), "Actual class", fill="#111827", font=font)

    max_value = max(int(matrix.max()), 1)

    for i, label in enumerate(labels):
        x = left_margin + i * cell
        draw.text((x + 4, top_margin - 48), label[:10], fill="#111827", font=small_font)
        y = top_margin + i * cell
        draw.text((20, y + 15), label[:14], fill="#111827", font=small_font)

    for ri, row in enumerate(matrix):
        for ci, value in enumerate(row):
            intensity = int(255 - (value / max_value) * 210)
            color = (255, min(intensity + 20, 255), intensity)
            x0 = left_margin + ci * cell
            y0 = top_margin + ri * cell
            draw.rectangle([(x0, y0), (x0 + cell, y0 + cell)], fill=color, outline="#d1d5db")
            if value > 0:
                draw.text((x0 + 14, y0 + 16), str(int(value)), fill="#111827", font=small_font)

    image.save(path)


def save_result_summary_chart(path, model_name, val_f1, test_f1):
    """Bar chart comparing validation and test macro F1."""
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 600, 400
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = get_font(20)
    font = get_font(14)
    small_font = get_font(12)

    draw.text((50, 20), f"Final Result Summary — {model_name}", fill="#111827", font=title_font)

    bar_left = 180
    bar_max_width = 350
    bar_height = 40

    # Validation F1 bar
    y1 = 100
    val_w = int(val_f1 * bar_max_width)
    draw.text((30, y1 + 10), "Val Macro F1", fill="#111827", font=font)
    draw.rectangle([(bar_left, y1), (bar_left + val_w, y1 + bar_height)], fill="#2563eb")
    draw.text((bar_left + val_w + 10, y1 + 10), f"{val_f1:.4f}", fill="#111827", font=font)

    # Test F1 bar
    y2 = 180
    test_w = int(test_f1 * bar_max_width)
    draw.text((30, y2 + 10), "Test Macro F1", fill="#111827", font=font)
    draw.rectangle([(bar_left, y2), (bar_left + test_w, y2 + bar_height)], fill="#dc2626")
    draw.text((bar_left + test_w + 10, y2 + 10), f"{test_f1:.4f}", fill="#111827", font=font)

    # Difference
    diff = test_f1 - val_f1
    sign = "+" if diff >= 0 else ""
    draw.text((30, 280), f"Difference: {sign}{diff:.4f}", fill="#374151", font=font)
    draw.text((30, 320),
              "A small drop from validation to test is normal.",
              fill="#6b7280", font=small_font)
    draw.text((30, 345),
              "The test set was used only once for this final evaluation.",
              fill="#6b7280", font=small_font)

    image.save(path)


def save_failure_examples(path, wrong_predictions, labels, image_size):
    """Create a grid of misclassified examples."""
    path.parent.mkdir(parents=True, exist_ok=True)

    # Show up to 12 failure examples.
    examples = wrong_predictions[:12]
    if not examples:
        return

    cols = min(4, len(examples))
    rows_count = (len(examples) + cols - 1) // cols
    thumb = 160
    label_h = 50
    cell_w = thumb + 10
    cell_h = thumb + label_h + 10
    width = cols * cell_w + 10
    height = rows_count * cell_h + 60

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = get_font(16)
    small_font = get_font(11)

    draw.text((10, 10), "Failure Examples (wrong predictions)", fill="#111827", font=title_font)

    for idx, (img_path, true_label, pred_label) in enumerate(examples):
        row_i = idx // cols
        col_i = idx % cols
        x = col_i * cell_w + 10
        y = row_i * cell_h + 50

        try:
            with Image.open(img_path) as img:
                img = ImageOps.exif_transpose(img).convert("RGB")
                img = img.resize((thumb, thumb))
                image.paste(img, (x, y))
        except Exception:
            draw.rectangle([(x, y), (x + thumb, y + thumb)], fill="#f3f4f6", outline="#d1d5db")
            draw.text((x + 20, y + 70), "Could not load", fill="#6b7280", font=small_font)

        draw.text((x, y + thumb + 2), f"True: {true_label}", fill="#16a34a", font=small_font)
        draw.text((x, y + thumb + 18), f"Pred: {pred_label}", fill="#dc2626", font=small_font)

    image.save(path)


# ---------------------------------------------------------------------------
# Failure analysis
# ---------------------------------------------------------------------------

def find_confused_pairs(labels, matrix, top_n=5):
    """Find the most confused class pairs from the confusion matrix."""
    pairs = []
    for i, actual in enumerate(labels):
        for j, predicted in enumerate(labels):
            if i != j and matrix[i][j] > 0:
                pairs.append((actual, predicted, int(matrix[i][j])))
    pairs.sort(key=lambda x: x[2], reverse=True)
    return pairs[:top_n]


def get_failure_explanations():
    """Return simple explanations for common confusions."""
    return {
        ("paper", "cardboard"): "Both can have similar brown or white flat textures.",
        ("cardboard", "paper"): "Both can have similar brown or white flat textures.",
        ("green-glass", "brown-glass"): "Glass bottles can look similar in shape, just different in color.",
        ("brown-glass", "green-glass"): "Glass bottles can look similar in shape, just different in color.",
        ("white-glass", "green-glass"): "Glass bottles share similar shapes despite color differences.",
        ("green-glass", "white-glass"): "Glass bottles share similar shapes despite color differences.",
        ("white-glass", "brown-glass"): "Glass items can have similar shapes and reflections.",
        ("brown-glass", "white-glass"): "Glass items can have similar shapes and reflections.",
        ("plastic", "white-glass"): "Both can appear transparent or translucent.",
        ("white-glass", "plastic"): "Both can appear transparent or translucent.",
        ("clothes", "shoes"): "Both are wearable items and can share fabric textures.",
        ("shoes", "clothes"): "Both are wearable items and can share fabric textures.",
        ("biological", "paper"): "Some food waste on paper or packaging can look similar.",
        ("paper", "biological"): "Some food waste on paper or packaging can look similar.",
        ("trash", "plastic"): "Miscellaneous trash often includes plastic items.",
        ("plastic", "trash"): "Miscellaneous trash often includes plastic items.",
        ("metal", "battery"): "Batteries are metallic, so they share surface appearance.",
        ("battery", "metal"): "Batteries are metallic, so they share surface appearance.",
    }


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def write_report(path, model_name, checkpoint_path, image_size, test_count,
                 metrics, labels, matrix, confused_pairs):
    """Generate the Phase 7 markdown report."""
    path.parent.mkdir(parents=True, exist_ok=True)

    # Build confusion pair analysis text.
    explanations = get_failure_explanations()
    confusion_lines = []
    for actual, predicted, count in confused_pairs:
        explanation = explanations.get((actual, predicted), "These classes may share visual features.")
        confusion_lines.append(f"- **{actual}** confused as **{predicted}** ({count} times): {explanation}")

    confusion_text = "\n".join(confusion_lines) if confusion_lines else "- No major confusions found."

    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    report = f"""# Phase 7: Final Test Evaluation Report

## What this phase does

Phase 7 evaluates the best model from Phase 6 on the held-out test set. This is the first and only time the test set is used. No training or tuning was done in this phase.

The "test set" means images that were kept aside and never seen during training or validation. This gives us an honest measure of how the model performs on new, unseen images.

## Model selected

We selected **{model_name}** because it had the best validation macro F1-score among the Phase 6 transfer learning models.

"Macro F1-score" gives equal importance to every class, which is important because our dataset is imbalanced (some classes have more images than others).

- Checkpoint: `{checkpoint_path}`
- Image size: {image_size} x {image_size}
- Date: {now}

## Final test results

| Metric | Value |
|---|---:|
| Test accuracy | {metrics['test_accuracy']:.4f} |
| Macro precision | {metrics['macro_precision']:.4f} |
| Macro recall | {metrics['macro_recall']:.4f} |
| Macro F1-score | {metrics['macro_f1']:.4f} |
| Total test images | {test_count} |

## Confusion matrix

The confusion matrix shows which classes are confused with each other. Darker cells mean more images were classified that way.

![Test confusion matrix](images/final_test_confusion_matrix_v1.png)

## Result summary

This chart compares the validation macro F1 (from Phase 6) with the test macro F1 (from this phase).

![Result summary](images/final_result_summary_v1.png)

## Failure Analysis

We looked at the most confused class pairs in the confusion matrix to understand where the model struggles.

{confusion_text}

### Failure examples

Below are some examples where the model predicted the wrong class. Each image shows the true label and the predicted label.

![Failure examples](images/failure_examples_v1.jpg)

### Why some classes are difficult

- **Glass types** (green-glass, brown-glass, white-glass) share very similar bottle and jar shapes. The model must rely on color, which can be tricky with lighting variation.
- **Paper and cardboard** both have flat, fibrous textures. The difference is mostly thickness, which is hard to see in a photo.
- **Trash** is a catch-all category with no consistent visual pattern, making it inherently hard.
- **Plastic** items come in many shapes and colors, overlapping with glass (transparent items) and trash.

## Important notes

- The test set was used **only once** in this phase.
- No training was done in Phase 7.
- No hyperparameter tuning was done using the test set.
- The model was not changed after seeing test results.
- These are the final, honest results.
"""
    path.write_text(report, encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Phase 7: Final evaluation on the test set."
    )
    parser.add_argument("--split-dir", type=Path, default=Path("data/processed/splits"))
    parser.add_argument("--model-path", type=Path, required=True,
                        help="Path to the Phase 6 model checkpoint (.pt)")
    parser.add_argument("--model-name", type=str, required=True,
                        choices=["mobilenet_v2", "efficientnet_b0"],
                        help="Architecture name used in Phase 6")
    parser.add_argument("--output-dir", type=Path, default=Path("docs/phase7"))
    parser.add_argument("--phase6-dir", type=Path, default=Path("docs/phase6"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Phase 7 Final Evaluation")
    print(f"Model: {args.model_name}")
    print(f"Checkpoint: {args.model_path}")
    print(f"Device: {device}")

    # ------------------------------------------------------------------
    # 1. Load checkpoint
    # ------------------------------------------------------------------
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    class_to_index = checkpoint["class_to_index"]
    image_size = checkpoint.get("image_size", 224)
    val_metrics = checkpoint.get("metrics", {})
    index_to_class = {v: k for k, v in class_to_index.items()}
    labels = [index_to_class[i] for i in range(len(index_to_class))]
    num_classes = len(labels)

    val_acc = val_metrics.get("accuracy", 0.0)
    val_f1 = val_metrics.get("macro_f1", 0.0)

    print(f"Classes: {num_classes}")
    print(f"Image size: {image_size}")
    print(f"Validation accuracy (Phase 6): {val_acc:.4f}")
    print(f"Validation macro F1 (Phase 6): {val_f1:.4f}")

    # ------------------------------------------------------------------
    # 2. Build model and load weights
    # ------------------------------------------------------------------
    model = build_model(args.model_name, num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    print("Model weights loaded.")

    # ------------------------------------------------------------------
    # 3. Load test set
    # ------------------------------------------------------------------
    test_rows = read_manifest(args.split_dir / "test_v1.csv")
    test_count = len(test_rows)
    print(f"Test images: {test_count}")

    test_dataset = WasteImageDataset(test_rows, class_to_index, image_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # ------------------------------------------------------------------
    # 4. Run inference — no training, no gradients
    # ------------------------------------------------------------------
    all_preds = []
    all_actuals = []
    all_paths = []

    with torch.no_grad():
        for images, targets, paths in test_loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_actuals.extend(targets.tolist())
            all_paths.extend(paths)

    actual_labels = [index_to_class[i] for i in all_actuals]
    pred_labels = [index_to_class[i] for i in all_preds]

    # ------------------------------------------------------------------
    # 5. Calculate metrics
    # ------------------------------------------------------------------
    test_acc = accuracy_score(actual_labels, pred_labels)
    macro_prec = precision_score(actual_labels, pred_labels, labels=labels,
                                  average="macro", zero_division=0)
    macro_rec = recall_score(actual_labels, pred_labels, labels=labels,
                              average="macro", zero_division=0)
    macro_f1 = f1_score(actual_labels, pred_labels, labels=labels,
                         average="macro", zero_division=0)

    report = classification_report(actual_labels, pred_labels, labels=labels,
                                    output_dict=True, zero_division=0)
    matrix = confusion_matrix(actual_labels, pred_labels, labels=labels)

    metrics = {
        "test_accuracy": round(test_acc, 4),
        "macro_precision": round(macro_prec, 4),
        "macro_recall": round(macro_rec, 4),
        "macro_f1": round(macro_f1, 4),
        "total_test_images": test_count,
        "model_name": args.model_name,
        "checkpoint_path": str(args.model_path),
        "image_size": image_size,
    }

    print(f"\n--- Final Test Results ---")
    print(f"Test accuracy:    {test_acc:.4f}")
    print(f"Macro precision:  {macro_prec:.4f}")
    print(f"Macro recall:     {macro_rec:.4f}")
    print(f"Macro F1-score:   {macro_f1:.4f}")

    # ------------------------------------------------------------------
    # 6. Save CSV outputs
    # ------------------------------------------------------------------
    write_metrics_csv(args.output_dir / "final_test_metrics_v1.csv", metrics)
    write_class_report_csv(args.output_dir / "final_test_class_report_v1.csv", labels, report)
    write_confusion_matrix_csv(args.output_dir / "final_test_confusion_matrix_v1.csv", labels, matrix)
    write_model_summary_csv(
        args.output_dir / "final_model_summary_v1.csv",
        args.model_name, val_acc, val_f1, test_acc, macro_f1,
    )

    # ------------------------------------------------------------------
    # 7. Save visual outputs
    # ------------------------------------------------------------------
    display_names = {
        "mobilenet_v2": "MobileNetV2",
        "efficientnet_b0": "EfficientNet-B0",
    }
    display_name = display_names.get(args.model_name, args.model_name)

    save_confusion_matrix_image(
        args.output_dir / "images" / "final_test_confusion_matrix_v1.png",
        labels, matrix, f"{display_name} Test Confusion Matrix",
    )
    save_result_summary_chart(
        args.output_dir / "images" / "final_result_summary_v1.png",
        display_name, val_f1, macro_f1,
    )

    # Failure examples: collect wrong predictions.
    wrong = []
    for img_path, true_lbl, pred_lbl in zip(all_paths, actual_labels, pred_labels):
        if true_lbl != pred_lbl:
            wrong.append((img_path, true_lbl, pred_lbl))

    # Shuffle to get variety, then take 12.
    rng = random.Random(args.seed)
    rng.shuffle(wrong)
    save_failure_examples(
        args.output_dir / "images" / "failure_examples_v1.jpg",
        wrong, labels, image_size,
    )

    # ------------------------------------------------------------------
    # 8. Identify confused class pairs
    # ------------------------------------------------------------------
    confused_pairs = find_confused_pairs(labels, matrix, top_n=5)

    # ------------------------------------------------------------------
    # 9. Write the markdown report
    # ------------------------------------------------------------------
    write_report(
        args.output_dir / "phase7_final_evaluation_report_v1.md",
        display_name, str(args.model_path), image_size, test_count,
        metrics, labels, matrix, confused_pairs,
    )

    print(f"\nAll Phase 7 outputs saved to: {args.output_dir}")
    print("Done.")


if __name__ == "__main__":
    main()
