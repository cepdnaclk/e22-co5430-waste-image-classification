import argparse
import csv
from pathlib import Path

import joblib
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def read_manifest(path):
    with path.open() as file:
        return list(csv.DictReader(file))


def get_font(size=14):
    font_paths = [
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


def extract_color_features(image_path, image_size=128, bins=16):
    with Image.open(image_path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        image = image.resize((image_size, image_size))
        array = np.asarray(image, dtype=np.float32)

    features = []

    # These features are simple on purpose. This is our first baseline.
    for channel in range(3):
        channel_values = array[:, :, channel].reshape(-1)
        hist, _ = np.histogram(channel_values, bins=bins, range=(0, 256))
        hist = hist.astype(np.float32)
        hist = hist / max(hist.sum(), 1)
        features.extend(hist)
        features.append(float(channel_values.mean() / 255.0))
        features.append(float(channel_values.std() / 255.0))

    return np.array(features, dtype=np.float32)


def build_features(rows, image_size, bins):
    features = []
    labels = []

    for index, row in enumerate(rows, start=1):
        features.append(extract_color_features(row["image_path"], image_size=image_size, bins=bins))
        labels.append(row["class_name"])

        if index % 1000 == 0:
            print(f"Processed {index} images")

    return np.vstack(features), np.array(labels)


def write_metrics(path, metrics):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["metric", "value"])
        writer.writeheader()
        for metric, value in metrics.items():
            writer.writerow({"metric": metric, "value": value})


def write_class_report(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["class_name", "precision", "recall", "f1_score", "support"]
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for class_name, values in report.items():
            if not isinstance(values, dict) or class_name in {"accuracy", "macro avg", "weighted avg"}:
                continue

            writer.writerow(
                {
                    "class_name": class_name,
                    "precision": round(values["precision"], 4),
                    "recall": round(values["recall"], 4),
                    "f1_score": round(values["f1-score"], 4),
                    "support": int(values["support"]),
                }
            )


def write_confusion_matrix_csv(path, labels, matrix):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["actual/predicted", *labels])
        for label, row in zip(labels, matrix):
            writer.writerow([label, *row.tolist()])


def save_confusion_matrix_image(path, labels, matrix):
    path.parent.mkdir(parents=True, exist_ok=True)

    cell = 48
    left = 150
    top = 130
    width = left + cell * len(labels) + 40
    height = top + cell * len(labels) + 40
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = get_font(22)
    font = get_font(12)
    small_font = get_font(10)

    draw.text((left, 30), "Baseline Validation Confusion Matrix", fill="#111827", font=title_font)
    draw.text((left, 70), "Predicted class", fill="#111827", font=font)
    draw.text((20, top - 30), "Actual class", fill="#111827", font=font)

    max_value = max(int(matrix.max()), 1)

    for index, label in enumerate(labels):
        x = left + index * cell
        draw.text((x + 4, top - 48), label[:10], fill="#111827", font=small_font)
        y = top + index * cell
        draw.text((20, y + 15), label[:14], fill="#111827", font=small_font)

    for row_index, row in enumerate(matrix):
        for col_index, value in enumerate(row):
            intensity = int(255 - (value / max_value) * 210)
            color = (intensity, min(intensity + 10, 255), 255)
            x0 = left + col_index * cell
            y0 = top + row_index * cell
            x1 = x0 + cell
            y1 = y0 + cell
            draw.rectangle([(x0, y0), (x1, y1)], fill=color, outline="#d1d5db")
            if value > 0:
                draw.text((x0 + 14, y0 + 16), str(int(value)), fill="#111827", font=small_font)

    image.save(path)


def get_class_rows(report):
    rows = []

    for class_name, values in report.items():
        if not isinstance(values, dict) or class_name in {"accuracy", "macro avg", "weighted avg"}:
            continue

        rows.append(
            {
                "class_name": class_name,
                "precision": values["precision"],
                "recall": values["recall"],
                "f1_score": values["f1-score"],
                "support": int(values["support"]),
            }
        )

    return rows


def format_class_rows(rows):
    lines = [
        "| Class | Precision | Recall | F1-score | Images |",
        "|---|---:|---:|---:|---:|",
    ]

    for row in rows:
        lines.append(
            f"| {row['class_name']} | {row['precision']:.4f} | {row['recall']:.4f} | "
            f"{row['f1_score']:.4f} | {row['support']} |"
        )

    return "\n".join(lines)


def write_report(path, metrics, train_count, val_count, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    class_rows = get_class_rows(report)
    best_rows = sorted(class_rows, key=lambda row: row["f1_score"], reverse=True)[:3]
    weak_rows = sorted(class_rows, key=lambda row: row["f1_score"])[:3]

    text = f"""# Phase 3 Baseline Report v1

## Phase 3 Goal

In this phase, we trained the first simple baseline model.

A baseline is the first result we use for comparison. Later, MobileNetV2 and EfficientNet-B0 should perform better than this baseline.

## Baseline Method

This baseline uses simple color features from each image.

For every image, the script reads the RGB values and creates color histograms. A color histogram counts how much of each color range appears in the image.

Then it trains a simple linear classifier. This classifier does not understand object shape deeply. It mostly learns color patterns. That is why it is a fair simple starting point.

## Data Used

| Split | Images |
|---|---:|
| Train | {train_count} |
| Validation | {val_count} |

The test set was not used in this phase. We keep it untouched for the final evaluation.

## Result

| Metric | Value |
|---|---:|
| Validation accuracy | {metrics['accuracy']:.4f} |
| Macro precision | {metrics['macro_precision']:.4f} |
| Macro recall | {metrics['macro_recall']:.4f} |
| Macro F1-score | {metrics['macro_f1']:.4f} |

## Per-Class Result

{format_class_rows(class_rows)}

## Strongest Classes

{format_class_rows(best_rows)}

These classes are easier for this baseline because color information gives useful clues.

## Weakest Classes

{format_class_rows(weak_rows)}

These classes need better shape and texture understanding. This is one reason we need CNN and transfer learning models in the next phase.

## What This Means

This result is our starting point. Since this method mainly uses color information, it may confuse classes with similar colors.

Expected weak areas:

- paper vs cardboard
- different glass colors
- plastic vs trash
- object classes where shape matters more than color

## Files Created

```text
docs/phase3/baseline_metrics_v1.csv
docs/phase3/baseline_class_report_v1.csv
docs/phase3/confusion_matrix_v1.csv
docs/phase3/images/confusion_matrix_v1.png
models/baseline_color_hist_v1.joblib
```

The model file is local only and ignored by Git.

## Command Used

```bash
python3 scripts/train_baseline.py --split-dir data/processed/splits --output-dir docs/phase3 --model-path models/baseline_color_hist_v1.joblib
```

## Next Phase

Next, we should train transfer learning models such as MobileNetV2 and EfficientNet-B0 and compare them with this baseline.
"""

    path.write_text(text)


def main():
    parser = argparse.ArgumentParser(description="Train the first simple waste classification baseline.")
    parser.add_argument("--split-dir", type=Path, default=Path("data/processed/splits"))
    parser.add_argument("--output-dir", type=Path, default=Path("docs/phase3"))
    parser.add_argument("--model-path", type=Path, default=Path("models/baseline_color_hist_v1.joblib"))
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--bins", type=int, default=16)
    args = parser.parse_args()

    train_rows = read_manifest(args.split_dir / "train_v1.csv")
    val_rows = read_manifest(args.split_dir / "val_v1.csv")

    print("Building train features")
    x_train, y_train = build_features(train_rows, image_size=args.image_size, bins=args.bins)

    print("Building validation features")
    x_val, y_val = build_features(val_rows, image_size=args.image_size, bins=args.bins)

    model = make_pipeline(
        StandardScaler(),
        SGDClassifier(
            loss="log_loss",
            class_weight="balanced",
            max_iter=2000,
            tol=1e-3,
            random_state=42,
        ),
    )

    print("Training baseline model")
    model.fit(x_train, y_train)

    print("Evaluating baseline model")
    predictions = model.predict(x_val)
    labels = sorted(set(y_train) | set(y_val))
    report = classification_report(y_val, predictions, labels=labels, output_dict=True, zero_division=0)
    matrix = confusion_matrix(y_val, predictions, labels=labels)

    metrics = {
        "accuracy": accuracy_score(y_val, predictions),
        "macro_precision": report["macro avg"]["precision"],
        "macro_recall": report["macro avg"]["recall"],
        "macro_f1": report["macro avg"]["f1-score"],
    }

    write_metrics(args.output_dir / "baseline_metrics_v1.csv", metrics)
    write_class_report(args.output_dir / "baseline_class_report_v1.csv", report)
    write_confusion_matrix_csv(args.output_dir / "confusion_matrix_v1.csv", labels, matrix)
    save_confusion_matrix_image(args.output_dir / "images" / "confusion_matrix_v1.png", labels, matrix)
    write_report(args.output_dir / "phase3_baseline_report_v1.md", metrics, len(train_rows), len(val_rows), report)

    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.model_path)

    print("Baseline complete")
    print(f"Validation accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro F1-score: {metrics['macro_f1']:.4f}")
    print(f"Output folder: {args.output_dir}")
    print(f"Model path: {args.model_path}")


if __name__ == "__main__":
    main()
