"""Phase 8: Error analysis from the final test results.

This script reads the Phase 7 CSV files and creates small tables,
charts, and a short report about the weak classes and common mistakes.
"""

import argparse
import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


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


def read_class_report(path):
    with path.open() as file:
        rows = list(csv.DictReader(file))

    for row in rows:
        row["precision"] = float(row["precision"])
        row["recall"] = float(row["recall"])
        row["f1_score"] = float(row["f1_score"])
        row["support"] = int(row["support"])
    return rows


def read_confusion_matrix(path):
    with path.open() as file:
        rows = list(csv.reader(file))

    labels = rows[0][1:]
    matrix = []
    for row in rows[1:]:
        matrix.append([int(value) for value in row[1:]])
    return labels, matrix


def read_metrics(path):
    with path.open() as file:
        return {row["metric"]: row["value"] for row in csv.DictReader(file)}


def write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def find_confused_pairs(labels, matrix):
    pairs = []
    for actual_index, actual in enumerate(labels):
        for pred_index, predicted in enumerate(labels):
            count = matrix[actual_index][pred_index]
            if actual != predicted and count > 0:
                pairs.append({
                    "actual_class": actual,
                    "predicted_class": predicted,
                    "count": count,
                })
    pairs.sort(key=lambda item: item["count"], reverse=True)
    return pairs


def save_f1_chart(path, class_rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(class_rows, key=lambda item: item["f1_score"])
    width = 900
    height = 520
    left = 160
    top = 70
    bar_h = 24
    gap = 12
    max_w = 620

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = get_font(22)
    font = get_font(13)

    draw.text((left, 25), "Per-class F1-score on test set", fill="#111827", font=title_font)

    for index, row in enumerate(rows):
        y = top + index * (bar_h + gap)
        f1 = row["f1_score"]
        bar_w = int(f1 * max_w)
        color = "#dc2626" if f1 < 0.75 else "#f59e0b" if f1 < 0.85 else "#16a34a"
        draw.text((30, y + 4), row["class_name"], fill="#111827", font=font)
        draw.rectangle([(left, y), (left + bar_w, y + bar_h)], fill=color)
        draw.text((left + bar_w + 10, y + 4), f"{f1:.4f}", fill="#111827", font=font)

    image.save(path)


def save_confusion_chart(path, pairs):
    path.parent.mkdir(parents=True, exist_ok=True)
    top_pairs = pairs[:8]
    width = 900
    height = 430
    left = 260
    top = 70
    bar_h = 26
    gap = 16
    max_w = 520
    max_count = max([pair["count"] for pair in top_pairs] or [1])

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = get_font(22)
    font = get_font(13)

    draw.text((left, 25), "Most common wrong predictions", fill="#111827", font=title_font)

    for index, pair in enumerate(top_pairs):
        y = top + index * (bar_h + gap)
        label = f"{pair['actual_class']} -> {pair['predicted_class']}"
        bar_w = int(pair["count"] / max_count * max_w)
        draw.text((30, y + 5), label, fill="#111827", font=font)
        draw.rectangle([(left, y), (left + bar_w, y + bar_h)], fill="#2563eb")
        draw.text((left + bar_w + 10, y + 5), str(pair["count"]), fill="#111827", font=font)

    image.save(path)


def explain_pair(actual, predicted):
    explanations = {
        ("white-glass", "plastic"): "both can be transparent or shiny in photos",
        ("plastic", "metal"): "some plastic and metal items have similar shapes and reflections",
        ("metal", "plastic"): "both can appear smooth and bright under indoor lighting",
        ("plastic", "paper"): "thin plastic wrappers can look like paper sheets",
        ("shoes", "biological"): "some shoe photos contain soil-like colors or textured backgrounds",
        ("paper", "cardboard"): "both are flat fibre-based materials",
        ("cardboard", "paper"): "both are flat fibre-based materials",
    }
    return explanations.get((actual, predicted), "these classes share some visual details")


def write_report(path, metrics, weak_rows, confused_pairs):
    path.parent.mkdir(parents=True, exist_ok=True)
    weakest_text = "\n".join(
        f"- **{row['class_name']}**: F1-score {row['f1_score']:.4f}, "
        f"precision {row['precision']:.4f}, recall {row['recall']:.4f}"
        for row in weak_rows[:5]
    )
    pair_text = "\n".join(
        f"- **{pair['actual_class']}** predicted as **{pair['predicted_class']}** "
        f"({pair['count']} images): {explain_pair(pair['actual_class'], pair['predicted_class'])}."
        for pair in confused_pairs[:5]
    )

    report = f"""# Phase 8: Error Analysis Report

## What this phase checks

This phase studies the final test results from Phase 7. The aim is to understand which classes are weak and why the model makes some wrong predictions.

The model used here is the final MobileNetV2 model selected in Phase 7.

## Final result used

| Metric | Value |
|---|---:|
| Test accuracy | {metrics.get('test_accuracy', '')} |
| Macro precision | {metrics.get('macro_precision', '')} |
| Macro recall | {metrics.get('macro_recall', '')} |
| Macro F1-score | {metrics.get('macro_f1', '')} |
| Test images | {metrics.get('total_test_images', '')} |

## Weakest classes

These classes have the lowest F1-scores. F1-score is useful here because it balances precision and recall.

{weakest_text}

![Per-class F1-score](images/per_class_f1_v1.png)

## Main confusion patterns

These are the most common wrong predictions in the test set.

{pair_text}

![Top confusion pairs](images/top_confusion_pairs_v1.png)

## What we learned

- **Plastic** is the hardest class. It appears in many shapes, colors, and materials, so it overlaps with metal, paper, glass, and trash.
- **White-glass** is difficult because transparent glass can look like transparent plastic.
- **Metal** can be confused with plastic when the object has a smooth or shiny surface.
- **Paper and cardboard** are visually close because both have flat fibre-like textures.
- **Trash** is not one clear object type. It is a mixed class, so some mistakes are expected.

## How this helps the next steps

This analysis shows that the model is already strong overall, but it still needs explanation for difficult classes. The next phase uses Grad-CAM to check whether the model looks at the waste object or at the background when it predicts a class.
"""
    path.write_text(report, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Create Phase 8 error analysis outputs.")
    parser.add_argument("--phase7-dir", type=Path, default=Path("docs/phase7"))
    parser.add_argument("--output-dir", type=Path, default=Path("docs/phase8"))
    args = parser.parse_args()

    class_rows = read_class_report(args.phase7_dir / "final_test_class_report_v1.csv")
    labels, matrix = read_confusion_matrix(args.phase7_dir / "final_test_confusion_matrix_v1.csv")
    metrics = read_metrics(args.phase7_dir / "final_test_metrics_v1.csv")

    weak_rows = sorted(class_rows, key=lambda item: item["f1_score"])
    confused_pairs = find_confused_pairs(labels, matrix)

    weak_output = [
        {
            "class_name": row["class_name"],
            "precision": f"{row['precision']:.4f}",
            "recall": f"{row['recall']:.4f}",
            "f1_score": f"{row['f1_score']:.4f}",
            "support": row["support"],
        }
        for row in weak_rows
    ]
    pair_output = [
        {
            "actual_class": pair["actual_class"],
            "predicted_class": pair["predicted_class"],
            "count": pair["count"],
            "reason": explain_pair(pair["actual_class"], pair["predicted_class"]),
        }
        for pair in confused_pairs
    ]

    write_csv(
        args.output_dir / "weak_classes_v1.csv",
        ["class_name", "precision", "recall", "f1_score", "support"],
        weak_output,
    )
    write_csv(
        args.output_dir / "top_confusions_v1.csv",
        ["actual_class", "predicted_class", "count", "reason"],
        pair_output,
    )
    save_f1_chart(args.output_dir / "images" / "per_class_f1_v1.png", weak_rows)
    save_confusion_chart(args.output_dir / "images" / "top_confusion_pairs_v1.png", confused_pairs)
    write_report(
        args.output_dir / "phase8_error_analysis_report_v1.md",
        metrics,
        weak_rows,
        confused_pairs,
    )

    print(f"Phase 8 outputs saved to {args.output_dir}")


if __name__ == "__main__":
    main()
