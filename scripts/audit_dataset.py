import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def get_font(size=16):
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


def image_files(data_dir):
    files = []
    for class_dir in sorted(path for path in data_dir.iterdir() if path.is_dir()):
        for file_path in sorted(class_dir.rglob("*")):
            if file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTENSIONS:
                files.append((class_dir.name, file_path))
    return files


def inspect_images(files):
    rows = []
    bad_files = []

    for class_name, file_path in files:
        try:
            with Image.open(file_path) as image:
                width, height = image.size
                mode = image.mode
                image.verify()
        except Exception as error:
            bad_files.append(
                {
                    "class_name": class_name,
                    "file_path": str(file_path),
                    "error": str(error),
                }
            )
            continue

        rows.append(
            {
                "class_name": class_name,
                "file_path": str(file_path),
                "file_name": file_path.name,
                "extension": file_path.suffix.lower(),
                "width": width,
                "height": height,
                "mode": mode,
            }
        )

    return rows, bad_files


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_class_table(rows):
    by_class = defaultdict(list)
    for row in rows:
        by_class[row["class_name"]].append(row)

    table = []
    for class_name in sorted(by_class):
        class_rows = by_class[class_name]
        widths = [row["width"] for row in class_rows]
        heights = [row["height"] for row in class_rows]
        table.append(
            {
                "class_name": class_name,
                "image_count": len(class_rows),
                "min_width": min(widths),
                "max_width": max(widths),
                "min_height": min(heights),
                "max_height": max(heights),
            }
        )

    return table


def make_split_plan(class_table):
    split_rows = []

    for row in class_table:
        total = row["image_count"]
        val_count = round(total * 0.15)
        test_count = round(total * 0.15)
        train_count = total - val_count - test_count
        split_rows.append(
            {
                "class_name": row["class_name"],
                "total": total,
                "train_70": train_count,
                "val_15": val_count,
                "test_15": test_count,
            }
        )

    return split_rows


def plot_class_counts(class_table, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sorted_table = sorted(class_table, key=lambda row: row["image_count"])
    width, height = 1000, 720
    margin_left, margin_right = 150, 90
    margin_top, margin_bottom = 70, 45
    chart_width = width - margin_left - margin_right
    chart_height = height - margin_top - margin_bottom

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = get_font(26)
    font = get_font(16)

    max_count = max(row["image_count"] for row in sorted_table)
    bar_gap = 10
    bar_height = (chart_height - bar_gap * (len(sorted_table) - 1)) / len(sorted_table)

    draw.text((margin_left, 25), "Images per Waste Class", fill="#111827", font=title_font)
    draw.line([(margin_left, margin_top), (margin_left, margin_top + chart_height)], fill="#374151", width=2)

    for index, row in enumerate(sorted_table):
        y0 = margin_top + index * (bar_height + bar_gap)
        x0 = margin_left
        x1 = margin_left + chart_width * row["image_count"] / max_count
        y1 = y0 + bar_height

        draw.rectangle([(x0, y0), (x1, y1)], fill="#2f80ed")
        draw.text((25, y0 + 6), row["class_name"], fill="#111827", font=font)
        draw.text((x1 + 8, y0 + 6), str(row["image_count"]), fill="#111827", font=font)

    image.save(output_path)


def plot_image_sizes(rows, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    width, height = 900, 700
    margin_left, margin_right = 80, 40
    margin_top, margin_bottom = 70, 80
    chart_width = width - margin_left - margin_right
    chart_height = height - margin_top - margin_bottom

    max_width = max(row["width"] for row in rows)
    max_height = max(row["height"] for row in rows)

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = get_font(26)
    font = get_font(15)

    draw.text((margin_left, 25), "Original Image Sizes", fill="#111827", font=title_font)
    draw.line(
        [(margin_left, margin_top + chart_height), (width - margin_right, margin_top + chart_height)],
        fill="#374151",
        width=2,
    )
    draw.line([(margin_left, margin_top), (margin_left, margin_top + chart_height)], fill="#374151", width=2)
    draw.text((margin_left, height - 35), "Width", fill="#111827", font=font)
    draw.text((15, margin_top), "Height", fill="#111827", font=font)

    colors = [
        "#2563eb",
        "#16a34a",
        "#dc2626",
        "#9333ea",
        "#f97316",
        "#0891b2",
        "#4b5563",
        "#db2777",
        "#65a30d",
        "#7c2d12",
        "#0f766e",
        "#ca8a04",
    ]
    color_by_class = {
        class_name: colors[index % len(colors)]
        for index, class_name in enumerate(sorted({row["class_name"] for row in rows}))
    }

    for row in rows:
        x = margin_left + (row["width"] / max_width) * chart_width
        y = margin_top + chart_height - (row["height"] / max_height) * chart_height
        draw.ellipse([(x - 2, y - 2), (x + 2, y + 2)], fill=color_by_class[row["class_name"]])

    legend_x = margin_left + 20
    legend_y = margin_top + 15
    for class_name, color in color_by_class.items():
        draw.rectangle([(legend_x, legend_y), (legend_x + 10, legend_y + 10)], fill=color)
        draw.text((legend_x + 16, legend_y), class_name, fill="#111827", font=font)
        legend_y += 17

    image.save(output_path)


def make_sample_grid(rows, output_path, samples_per_class=2, seed=42):
    random.seed(seed)
    classes = sorted({row["class_name"] for row in rows})
    selected = []

    for class_name in classes:
        class_rows = [row for row in rows if row["class_name"] == class_name]
        selected.extend(random.sample(class_rows, min(samples_per_class, len(class_rows))))

    thumb_size = 150
    label_height = 26
    columns = 6
    rows_count = math.ceil(len(selected) / columns)
    canvas = Image.new(
        "RGB",
        (columns * thumb_size, rows_count * (thumb_size + label_height)),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    font = get_font(13)

    for index, row in enumerate(selected):
        x = (index % columns) * thumb_size
        y = (index // columns) * (thumb_size + label_height)

        with Image.open(row["file_path"]) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            image.thumbnail((thumb_size, thumb_size))
            tile = Image.new("RGB", (thumb_size, thumb_size), "white")
            tile_x = (thumb_size - image.width) // 2
            tile_y = (thumb_size - image.height) // 2
            tile.paste(image, (tile_x, tile_y))

        canvas.paste(tile, (x, y))

        # The class label is useful when we quickly inspect the sample grid.
        label_image = Image.new("RGB", (thumb_size, label_height), "#f3f4f6")
        canvas.paste(label_image, (x, y + thumb_size))
        draw.text((x + 6, y + thumb_size + 6), row["class_name"], fill="#111827", font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=95)


def write_markdown_table(path, class_table, total_images, bad_files, extension_counts, mode_counts):
    split_plan = make_split_plan(class_table)
    lines = [
        "# Dataset Audit Table v1",
        "",
        f"Total readable images: {total_images}",
        f"Unreadable image files: {len(bad_files)}",
        "",
        "## Class Counts",
        "",
        "| Class | Images | Width Range | Height Range |",
        "|---|---:|---:|---:|",
    ]

    for row in class_table:
        lines.append(
            f"| {row['class_name']} | {row['image_count']} | "
            f"{row['min_width']}-{row['max_width']} | {row['min_height']}-{row['max_height']} |"
        )

    lines.extend(
        [
            "",
            "## File Extensions",
            "",
            "| Extension | Count |",
            "|---|---:|",
        ]
    )

    for extension, count in sorted(extension_counts.items()):
        lines.append(f"| {extension} | {count} |")

    lines.extend(
        [
            "",
            "## Image Modes",
            "",
            "| Mode | Count |",
            "|---|---:|",
        ]
    )

    for mode, count in sorted(mode_counts.items()):
        lines.append(f"| {mode} | {count} |")

    lines.extend(
        [
            "",
            "## Planned 70:15:15 Split Counts",
            "",
            "| Class | Total | Train | Validation | Test |",
            "|---|---:|---:|---:|---:|",
        ]
    )

    for row in split_plan:
        lines.append(
            f"| {row['class_name']} | {row['total']} | {row['train_70']} | "
            f"{row['val_15']} | {row['test_15']} |"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Create a Phase 1 dataset audit.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("docs/phase1"))
    args = parser.parse_args()

    files = image_files(args.data_dir)
    rows, bad_files = inspect_images(files)
    class_table = make_class_table(rows)
    split_plan = make_split_plan(class_table)

    extension_counts = Counter(row["extension"] for row in rows)
    mode_counts = Counter(row["mode"] for row in rows)

    write_csv(
        args.output_dir / "dataset_inventory_v1.csv",
        rows,
        ["class_name", "file_path", "file_name", "extension", "width", "height", "mode"],
    )
    write_csv(
        args.output_dir / "class_counts_v1.csv",
        class_table,
        ["class_name", "image_count", "min_width", "max_width", "min_height", "max_height"],
    )
    write_csv(
        args.output_dir / "planned_split_counts_v1.csv",
        split_plan,
        ["class_name", "total", "train_70", "val_15", "test_15"],
    )
    write_csv(
        args.output_dir / "bad_files_v1.csv",
        bad_files,
        ["class_name", "file_path", "error"],
    )

    plot_class_counts(class_table, args.output_dir / "images" / "class_counts_v1.png")
    plot_image_sizes(rows, args.output_dir / "images" / "image_sizes_v1.png")
    make_sample_grid(rows, args.output_dir / "images" / "sample_grid_v1.jpg")

    write_markdown_table(
        args.output_dir / "dataset_audit_table_v1.md",
        class_table,
        len(rows),
        bad_files,
        extension_counts,
        mode_counts,
    )

    summary = {
        "data_dir": str(args.data_dir),
        "class_count": len(class_table),
        "total_readable_images": len(rows),
        "bad_file_count": len(bad_files),
        "extensions": dict(extension_counts),
        "image_modes": dict(mode_counts),
    }
    (args.output_dir / "dataset_summary_v1.json").write_text(json.dumps(summary, indent=2))

    print("Dataset audit complete")
    print(f"Classes: {len(class_table)}")
    print(f"Readable images: {len(rows)}")
    print(f"Unreadable files: {len(bad_files)}")
    print(f"Output folder: {args.output_dir}")


if __name__ == "__main__":
    main()
