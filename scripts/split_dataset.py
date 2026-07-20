import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def collect_images(data_dir):
    images_by_class = defaultdict(list)

    for class_dir in sorted(path for path in data_dir.iterdir() if path.is_dir()):
        for image_path in sorted(class_dir.rglob("*")):
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                images_by_class[class_dir.name].append(image_path)

    return images_by_class


def split_class_images(image_paths, train_ratio, val_ratio, seed):
    shuffled = list(image_paths)
    random.Random(seed).shuffle(shuffled)

    total = len(shuffled)
    val_count = round(total * val_ratio)
    test_count = round(total * (1 - train_ratio - val_ratio))
    train_count = total - val_count - test_count

    train_files = shuffled[:train_count]
    val_files = shuffled[train_count : train_count + val_count]
    test_files = shuffled[train_count + val_count :]

    return {
        "train": train_files,
        "val": val_files,
        "test": test_files,
    }


def write_manifest(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["image_path", "class_name"])
        writer.writeheader()
        writer.writerows(rows)


def write_split_counts(path, split_counts):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["class_name", "total", "train", "val", "test"],
        )
        writer.writeheader()
        writer.writerows(split_counts)


def main():
    parser = argparse.ArgumentParser(description="Create stratified dataset split CSV files.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/splits"))
    parser.add_argument("--report-dir", type=Path, default=Path("docs/phase2"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    args = parser.parse_args()

    if not args.data_dir.exists():
        raise FileNotFoundError(f"Dataset folder not found: {args.data_dir}")

    images_by_class = collect_images(args.data_dir)

    if not images_by_class:
        raise ValueError("No class folders with images were found.")

    manifests = {
        "train": [],
        "val": [],
        "test": [],
    }
    split_counts = []

    for class_name, image_paths in sorted(images_by_class.items()):
        class_splits = split_class_images(
            image_paths=image_paths,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            seed=args.seed,
        )

        for split_name, split_files in class_splits.items():
            for image_path in split_files:
                manifests[split_name].append(
                    {
                        "image_path": str(image_path),
                        "class_name": class_name,
                    }
                )

        split_counts.append(
            {
                "class_name": class_name,
                "total": len(image_paths),
                "train": len(class_splits["train"]),
                "val": len(class_splits["val"]),
                "test": len(class_splits["test"]),
            }
        )

    for split_name, rows in manifests.items():
        rows.sort(key=lambda row: (row["class_name"], row["image_path"]))
        write_manifest(args.output_dir / f"{split_name}_v1.csv", rows)

    write_split_counts(args.output_dir / "split_counts_v1.csv", split_counts)
    write_split_counts(args.report_dir / "split_counts_v1.csv", split_counts)

    print("Dataset split complete")
    print(f"Classes: {len(split_counts)}")
    print(f"Train images: {len(manifests['train'])}")
    print(f"Validation images: {len(manifests['val'])}")
    print(f"Test images: {len(manifests['test'])}")
    print(f"Split files: {args.output_dir}")


if __name__ == "__main__":
    main()
