import argparse
from pathlib import Path

from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def count_images(class_dir):
    return sum(
        1
        for file_path in class_dir.rglob("*")
        if file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTENSIONS
    )


def find_image_folders(data_dir):
    folders = []

    for item in sorted(data_dir.iterdir()):
        if item.is_dir():
            image_count = count_images(item)
            if image_count > 0:
                folders.append((item.name, image_count))

    return folders


def check_one_image(class_dir):
    for file_path in class_dir.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTENSIONS:
            with Image.open(file_path) as image:
                return image.size

    return None


def main():
    parser = argparse.ArgumentParser(description="Check waste dataset folders.")
    parser.add_argument("--data-dir", type=Path, required=True)
    args = parser.parse_args()

    data_dir = args.data_dir

    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset folder not found: {data_dir}")

    folders = find_image_folders(data_dir)

    if not folders:
        raise ValueError("No class folders with images were found.")

    total_images = sum(count for _, count in folders)

    print("Dataset check")
    print("-------------")
    print(f"Data folder: {data_dir}")
    print(f"Class folders: {len(folders)}")
    print(f"Total images: {total_images}")
    print()

    for class_name, image_count in folders:
        image_size = check_one_image(data_dir / class_name)
        print(f"{class_name:15s} {image_count:5d} images  sample size: {image_size}")


if __name__ == "__main__":
    main()
