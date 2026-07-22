import argparse
import csv
import random
import time
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF
from torchvision import models
from PIL import Image, ImageDraw, ImageEnhance, ImageFont, ImageOps
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader, Dataset


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def read_manifest(path, max_items=None, max_per_class=None, seed=42):
    with path.open() as file:
        rows = list(csv.DictReader(file))

    if max_per_class:
        by_class = {}
        for row in rows:
            by_class.setdefault(row["class_name"], []).append(row)

        sampled = []
        rng = random.Random(seed)
        for class_name in sorted(by_class):
            class_rows = by_class[class_name]
            sampled.extend(rng.sample(class_rows, min(max_per_class, len(class_rows))))
        rows = sampled

    if max_items:
        rows = random.Random(seed).sample(rows, min(max_items, len(rows)))

    return rows


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


def apply_augmentation(image):
    if random.random() < 0.5:
        image = ImageOps.mirror(image)

    if random.random() < 0.7:
        angle = random.uniform(-12, 12)
        image = image.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=(255, 255, 255))

    if random.random() < 0.7:
        brightness = random.uniform(0.85, 1.15)
        contrast = random.uniform(0.85, 1.15)
        image = ImageEnhance.Brightness(image).enhance(brightness)
        image = ImageEnhance.Contrast(image).enhance(contrast)

    return image


class WasteImageDataset(Dataset):
    def __init__(self, rows, class_to_index, image_size, augment=False):
        self.rows = rows
        self.class_to_index = class_to_index
        self.image_size = image_size
        self.augment = augment

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        image_path = row["image_path"]
        class_name = row["class_name"]

        with Image.open(image_path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            image = image.resize((self.image_size, self.image_size))
            if self.augment:
                image = apply_augmentation(image)
            array = np.asarray(image, dtype=np.float32) / 255.0

        # PyTorch expects channels first: C x H x W.
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        # ImageNet normalization
        tensor = TF.normalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        label = self.class_to_index[class_name]

        return tensor, label


def get_transfer_model(model_name, num_classes):
    if model_name == "mobilenet_v2":
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        for param in model.features.parameters():
            param.requires_grad = False
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        for param in model.features.parameters():
            param.requires_grad = False
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return model


def train_one_epoch(model, loader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    predictions = []
    actuals = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = loss_fn(outputs, labels)
            total_loss += loss.item() * labels.size(0)

            predictions.extend(outputs.argmax(dim=1).cpu().tolist())
            actuals.extend(labels.cpu().tolist())

    accuracy = accuracy_score(actuals, predictions)
    return total_loss / len(actuals), accuracy, actuals, predictions


def make_class_weights(rows, labels, class_to_index):
    counts = {label: 0 for label in labels}
    for row in rows:
        counts[row["class_name"]] += 1

    total = sum(counts.values())
    weights = []
    for label in labels:
        weights.append(total / max(counts[label], 1))

    weights = torch.tensor(weights, dtype=torch.float32)
    weights = weights / weights.mean()
    return weights


def write_history(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["epoch", "train_loss", "train_accuracy", "val_loss", "val_accuracy"])
        writer.writeheader()
        writer.writerows(rows)


def write_metrics(path, metrics):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["metric", "value"])
        writer.writeheader()
        for key, value in metrics.items():
            writer.writerow({"metric": key, "value": value})


def write_class_report(path, labels, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["class_name", "precision", "recall", "f1_score", "support"])
        writer.writeheader()
        for label in labels:
            values = report[label]
            writer.writerow({
                "class_name": label,
                "precision": round(values["precision"], 4),
                "recall": round(values["recall"], 4),
                "f1_score": round(values["f1-score"], 4),
                "support": int(values["support"]),
            })


def write_confusion_matrix_csv(path, labels, matrix):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["actual/predicted", *labels])
        for label, row in zip(labels, matrix):
            writer.writerow([label, *row.tolist()])


def save_training_curve(path, history, model_name):
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 800, 480
    left, top = 80, 60
    chart_width, chart_height = 660, 320
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = get_font(22)
    font = get_font(13)

    draw.text((left, 20), f"{model_name} Training Curve", fill="#111827", font=title_font)
    draw.line([(left, top), (left, top + chart_height)], fill="#374151", width=2)
    draw.line([(left, top + chart_height), (left + chart_width, top + chart_height)], fill="#374151", width=2)
    draw.text((left, height - 45), "Epoch", fill="#111827", font=font)
    draw.text((18, top), "Accuracy", fill="#111827", font=font)

    if len(history) == 1:
        x_positions = [left + chart_width // 2]
    else:
        x_positions = [left + int(index * chart_width / (len(history) - 1)) for index in range(len(history))]

    def y_for(value):
        return top + chart_height - int(float(value) * chart_height)

    train_points = [(x, y_for(row["train_accuracy"])) for x, row in zip(x_positions, history)]
    val_points = [(x, y_for(row["val_accuracy"])) for x, row in zip(x_positions, history)]

    if len(train_points) > 1:
        draw.line(train_points, fill="#2563eb", width=3)
        draw.line(val_points, fill="#dc2626", width=3)

    for x, y in train_points:
        draw.ellipse([(x - 4, y - 4), (x + 4, y + 4)], fill="#2563eb")
    for x, y in val_points:
        draw.ellipse([(x - 4, y - 4), (x + 4, y + 4)], fill="#dc2626")

    draw.rectangle([(left + 470, 35), (left + 485, 50)], fill="#2563eb")
    draw.text((left + 495, 34), "train accuracy", fill="#111827", font=font)
    draw.rectangle([(left + 470, 58), (left + 485, 73)], fill="#dc2626")
    draw.text((left + 495, 57), "validation accuracy", fill="#111827", font=font)

    image.save(path)


def save_confusion_matrix_image(path, labels, matrix, model_name):
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

    draw.text((left, 30), f"{model_name} Validation Confusion Matrix", fill="#111827", font=title_font)
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
            color = (255, min(intensity + 20, 255), intensity)
            x0 = left + col_index * cell
            y0 = top + row_index * cell
            x1 = x0 + cell
            y1 = y0 + cell
            draw.rectangle([(x0, y0), (x1, y1)], fill=color, outline="#d1d5db")
            if value > 0:
                draw.text((x0 + 14, y0 + 16), str(int(value)), fill="#111827", font=small_font)

    image.save(path)


def main():
    parser = argparse.ArgumentParser(description="Train transfer learning models for waste image classification.")
    parser.add_argument("--model", type=str, choices=["mobilenet_v2", "efficientnet_b0"], default="mobilenet_v2")
    parser.add_argument("--split-dir", type=Path, default=Path("data/processed/splits"))
    parser.add_argument("--output-dir", type=Path, default=Path("docs/phase6"))
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--learning-rate", type=float, default=0.0003)
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-val", type=int, default=None)
    parser.add_argument("--max-train-per-class", type=int, default=None)
    parser.add_argument("--max-val-per-class", type=int, default=None)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}, Model: {args.model}")

    train_rows = read_manifest(
        args.split_dir / "train_v1.csv",
        max_items=args.max_train,
        max_per_class=args.max_train_per_class,
        seed=args.seed,
    )
    val_rows = read_manifest(
        args.split_dir / "val_v1.csv",
        max_items=args.max_val,
        max_per_class=args.max_val_per_class,
        seed=args.seed,
    )

    labels = sorted({row["class_name"] for row in train_rows + val_rows})
    class_to_index = {label: index for index, label in enumerate(labels)}
    index_to_class = {index: label for label, index in class_to_index.items()}

    train_dataset = WasteImageDataset(train_rows, class_to_index, args.image_size, augment=args.augment)
    val_dataset = WasteImageDataset(val_rows, class_to_index, args.image_size, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = get_transfer_model(args.model, len(labels)).to(device)
    class_weights = make_class_weights(train_rows, labels, class_to_index).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    
    # Train only the classifier parameters since features are frozen
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=args.learning_rate)

    history = []
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        val_loss, val_accuracy, _, _ = evaluate(model, val_loader, loss_fn, device)

        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "train_accuracy": round(train_accuracy, 6),
            "val_loss": round(val_loss, 6),
            "val_accuracy": round(val_accuracy, 6),
        }
        history.append(row)
        print(f"Epoch {epoch}: train_acc={train_accuracy:.4f}, val_acc={val_accuracy:.4f}, val_loss={val_loss:.4f}")

    _, _, actual_indexes, predicted_indexes = evaluate(model, val_loader, loss_fn, device)
    actual = [index_to_class[index] for index in actual_indexes]
    predicted = [index_to_class[index] for index in predicted_indexes]

    report = classification_report(actual, predicted, labels=labels, output_dict=True, zero_division=0)
    matrix = confusion_matrix(actual, predicted, labels=labels)

    metrics = {
        "accuracy": accuracy_score(actual, predicted),
        "macro_precision": report["macro avg"]["precision"],
        "macro_recall": report["macro avg"]["recall"],
        "macro_f1": report["macro avg"]["f1-score"],
        "training_seconds": round(time.time() - start_time, 2),
    }

    base_name = f"{args.model}"
    
    write_history(args.output_dir / f"{base_name}_history_v1.csv", history)
    write_metrics(args.output_dir / f"{base_name}_metrics_v1.csv", metrics)
    write_class_report(args.output_dir / f"{base_name}_class_report_v1.csv", labels, report)
    write_confusion_matrix_csv(args.output_dir / f"{base_name}_confusion_matrix_v1.csv", labels, matrix)
    
    save_confusion_matrix_image(args.output_dir / "images" / f"{base_name}_confusion_matrix_v1.png", labels, matrix, args.model)
    save_training_curve(args.output_dir / "images" / f"{base_name}_training_curve_v1.png", history, args.model)

    model_path = args.models_dir / f"{base_name}_v1.pt"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_to_index": class_to_index,
            "image_size": args.image_size,
            "metrics": metrics,
        },
        model_path,
    )

    print(f"{args.model} training complete")
    print(f"Validation accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro F1-score: {metrics['macro_f1']:.4f}")
    print(f"Training seconds: {metrics['training_seconds']}")
    print(f"Output folder: {args.output_dir}")
    print(f"Model path: {model_path}")

if __name__ == "__main__":
    main()
