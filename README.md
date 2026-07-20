# Waste Image Classification

Group G20 - Project P04

This project classifies household waste images into 12 waste classes. The goal is to build a clear computer vision pipeline, compare a simple baseline with improved models, and explain the results with proper evaluation.

## Team

- E/22/051
- E/22/385
- E/22/227
- E/22/004

## Dataset

We use the Kaggle Garbage Classification dataset:

https://www.kaggle.com/datasets/mostafaabla/garbage-classification

The Kaggle page describes the dataset as a 12-class household waste image dataset. The local copy used in this project has 15,515 readable JPG images.

- paper
- cardboard
- biological
- metal
- plastic
- green-glass
- brown-glass
- white-glass
- clothes
- shoes
- battery
- trash

The full dataset is not committed to this repository. Download it from Kaggle and place it inside:

```text
data/raw/
```

## Project Plan

We will do the project in this order:

1. Inspect the dataset and check class counts.
2. Split images into train, validation, and test sets.
3. Train a simple baseline model.
4. Fine-tune MobileNetV2 and EfficientNet-B0.
5. Compare results with accuracy, precision, recall, F1-score, and confusion matrix.
6. Study the effect of data augmentation.
7. Generate Grad-CAM examples for correct and wrong predictions.
8. Build a small image upload demo if the main experiments are complete.

## Folder Structure

```text
data/              Dataset files. Not committed.
docs/              Project notes and planning.
notebooks/         Small exploration notebooks.
src/               Main Python source code.
scripts/           Command line scripts.
results/           Tables, plots, and example outputs.
models/            Trained model files. Not committed.
reports/           Report drafts and final writing.
app/               Simple demo app later.
```

## Setup

Create a Python environment and install the packages:

```bash
pip install -r requirements.txt
```

## First Check

After downloading the dataset, run:

```bash
python scripts/check_dataset.py --data-dir data/raw
```

This prints the class names and image counts. It helps us confirm that the dataset is in the expected format.

## Phase 1 Outputs

Phase 1 prepares and checks the dataset before model training.

Main files:

- `docs/phase1/phase1_learning_report_v1.md`
- `docs/phase1/dataset_audit_table_v1.md`
- `docs/phase1/class_counts_v1.csv`
- `docs/phase1/planned_split_counts_v1.csv`
- `docs/phase1/images/class_counts_v1.png`
- `docs/phase1/images/image_sizes_v1.png`
- `docs/phase1/images/sample_grid_v1.jpg`

Run the full Phase 1 audit with:

```bash
python scripts/audit_dataset.py --data-dir data/raw --output-dir docs/phase1
```

## Phase 2 Outputs

Phase 2 creates a stratified train, validation, and test split.

Main files:

- `scripts/split_dataset.py`
- `docs/phase2/phase2_split_report_v1.md`
- `docs/phase2/split_counts_v1.csv`

Run the split with:

```bash
python scripts/split_dataset.py --data-dir data/raw --output-dir data/processed/splits --report-dir docs/phase2 --seed 42
```

## AI Use Note

AI tools may be used for planning, writing support, code suggestions, and debugging. All code and results must be reviewed and understood by the team.
