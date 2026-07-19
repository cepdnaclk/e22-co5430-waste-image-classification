# Project Plan v1

## Project Summary

We are building an image classification system for household waste. The system takes one waste image as input and predicts one of 12 waste classes.

This is an application track project because it applies computer vision to an environmental problem. It also includes a small model comparison because we compare a baseline with fine-tuned pretrained models.

## Main Goal

Build a working and evaluated waste image classifier.

The project should not only give a prediction. It should also show how well the model works, where it fails, and why some classes are difficult.

## Minimum Goal

- Download and organize the dataset.
- Check class names and class counts.
- Create train, validation, and test splits.
- Train a simple baseline model.
- Report accuracy, F1-score, and confusion matrix.

## Expected Goal

- Fine-tune MobileNetV2.
- Fine-tune EfficientNet-B0.
- Compare both models with the baseline.
- Run an augmentation comparison.
- Show correct predictions and wrong predictions.

## Stretch Goal

- Add Grad-CAM heatmaps.
- Build a small image upload demo.

## Work Breakdown

### Phase 1 - Setup and Dataset

Prepare the repository, download the dataset, inspect class folders, and make sure the labels are correct.

Output:

- Clean repository structure.
- Dataset class count table.
- Notes about dataset source and license.

### Phase 2 - Baseline Model

Train a simple first model. This gives us a result to compare against.

Output:

- Baseline training script.
- Baseline result table.
- Baseline confusion matrix.

### Phase 3 - Improved Models

Use pretrained MobileNetV2 and EfficientNet-B0. Replace the final layer so each model predicts our 12 waste classes.

Output:

- MobileNetV2 results.
- EfficientNet-B0 results.
- Comparison table.

### Phase 4 - Augmentation Study

Train a model with and without image augmentation. This shows whether augmentation helps the model learn better.

Output:

- Augmentation comparison table.
- Short explanation of the result.

### Phase 5 - Error Analysis

Look at wrong predictions and common class confusion.

Output:

- Confusion matrix.
- Example failure images.
- Explanation of difficult classes.

### Phase 6 - Grad-CAM

Create visual heatmaps to show which parts of the image the model used for prediction.

Output:

- Heatmaps for correct predictions.
- Heatmaps for wrong predictions.

### Phase 7 - Demo

Build a small app only after the main experiments are ready.

Output:

- Image upload demo.
- Prediction class and confidence score.

## Branch Plan

Use simple branch names:

- `main`
- `setup-v1`
- `dataset-v1`
- `baseline-v1`
- `models-v1`
- `evaluation-v1`
- `gradcam-v1`
- `app-v1`

## File Naming Style

Use short technical names:

- `check_dataset.py`
- `split_dataset.py`
- `train_baseline.py`
- `train_transfer.py`
- `evaluate_model.py`
- `make_gradcam.py`
- `app.py`

## Important Rules

- Do not commit the full dataset.
- Do not commit large model files.
- Keep code simple and explainable.
- Add short comments only where they help.
- Record every result clearly.
- Cite the dataset, papers, pretrained models, and AI use.
