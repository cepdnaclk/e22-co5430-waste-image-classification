# Phase 3 Baseline Report v1

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
| Train | 10857 |
| Validation | 2329 |

The test set was not used in this phase. We keep it untouched for the final evaluation.

## Result

| Metric | Value |
|---|---:|
| Validation accuracy | 0.4848 |
| Macro precision | 0.3870 |
| Macro recall | 0.4330 |
| Macro F1-score | 0.3928 |

## Per-Class Result

| Class | Precision | Recall | F1-score | Images |
|---|---:|---:|---:|---:|
| battery | 0.3345 | 0.6620 | 0.4444 | 142 |
| biological | 0.4958 | 0.3986 | 0.4419 | 148 |
| brown-glass | 0.5400 | 0.5934 | 0.5654 | 91 |
| cardboard | 0.3081 | 0.4851 | 0.3768 | 134 |
| clothes | 0.7393 | 0.7347 | 0.7370 | 799 |
| green-glass | 0.4792 | 0.7340 | 0.5798 | 94 |
| metal | 0.3269 | 0.1478 | 0.2036 | 115 |
| paper | 0.3228 | 0.2595 | 0.2877 | 158 |
| plastic | 0.3772 | 0.3308 | 0.3525 | 130 |
| shoes | 0.1667 | 0.0438 | 0.0693 | 297 |
| trash | 0.3851 | 0.5905 | 0.4662 | 105 |
| white-glass | 0.1689 | 0.2155 | 0.1894 | 116 |

## Strongest Classes

| Class | Precision | Recall | F1-score | Images |
|---|---:|---:|---:|---:|
| clothes | 0.7393 | 0.7347 | 0.7370 | 799 |
| green-glass | 0.4792 | 0.7340 | 0.5798 | 94 |
| brown-glass | 0.5400 | 0.5934 | 0.5654 | 91 |

These classes are easier for this baseline because color information gives useful clues.

## Weakest Classes

| Class | Precision | Recall | F1-score | Images |
|---|---:|---:|---:|---:|
| shoes | 0.1667 | 0.0438 | 0.0693 | 297 |
| white-glass | 0.1689 | 0.2155 | 0.1894 | 116 |
| metal | 0.3269 | 0.1478 | 0.2036 | 115 |

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
