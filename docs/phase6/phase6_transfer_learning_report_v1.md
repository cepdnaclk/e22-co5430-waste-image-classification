# Phase 6: Transfer Learning Models Report v1

## Goal

In this phase, we trained and compared stronger transfer learning models for waste image classification. Transfer learning means using a model that already learned general image patterns from a large dataset like ImageNet. We replace the final layer so it predicts our 12 waste classes instead of its original classes.

## Models Trained

We utilized pretrained weights from `torchvision` and trained the following models:
- **MobileNetV2**: A lightweight and fast CNN architecture.
- **EfficientNet-B0**: A well-balanced CNN model that scales efficiently.

For each model, the base features were frozen and only the final classification layer was trained to prevent destroying the pretrained weights with large initial loss gradients.

## Evaluation and Comparison

The test set was **not** used during Phase 6. All models were evaluated strictly on the validation set. 

Here is the comparison of our new models against previous baselines:

| Model | Validation accuracy | Macro F1-score |
|---|---:|---:|
| Color histogram baseline | 0.4848 | 0.3928 |
| Small CNN | 0.4343 | 0.3950 |
| Small CNN with augmentation | 0.4407 | 0.3992 |
| MobileNetV2 (Transfer Learning) | 0.8927 | 0.8929 |
| EfficientNet-B0 (Transfer Learning) | 0.8913 | 0.8909 |

Both transfer learning models significantly outperform the simple CNN models because they leverage features learned from millions of real-world images. MobileNetV2 slightly edges out EfficientNet-B0 in macro F1-score.

## Files Created

```text
docs/phase6/transfer_model_comparison_v1.csv
docs/phase6/images/transfer_model_comparison_v1.png
docs/phase6/mobilenet_v2_metrics_v1.csv
docs/phase6/mobilenet_v2_class_report_v1.csv
docs/phase6/mobilenet_v2_confusion_matrix_v1.csv
docs/phase6/images/mobilenet_v2_confusion_matrix_v1.png
docs/phase6/images/mobilenet_v2_training_curve_v1.png
docs/phase6/efficientnet_b0_metrics_v1.csv
docs/phase6/efficientnet_b0_class_report_v1.csv
docs/phase6/efficientnet_b0_confusion_matrix_v1.csv
docs/phase6/images/efficientnet_b0_confusion_matrix_v1.png
docs/phase6/images/efficientnet_b0_training_curve_v1.png
```
