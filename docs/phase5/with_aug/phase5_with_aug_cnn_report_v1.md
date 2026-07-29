# Phase 5 CNN With Augmentation Report v1

## Goal

In this phase, we trained the small CNN model with augmentation.

The Phase 3 baseline used only color histograms. This CNN learns directly from image pixels, so it can learn simple shape and texture patterns too.

## Model

The model has three convolution blocks. Each block looks for image patterns and then reduces the image size. At the end, a small classifier predicts one of the 12 waste classes.

This is still a lightweight model. It is not MobileNetV2 or EfficientNet-B0 yet, because `torchvision` is not available in the local environment. This script gives us a working deep learning model first. MobileNetV2 and EfficientNet-B0 can be added in the next phase when `torchvision` or pretrained weights are available.

## Data Used

| Split | Images |
|---|---:|
| Train | 4200 |
| Validation | 1080 |

The test set was not used.

## Training Settings

| Setting | Value |
|---|---:|
| Epochs | 8 |
| Batch size | 64 |
| Image size | 96 |
| Learning rate | 0.001 |
| Max train images | full train split |
| Max validation images | full validation split |
| Max train images per class | 350 |
| Max validation images per class | 90 |
| Augmentation | True |

## Result

| Metric | Value |
|---|---:|
| Validation accuracy | 0.4315 |
| Macro precision | 0.3902 |
| Macro recall | 0.4315 |
| Macro F1-score | 0.3888 |

## Baseline Comparison

| Model | Validation accuracy | Macro F1-score |
|---|---:|---:|
| Color histogram baseline | 0.4848 | 0.3928 |
| Small CNN | 0.4315 | 0.3888 |

The CNN has lower accuracy than the color baseline, but it slightly improves macro F1-score. Macro F1-score is important here because the dataset is imbalanced and we want each class to matter.

## Simple Explanation

This model is better than the color baseline in method, because it sees image structure and not only color counts.

The score is still not high. That means the model needs more training time, pretrained weights, augmentation, or a stronger architecture.

## Files Created

```text
docs/phase5/with_aug/cnn_metrics_v1.csv
docs/phase5/with_aug/cnn_class_report_v1.csv
docs/phase5/with_aug/cnn_history_v1.csv
docs/phase5/with_aug/cnn_confusion_matrix_v1.csv
docs/phase5/with_aug/images/cnn_confusion_matrix_v1.png
docs/phase5/with_aug/images/cnn_training_curve_v1.png
models/small_cnn_aug_v1.pt
```

The model file is local only and ignored by Git.

## Reference Note

The CNN structure follows the usual PyTorch style of defining an `nn.Module`, using convolution layers, training with a loss function, and evaluating predictions. Useful official references:

- https://docs.pytorch.org/tutorials/recipes/recipes/defining_a_neural_network.html
- https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
