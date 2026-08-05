# Phase 9: Grad-CAM Explainability Report

## What this phase does

This phase uses Grad-CAM to explain the final MobileNetV2 predictions. Grad-CAM creates a heatmap over the image. The warmer areas show the parts that affected the model decision more strongly.

## Outputs created

- Correct prediction examples: 6
- Wrong prediction examples: 6
- Correct prediction grid: `images/gradcam_correct_grid_v1.jpg`
- Wrong prediction grid: `images/gradcam_wrong_grid_v1.jpg`

## How to read the images

If the warm area is on the waste item, the model is using useful visual evidence. If the warm area is on the background, label text, or another object, the prediction is less reliable.

## What we expect to learn

- Correct predictions should mostly focus on the object shape, color, and texture.
- Wrong predictions may focus on shared visual parts, such as shiny surfaces in plastic, glass, and metal.
- These examples help explain why the final model performs well overall but still struggles on plastic, white-glass, and metal.
