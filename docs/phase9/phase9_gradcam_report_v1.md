# Phase 9: Grad-CAM Explainability Report

## What this phase adds

This phase adds Grad-CAM support for the final transfer learning model. Grad-CAM is used to show which image regions influenced the model prediction.

In simple terms, it gives a heatmap over the image. Warm areas mean the model paid more attention to those parts when making the prediction.

## Why this is useful

The final model has strong test results, but the numbers alone do not explain its behaviour. Grad-CAM helps us check whether the model is looking at the actual waste item or at some background detail.

This is useful for:

- explaining correct predictions
- understanding wrong predictions
- checking difficult classes like plastic, white-glass, and metal
- supporting the final report and viva discussion

## Script added

Main file:

- `scripts/generate_gradcam.py`

Run it after the final MobileNetV2 checkpoint is available locally:

```bash
python3 scripts/generate_gradcam.py --split-dir data/processed/splits --model-path models/mobilenet_v2_v1.pt --model-name mobilenet_v2 --output-dir docs/phase9
```

The script creates:

- `docs/phase9/images/gradcam_correct_grid_v1.jpg`
- `docs/phase9/images/gradcam_wrong_grid_v1.jpg`
- individual Grad-CAM images for correct predictions
- individual Grad-CAM images for wrong predictions

## What to look for

For correct predictions, the heatmap should mostly cover the waste object. For example, a battery prediction should focus on the battery body, and a shoe prediction should focus on the shoe shape.

For wrong predictions, the heatmap may focus on confusing parts. For example, white-glass and plastic can both be transparent, so the model may focus on reflections instead of clear object shape.

## Link with Phase 8

Phase 8 showed that the weaker classes are mainly:

- plastic
- white-glass
- metal

So the Grad-CAM examples should include some images from these classes. That will make the error analysis easier to explain in the final report.

## Notes

The model checkpoint is not committed to GitHub because model files are ignored. Keep `models/mobilenet_v2_v1.pt` locally and run the script from the project root.
