# Phase 8: Error Analysis Report

## What this phase checks

This phase studies the final test results from Phase 7. The aim is to understand which classes are weak and why the model makes some wrong predictions.

The model used here is the final MobileNetV2 model selected in Phase 7.

## Final result used

| Metric | Value |
|---|---:|
| Test accuracy | 0.9038 |
| Macro precision | 0.8652 |
| Macro recall | 0.8688 |
| Macro F1-score | 0.8648 |
| Test images | 2329 |

## Weakest classes

These classes have the lowest F1-scores. F1-score is useful here because it balances precision and recall.

- **plastic**: F1-score 0.6859, precision 0.6463, recall 0.7308
- **white-glass**: F1-score 0.7291, precision 0.8506, recall 0.6379
- **metal**: F1-score 0.7467, precision 0.7636, recall 0.7304
- **paper**: F1-score 0.8589, precision 0.8333, recall 0.8861
- **brown-glass**: F1-score 0.8729, precision 0.8778, recall 0.8681

![Per-class F1-score](images/per_class_f1_v1.png)

## Main confusion patterns

These are the most common wrong predictions in the test set.

- **white-glass** predicted as **plastic** (28 images): both can be transparent or shiny in photos.
- **plastic** predicted as **metal** (12 images): some plastic and metal items have similar shapes and reflections.
- **metal** predicted as **plastic** (10 images): both can appear smooth and bright under indoor lighting.
- **plastic** predicted as **paper** (10 images): thin plastic wrappers can look like paper sheets.
- **shoes** predicted as **biological** (9 images): some shoe photos contain soil-like colors or textured backgrounds.

![Top confusion pairs](images/top_confusion_pairs_v1.png)

## What we learned

- **Plastic** is the hardest class. It appears in many shapes, colors, and materials, so it overlaps with metal, paper, glass, and trash.
- **White-glass** is difficult because transparent glass can look like transparent plastic.
- **Metal** can be confused with plastic when the object has a smooth or shiny surface.
- **Paper and cardboard** are visually close because both have flat fibre-like textures.
- **Trash** is not one clear object type. It is a mixed class, so some mistakes are expected.

## How this helps the next steps

This analysis shows that the model is already strong overall, but it still needs explanation for difficult classes. The next phase uses Grad-CAM to check whether the model looks at the waste object or at the background when it predicts a class.
