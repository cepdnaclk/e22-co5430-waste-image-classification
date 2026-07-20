# Dataset Notes v1

## Dataset Source

Dataset name: Garbage Classification

Source:

https://www.kaggle.com/datasets/mostafaabla/garbage-classification

Creator listed on Kaggle: Mostafa Mohamed

## Classes

The dataset contains 12 waste classes. In our downloaded folder, the battery class is named `battery`.

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

## License Note

The Kaggle page lists the license as:

Database: Open Database, Contents: Original Authors

We will cite the dataset in the report and README. We will not upload the full dataset to GitHub.

## Dataset Limitations

Some images were collected from the web. This means the dataset may not fully match real waste images from a recycling plant.

Possible issues:

- Some images may have clean objects instead of dirty waste.
- Some labels may be noisy.
- Some classes may look similar.
- Backgrounds may affect prediction.

We will mention these limitations in the final report.

## Local Dataset Audit

The dataset was copied from:

```text
/Users/bhaveenthankajanikanth/Downloads/garbage_classification
```

to:

```text
data/raw/
```

Phase 1 audit result:

- Class folders: 12
- Readable image files: 15,515
- Unreadable image files: 0
- File type: JPG
- Main image modes: RGB and a small number of P-mode images

The local count is 15,515 images. The proposal mentioned approximately 15,150 images based on the dataset page text. In the project report, we should use the local audit count when describing our actual experiments.
