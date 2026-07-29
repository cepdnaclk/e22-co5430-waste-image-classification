# Scripts

Run scripts from the project root.

Example:

```bash
python scripts/check_dataset.py --data-dir data/raw
```

Create the train, validation, and test split:

```bash
python scripts/split_dataset.py --data-dir data/raw --output-dir data/processed/splits --report-dir docs/phase2 --seed 42
```

Train the first simple baseline:

```bash
python scripts/train_baseline.py --split-dir data/processed/splits --output-dir docs/phase3 --model-path models/baseline_color_hist_v1.joblib
```

Train the first CNN model:

```bash
python scripts/train_cnn.py --split-dir data/processed/splits --output-dir docs/phase4 --model-path models/small_cnn_v1.pt --epochs 8 --batch-size 64 --image-size 96 --max-train 0 --max-val 0 --max-train-per-class 350 --max-val-per-class 90
```

Train the CNN with augmentation:

```bash
python scripts/train_cnn.py --split-dir data/processed/splits --output-dir docs/phase5/with_aug --model-path models/small_cnn_aug_v1.pt --epochs 8 --batch-size 64 --image-size 96 --max-train 0 --max-val 0 --max-train-per-class 350 --max-val-per-class 90 --augment --report-file-name phase5_with_aug_cnn_report_v1.md
```

Train transfer learning models (MobileNetV2 and EfficientNet-B0):

```bash
python scripts/train_transfer.py --model mobilenet_v2 --epochs 5 --batch-size 32 --image-size 224 --max-train-per-class 500 --max-val-per-class 120 --augment
python scripts/train_transfer.py --model efficientnet_b0 --epochs 5 --batch-size 32 --image-size 224 --max-train-per-class 500 --max-val-per-class 120 --augment
```

Run final evaluation on the test set (Phase 7 — no training, test set used only once):

```bash
python scripts/evaluate_final.py --split-dir data/processed/splits --model-path models/mobilenet_v2_v1.pt --model-name mobilenet_v2 --output-dir docs/phase7
```

