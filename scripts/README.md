# Scripts

Run all scripts from the project root.

Check the dataset:

```bash
python3 scripts/check_dataset.py --data-dir data/raw
```

Create the train, validation, and test split:

```bash
python3 scripts/split_dataset.py --data-dir data/raw --output-dir data/processed/splits --report-dir docs/phase2 --seed 42
```

Train the first baseline:

```bash
python3 scripts/train_baseline.py --split-dir data/processed/splits --output-dir docs/phase3 --model-path models/baseline_color_hist_v1.joblib
```

Train the small CNN:

```bash
python3 scripts/train_cnn.py --split-dir data/processed/splits --output-dir docs/phase4 --model-path models/small_cnn_v1.pt --epochs 8 --batch-size 64 --image-size 96 --max-train-per-class 350 --max-val-per-class 90
```

Train the CNN with augmentation:

```bash
python3 scripts/train_cnn.py --split-dir data/processed/splits --output-dir docs/phase5/with_aug --model-path models/small_cnn_aug_v1.pt --epochs 8 --batch-size 64 --image-size 96 --max-train-per-class 350 --max-val-per-class 90 --augment --report-file-name phase5_with_aug_cnn_report_v1.md
```

Train transfer learning models:

```bash
python3 scripts/train_transfer.py --model mobilenet_v2 --epochs 5 --batch-size 32 --image-size 224 --max-train-per-class 500 --max-val-per-class 120 --augment
python3 scripts/train_transfer.py --model efficientnet_b0 --epochs 5 --batch-size 32 --image-size 224 --max-train-per-class 500 --max-val-per-class 120 --augment
```

Run final evaluation:

```bash
python3 scripts/evaluate_final.py --split-dir data/processed/splits --model-path models/mobilenet_v2_v1.pt --model-name mobilenet_v2 --output-dir docs/phase7
```

Run error analysis:

```bash
python3 scripts/analyze_errors.py --phase7-dir docs/phase7 --output-dir docs/phase8
```

Generate Grad-CAM examples:

```bash
python3 scripts/generate_gradcam.py --split-dir data/processed/splits --model-path models/mobilenet_v2_v1.pt --model-name mobilenet_v2 --output-dir docs/phase9
```
