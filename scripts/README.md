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
