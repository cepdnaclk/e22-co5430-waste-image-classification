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
