from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"

MODELS_DIR = PROJECT_ROOT / "models"

IMAGE_SIZE = 224
RANDOM_SEED = 42

CLASS_NAMES = [
    "paper",
    "cardboard",
    "biological",
    "metal",
    "plastic",
    "green-glass",
    "brown-glass",
    "white-glass",
    "clothes",
    "shoes",
    "battery",
    "trash",
]
