"""Project-wide paths and settings."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

# Create the folders if they do not exist yet
for _d in (DATA_RAW, DATA_PROCESSED, MODELS_DIR, RESULTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

SEED = 42                # fixed random seed for reproducible splits
LABELS = ["NOT", "OFF"]  # 0 = clean, 1 = offensive