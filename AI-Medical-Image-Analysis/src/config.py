from pathlib import Path
DATA_ROOT = Path("data/raw/chest_xray")
MODEL_DIR = Path("models")
METRICS_DIR = Path("outputs/metrics")
FIGURES_DIR = Path("outputs/figures")
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
SEED = 42
