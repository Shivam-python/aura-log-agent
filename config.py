from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

DATASET_PATH = BASE_DIR / "training" / "dataset" / "synthetic_logs.csv"
REGEX_CONFIG_PATH = BASE_DIR / "configs" / "regex_patterns.yaml"
MODEL_OUTPUT_PATH = BASE_DIR / "models" / "log_classifier.joblib"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

DBSCAN_EPS = 0.2
DBSCAN_MIN_SAMPLES = 1