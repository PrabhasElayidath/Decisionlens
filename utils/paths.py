"""Project root resolution for models, data cache, and assets."""

from pathlib import Path


def project_root() -> Path:
    # utils/paths.py → parent = utils/ → parent = repo root
    return Path(__file__).resolve().parent.parent


MODELS_DIR = project_root() / "models"
DATA_RAW_DIR = project_root() / "data" / "raw"

MODEL_PATH = MODELS_DIR / "xgb_model.json"
META_PATH = MODELS_DIR / "metadata.json"
GLOBAL_SHAP_PATH = MODELS_DIR / "global_shap.json"
BACKGROUND_PATH = MODELS_DIR / "background_X.npy"
