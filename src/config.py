"""
Configuration management for the Music Similarity Research Pipeline.

Provides centralized path management, analysis thresholds, and runtime settings
loaded from environment variables with safe defaults.
"""

import os
from pathlib import Path
from typing import Any, Dict

# Base Directories
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("MUSIC_DATA_DIR", str(REPO_ROOT / "data")))
HUMAN_DIR = DATA_DIR / "human_songs"
AI_DIR = DATA_DIR / "ai_generated"
REF_DIR = DATA_DIR / "reference_pool"
FEATURES_DIR = Path(os.getenv("MUSIC_FEATURES_DIR", str(REPO_ROOT / "features")))
MODELS_DIR = Path(os.getenv("MUSIC_MODELS_DIR", str(REPO_ROOT / "models")))
RESULTS_DIR = Path(os.getenv("MUSIC_RESULTS_DIR", str(REPO_ROOT / "results")))
METADATA_PATH = Path(os.getenv("MUSIC_METADATA_PATH", str(DATA_DIR / "metadata.csv")))
TEMP_UPLOADS_DIR = Path(os.getenv("MUSIC_TEMP_DIR", str(REPO_ROOT / "temp_uploads")))

# Audio Analysis Parameters
SAMPLE_RATE = int(os.getenv("MUSIC_SAMPLE_RATE", "22050"))
MIN_UPLOAD_DURATION_SEC = float(os.getenv("MUSIC_MIN_UPLOAD_DURATION_SEC", "0.0"))  # 0.0 = unlimited
MAX_UPLOAD_SIZE_BYTES = int(os.getenv("MUSIC_MAX_UPLOAD_SIZE_BYTES", str(100 * 1024 * 1024)))  # 100 MB

# Pipeline Thresholds & Cutoffs
COSINE_PREFILTER_THRESHOLD = float(os.getenv("MUSIC_COSINE_THRESHOLD", "0.30"))
SIMILARITY_FLOOR = float(os.getenv("MUSIC_SIMILARITY_FLOOR", "0.45"))
P_VALUE_CRITICAL_THRESHOLD = float(os.getenv("MUSIC_P_CRITICAL", "0.01"))
P_VALUE_BORDERLINE_THRESHOLD = float(os.getenv("MUSIC_P_BORDERLINE", "0.10"))
CREATIVE_SCORE_VERDICT_THRESHOLD = float(os.getenv("MUSIC_CREATIVE_THRESHOLD", "0.24"))  # normalized [0, 1]
MAX_SURVIVORS = int(os.getenv("MUSIC_MAX_SURVIVORS", "15"))
GENRE_MIN_SAMPLE_SIZE = int(os.getenv("MUSIC_GENRE_MIN_SAMPLE", "30"))
NARROW_CREATIVE_MIN_SURVIVORS = int(os.getenv("MUSIC_NARROW_MIN_SURVIVORS", "5"))
NULL_DIST_SAMPLE_PAIRS = int(os.getenv("MUSIC_NULL_SAMPLE_PAIRS", "2000"))
NULL_DIST_CACHE_FILE = FEATURES_DIR / "null_distribution.json"
DURATION_DISPARITY_RATIO = float(os.getenv("MUSIC_DURATION_DISPARITY_RATIO", "2.5"))

MIN_NULL_PAIRS = int(os.getenv("MUSIC_MIN_NULL_PAIRS", "5"))
GENRE_MODEL_PATH = Path(os.getenv("MUSIC_GENRE_MODEL_PATH", str(MODELS_DIR / "genre_classifier.joblib")))

# Remote Dataset Configuration
DEFAULT_REMOTE_DATASET = os.getenv("MUSIC_REMOTE_DATASET", "SleepyJesse/ai_music_large")
DEFAULT_REMOTE_SPLIT = os.getenv("MUSIC_REMOTE_SPLIT", "train")
REMOTE_INDEX_FILE = FEATURES_DIR / "remote_index.json"
DEFAULT_REMOTE_SAMPLE_SIZE = int(os.getenv("MUSIC_REMOTE_SAMPLE_SIZE", "10"))

# Optional LLM Configuration
LLM_API_KEY = os.getenv("MUSIC_SIMILARITY_LLM_KEY", os.getenv("OPENAI_API_KEY", ""))
LLM_MODEL = os.getenv("MUSIC_SIMILARITY_LLM_MODEL", "gpt-4o-mini")


def get_runtime_config() -> Dict[str, Any]:
    """
    Export safe, non-secret runtime configuration settings for the API client.
    """
    return {
        "sample_rate": SAMPLE_RATE,
        "min_upload_duration_sec": MIN_UPLOAD_DURATION_SEC,
        "max_upload_size_bytes": MAX_UPLOAD_SIZE_BYTES,
        "cosine_prefilter_threshold": COSINE_PREFILTER_THRESHOLD,
        "similarity_floor": SIMILARITY_FLOOR,
        "min_null_pairs": MIN_NULL_PAIRS,
        "genre_model_available": GENRE_MODEL_PATH.is_file(),
        "genre_model_path": str(GENRE_MODEL_PATH),
        "remote_dataset": {
            "default_dataset": DEFAULT_REMOTE_DATASET,
            "default_split": DEFAULT_REMOTE_SPLIT,
            "default_sample_size": DEFAULT_REMOTE_SAMPLE_SIZE,
            "index_file": str(REMOTE_INDEX_FILE),
        },
        "p_value_thresholds": {
            "critical": P_VALUE_CRITICAL_THRESHOLD,
            "borderline": P_VALUE_BORDERLINE_THRESHOLD,
        },
        "creative_score_verdict_threshold": CREATIVE_SCORE_VERDICT_THRESHOLD,
        "max_survivors": MAX_SURVIVORS,
        "genre_min_sample_size": GENRE_MIN_SAMPLE_SIZE,
        "narrow_creative_min_survivors": NARROW_CREATIVE_MIN_SURVIVORS,
        "llm_enabled": bool(LLM_API_KEY),
    }
