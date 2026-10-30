"""
GTZAN-Compatible Ten-Class Genre Estimation Classifier Module.

Supports acoustic genre estimation utilizing MFCC features extracted from raw audio waveforms.
Uses identical preprocessing for training and inference (20 MFCC coefficients with mean and
standard deviation temporal aggregation producing a 40-dimensional feature vector).

Model Loading & Fallback Safety:
  - Loads real model artifact from models/genre_classifier.joblib if available.
  - If no model artifact exists, maintains an explicit, honest unconfigured state
    (is_available == False, returns (None, None)).
  - Never fabricates genre labels or hardcodes ambient defaults.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import librosa

from src.config import GENRE_MODEL_PATH, SAMPLE_RATE

logger = logging.getLogger(__name__)

# Standard 10 GTZAN music genre classes
GTZAN_GENRES = [
    "blues",
    "classical",
    "country",
    "disco",
    "hiphop",
    "jazz",
    "metal",
    "pop",
    "reggae",
    "rock",
]
GENRE_LABELS = GTZAN_GENRES

N_MFCC = 20
FEATURE_DIM = 40  # 20 mean + 20 std


def extract_genre_features(
    waveform: Optional[np.ndarray],
    sr: int = SAMPLE_RATE,
    n_mfcc: int = N_MFCC,
) -> Optional[np.ndarray]:
    """
    Extract 40-dimensional MFCC summary feature vector from raw audio waveform.

    Computes:
      - n_mfcc (default 20) MFCC coefficients over the waveform
      - n_mfcc temporal mean values
      - n_mfcc temporal standard deviation values
      - Concatenated into a (2 * n_mfcc) element float32 vector (default 40-dim)

    Identical preprocessing is applied for both training and inference.
    Returns None if waveform is empty, non-finite, or too short.
    """
    if waveform is None or len(waveform) == 0:
        return None

    y = np.asarray(waveform, dtype=np.float32)
    if not np.all(np.isfinite(y)):
        return None

    n_samples = len(y)
    n_fft = min(2048, n_samples)
    if n_fft < 256:
        return None

    hop_length = max(64, n_fft // 4)

    try:
        mfcc = librosa.feature.mfcc(
            y=y,
            sr=sr,
            n_mfcc=n_mfcc,
            n_fft=n_fft,
            hop_length=hop_length,
        )
    except Exception as exc:
        logger.debug(f"MFCC feature extraction failed: {exc}")
        return None

    if mfcc.shape[0] != n_mfcc or mfcc.shape[1] == 0:
        return None

    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)

    features = np.concatenate([mfcc_mean, mfcc_std]).astype(np.float32)
    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)


class GenreClassifier:
    """
    GTZAN-compatible ten-class genre classifier interface.

    Loads real trained model artifact from disk if present. Otherwise safely defaults
    to an unconfigured state without fabricating genre predictions.
    """

    def __init__(self, model_path: Optional[Union[str, Path]] = None, model: Any = None, scaler: Any = None, is_available: Optional[bool] = None):
        self.model_path = Path(model_path) if model_path is not None else GENRE_MODEL_PATH
        self.model: Optional[Any] = model
        self.scaler: Optional[Any] = scaler
        self.classes_: List[str] = list(GTZAN_GENRES)
        if model is None and is_available is None:
            self._load_model()
        elif is_available is not None:
            self._explicit_available = is_available
        else:
            self._explicit_available = None

    def _load_model(self) -> None:
        """Attempt to load trained classifier artifact."""
        if self.model_path.is_file():
            try:
                import joblib
                loaded = joblib.load(str(self.model_path))
                if isinstance(loaded, dict):
                    self.model = loaded.get("model")
                    self.scaler = loaded.get("scaler")
                    if "labels" in loaded:
                        self.classes_ = list(loaded["labels"])
                else:
                    self.model = loaded
                    self.scaler = None
                if hasattr(self.model, "classes_"):
                    self.classes_ = list(self.model.classes_)
                logger.info(f"Successfully loaded trained genre classifier from {self.model_path}")
            except Exception as exc:
                logger.warning(f"Failed to load genre classifier artifact at {self.model_path}: {exc}")
                self.model = None
        else:
            self.model = None

    @property
    def is_available(self) -> bool:
        """True only if a valid trained model artifact is active in memory."""
        if hasattr(self, "_explicit_available") and self._explicit_available is not None:
            return self._explicit_available
        return self.model is not None


    def predict(
        self,
        waveform: np.ndarray,
        sr: int = SAMPLE_RATE,
    ) -> Tuple[Optional[str], Optional[float]]:
        """
        Predict musical genre and confidence from audio waveform.

        Returns
        -------
        Tuple[Optional[str], Optional[float]]
            (estimated_genre, genre_confidence) in [0.0, 1.0], or (None, None)
            when unconfigured or on processing error.
        """
        if not self.is_available:
            return (None, None)

        try:
            raw_feats = extract_genre_features(waveform, sr=sr)
            if raw_feats is None:
                return (None, None)

            features = raw_feats.reshape(1, -1)
            if self.scaler is not None:
                features = self.scaler.transform(features)

            pred_class = self.model.predict(features)[0]

            confidence: Optional[float] = None
            if hasattr(self.model, "predict_proba"):
                probs = self.model.predict_proba(features)[0]
                confidence = float(np.max(probs))

            return (str(pred_class), round(confidence, 4) if confidence is not None else None)
        except Exception as exc:
            logger.warning(f"Genre prediction failed: {exc}")
            return (None, None)


def load_genre_classifier(model_path: Optional[Union[str, Path]] = None) -> GenreClassifier:
    """Load or initialize a GenreClassifier from the given model path."""
    return GenreClassifier(model_path=model_path)


def get_genre_classifier(model_path: Optional[Union[str, Path]] = None) -> GenreClassifier:
    """Get the singleton or new instance of GenreClassifier."""
    return load_genre_classifier(model_path)


def train_genre_classifier(
    dataset_dir: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    sr: int = SAMPLE_RATE,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Train a GTZAN-compatible genre classifier from audio files organized by genre subfolders.

    Expected directory structure:
        dataset_dir/
            blues/
                track1.wav, ...
            classical/
                ...
            rock/
                ...

    Parameters
    ----------
    dataset_dir : Union[str, Path]
        Path to dataset containing genre subdirectories.
    output_path : Optional[Union[str, Path]]
        Destination path for joblib model artifact (default: GENRE_MODEL_PATH).
    sr : int
        Audio sampling rate.
    test_size : float
        Fraction of data for test split.
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    Dict[str, Any]
        Training summary with accuracy and classification report.
    """
    import joblib
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    data_path = Path(dataset_dir)
    if not data_path.is_dir():
        raise FileNotFoundError(f"Training dataset directory not found: {data_path}")

    X: List[np.ndarray] = []
    y: List[str] = []

    valid_exts = {".wav", ".mp3", ".flac", ".ogg", ".au"}

    for genre_folder in data_path.iterdir():
        if not genre_folder.is_dir():
            continue
        genre_name = genre_folder.name.lower()
        for audio_file in genre_folder.iterdir():
            if audio_file.suffix.lower() not in valid_exts:
                continue
            try:
                wave, sr_loaded = librosa.load(str(audio_file), sr=sr, mono=True)
                feat = extract_genre_features(wave, sr=sr_loaded)
                X.append(feat)
                y.append(genre_name)
            except Exception as exc:
                logger.debug(f"Skipping {audio_file}: {exc}")

    if len(X) < 10:
        raise ValueError(f"Insufficient training data in {dataset_dir}: extracted only {len(X)} samples.")

    X_arr = np.array(X, dtype=np.float32)
    y_arr = np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X_arr, y_arr, test_size=test_size, random_state=random_state, stratify=y_arr
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(n_estimators=100, random_state=random_state)),
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    acc = float(accuracy_score(y_test, y_pred))
    report = classification_report(y_test, y_pred, output_dict=True)

    dest = Path(output_path) if output_path is not None else GENRE_MODEL_PATH
    dest.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, str(dest))
    logger.info(f"Saved trained genre classifier to {dest} with test accuracy: {acc:.4f}")

    return {
        "status": "success",
        "model_path": str(dest),
        "total_samples": len(X),
        "test_accuracy": round(acc, 4),
        "classification_report": report,
    }


# Singleton instance accessor
_classifier_instance: Optional[GenreClassifier] = None


def get_genre_classifier(model_path: Optional[Union[str, Path]] = None) -> GenreClassifier:
    """Return singleton GenreClassifier instance."""
    global _classifier_instance
    if _classifier_instance is None or model_path is not None:
        _classifier_instance = GenreClassifier(model_path=model_path)
    return _classifier_instance
