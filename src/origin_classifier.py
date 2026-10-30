"""
Feature 4: Independent Human/AI Origin Classifier.

Independent statistical classifier utilizing audio-derived spectral and acoustic features
(MFCCs, spectral centroid, spectral rolloff, zero-crossing rate, and chroma statistics).
Completely decoupled from Feature 1's DTW similarity engine, empirical p-values,
and creative scores.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import joblib
import librosa
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import MODELS_DIR, SAMPLE_RATE
from src.schemas import OriginClassifierResult
from src.utils import ensure_dir, list_audio_files


MODEL_FILE_NAME = "origin_classifier.joblib"
MODEL_VERSION = "1.0.0"


def extract_origin_features_from_audio(
    audio_path: Union[str, Path],
    sr: int = SAMPLE_RATE
) -> np.ndarray:
    """
    Extract a 56-dimensional acoustic feature vector for origin classification.

    Features:
    - MFCCs (13 coefficients): mean & std (26 values)
    - Spectral Centroid: mean & std (2 values)
    - Spectral Rolloff: mean & std (2 values)
    - Zero Crossing Rate: mean & std (2 values)
    - Chroma STFT: mean & std across 12 pitch classes (24 values)

    Total: 56 features.
    """
    path = Path(audio_path)
    y, sr_loaded = librosa.load(str(path), sr=sr, mono=True)
    if len(y) == 0:
        raise ValueError(f"Empty audio signal in {path}")

    # MFCCs
    mfcc = librosa.feature.mfcc(y=y, sr=sr_loaded, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)

    # Spectral Centroid
    cent = librosa.feature.spectral_centroid(y=y, sr=sr_loaded)
    cent_mean = np.array([np.mean(cent)])
    cent_std = np.array([np.std(cent)])

    # Spectral Rolloff
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr_loaded)
    rolloff_mean = np.array([np.mean(rolloff)])
    rolloff_std = np.array([np.std(rolloff)])

    # Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y)
    zcr_mean = np.array([np.mean(zcr)])
    zcr_std = np.array([np.std(zcr)])

    # Chroma STFT
    chroma = librosa.feature.chroma_stft(y=y, sr=sr_loaded, n_fft=min(1024, len(y)))
    chroma_mean = np.mean(chroma, axis=1)
    chroma_std = np.std(chroma, axis=1)

    feature_vec = np.concatenate([
        mfcc_mean,
        mfcc_std,
        cent_mean,
        cent_std,
        rolloff_mean,
        rolloff_std,
        zcr_mean,
        zcr_std,
        chroma_mean,
        chroma_std,
    ])

    return np.nan_to_num(feature_vec, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def get_default_model_path() -> Path:
    """Return the resolved path to the trained origin classifier artifact."""
    return MODELS_DIR / MODEL_FILE_NAME


def load_origin_classifier(
    model_path: Optional[Union[str, Path]] = None
) -> Optional[Any]:
    """
    Safely load the fitted classifier artifact from disk.

    Returns None if model file does not exist or fails to deserialize.
    """
    path = Path(model_path) if model_path is not None else get_default_model_path()
    if not path.is_file():
        return None

    try:
        model = joblib.load(str(path))
        return model
    except Exception:
        return None


def predict_origin(
    audio_path: Union[str, Path],
    model_path: Optional[Union[str, Path]] = None
) -> OriginClassifierResult:
    """
    Predict track origin (Human vs. AI) using the independent classifier.

    If model is missing or unconfigured, returns an honest 'Unconfigured' status.
    Never invents predictions or infers origin from DTW similarity scores.

    Parameters
    ----------
    audio_path : Union[str, Path]
        Path to target audio file.
    model_path : Optional[Union[str, Path]]
        Optional explicit model path override.

    Returns
    -------
    OriginClassifierResult
        Prediction details or explicit unconfigured notice.
    """
    model = load_origin_classifier(model_path)
    if model is None:
        return OriginClassifierResult(
            predicted_origin="Unconfigured",
            confidence=None,
            probability_ai=None,
            model_status="Unconfigured / Missing Labeled Data",
            model_version=None,
        )

    try:
        features = extract_origin_features_from_audio(audio_path)
        features_2d = features.reshape(1, -1)

        pred_class = model.predict(features_2d)[0]
        label = "AI" if pred_class in (1, "AI", "ai") else "Human"

        prob_ai: Optional[float] = None
        confidence: Optional[float] = None

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(features_2d)[0]
            # Assume class 1 is AI if classes are [0, 1]
            classes = list(model.classes_)
            if 1 in classes:
                ai_idx = classes.index(1)
            elif "AI" in classes:
                ai_idx = classes.index("AI")
            else:
                ai_idx = len(classes) - 1

            prob_ai = float(probs[ai_idx])
            confidence = float(max(probs))

        return OriginClassifierResult(
            predicted_origin=label,
            confidence=round(confidence, 4) if confidence is not None else None,
            probability_ai=round(prob_ai, 4) if prob_ai is not None else None,
            model_status="Trained & Active",
            model_version=MODEL_VERSION,
        )
    except Exception:
        return OriginClassifierResult(
            predicted_origin="Unconfigured",
            confidence=None,
            probability_ai=None,
            model_status="Unconfigured / Missing Labeled Data",
            model_version=None,
        )


def train_origin_classifier(
    human_dir: Union[str, Path],
    ai_dir: Union[str, Path],
    model_save_path: Optional[Union[str, Path]] = None,
    test_size: float = 0.25,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Train and save an independent Random Forest origin classifier on labeled audio.

    Parameters
    ----------
    human_dir : Union[str, Path]
        Directory containing labeled human-composed/performed audio.
    ai_dir : Union[str, Path]
        Directory containing labeled AI-generated audio.
    model_save_path : Optional[Union[str, Path]]
        Target path to save fitted joblib model.
    test_size : float
        Fraction of tracks to reserve for validation.
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    Dict[str, Any]
        Training summary with sample counts and evaluation metrics.

    Raises
    ------
    ValueError
        If either class has fewer than 2 audio files.
    """
    human_files = list_audio_files(human_dir)
    ai_files = list_audio_files(ai_dir)

    if len(human_files) < 2 or len(ai_files) < 2:
        raise ValueError(
            f"Insufficient labeled training audio. Found {len(human_files)} human files "
            f"and {len(ai_files)} AI files. Minimum 2 required per class."
        )

    X_list: List[np.ndarray] = []
    y_list: List[int] = []

    for hf in human_files:
        try:
            feats = extract_origin_features_from_audio(hf)
            X_list.append(feats)
            y_list.append(0)  # Human = 0
        except Exception:
            continue

    for af in ai_files:
        try:
            feats = extract_origin_features_from_audio(af)
            X_list.append(feats)
            y_list.append(1)  # AI = 1
        except Exception:
            continue

    if len(X_list) < 4 or len(set(y_list)) < 2:
        raise ValueError("Could not extract sufficient valid feature vectors for both classes.")

    X = np.array(X_list)
    y = np.array(y_list)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(n_estimators=100, random_state=random_state)),
    ])
    pipeline.fit(X_train, y_train)

    train_acc = float(pipeline.score(X_train, y_train))
    val_acc = float(pipeline.score(X_val, y_val))

    target_path = Path(model_save_path) if model_save_path is not None else get_default_model_path()
    ensure_dir(target_path.parent)
    joblib.dump(pipeline, str(target_path))

    return {
        "status": "success",
        "human_samples": len([y for y in y_list if y == 0]),
        "ai_samples": len([y for y in y_list if y == 1]),
        "total_samples": len(y_list),
        "train_accuracy": round(train_acc, 4),
        "val_accuracy": round(val_acc, 4),
        "model_path": str(target_path),
        "model_version": MODEL_VERSION,
    }


def compare_and_train_origin_classifiers(
    human_dir: Union[str, Path],
    ai_dir: Union[str, Path],
    model_save_path: Optional[Union[str, Path]] = None,
    test_size: float = 0.25,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Multi-model comparative evaluation across Logistic Regression, SVM, and Random Forest.
    Reports precision, recall, F1, accuracy, and confusion matrix on stratified validation set.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC
    from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

    human_files = list_audio_files(human_dir)
    ai_files = list_audio_files(ai_dir)

    if len(human_files) < 2 or len(ai_files) < 2:
        return {
            "status": "unconfigured",
            "message": f"Insufficient labeled training audio (human: {len(human_files)}, AI: {len(ai_files)}). Minimum 2 required per class.",
            "comparison": {},
        }

    X_list: List[np.ndarray] = []
    y_list: List[int] = []

    for hf in human_files:
        try:
            X_list.append(extract_origin_features_from_audio(hf))
            y_list.append(0)
        except Exception:
            continue

    for af in ai_files:
        try:
            X_list.append(extract_origin_features_from_audio(af))
            y_list.append(1)
        except Exception:
            continue

    if len(X_list) < 4 or len(set(y_list)) < 2:
        return {
            "status": "unconfigured",
            "message": "Could not extract sufficient valid feature vectors for both classes.",
            "comparison": {},
        }

    X = np.array(X_list)
    y = np.array(y_list)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    candidate_models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=random_state),
        "LinearSVC": LinearSVC(random_state=random_state, dual=False),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=random_state),
    }

    comparison: Dict[str, Any] = {}
    best_name = "RandomForest"
    best_f1 = -1.0
    best_pipeline = None

    for name, clf in candidate_models.items():
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_val)

        prec = float(precision_score(y_val, preds, zero_division=0))
        rec = float(recall_score(y_val, preds, zero_division=0))
        f1 = float(f1_score(y_val, preds, zero_division=0))
        acc = float(pipe.score(X_val, y_val))
        cm = confusion_matrix(y_val, preds).tolist()

        comparison[name] = {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "confusion_matrix": cm,
        }

        if f1 > best_f1:
            best_f1 = f1
            best_name = name
            best_pipeline = pipe

    # Save best performing pipeline
    target_path = Path(model_save_path) if model_save_path is not None else get_default_model_path()
    if best_pipeline is not None:
        ensure_dir(target_path.parent)
        joblib.dump(best_pipeline, str(target_path))

    return {
        "status": "success",
        "best_model": best_name,
        "best_f1": best_f1,
        "comparison": comparison,
        "human_samples": len([y for y in y_list if y == 0]),
        "ai_samples": len([y for y in y_list if y == 1]),
        "total_samples": len(y_list),
        "model_path": str(target_path),
        "model_version": MODEL_VERSION,
    }
