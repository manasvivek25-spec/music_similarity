"""
Unit and integration tests for Genre Estimation (Authoritative Decision 2).

Tests:
- Missing genre metadata handling
- Missing classifier artifact behavior (honest fallback)
- MFCC extraction and feature aggregation
- Invalid or empty audio waveforms
- Nullable estimated genre and confidence fields
- Distinguishing verified metadata, model estimates, and unknown genre
- Application operational resilience when genre estimation is unconfigured
"""

from pathlib import Path
import numpy as np
import pytest

from src.genre import resolve_track_genre
from src.genre_classifier import (
    GENRE_LABELS,
    GenreClassifier,
    extract_genre_features,
    get_genre_classifier,
    load_genre_classifier,
)


def test_missing_genre_metadata_returns_unknown_when_no_model():
    """When metadata is missing/ambient and classifier is unconfigured, genre is unknown."""
    # Never use 'ambient' as genuine genre
    info_none = resolve_track_genre(None, waveform=None)
    assert info_none["genre"] is None
    assert info_none["estimated_genre"] is None
    assert info_none["genre_confidence"] is None
    assert info_none["genre_source"] == "unknown"

    info_ambient = resolve_track_genre("ambient", waveform=None)
    assert info_ambient["genre"] is None
    assert info_ambient["genre_source"] == "unknown"

    info_blank = resolve_track_genre("   ", waveform=None)
    assert info_blank["genre"] is None
    assert info_blank["genre_source"] == "unknown"


def test_verified_metadata_is_preserved_as_metadata():
    """Verified genuine metadata is preserved with source='metadata'."""
    info_rock = resolve_track_genre("Rock", waveform=None)
    assert info_rock["genre"] == "Rock"
    assert info_rock["estimated_genre"] is None
    assert info_rock["genre_confidence"] is None
    assert info_rock["genre_source"] == "metadata"

    info_classical = resolve_track_genre("classical", waveform=None)
    assert info_classical["genre"] == "classical"
    assert info_classical["genre_source"] == "metadata"


def test_missing_classifier_artifact_behavior():
    """When model artifact does not exist on disk, returns unconfigured classifier without error."""
    non_existent = Path("models/non_existent_genre_model.joblib")
    clf = load_genre_classifier(non_existent)
    assert clf.is_available is False
    assert clf.model is None

    # Prediction on unconfigured classifier returns (None, None)
    est_label, est_conf = clf.predict(np.zeros(22050, dtype=np.float32), sr=22050)
    assert est_label is None
    assert est_conf is None


def test_mfcc_feature_extraction_shape_and_aggregation():
    """Audio waveform is converted to 40-dimensional MFCC feature vector (20 mean + 20 std)."""
    sr = 22050
    # 2 seconds synthetic tone
    t = np.linspace(0, 2.0, sr * 2, endpoint=False)
    waveform = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    feats = extract_genre_features(waveform, sr=sr, n_mfcc=20)
    assert isinstance(feats, np.ndarray)
    assert feats.shape == (40,)
    assert np.all(np.isfinite(feats))


def test_invalid_or_empty_audio_returns_none_features():
    """Empty or non-finite audio returns None features safely."""
    assert extract_genre_features(None, sr=22050) is None
    assert extract_genre_features(np.array([], dtype=np.float32), sr=22050) is None
    assert extract_genre_features(np.array([np.nan, np.inf]), sr=22050) is None

    clf = GenreClassifier(model=None, scaler=None, is_available=False)
    label, conf = clf.predict(np.array([]), sr=22050)
    assert label is None
    assert conf is None


def test_gtzan_genre_labels_defined():
    """Verify standard 10 GTZAN classes are specified."""
    assert len(GENRE_LABELS) == 10
    assert "classical" in GENRE_LABELS
    assert "jazz" in GENRE_LABELS
    assert "rock" in GENRE_LABELS
    assert "pop" in GENRE_LABELS


def test_synthetic_trained_model_predicts_and_resolves_genre(tmp_path):
    """When a trained model exists and loads, genre is resolved with source='estimated'."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    import joblib

    # Train a minimal synthetic model on 10 GTZAN classes
    rng = np.random.RandomState(42)
    X = rng.randn(30, 40)
    y = np.array([GENRE_LABELS[i % 10] for i in range(30)])

    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)
    rf = RandomForestClassifier(n_estimators=5, random_state=42).fit(X_scaled, y)

    artifact_path = tmp_path / "test_genre_model.joblib"
    joblib.dump({"model": rf, "scaler": scaler, "labels": GENRE_LABELS}, artifact_path)

    # Load and test prediction
    clf = load_genre_classifier(artifact_path)
    assert clf.is_available is True

    # 1 second test audio
    waveform = rng.randn(22050).astype(np.float32)
    pred_label, pred_conf = clf.predict(waveform, sr=22050)

    assert pred_label in GENRE_LABELS
    assert 0.0 <= pred_conf <= 1.0

    # Resolve track genre with this classifier
    info = resolve_track_genre(None, waveform=waveform, sr=22050, classifier=clf)
    assert info["genre"] is None  # verified metadata is None
    assert info["estimated_genre"] == pred_label
    assert info["genre_confidence"] == pred_conf
    assert info["genre_source"] == "estimated"
