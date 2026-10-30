"""
Unit tests for data validation, audio sanity, provenance, and index migration.
"""
import numpy as np
import pytest
from src.validation import (
    validate_origin_label,
    validate_audio_waveform,
    validate_genre_metadata,
    validate_feature_matrix,
    compute_full_audio_content_hash,
)


def test_validate_origin_label():
    assert validate_origin_label("human").normalized_label == "human"
    assert validate_origin_label("HUMAN").normalized_label == "human"
    assert validate_origin_label("ai").normalized_label == "ai_generated"
    assert validate_origin_label("synthetic").normalized_label == "ai_generated"
    assert validate_origin_label("False").normalized_label == "human"
    assert validate_origin_label("True").normalized_label == "ai_generated"
    assert validate_origin_label(False).normalized_label == "human"
    assert validate_origin_label(True).normalized_label == "ai_generated"
    assert validate_origin_label("unknown").normalized_label == "unknown"
    assert validate_origin_label("").normalized_label == "unknown"
    assert validate_origin_label(None).normalized_label == "unknown"


def test_validate_audio_waveform():
    sr = 22050
    # Valid sine wave (3.5 seconds to exceed 3.0s threshold)
    t = np.linspace(0, 3.5, int(sr * 3.5), endpoint=False)
    y = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    val = validate_audio_waveform(y, sr, min_duration_sec=3.0)
    assert val.is_valid is True
    assert val.error_message is None
    assert val.duration_sec >= 3.49

    # Empty audio
    val_empty = validate_audio_waveform(np.array([], dtype=np.float32), sr)
    assert val_empty.is_valid is False

    # Pure silence
    y_silent = np.zeros(int(sr * 3.5), dtype=np.float32)
    val_silent = validate_audio_waveform(y_silent, sr)
    assert val_silent.is_valid is False
    assert "silent" in val_silent.error_message.lower()

    # Audio with NaNs
    y_nan = y.copy()
    y_nan[10] = np.nan
    val_nan = validate_audio_waveform(y_nan, sr)
    assert val_nan.is_valid is False
    assert "non-finite" in val_nan.error_message.lower()


def test_validate_genre_provenance():
    # Present metadata
    res = validate_genre_metadata("Rock", genre_source="metadata")
    assert res.normalized_genre == "Rock"
    assert res.genre_source == "metadata"

    # Missing metadata
    res_none = validate_genre_metadata(None)
    assert res_none.normalized_genre is None
    assert res_none.genre_source == "unknown"


def test_validate_feature_matrix():
    valid_chroma = np.ones((12, 100), dtype=np.float32) / np.sqrt(12)
    res = validate_feature_matrix(valid_chroma, expected_dim=12)
    assert res.is_valid is True

    # Bad shape
    bad_shape = np.ones((10, 100), dtype=np.float32)
    res_bad = validate_feature_matrix(bad_shape, expected_dim=12)
    assert res_bad.is_valid is False

    # Contains NaNs
    nan_chroma = valid_chroma.copy()
    nan_chroma[0, 5] = np.nan
    res_nan = validate_feature_matrix(nan_chroma, expected_dim=12)
    assert res_nan.is_valid is False
    assert res_nan.has_nans is True


def test_compute_content_hash():
    data1 = b"abc123audiobytes"
    data2 = b"abc123audiobytes"
    data3 = b"differentbytes"

    h1 = compute_full_audio_content_hash(data1)
    h2 = compute_full_audio_content_hash(data2)
    h3 = compute_full_audio_content_hash(data3)

    assert h1 == h2
    assert h1 != h3
    assert len(h1) == 64  # SHA-256 hex length
