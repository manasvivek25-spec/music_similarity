"""
Unit tests for independent audio feature extraction (MFCCs, HPSS, spectral descriptors, rhythm).
"""
import numpy as np
import pytest
from src.features_extra import extract_extra_features_from_waveform, ExtraAudioFeatures


def test_extract_extra_features_synthetic_sine():
    sr = 22050
    t = np.linspace(0, 3.5, int(sr * 3.5), endpoint=False)
    # Sine wave (harmonic tone)
    y = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    feat = extract_extra_features_from_waveform(y, sr=sr, track_id="test_tone")
    assert isinstance(feat, ExtraAudioFeatures)
    assert len(feat.mfcc_mean) == 13
    assert len(feat.mfcc_std) == 13
    assert len(feat.spectral_contrast_mean) == 7
    assert feat.harmonic_ratio >= 0.0
    assert feat.percussive_ratio >= 0.0
    assert feat.spectral_rolloff_hz > 0.0
    assert feat.spectral_bandwidth_hz > 0.0

    d = feat.to_dict()
    assert isinstance(d, dict)
    assert d["track_id"] == "test_tone"


def test_extract_extra_features_short_signal():
    sr = 22050
    # Empty or very short signal returns zero-filled features safely
    y = np.zeros(100, dtype=np.float32)
    feat = extract_extra_features_from_waveform(y, sr=sr, track_id="test_zero")
    assert isinstance(feat, ExtraAudioFeatures)
    assert feat.harmonic_ratio == 0.0
    assert feat.percussive_ratio == 0.0
