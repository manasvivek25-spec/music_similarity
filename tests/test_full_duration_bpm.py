"""
Regression and Verification Tests for Full-Duration Audio Processing and BPM Estimation.

Ensures:
1. No arbitrary duration caps (60.0s or 30.0s) truncate audio analysis in Feature 5 or API endpoints.
2. Accurate tempo (BPM) and rhythm regularity extraction on synthetic signals with known ground truth.
3. Robust edge-case behavior on drones, silence, pure sinusoidal tones, and extreme durations (>65s).
"""

import numpy as np
import pytest
import soundfile as sf
from pathlib import Path

from src.feature5 import (
    extract_metadata_context,
    estimate_tempo,
    estimate_musical_key,
)
from src.features_extra import extract_extra_features_from_waveform
from src.config import SAMPLE_RATE


def test_short_audio_musical_context(tmp_path: Path):
    """Verify context extraction on short audio (<60s, 5s duration)."""
    sr = SAMPLE_RATE
    duration = 5.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False, dtype=np.float32)
    # 120 BPM = 2 beats per second = pulse every 0.5s
    beat_pulse = np.zeros_like(t)
    beat_pulse[:: int(sr * 0.5)] = 1.0
    y = (0.7 * np.sin(2 * np.pi * 220.0 * t) + 0.3 * beat_pulse).astype(np.float32)

    audio_file = tmp_path / "test_5s.wav"
    sf.write(str(audio_file), y, sr)

    res = extract_metadata_context(str(audio_file), provided_metadata={"track_id": "short_test"})
    assert res.estimated_tempo is not None
    assert 90.0 <= res.estimated_tempo <= 150.0
    assert res.duration_sec == pytest.approx(5.0, abs=0.2)


def test_long_audio_musical_context_exceeds_60s(tmp_path: Path):
    """Verify audio exceeding 60s is analyzed in full without duration truncation."""
    sr = SAMPLE_RATE
    duration = 65.0  # 65 seconds exceeds the old 60.0s cap
    t = np.linspace(0, duration, int(sr * duration), endpoint=False, dtype=np.float32)
    
    # 100 BPM rhythm
    beat_interval = 60.0 / 100.0  # 0.6s
    beat_samples = int(sr * beat_interval)
    y = (0.5 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    y[::beat_samples] += 0.8
    y = np.clip(y, -1.0, 1.0)

    audio_file = tmp_path / "test_65s.wav"
    sf.write(str(audio_file), y, sr)

    res = extract_metadata_context(str(audio_file), provided_metadata={"track_id": "long_test"})
    assert res.duration_sec is not None
    assert res.duration_sec >= 64.8  # Confirms full 65s duration is preserved without 60s cap
    assert res.estimated_tempo is not None


def test_known_tempo_click_train(tmp_path: Path):
    """Verify estimate_tempo recovers target tempo on synthetic clicks."""
    sr = 22050
    duration = 10.0
    target_bpm = 120.0
    beat_sec = 60.0 / target_bpm  # 0.5s
    
    y = np.zeros(int(sr * duration), dtype=np.float32)
    pulse_indices = np.arange(0, len(y), int(sr * beat_sec))
    y[pulse_indices] = 1.0

    audio_file = tmp_path / "clicks_120bpm.wav"
    sf.write(str(audio_file), y, sr)

    bpm = estimate_tempo(str(audio_file), sr)
    assert bpm is not None
    # 120 BPM or octave harmonic (60 / 240)
    assert np.isclose(bpm, 120.0, atol=6.0) or np.isclose(bpm, 60.0, atol=4.0)


def test_drone_signal_tempo_handling(tmp_path: Path):
    """Verify drone signal (flat continuous sinusoid with zero onset flux) behaves safely."""
    sr = 22050
    duration = 6.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False, dtype=np.float32)
    # Pure constant sine wave drone
    drone = (0.5 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)

    audio_file = tmp_path / "drone.wav"
    sf.write(str(audio_file), drone, sr)

    bpm = estimate_tempo(str(audio_file), sr)
    # Flat continuous sine wave should be detected as having no rhythmic onsets (None)
    assert bpm is None


def test_pure_silence_extra_features():
    """Verify zero audio signal does not cause division by zero or crashes."""
    y_silence = np.zeros(22050 * 3, dtype=np.float32)
    feats = extract_extra_features_from_waveform(y_silence, 22050, "silence")
    assert feats.tempo_bpm is not None
    assert feats.harmonic_ratio is not None
    assert feats.percussive_ratio is not None


def test_no_60s_cap_in_source_code():
    """Inspect source code of feature5.py and api.py to ensure 60.0s caps are not present."""
    import inspect
    from src import feature5, api

    src_f5 = inspect.getsource(feature5)
    src_api = inspect.getsource(api)

    # Neither should have duration=60.0 or duration=30.0 in audio loading calls
    assert "duration=60.0" not in src_f5, "Found hardcoded duration=60.0 in src/feature5.py"
    assert "duration=60" not in src_f5, "Found hardcoded duration=60 in src/feature5.py"
    assert "duration=30.0" not in src_api, "Found hardcoded duration=30.0 in src/api.py"
