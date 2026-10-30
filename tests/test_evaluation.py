"""
Unit tests for evaluation framework, audio perturbations, and retrieval metric calculations.
"""
import numpy as np
import pytest
from src.evaluation import (
    perturb_pitch,
    perturb_tempo,
    perturb_additive_noise,
    calculate_retrieval_metrics,
)


def test_audio_perturbations():
    sr = 22050
    t = np.linspace(0, 1.0, sr, endpoint=False)
    y = np.sin(2 * np.pi * 440 * t).astype(np.float32)

    # Pitch shift
    y_shifted = perturb_pitch(y, sr=sr, n_semitones=2.0)
    assert len(y_shifted) == len(y)
    assert not np.allclose(y_shifted, y)

    # Tempo stretch
    y_stretched = perturb_tempo(y, rate=1.2)
    assert len(y_stretched) < len(y)  # Faster tempo = shorter duration

    # Additive noise
    y_noisy = perturb_additive_noise(y, snr_db=15.0)
    assert len(y_noisy) == len(y)
    assert not np.allclose(y_noisy, y)


def test_calculate_retrieval_metrics():
    # Candidates ranked:
    # 1. 'track_rel_1' (relevant)
    # 2. 'track_other' (not relevant)
    # 3. 'track_rel_2' (relevant)
    # 4. 'track_other_2' (not relevant)
    ranked = [
        ("track_rel_1", 0.95),
        ("track_other", 0.80),
        ("track_rel_2", 0.70),
        ("track_other_2", 0.50),
    ]
    ground_truth = {"track_rel_1", "track_rel_2"}

    p1, p3, r1, r3, mrr = calculate_retrieval_metrics(ranked, ground_truth)
    # Top 1: 'track_rel_1' is in ground_truth -> P@1 = 1.0, R@1 = 1/2 = 0.5
    assert p1 == 1.0
    assert r1 == 0.5
    # Top 3: 2 relevant out of 3 -> P@3 = 2/3 = 0.6667, R@3 = 2/2 = 1.0
    assert np.isclose(p3, 2 / 3, atol=1e-3)
    assert r3 == 1.0
    # First relevant item is at rank 1 -> MRR = 1.0
    assert mrr == 1.0
