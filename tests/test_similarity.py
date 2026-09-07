"""
Unit tests for similarity and statistical testing stubs.
"""

import numpy as np
import pytest
from src.similarity import dtw_similarity
from src.stats_test import null_distribution_test


def test_dtw_similarity_identical():
    """
    Comparing two identical chroma matrices should yield similarity ≈ 1.0.
    """
    np.random.seed(42)
    # Synthetic chroma matrix: 12 pitch classes, 50 time frames, normalized in [0, 1]
    chroma_a = np.random.uniform(0.0, 1.0, size=(12, 50))
    chroma_b = chroma_a.copy()

    score = dtw_similarity(chroma_a, chroma_b)

    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0
    assert np.isclose(score, 1.0, atol=1e-5), f"Expected similarity ≈ 1.0, got {score}"


def test_dtw_similarity_transposed_key():
    """
    Circularly pitch-shifted (transposed key) chroma matrix should still yield similarity ≈ 1.0.
    """
    np.random.seed(123)
    chroma_a = np.random.uniform(0.0, 1.0, size=(12, 40))
    # Shift by 4 semitones (major third transposition)
    chroma_b = np.roll(chroma_a, shift=4, axis=0)

    score = dtw_similarity(chroma_a, chroma_b)

    assert np.isclose(score, 1.0, atol=1e-5), f"Expected transposition-invariant similarity ≈ 1.0, got {score}"


def test_dtw_similarity_random_noise():
    """
    Comparing two independent random noise matrices should return low similarity.
    """
    np.random.seed(999)
    # Generate two independent random matrices
    chroma_a = np.random.uniform(0.0, 1.0, size=(12, 60))
    chroma_b = np.random.uniform(0.0, 1.0, size=(12, 60))

    score = dtw_similarity(chroma_a, chroma_b)

    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0
    # Random chroma matrices with Euclidean distance between 12-D vectors will have
    # positive distance, significantly lower than 1.0.
    assert score < 0.65, f"Expected low similarity (< 0.65) for random noise, got {score}"


def test_null_distribution_test():
    """
    Verify empirical p-value calculation against a known null distribution.
    """
    null_scores = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    # Query score of 0.85: only 0.9 and 1.0 are >= 0.85 -> p-value = 2 / 10 = 0.2
    p_val = null_distribution_test(0.85, null_scores)
    assert np.isclose(p_val, 0.2)

    # Query score of 1.1: 0 items >= 1.1 -> p-value = 0.0
    assert np.isclose(null_distribution_test(1.1, null_scores), 0.0)

    # Query score of 0.05: all 10 items >= 0.05 -> p-value = 1.0
    assert np.isclose(null_distribution_test(0.05, null_scores), 1.0)
