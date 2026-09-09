"""
Unit tests for similarity engine and statistical testing modules.

Validates key-invariance, tempo-invariance, monotonic decay under corruption,
and granular alignment details (paths and local step costs) required by downstream modules.
"""

import numpy as np
import pytest
from src.similarity import DTWResult, dtw_align_detailed, dtw_similarity
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
    chroma_a = np.random.uniform(0.0, 1.0, size=(12, 60))
    chroma_b = np.random.uniform(0.0, 1.0, size=(12, 60))

    score = dtw_similarity(chroma_a, chroma_b)

    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0
    assert score < 0.65, f"Expected low similarity (< 0.65) for random noise, got {score}"


def test_dtw_similarity_partial_corruption():
    """
    Partially corrupting a chroma matrix should yield intermediate similarity
    monotonically between identical (≈1.0) and random noise (<0.65).
    """
    np.random.seed(42)
    n_frames = 60
    chroma_orig = np.random.uniform(0.0, 1.0, size=(12, n_frames))
    chroma_random = np.random.uniform(0.0, 1.0, size=(12, n_frames))

    # Corrupt 30% of frames in the middle
    chroma_corrupted = chroma_orig.copy()
    corrupt_indices = slice(20, 38)
    chroma_corrupted[:, corrupt_indices] = np.random.uniform(0.0, 1.0, size=(12, 18))

    score_identical = dtw_similarity(chroma_orig, chroma_orig)
    score_corrupted = dtw_similarity(chroma_orig, chroma_corrupted)
    score_random = dtw_similarity(chroma_orig, chroma_random)

    assert score_identical > score_corrupted > score_random, (
        f"Monotonicity violation: identical ({score_identical:.4f}) > "
        f"corrupted ({score_corrupted:.4f}) > random ({score_random:.4f})"
    )
    # Check that mid-range behavior is within realistic boundaries
    assert 0.65 < score_corrupted < 0.95, f"Expected mid-range score, got {score_corrupted}"


def test_dtw_align_detailed_structure_and_path():
    """
    Verify DTWResult structure, alignment path continuity, and local step cost sum.
    This contract is required by Feature 7 (Suggestor Box) for segment localization.
    """
    np.random.seed(77)
    n_a, n_b = 30, 35
    chroma_a = np.random.uniform(0.0, 1.0, size=(12, n_a))
    chroma_b = np.random.uniform(0.0, 1.0, size=(12, n_b))

    result = dtw_align_detailed(chroma_a, chroma_b)

    assert isinstance(result, DTWResult)
    assert np.isclose(result.similarity, dtw_similarity(chroma_a, chroma_b), atol=1e-6)
    assert 0 <= result.best_shift < 12
    assert len(result.alignment_path) > 0
    assert len(result.local_costs) == len(result.alignment_path)

    # Path boundary conditions
    assert result.alignment_path[0] == (0, 0)
    assert result.alignment_path[-1] == (n_a - 1, n_b - 1)

    # Path monotonicity / step continuity (each step moves by at most 1 in each dimension)
    for k in range(1, len(result.alignment_path)):
        prev_i, prev_j = result.alignment_path[k - 1]
        curr_i, curr_j = result.alignment_path[k]
        assert 0 <= curr_i - prev_i <= 1
        assert 0 <= curr_j - prev_j <= 1
        assert (curr_i - prev_i) + (curr_j - prev_j) >= 1

    # Local costs must sum to the raw DTW distance
    assert np.isclose(np.sum(result.local_costs), result.raw_distance, atol=1e-5)


def test_dtw_align_detailed_transposition_detection():
    """
    Verify that dtw_align_detailed correctly identifies the exact semitone shift applied.
    """
    np.random.seed(88)
    chroma_a = np.random.uniform(0.0, 1.0, size=(12, 25))
    # Transpose track B upwards by 5 semitones
    shift_applied = 5
    chroma_b = np.roll(chroma_a, shift=shift_applied, axis=0)

    result = dtw_align_detailed(chroma_a, chroma_b)

    # To realign chroma_b to chroma_a, the required circular shift is (12 - 5) % 12 = 7
    expected_shift = (12 - shift_applied) % 12
    assert result.best_shift == expected_shift
    assert np.isclose(result.similarity, 1.0, atol=1e-5)
    assert np.isclose(result.raw_distance, 0.0, atol=1e-5)


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
