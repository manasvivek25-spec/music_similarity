"""
Unit tests for similarity engine and alignment modules.

Validates key-invariance, tempo-invariance, monotonic decay under corruption,
granular alignment details (paths and local step costs), coarse cosine prefiltering,
zero vector handling, and boundary checks.
"""

import numpy as np
import pytest
from src.similarity import (
    DTWResult,
    coarse_cosine_prefilter,
    cosine_similarity_vectors,
    dtw_align_detailed,
    dtw_similarity,
    get_similarity_tier,
)


def test_dtw_similarity_identical():
    """Comparing two identical chroma matrices should yield similarity ≈ 1.0."""
    np.random.seed(42)
    chroma_a = np.random.uniform(0.0, 1.0, size=(12, 50))
    chroma_b = chroma_a.copy()

    score = dtw_similarity(chroma_a, chroma_b)

    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0
    assert np.isclose(score, 1.0, atol=1e-5), f"Expected similarity ≈ 1.0, got {score}"


def test_dtw_similarity_transposed_key():
    """Circularly pitch-shifted (transposed key) chroma matrix should still yield similarity ≈ 1.0."""
    np.random.seed(123)
    chroma_a = np.random.uniform(0.0, 1.0, size=(12, 40))
    # Shift by 4 semitones (major third transposition)
    chroma_b = np.roll(chroma_a, shift=4, axis=0)

    score = dtw_similarity(chroma_a, chroma_b)

    assert np.isclose(score, 1.0, atol=1e-5), f"Expected transposition-invariant similarity ≈ 1.0, got {score}"


def test_dtw_similarity_random_noise():
    """Comparing two independent random noise matrices should return low similarity."""
    np.random.seed(999)
    chroma_a = np.random.uniform(0.0, 1.0, size=(12, 60))
    chroma_b = np.random.uniform(0.0, 1.0, size=(12, 60))

    score = dtw_similarity(chroma_a, chroma_b)

    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0
    assert score < 0.65, f"Expected low similarity (< 0.65) for random noise, got {score}"


def test_dtw_similarity_partial_corruption():
    """Partially corrupting a chroma matrix should yield intermediate similarity."""
    np.random.seed(42)
    n_frames = 60
    chroma_orig = np.random.uniform(0.0, 1.0, size=(12, n_frames))
    chroma_corrupted = chroma_orig.copy()
    chroma_corrupted[:, slice(20, 38)] = np.random.uniform(0.0, 1.0, size=(12, 18))

    score_identical = dtw_similarity(chroma_orig, chroma_orig)
    score_corrupted = dtw_similarity(chroma_orig, chroma_corrupted)

    assert score_identical > score_corrupted
    assert 0.65 < score_corrupted < 0.95


def test_dtw_align_detailed_structure_and_path():
    """Verify DTWResult structure, alignment path continuity, and local step cost sum."""
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
    assert result.boundary_verified is True

    # Path boundary conditions
    assert result.alignment_path[0] == (0, 0)
    assert result.alignment_path[-1] == (n_a - 1, n_b - 1)

    # Local costs must sum to the raw DTW distance
    assert np.isclose(np.sum(result.local_costs), result.raw_distance, atol=1e-5)


def test_coarse_cosine_prefilter_boundary():
    """Test coarse cosine prefilter behavior around the 0.30 cutoff."""
    np.random.seed(10)
    # Query: mostly C and G active
    query_chroma = np.zeros((12, 30))
    query_chroma[0, :] = 1.0  # C
    query_chroma[7, :] = 0.8  # G

    # Candidate 1: identical harmonic profile -> high cosine similarity
    cand1 = query_chroma.copy()

    # Candidate 2: completely orthogonal profile (F# active, index 6) -> cosine = 0.0 < 0.30
    cand2 = np.zeros((12, 30))
    cand2[6, :] = 1.0

    candidates = {"cand1": cand1, "cand2": cand2}
    surviving, stats = coarse_cosine_prefilter(query_chroma, candidates, threshold=0.30)

    assert "cand1" in surviving
    assert "cand2" not in surviving
    assert stats.before_count == 2
    assert stats.after_count == 1
    assert np.isclose(stats.percentage_removed, 50.0)


def test_zero_vector_and_invalid_numerical_input():
    """Ensure zero vectors and non-finite numbers return 0.0 without throwing errors or NaNs."""
    vec_zero = np.zeros(12, dtype=float)
    vec_valid = np.ones(12, dtype=float)
    vec_nan = np.array([np.nan] * 12, dtype=float)

    # Zero vector cosine similarity
    sim_zero = cosine_similarity_vectors(vec_zero, vec_valid)
    assert sim_zero == 0.0

    # Non-finite vector cosine similarity
    sim_nan = cosine_similarity_vectors(vec_nan, vec_valid)
    assert sim_nan == 0.0

    # DTW with non-2D matrices raises clear ValueError
    with pytest.raises(ValueError):
        dtw_align_detailed(np.array([1.0, 2.0]), np.array([1.0, 2.0]))


def test_similarity_tier_boundaries():
    """Verify exact numerical boundaries for similarity tiers without display gaps."""
    assert get_similarity_tier(1.0) == "High Structural Similarity"
    assert get_similarity_tier(0.85) == "High Structural Similarity"
    assert get_similarity_tier(0.8499) == "Moderate Similarity"
    assert get_similarity_tier(0.65) == "Moderate Similarity"
    assert get_similarity_tier(0.6499) == "Low Similarity"
    assert get_similarity_tier(0.45) == "Low Similarity"
    assert get_similarity_tier(0.4499) == "Negligible"
    assert get_similarity_tier(0.0) == "Negligible"
