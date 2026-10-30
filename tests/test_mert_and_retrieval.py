"""
Unit tests for MERT neural representation isolation, Subsequence DTW, and exact vector retrieval.
"""
import numpy as np
import pytest
from src.mert_embedding import detect_mert_environment, mert_cosine_similarity
from src.subsequence_dtw import subsequence_dtw_align, SubsequenceDTWResult
from src.retrieval import exact_vector_cosine_search, run_retrieval_benchmark, RetrievalBenchmarkResult


def test_mert_status_probe():
    status = detect_mert_environment()
    d = status.to_dict()
    assert isinstance(d, dict)
    assert "is_supported" in d
    assert "device" in d
    assert "model_id" in d
    assert d["model_id"] == "m-a-p/MERT-v1-95M"


def test_mert_cosine_similarity():
    v1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    v2 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    v3 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    v_zero = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    # Identical: (1 + 1)/2 = 1.0
    assert np.isclose(mert_cosine_similarity(v1, v2), 1.0)
    # Orthogonal: (0 + 1)/2 = 0.5
    assert np.isclose(mert_cosine_similarity(v1, v3), 0.5)
    # Zero vector returns 0.0
    assert mert_cosine_similarity(v1, v_zero) == 0.0


def test_subsequence_dtw_exact_subsequence():
    np.random.seed(42)
    # Candidate track is 100 frames
    candidate = np.random.randn(12, 100).astype(np.float32)
    candidate = candidate / np.linalg.norm(candidate, axis=0, keepdims=True)

    # Query track is an exact 30-frame slice of candidate from frame 20 to 50
    query = candidate[:, 20:50]

    result = subsequence_dtw_align(query, candidate)
    assert isinstance(result, SubsequenceDTWResult)
    assert result.best_shift == 0  # No pitch shift needed
    assert result.similarity > 0.90  # Extremely high similarity
    # Check start and end frames are recovered close to 20 and 50
    assert abs(result.candidate_start_frame - 20) <= 2
    assert abs(result.candidate_end_frame - 50) <= 2


def test_exact_vector_cosine_search():
    query_vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    candidate_matrix = np.array([
        [0.9, 0.1, 0.0],
        [0.0, 0.0, 1.0],
        [0.5, 0.5, 0.0],
    ], dtype=np.float32)
    candidate_ids = ["track_close", "track_far", "track_mid"]

    results = exact_vector_cosine_search(query_vec, candidate_matrix, candidate_ids, top_k=2)
    assert len(results) == 2
    assert results[0][0] == "track_close"
    assert results[1][0] == "track_mid"
    assert results[0][1] > results[1][1]


def test_run_retrieval_benchmark():
    res = run_retrieval_benchmark(num_candidates=20, dimension=64, top_k=5)
    assert isinstance(res, RetrievalBenchmarkResult)
    d = res.to_dict()
    assert "numpy_exact_latency_ms" in d
    assert d["num_candidates"] == 20
    assert d["vector_dimension"] == 64
