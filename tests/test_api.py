"""
Integration and functional tests for the FastAPI backend endpoints.
"""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["version"] == "0.1.0"


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert "Music Similarity" in response.text or "/docs" in response.text


def test_pairwise_similarity_matrix_identical():
    np.random.seed(42)
    chroma_a = np.random.uniform(0.0, 1.0, size=(12, 30)).tolist()

    response = client.post(
        "/api/similarity/pairwise/matrix",
        json={"chroma_a": chroma_a, "chroma_b": chroma_a},
    )
    assert response.status_code == 200
    data = response.json()
    assert np.isclose(data["similarity"], 1.0, atol=1e-4)
    assert data["best_shift"] == 0


def test_pairwise_similarity_matrix_transposed():
    np.random.seed(42)
    mat_a = np.random.uniform(0.0, 1.0, size=(12, 25))
    # Shift up by 3 semitones
    mat_b = np.roll(mat_a, shift=3, axis=0)

    response = client.post(
        "/api/similarity/pairwise/matrix",
        json={"chroma_a": mat_a.tolist(), "chroma_b": mat_b.tolist()},
    )
    assert response.status_code == 200
    data = response.json()
    assert np.isclose(data["similarity"], 1.0, atol=1e-4)
    # Realignment requires (12 - 3) % 12 = 9
    assert data["best_shift"] == 9


def test_detailed_alignment_matrix():
    np.random.seed(99)
    mat_a = np.random.uniform(0.0, 1.0, size=(12, 20)).tolist()
    mat_b = np.random.uniform(0.0, 1.0, size=(12, 22)).tolist()

    response = client.post(
        "/api/similarity/detailed/matrix",
        json={"chroma_a": mat_a, "chroma_b": mat_b},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["total_steps"] > 0
    assert len(data["alignment_path"]) == data["total_steps"]
    assert len(data["local_costs"]) == data["total_steps"]


def test_suggestor_box_severity_tiers():
    np.random.seed(101)
    mat_a = np.random.uniform(0.0, 1.0, size=(12, 25)).tolist()

    # High similarity test with low p-value override -> High tier warning
    resp_high = client.post(
        "/api/similarity/suggest/matrix?p_value_override=0.005",
        json={"chroma_a": mat_a, "chroma_b": mat_a},
    )
    assert resp_high.status_code == 200
    data_high = resp_high.json()
    assert data_high["severity_tier"] == "high"
    assert len(data_high["suggestions"]) > 0

    # Mild tier test (0.01 < p <= 0.10)
    resp_mild = client.post(
        "/api/similarity/suggest/matrix?p_value_override=0.05",
        json={"chroma_a": mat_a, "chroma_b": mat_a},
    )
    assert resp_mild.status_code == 200
    assert resp_mild.json()["severity_tier"] == "mild"

    # None tier test (p > 0.10)
    resp_none = client.post(
        "/api/similarity/suggest/matrix?p_value_override=0.25",
        json={"chroma_a": mat_a, "chroma_b": mat_a},
    )
    assert resp_none.status_code == 200
    assert resp_none.json()["severity_tier"] == "none"


def test_null_test_endpoint():
    response = client.post(
        "/api/stats/null-test",
        json={"query_score": 0.85, "null_scores": [0.2, 0.4, 0.6, 0.8, 0.9]},
    )
    assert response.status_code == 200
    data = response.json()
    # 0.9 >= 0.85 -> 1 out of 5 -> p = 0.2
    assert np.isclose(data["p_value"], 0.2)
    assert data["is_significant"] is False


def test_population_comparison_endpoint():
    human_scores = [0.35, 0.38, 0.42, 0.40, 0.39, 0.41]
    ai_scores = [0.70, 0.72, 0.68, 0.75, 0.73, 0.71]

    response = client.post(
        "/api/stats/population-comparison",
        json={"human_scores": human_scores, "ai_scores": ai_scores},
    )
    assert response.status_code == 200
    data = response.json()
    assert "u_statistic" in data
    assert data["p_value"] < 0.05
    assert data["ai_median"] > data["human_median"]


def test_candidate_ranking_endpoint():
    np.random.seed(55)
    query = np.random.uniform(0.0, 1.0, size=(12, 25)).tolist()
    cand_identical = query
    cand_noise_1 = np.random.uniform(0.0, 1.0, size=(12, 25)).tolist()
    cand_noise_2 = np.random.uniform(0.0, 1.0, size=(12, 25)).tolist()

    payload = {
        "query_id": "original_track_01",
        "query_chroma": query,
        "candidates": [
            {"candidate_id": "cand_unrelated_a", "chroma": cand_noise_1},
            {"candidate_id": "cand_mimic_target", "chroma": cand_identical},
            {"candidate_id": "cand_unrelated_b", "chroma": cand_noise_2},
        ],
    }

    response = client.post("/api/candidates/rank/matrix", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["top_match"]["candidate_id"] == "cand_mimic_target"
    assert data["top_match"]["is_top_match"] is True
    assert np.isclose(data["top_match"]["similarity"], 1.0, atol=1e-4)
    assert len(data["candidates"]) == 3


def test_candidate_ranking_audio_dataset():
    import io
    import scipy.io.wavfile as wav

    sr = 22050
    t = np.linspace(0, 0.4, int(sr * 0.4), endpoint=False)
    audio_a = (np.sin(2 * np.pi * 440 * t) * 16000).astype(np.int16)
    audio_b = (np.sin(2 * np.pi * 880 * t) * 16000).astype(np.int16)

    buf_query = io.BytesIO()
    wav.write(buf_query, sr, audio_a)
    buf_query.seek(0)

    buf_cand1 = io.BytesIO()
    wav.write(buf_cand1, sr, audio_a)
    buf_cand1.seek(0)

    buf_cand2 = io.BytesIO()
    wav.write(buf_cand2, sr, audio_b)
    buf_cand2.seek(0)

    files = [
        ("query_file", ("query.wav", buf_query, "audio/wav")),
        ("candidate_files", ("match_exact.wav", buf_cand1, "audio/wav")),
        ("candidate_files", ("different.wav", buf_cand2, "audio/wav")),
    ]

    response = client.post("/api/candidates/rank/audio", files=files)
    assert response.status_code == 200
    data = response.json()
    assert len(data["candidates"]) == 2
    assert data["top_match"]["candidate_id"] == "match_exact"
    assert data["top_match"]["is_top_match"] is True
    assert data["top_match"]["similarity"] > data["candidates"][1]["similarity"]
