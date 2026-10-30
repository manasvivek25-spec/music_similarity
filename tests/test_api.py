"""
Integration and functional tests for the FastAPI backend endpoints.
"""

import io
import numpy as np
import pytest
import scipy.io.wavfile as wav
from fastapi.testclient import TestClient

from src.api import app

client = TestClient(app)


def test_health_endpoint():
    """Test health probes at /health and /api/health."""
    for path in ["/health", "/api/health"]:
        response = client.get(path)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "service_available" in data
        assert "analysis_ready" in data
        assert "origin_classifier_status" in data


def test_config_endpoint():
    """Test public runtime configuration endpoint /api/config."""
    response = client.get("/api/config")
    assert response.status_code == 200
    data = response.json()
    assert data["sample_rate"] == 22050
    assert data["min_upload_duration_sec"] == 0.0  # Unlimited duration
    assert data["similarity_floor"] == 0.45
    assert data["creative_score_verdict_threshold"] == 0.24


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200


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


def test_upload_accepts_any_clip_duration_unlimited():
    """Verify that uploading an audio track of short duration is accepted without 60s restriction."""
    sr = 22050
    # 5-second audio
    t = np.linspace(0, 5.0, int(sr * 5.0), endpoint=False)
    audio = (np.sin(2 * np.pi * 440 * t) * 16000).astype(np.int16)

    buf = io.BytesIO()
    wav.write(buf, sr, audio)
    buf.seek(0)

    files = {"query_audio": ("short_track.wav", buf, "audio/wav")}
    response = client.post("/api/analyze", files=files)

    assert response.status_code == 200
    data = response.json()
    assert "final_verdict" in data
    assert data["query_track"]["duration_sec"] == pytest.approx(5.0, abs=0.2)
    assert "total_corpus_size" in data
    assert "candidates_evaluated" in data
    assert "candidates_surviving_floor" in data
    assert "candidates_surviving_pvalue" in data
    assert "null_distribution_sample_count" in data
    top_cands = data["mimic_analysis"]["top_mimic_candidates"]
    if top_cands:
        assert "normalized_distance" in top_cands[0]


def test_mimic_requires_at_least_two_candidates():
    """Verify that /api/mimic requires at least two candidate files, returning 400 otherwise."""
    sr = 22050
    t = np.linspace(0, 1.0, int(sr * 1.0), endpoint=False)
    audio = (np.sin(2 * np.pi * 440 * t) * 16000).astype(np.int16)

    buf_q = io.BytesIO()
    wav.write(buf_q, sr, audio)
    buf_q.seek(0)

    buf_cand = io.BytesIO()
    wav.write(buf_cand, sr, audio)
    buf_cand.seek(0)

    # Only 1 candidate file
    files = [
        ("query_audio", ("q.wav", buf_q, "audio/wav")),
        ("candidate_audio_files", ("c1.wav", buf_cand, "audio/wav")),
    ]

    response = client.post("/api/mimic", files=files)
    assert response.status_code == 400
    assert "requires at least two candidate audio files" in response.json()["detail"]


def test_candidate_count_reporting_and_health_integrity():
    """Verify distinct candidate counts and health reporting without double-counting (Decision 4)."""
    res = client.get("/api/health")
    assert res.status_code == 200
    h_data = res.json()
    assert "total_indexed_tracks" in h_data
    assert "human_tracks" in h_data
    assert "ai_tracks" in h_data
    assert "tracks_with_estimated_genre" in h_data
    assert "null_pairs_global" in h_data

    from src.remote_dataset import load_remote_index
    r_idx = load_remote_index().get("tracks", {})
    expected_total = len(r_idx)
    expected_human = sum(1 for r in r_idx.values() if str(r.get("origin_label", "")).lower() in ("human", "0", "false"))
    expected_ai = sum(1 for r in r_idx.values() if str(r.get("origin_label", "")).lower() in ("ai", "ai_generated", "1", "true"))

    # Total tracks must equal real source count without double-counting (normalized 'False' -> human)
    assert h_data["total_indexed_tracks"] == expected_total
    assert h_data["human_tracks"] == expected_human
    assert h_data["ai_tracks"] == expected_ai
    assert h_data["tracks_with_estimated_genre"] == 0
    # Global null distribution not built yet -> None / null
    assert h_data["null_pairs_global"] is None


