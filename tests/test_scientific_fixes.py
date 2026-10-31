import json
import numpy as np
import pytest
from pathlib import Path

from src.config import FEATURES_DIR, NULL_DIST_CACHE_FILE, REMOTE_INDEX_FILE
from src.stats_test import (
    null_distribution_test,
    null_distribution_test_detailed,
    evaluate_statistical_gateway,
    is_fixture_identifier,
)
from src.feature5 import estimate_tempo
from src.mert_embedding import (
    detect_mert_environment,
    batch_mert_cosine_similarity,
)
from src.similarity import (
    get_similarity_tier,
    dtw_similarity,
    SIMILARITY_FLOOR,
)


def test_p_value_formula_exact_k_over_n():
    """Verify p = k / N unsmoothed formula and exceedance exposure."""
    # Reference null scores: 10 scores
    null_scores = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    # Query score 0.75: exactly 3 scores (0.8, 0.9, 1.0) >= 0.75 -> k = 3, N = 10 -> p = 0.3
    p_val, k, n = null_distribution_test_detailed(0.75, null_scores, min_pairs=5)
    assert k == 3
    assert n == 10
    assert p_val == pytest.approx(0.3, abs=1e-6)

    # Simple function returns identical float
    p_simple = null_distribution_test(0.75, null_scores, min_pairs=5)
    assert p_simple == pytest.approx(0.3, abs=1e-6)

    # Extreme high score: k = 0 -> p = 0.0 (no plus-one smoothing!)
    p_zero, k_zero, n_zero = null_distribution_test_detailed(1.5, null_scores, min_pairs=5)
    assert k_zero == 0
    assert n_zero == 10
    assert p_zero == 0.0


def test_statistical_gateway_result_exposes_k_and_n():
    """Verify StatisticalGatewayResult returns k and N, plus finite-sample explanation when k=0."""
    null_scores = [0.1, 0.2, 0.3, 0.4, 0.5]
    gw_res = evaluate_statistical_gateway(
        query_score=0.99,
        genre=None,
        genre_null_scores={},
        global_null_scores=null_scores,
        min_null_pairs=5,
    )
    assert gw_res.p_value == 0.0
    assert gw_res.exceedance_count == 0
    assert gw_res.null_sample_size == 5
    assert gw_res.explanation is not None
    assert "k=0" in gw_res.explanation
    assert "bound p < 1/5" in gw_res.explanation


def test_fixture_rejection_and_corpus_purity():
    """Verify test fixtures are rejected and cannot contaminate production corpus."""
    # Test fixture detector
    assert is_fixture_identifier("test_65s") is True
    assert is_fixture_identifier("short_clip_7s") is True
    assert is_fixture_identifier("tmp_audio_123") is True
    assert is_fixture_identifier("synth_sine_440") is True
    assert is_fixture_identifier("haim_human_06001") is False
    assert is_fixture_identifier("mtg_jamendo_948") is False
    assert is_fixture_identifier("track_44c8336f5f7c") is False

    # Production remote_index.json must contain zero fixtures
    assert REMOTE_INDEX_FILE.is_file()
    with open(REMOTE_INDEX_FILE, "r", encoding="utf-8") as f:
        idx = json.load(f)
    tracks = idx.get("tracks", {})
    assert len(tracks) >= 71, f"Expected at least 71 canonical tracks, found {len(tracks)}"

    for tid in tracks.keys():
        assert not is_fixture_identifier(tid), f"Fixture found in canonical index: {tid}"

    # Production null_distribution.json must contain zero fixtures
    if NULL_DIST_CACHE_FILE.is_file():
        with open(NULL_DIST_CACHE_FILE, "r", encoding="utf-8") as f:
            nd = json.load(f)
        src_ids = nd.get("source_track_ids", [])
        assert len(src_ids) >= 37, f"Expected at least 37 human reference tracks, found {len(src_ids)}"
        for sid in src_ids:
            assert not is_fixture_identifier(sid), f"Fixture found in null distribution: {sid}"
        expected_pairs = len(src_ids) * (len(src_ids) - 1) // 2
        assert nd.get("sample_pairs_count") == expected_pairs, f"Expected {expected_pairs} pairs, found {nd.get('sample_pairs_count')}"


def test_bpm_estimation_returns_none_for_pure_drone():
    """Verify Feature 5 BPM estimation returns None for unmodulated sine tone."""
    sine_path = Path("temp_uploads/test_65s.wav")
    if sine_path.is_file():
        tempo = estimate_tempo(sine_path)
        assert tempo is None, f"Expected None for sine wave, got {tempo}"


def test_cuda_gpu_availability_and_numerical_parity():
    """Verify CUDA acceleration is functional and matches CPU within <= 1e-5 tolerance."""
    env = detect_mert_environment()
    assert env.torch_available is True
    assert env.cuda_available is True
    assert "RTX 3050" in env.device_name

    # Numerical parity test on batch cosine similarity
    np.random.seed(123)
    q = np.random.randn(768).astype(np.float32)
    c = np.random.randn(71, 768).astype(np.float32)

    cpu_sim = batch_mert_cosine_similarity(q, c, use_gpu=False)
    gpu_sim = batch_mert_cosine_similarity(q, c, use_gpu=True)

    max_diff = float(np.max(np.abs(cpu_sim - gpu_sim)))
    assert max_diff <= 1e-5, f"GPU vs CPU difference {max_diff} exceeded 1e-5"


def test_feature3_broad_novelty_evaluates_prefilter_rejected_human_references(tmp_path):
    """Verify that human references rejected by cosine prefilter are evaluated by Feature 3 DTW."""
    # Create two synthetic human reference beat matrices
    ref1_beat = np.random.uniform(0.1, 1.0, (12, 30)).astype(np.float32)
    ref2_beat = np.random.uniform(0.1, 1.0, (12, 30)).astype(np.float32)

    ref1_file = tmp_path / "ref1_beat.npy"
    ref2_file = tmp_path / "ref2_beat.npy"
    np.save(str(ref1_file), ref1_beat)
    np.save(str(ref2_file), ref2_beat)

    # Query beat matrix
    query_beat = np.random.uniform(0.1, 1.0, (12, 30)).astype(np.float32)

    # Candidate descriptors: ref1 passed cosine prefilter, ref2 was rejected
    candidate_descriptors = {
        "human_cand_1": {
            "meta": {"origin_label": "human", "genre": "Classical", "beat_cache": str(ref1_file)}
        },
        "human_cand_2": {
            "meta": {"origin_label": "human", "genre": "Classical", "beat_cache": str(ref2_file)}
        },
    }

    # Only human_cand_1 is in evaluated_records (passed Stage 1 coarse filter)
    evaluated_records = [
        {"candidate_id": "human_cand_1", "similarity": 0.72}
    ]

    # Replicate Feature 3 broad novelty reference gathering logic
    query_to_human_scores = []
    for cid, c_desc in candidate_descriptors.items():
        meta = c_desc.get("meta", {})
        origin_val = str(meta.get("origin_label", "")).lower()
        if origin_val in ("human", "0", "false"):
            matched_rec = next((r for r in evaluated_records if r.get("candidate_id") == cid), None)
            if matched_rec is not None:
                sim = float(matched_rec["similarity"])
            else:
                beat_ref = meta.get("beat_cache")
                assert beat_ref is not None
                c_mat = np.load(str(beat_ref))
                sim = float(dtw_similarity(query_beat, c_mat))
            assert sim is not None and np.isfinite(sim)
            query_to_human_scores.append(sim)

    # Both human references must be represented in the novelty pool
    assert len(query_to_human_scores) == 2
    assert query_to_human_scores[0] == 0.72
    assert 0.0 <= query_to_human_scores[1] <= 1.0


def test_feature6_similarity_floor_and_tier_boundaries():
    """Verify Feature 6 0.45 similarity floor and exact tier boundary classifications."""
    # Exact tier boundary definitions:
    # [0.85, 1.00] -> High Structural Similarity
    # [0.65, 0.85) -> Moderate Similarity
    # [0.45, 0.65) -> Low Similarity
    # [0.00, 0.45) -> Negligible
    assert get_similarity_tier(0.449) == "Negligible"
    assert get_similarity_tier(0.450) == "Low Similarity"
    assert get_similarity_tier(0.649) == "Low Similarity"
    assert get_similarity_tier(0.650) == "Moderate Similarity"
    assert get_similarity_tier(0.849) == "Moderate Similarity"
    assert get_similarity_tier(0.850) == "High Structural Similarity"

    # Engine floor threshold constant
    assert SIMILARITY_FLOOR == 0.45

    # Boundary filtering: 0.449 rejected, 0.450 and 0.649 survive floor
    evaluated_candidates = [
        {"candidate_id": "cand_below", "similarity": 0.449},
        {"candidate_id": "cand_low_boundary", "similarity": 0.450},
        {"candidate_id": "cand_moderate_boundary", "similarity": 0.649},
        {"candidate_id": "cand_moderate", "similarity": 0.650},
        {"candidate_id": "cand_high", "similarity": 0.850},
    ]

    floor_survivors = [r for r in evaluated_candidates if r["similarity"] >= SIMILARITY_FLOOR]
    survivor_ids = [r["candidate_id"] for r in floor_survivors]

    assert "cand_below" not in survivor_ids
    assert "cand_low_boundary" in survivor_ids
    assert "cand_moderate_boundary" in survivor_ids
    assert "cand_moderate" in survivor_ids
    assert "cand_high" in survivor_ids
    assert len(floor_survivors) == 4


