"""
Unit & Engineering Invariant Tests for v2 Pipeline, F6 Structural Mimic Evidence,
Case Routing Consistency, Null Cache Invalidation, Label Polarity, Dedupe/Leakage,
Resumable Index State, and Content-Hash Ground-Truth Lookup.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.config import V2_INDEX_STATE_FILE
from src.mimic import (
    build_mimic_analysis,
    calculate_path_mimic_evidence,
    classify_mimic_evidence_level,
)
from src.pipeline import evaluate_final_verdict
from src.schemas import MetadataContext
from src.stats_test import compute_null_cache_key
from src.truth_lookup import lookup_upload_ground_truth
from src.v2_indexer import verify_label_and_provenance


# ---------------------------------------------------------------------------
# 1. F6 Structural Mimic Evidence Tests
# ---------------------------------------------------------------------------

def test_f6_path_evidence_high_continuity_diagonal():
    # A clean diagonal path of 24 steps: first 16 low-cost (0.1), last 8 high-cost (1.2)
    path = [(i, i) for i in range(24)]
    costs = [0.1] * 16 + [1.2] * 8

    ev = calculate_path_mimic_evidence(
        alignment_path=path,
        local_costs=costs,
    )
    assert ev["significant_region_count"] >= 1
    assert 0.8 <= ev["region_continuity"] <= 1.0
    assert 0.8 <= ev["sequential_consistency"] <= 1.0
    assert 0.5 <= ev["local_similarity_concentration"] <= 1.0


def test_f6_evidence_level_classification_and_suppression():
    # High: Case A/B, p <= 0.01, coverage >= 0.30, continuity >= 0.45, seq >= 0.70, conc >= 0.35, >=1 region
    assert classify_mimic_evidence_level(
        similarity=0.90,
        p_value=0.005,
        artist_relationship="Different Artist",
        alignment_coverage=0.45,
        significant_region_count=2,
        region_continuity=0.80,
        sequential_consistency=0.85,
        local_similarity_concentration=0.60,
        case_type="Case B",
    ) == "High"

    # Moderate: Case A/B, p <= 0.10, moderate coverage/continuity
    assert classify_mimic_evidence_level(
        similarity=0.72,
        p_value=0.04,
        artist_relationship="Different Artist",
        alignment_coverage=0.20,
        significant_region_count=1,
        region_continuity=0.55,
        sequential_consistency=0.60,
        local_similarity_concentration=0.30,
        case_type="Case A",
    ) == "Moderate"

    # Low: Case A/B, p > 0.10 with region present
    assert classify_mimic_evidence_level(
        similarity=0.60,
        p_value=0.25,
        artist_relationship="Different Artist",
        alignment_coverage=0.50,
        significant_region_count=1,
        region_continuity=0.80,
        sequential_consistency=0.80,
        local_similarity_concentration=0.50,
        case_type="Case A",
    ) == "Low"

    # Case C & Case U MUST return "Not applicable"
    assert classify_mimic_evidence_level(
        similarity=0.92,
        p_value=0.001,
        artist_relationship="Different Artist",
        alignment_coverage=0.90,
        significant_region_count=3,
        region_continuity=0.95,
        sequential_consistency=0.95,
        local_similarity_concentration=0.80,
        case_type="Case C",
    ) == "Not applicable"

    assert classify_mimic_evidence_level(
        similarity=0.92,
        p_value=0.001,
        artist_relationship="Different Artist",
        alignment_coverage=0.90,
        significant_region_count=3,
        region_continuity=0.95,
        sequential_consistency=0.95,
        local_similarity_concentration=0.80,
        case_type="Case U",
    ) == "Not applicable"

    # Same artist ("Consistent Artist Style") MUST be suppressed -> "Not applicable"
    assert classify_mimic_evidence_level(
        similarity=0.92,
        p_value=0.001,
        artist_relationship="Consistent Artist Style",
        alignment_coverage=0.90,
        significant_region_count=3,
        region_continuity=0.95,
        sequential_consistency=0.95,
        local_similarity_concentration=0.80,
        case_type="Case B",
    ) == "Not applicable"

    # Missing p_value MUST NEVER be High/Moderate/Low -> "Not applicable"
    assert classify_mimic_evidence_level(
        similarity=0.92,
        p_value=None,
        artist_relationship="Different Artist",
        alignment_coverage=0.90,
        significant_region_count=3,
        region_continuity=0.95,
        sequential_consistency=0.95,
        local_similarity_concentration=0.80,
        case_type="Case B",
    ) == "Not applicable"


def test_f6_build_mimic_analysis_missing_p_and_same_artist():
    q_ctx = MetadataContext(
        track_id="q0", duration_sec=10.0, title="Query Track", artist="Alice", genre="Electronic"
    )
    path = [(i, i) for i in range(24)]
    costs = [0.1] * 16 + [1.2] * 8
    candidates = [
        {
            "candidate_id": "t1",
            "title": "Same Artist Song",
            "artist": "Alice",
            "genre": "Electronic",
            "similarity": 0.91,
            "similarity_tier": "High",
            "best_shift": 0,
            "p_value": 0.002,
            "alignment_coverage": 0.65,
            "alignment_path": path,
            "local_costs": costs,
        },
        {
            "candidate_id": "t2",
            "title": "Different Artist Missing P",
            "artist": "Bob",
            "genre": "Electronic",
            "similarity": 0.93,
            "similarity_tier": "High",
            "best_shift": 2,
            "p_value": None,
            "alignment_coverage": 0.70,
            "alignment_path": path,
            "local_costs": costs,
        },
        {
            "candidate_id": "t3",
            "title": "Different Artist Significant",
            "artist": "Charlie",
            "genre": "Electronic",
            "similarity": 0.88,
            "similarity_tier": "High",
            "best_shift": 9,
            "p_value": 0.004,
            "alignment_coverage": 0.55,
            "alignment_path": path,
            "local_costs": costs,
        },
    ]
    resp = build_mimic_analysis(q_ctx, candidates, max_mimic_candidates=3, case_type="Case B")
    by_id = {c.candidate_id: c for c in resp.top_mimic_candidates}

    # t1: same artist suppressed
    assert by_id["t1"].artist_relationship == "Consistent Artist Style"
    assert by_id["t1"].mimic_warning_active is False
    assert by_id["t1"].mimic_evidence_level == "Not applicable"

    # t2: missing p_value MUST NOT trigger mimic_warning or High evidence
    assert by_id["t2"].mimic_warning_active is False
    assert by_id["t2"].mimic_evidence_level == "Not applicable"

    # t3: genuine high mimic evidence
    assert by_id["t3"].mimic_warning_active is True
    assert by_id["t3"].mimic_evidence_level == "High"


# ---------------------------------------------------------------------------
# 2. Case C Verdict Consistency (regression test for previous Incomplete bug)
# ---------------------------------------------------------------------------

def test_case_c_with_valid_null_returns_clear_not_incomplete():
    verdict = evaluate_final_verdict(
        best_similarity=0.58,
        best_p_value=0.22,
        normalized_creative_score=0.80,
        confidence="High Statistical Confidence",
        candidates_evaluated=100,
        null_sample_count=2000,
    )
    assert verdict.verdict_code == "clear"
    assert "incomplete" not in verdict.verdict_label.lower()

    verdict_u = evaluate_final_verdict(
        best_similarity=0.58,
        best_p_value=None,
        normalized_creative_score=None,
        confidence=None,
        candidates_evaluated=100,
        null_sample_count=0,
    )
    assert verdict_u.verdict_code == "incomplete"


# ---------------------------------------------------------------------------
# 3. Null Cache Invalidation Key Determinism & Sensitivity
# ---------------------------------------------------------------------------

def test_null_cache_key_deterministic_and_sensitive():
    k1 = compute_null_cache_key(
        reference_ids=["b", "a", "c"],
        max_pairs=2000,
        seed=42,
        feature_version="v2.0",
        algorithm_version="v2.0",
        index_max_seconds=10.0,
    )
    k2 = compute_null_cache_key(
        reference_ids=["c", "b", "a"],  # different order, same set
        max_pairs=2000,
        seed=42,
        feature_version="v2.0",
        algorithm_version="v2.0",
        index_max_seconds=10.0,
    )
    assert k1 == k2

    # Any config or population change MUST invalidate the key
    assert k1 != compute_null_cache_key(["a", "b", "d"], 2000, 42, "v2.0", "v2.0", None, 10.0)
    assert k1 != compute_null_cache_key(["a", "b", "c"], 1000, 42, "v2.0", "v2.0", None, 10.0)
    assert k1 != compute_null_cache_key(["a", "b", "c"], 2000, 99, "v2.0", "v2.0", None, 10.0)
    assert k1 != compute_null_cache_key(["a", "b", "c"], 2000, 42, "v2.1", "v2.0", None, 10.0)
    assert k1 != compute_null_cache_key(["a", "b", "c"], 2000, 42, "v2.0", "v2.0", None, 30.0)


# ---------------------------------------------------------------------------
# 4. Label Polarity Verification
# ---------------------------------------------------------------------------

def test_label_polarity_verification():
    assert verify_label_and_provenance("human_fma_small", {})[0] == "human"
    assert verify_label_and_provenance(
        "sleepyjesse", {"source": "free-music-archive-full", "ai_generated": False}
    )[0] == "human"
    assert verify_label_and_provenance(
        "sleepyjesse", {"source": "ManzhenWei/MusicSet", "ai_generated": True}
    )[:2] == ("ai", "MusicSet")
    assert verify_label_and_provenance(
        "sleepyjesse", {"source": "SunoCaps", "ai_generated": True}
    )[:2] == ("ai", "SunoCaps")
    assert verify_label_and_provenance(
        "sleepyjesse", {"source": "blanchon/udio_dataset", "ai_generated": True}
    )[:2] == ("ai", "Udio")

    # Polarity conflicts MUST be rejected as 'unknown'
    assert verify_label_and_provenance(
        "sleepyjesse", {"source": "free-music-archive-full", "ai_generated": True}
    )[0] == "unknown"
    assert verify_label_and_provenance(
        "sleepyjesse", {"source": "SunoCaps", "ai_generated": False}
    )[0] == "unknown"


# ---------------------------------------------------------------------------
# 5. Content-Hash Ground-Truth Lookup
# ---------------------------------------------------------------------------

def test_truth_lookup_unmatched_and_real_held_out():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        audio_file = tmp_path / "dummy.wav"
        audio_file.write_bytes(b"RIFF0000WAVEfmt_test_payload_unique_999999")

        res_none = lookup_upload_ground_truth(audio_file, predicted_origin="Human")
        assert res_none["available"] is False
        assert "No ground truth available" in res_none["message"]

    # If v2 held-out audio exists, verify real content-hash match works
    ho_json = Path(__file__).resolve().parent.parent / "data" / "v2" / "v2_held_out.json"
    if ho_json.is_file():
        tracks = json.loads(ho_json.read_text(encoding="utf-8")).get("tracks", {})
        for rec in tracks.values():
            wp = Path(__file__).resolve().parent.parent / rec["norm_wav_path"]
            if wp.is_file():
                expected = "AI" if rec["origin_label"] == "ai" else "Human"
                res_hit = lookup_upload_ground_truth(wp, predicted_origin=expected)
                assert res_hit["available"] is True
                assert res_hit["origin_comparison"]["true_origin"] == expected
                assert res_hit["origin_comparison"]["match"] is True
                break


# ---------------------------------------------------------------------------
# 6. v2 Dataset Audit & Resumable Index State Verification (if v2 present)
# ---------------------------------------------------------------------------

def test_v2_leakage_audit_and_index_state_if_present():
    audit_path = Path(__file__).resolve().parent.parent / "reports" / "dataset_audit.json"
    if audit_path.is_file():
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        assert audit["leakage_audit"]["status"] == "PASSED_ZERO_LEAKAGE"
        assert audit["leakage_audit"]["total_leakage_violations"] == 0
        for key in ("track_id_overlaps", "group_id_overlaps", "content_hash_overlaps"):
            for pair, cnt in audit["leakage_audit"][key].items():
                assert cnt == 0, f"Overlap in {key}/{pair}"
        post_lr = audit["confound_probes"]["post_normalization_with_silence_ratio"]["lr_roc_auc"]
        post_rf = audit["confound_probes"]["post_normalization_with_silence_ratio"]["rf_roc_auc"]
        assert 0.40 <= post_lr <= 0.60, f"Post-norm LR confound AUC too far from chance: {post_lr}"
        assert 0.40 <= post_rf <= 0.60, f"Post-norm RF confound AUC too far from chance: {post_rf}"

    if V2_INDEX_STATE_FILE.is_file():
        state = json.loads(V2_INDEX_STATE_FILE.read_text(encoding="utf-8"))
        assert state["status"] in {"NOT_STARTED", "INDEXING", "PAUSED", "COMPLETED", "FAILED"}
        assert state["indexed"] >= 400
        assert state["is_complete"] is True


# ---------------------------------------------------------------------------
# 7. Terminology Guard (No "fraud" or "plagiarism" in mimic/verdict wording)
# ---------------------------------------------------------------------------

def test_no_fraud_or_plagiarism_in_diagnostic_code():
    mimic_src = (Path(__file__).resolve().parent.parent / "src" / "mimic.py").read_text(encoding="utf-8").lower()
    assert "fraud" not in mimic_src
    assert "plagiarism" not in mimic_src


def test_corrupt_audio_upload_returns_400_not_500():
    from fastapi.testclient import TestClient
    from src.api import app
    client = TestClient(app)
    resp = client.post(
        "/api/analyze",
        files={"query_audio": ("corrupt.wav", b"RIFF\x00\x00\x00\x00WAVEfmt_corrupted_garbage", "audio/wav")},
    )
    assert resp.status_code == 400

