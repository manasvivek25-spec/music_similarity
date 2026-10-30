"""
Comprehensive regression tests for the entire-dataset scanning architecture:
- Full-index coverage accounting across SleepyJesse/ai_music_large
- Resumable indexing without duplicate processing or re-decoding
- Unresolved record error tracking and retry mechanisms
- Normalized origin labels ("False" -> human, "True" -> ai_generated)
- Unknown genre handling without hardcoded ambient defaults
- Two-stage retrieval: 100% prefilter coverage, DTW only for survivors
- Missing/corrupted candidate feature file fault tolerance
- Duration disparity detection (>= 2.5x ratio)
- Atomic disk caching for empirical null distributions
- Insufficient statistical evidence (N < 5) and State U routing
- Asynchronous 13-stage background job management and status polling
"""

import io
import json
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from src.api import app, job_manager
from src.config import DURATION_DISPARITY_RATIO
from src.pipeline import run_pipeline
from src.remote_dataset import (
    index_remote_dataset,
    load_remote_index,
    parse_origin_label,
    save_remote_index,
)
from src.stats_test import (
    get_or_build_cached_null_distribution,
    invalidate_null_distribution_cache,
    null_distribution_test,
)


# ---------------------------------------------------------------------------
# Test Helpers
# ---------------------------------------------------------------------------

def make_wav_bytes(duration_sec: float = 1.0, sr: int = 22050, freq: float = 440.0) -> bytes:
    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)
    sig = 0.5 * np.sin(2 * np.pi * freq * t)
    bio = io.BytesIO()
    sf.write(bio, sig.astype(np.float32), sr, format="WAV")
    return bio.getvalue()


# ---------------------------------------------------------------------------
# 1. Origin Label Normalization Tests
# ---------------------------------------------------------------------------

def test_parse_origin_label_variations():
    """Verify raw boolean strings and types normalize strictly to human or ai_generated."""
    # Human cases (including raw string "False" from SleepyJesse/ai_music_large)
    assert parse_origin_label("False") == "human"
    assert parse_origin_label("false") == "human"
    assert parse_origin_label("FALSE") == "human"
    assert parse_origin_label(False) == "human"
    assert parse_origin_label("0") == "human"
    assert parse_origin_label(0) == "human"
    assert parse_origin_label("human") == "human"
    assert parse_origin_label("Human") == "human"

    # AI cases
    assert parse_origin_label("True") == "ai_generated"
    assert parse_origin_label("true") == "ai_generated"
    assert parse_origin_label("TRUE") == "ai_generated"
    assert parse_origin_label(True) == "ai_generated"
    assert parse_origin_label("1") == "ai_generated"
    assert parse_origin_label(1) == "ai_generated"
    assert parse_origin_label("ai") == "ai_generated"
    assert parse_origin_label("ai_generated") == "ai_generated"

    # Unknown cases
    assert parse_origin_label(None) == "unknown"
    assert parse_origin_label("") == "unknown"
    assert parse_origin_label("unspecified") == "unknown"


# ---------------------------------------------------------------------------
# 2. Resumable Indexing & No-Duplicate Tests
# ---------------------------------------------------------------------------

def test_resumable_indexing_preserves_existing_and_skips_duplicates():
    """Verify that indexing resumes from the existing index without duplicate entries."""
    temp_dir = Path(tempfile.mkdtemp(prefix="test_idx_resume_"))
    try:
        index_path = temp_dir / "test_remote_index.json"
        features_dir = temp_dir / "features"
        features_dir.mkdir(parents=True, exist_ok=True)

        wav_bytes_1 = make_wav_bytes(duration_sec=0.5, freq=440.0)
        wav_bytes_2 = make_wav_bytes(duration_sec=0.5, freq=554.37)
        wav_bytes_3 = make_wav_bytes(duration_sec=0.5, freq=659.25)

        initial_stream = [
            {"audio": {"bytes": wav_bytes_1}, "title": "Track 0", "ai_generated": "False", "id": "t0"},
            {"audio": {"bytes": wav_bytes_2}, "title": "Track 1", "ai_generated": "True", "id": "t1"},
        ]

        rep1 = index_remote_dataset(
            sample_size=2,
            is_full=False,
            force_reindex=False,
            index_file=index_path,
            features_cache_dir=features_dir,
            dataset_override=initial_stream,
        )

        assert rep1.processed_records == 2
        assert rep1.cache_hits == 0
        idx1 = load_remote_index(index_path)
        assert len(idx1["tracks"]) == 2

        # Second pass with 3 items (2 already indexed, 1 new)
        second_stream = [
            {"audio": {"bytes": wav_bytes_1}, "title": "Track 0", "ai_generated": "False", "id": "t0"},
            {"audio": {"bytes": wav_bytes_2}, "title": "Track 1", "ai_generated": "True", "id": "t1"},
            {"audio": {"bytes": wav_bytes_3}, "title": "Track 2", "ai_generated": "False", "id": "t2"},
        ]

        rep2 = index_remote_dataset(
            sample_size=3,
            is_full=False,
            force_reindex=False,
            index_file=index_path,
            features_cache_dir=features_dir,
            dataset_override=second_stream,
        )

        assert rep2.cache_hits == 2
        assert rep2.processed_records == 1
        idx2 = load_remote_index(index_path)
        assert len(idx2["tracks"]) == 3
        # Unique track IDs without duplicate processing
        assert len(set(idx2["tracks"].keys())) == 3
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# 3. Corrupted Stream Record & Unresolved Failure Handling
# ---------------------------------------------------------------------------

def test_unresolved_failure_tracking_and_completeness_flag():
    """Verify that corrupt stream records are tracked in unresolved_records and completeness is guarded."""
    temp_dir = Path(tempfile.mkdtemp(prefix="test_idx_err_"))
    try:
        index_path = temp_dir / "test_remote_index.json"
        features_dir = temp_dir / "features"
        features_dir.mkdir(parents=True, exist_ok=True)

        valid_wav = make_wav_bytes(duration_sec=0.5, freq=440.0)
        corrupt_stream = [
            {"audio": {"bytes": valid_wav}, "title": "Good Track", "id": "good_1"},
            {"audio": {"bytes": b"BAD CORRUPT AUDIO BYTES"}, "title": "Corrupt Track", "id": "bad_1"},
        ]

        rep = index_remote_dataset(
            sample_size=2,
            is_full=False,
            force_reindex=False,
            index_file=index_path,
            features_cache_dir=features_dir,
            dataset_override=corrupt_stream,
        )

        assert rep.processed_records == 1
        assert rep.failed_records == 1

        idx = load_remote_index(index_path)
        assert idx["is_complete"] is False
        assert len(idx["tracks"]) == 1
        assert len(idx["unresolved_records"]) == 1
        unresolved_item = list(idx["unresolved_records"].values())[0]
        assert "error" in unresolved_item
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# 4. Unknown Genre Handling (No Hardcoded Fallback)
# ---------------------------------------------------------------------------

def test_unknown_genre_handling_without_ambient_fallback():
    """Verify unknown genre remains Unknown / unknown source rather than ambient."""
    temp_dir = Path(tempfile.mkdtemp(prefix="test_idx_genre_"))
    try:
        index_path = temp_dir / "test_remote_index.json"
        features_dir = temp_dir / "features"
        features_dir.mkdir(parents=True, exist_ok=True)

        stream = [
            {"audio": {"bytes": make_wav_bytes(0.5)}, "title": "No Genre Track", "id": "nog_1"},
        ]

        index_remote_dataset(
            sample_size=1,
            is_full=False,
            index_file=index_path,
            features_cache_dir=features_dir,
            dataset_override=stream,
        )

        idx = load_remote_index(index_path)
        track = list(idx["tracks"].values())[0]
        assert track.get("genre") is None or track.get("genre") == "Unknown"
        assert track.get("genre_source") == "unknown"
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# 5. Two-Stage Candidate Retrieval & Missing File Fault Tolerance
# ---------------------------------------------------------------------------

def test_two_stage_prefilter_accounting_and_missing_file_fault_tolerance():
    """
    Verify:
    1. 100% of indexed tracks are prefilter-checked in memory.
    2. Only surviving candidates (>= 0.30) undergo DTW.
    3. Missing cached files are accounted for in candidates_failed_to_load without crashing.
    """
    temp_dir = Path(tempfile.mkdtemp(prefix="test_twostage_"))
    try:
        features_dir = temp_dir / "features"
        features_dir.mkdir(parents=True, exist_ok=True)

        # Create query audio
        q_path = temp_dir / "query.wav"
        sf.write(str(q_path), np.random.randn(22050 * 3).astype(np.float32), 22050)

        # Build index data with 4 tracks:
        # cand_0: Highly correlated chroma -> passes prefilter, valid feature file
        # cand_1: Highly correlated chroma -> passes prefilter, MISSING feature file
        # cand_2: Orthogonal chroma -> filtered out by cosine prefilter (< 0.30)
        # cand_3: Orthogonal chroma -> filtered out by cosine prefilter (< 0.30)
        vector_match = [1.0] * 12
        vector_unmatch = [0.0] * 12

        idx_data = {
            "dataset_name": "test/dataset",
            "split": "train",
            "is_complete": True,
            "tracks": {
                "cand_0": {
                    "track_id": "cand_0",
                    "title": "Match Surviving",
                    "artist": "Artist A",
                    "genre": "Classical",
                    "genre_source": "metadata",
                    "origin_label": "human",
                    "mean_chroma": vector_match,
                    "beat_feature_path": str(features_dir / "cand_0_beat.npy"),
                    "duration_sec": 3.0,
                },
                "cand_1": {
                    "track_id": "cand_1",
                    "title": "Match Missing File",
                    "artist": "Artist B",
                    "genre": "Classical",
                    "genre_source": "metadata",
                    "origin_label": "ai_generated",
                    "mean_chroma": vector_match,
                    "beat_feature_path": str(features_dir / "cand_1_missing_beat.npy"),  # Does not exist!
                    "duration_sec": 3.0,
                },
                "cand_2": {
                    "track_id": "cand_2",
                    "title": "Unmatched 1",
                    "artist": "Artist C",
                    "genre": "Electronic",
                    "genre_source": "metadata",
                    "origin_label": "human",
                    "mean_chroma": vector_unmatch,
                    "beat_feature_path": str(features_dir / "cand_2_beat.npy"),
                    "duration_sec": 3.0,
                },
                "cand_3": {
                    "track_id": "cand_3",
                    "title": "Unmatched 2",
                    "artist": "Artist D",
                    "genre": "Rock",
                    "genre_source": "metadata",
                    "origin_label": "human",
                    "mean_chroma": vector_unmatch,
                    "beat_feature_path": str(features_dir / "cand_3_beat.npy"),
                    "duration_sec": 3.0,
                },
            },
            "unresolved_records": {},
        }

        # Create the valid .npy for cand_0
        np.save(str(features_dir / "cand_0_beat.npy"), np.ones((12, 10), dtype=np.float32))

        # Run pipeline with progress callback
        stages_seen = []
        def on_progress(stage, name, pct, cur, tot, msg):
            stages_seen.append((stage, name))

        with patch("src.remote_dataset.load_remote_index", return_value=idx_data):
            res = run_pipeline(q_path, progress_callback=on_progress)

        assert res.candidates_considered == 4
        assert res.candidates_retained_prefilter == 2
        assert res.candidates_discarded_prefilter == 2
        assert res.candidates_failed_to_load == 1
        assert len(res.failed_candidate_details) == 1
        assert res.failed_candidate_details[0]["candidate_id"] == "cand_1"
        assert res.candidates_evaluated == 1
        assert res.is_index_complete is True
        assert "Complete" in res.corpus_status

        # Check that stages were executed
        unique_stages = set(s[0] for s in stages_seen)
        assert len(unique_stages) >= 12
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# 6. Duration Disparity Detection Tests
# ---------------------------------------------------------------------------

def test_duration_disparity_ratio_threshold():
    """Verify that candidate duration >= 2.5x query duration triggers duration disparity warning."""
    temp_dir = Path(tempfile.mkdtemp(prefix="test_disparity_"))
    try:
        features_dir = temp_dir / "features"
        features_dir.mkdir(parents=True, exist_ok=True)

        # Query is 10 seconds
        q_path = temp_dir / "query.wav"
        sf.write(str(q_path), np.random.randn(22050 * 10).astype(np.float32), 22050)

        # Candidate is 30 seconds (3.0x ratio >= 2.5)
        np.save(str(features_dir / "long_beat.npy"), np.ones((12, 60), dtype=np.float32))

        idx_data = {
            "dataset_name": "test/dataset",
            "tracks": {
                "long_track": {
                    "track_id": "long_track",
                    "title": "Long Symphony",
                    "artist": "Orchestra",
                    "genre": "Classical",
                    "genre_source": "metadata",
                    "origin_label": "human",
                    "mean_chroma": [1.0] * 12,
                    "beat_feature_path": str(features_dir / "long_beat.npy"),
                    "duration_sec": 30.0,
                }
            },
            "unresolved_records": {},
        }

        with patch("src.remote_dataset.load_remote_index", return_value=idx_data):
            res = run_pipeline(q_path)

        assert res.duration_disparity_detected is True
        top_cand = res.mimic_analysis.top_mimic_candidates[0]
        assert top_cand.duration_disparity_warning is True
        assert top_cand.disparity_explanation is not None
        assert "duration disparity" in top_cand.disparity_explanation.lower()
        assert "stretching" in top_cand.disparity_explanation.lower()
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# 7. Null Distribution Disk Caching Tests
# ---------------------------------------------------------------------------

def test_null_distribution_disk_caching_and_invalidation():
    """Verify that empirical null distribution is saved to disk and invalidation works."""
    temp_dir = Path(tempfile.mkdtemp(prefix="test_null_cache_"))
    try:
        cache_file = temp_dir / "null_distribution.json"
        assert not cache_file.exists()

        candidates = {
            "t1": np.ones((12, 10)),
            "t2": np.ones((12, 10)) * 0.8,
            "t3": np.ones((12, 10)) * 0.6,
        }

        def mock_sim(c1, c2):
            return 0.75

        scores, _ = get_or_build_cached_null_distribution(candidates, similarity_func=mock_sim, cache_file=cache_file)
        assert len(scores) > 0
        assert cache_file.is_file()

        # Modify cache file content to verify subsequent read uses the cache
        with open(cache_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        data["null_scores"] = [0.999]
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(data, f)

        scores_cached, _ = get_or_build_cached_null_distribution(candidates, similarity_func=mock_sim, cache_file=cache_file)
        assert scores_cached == [0.999]

        # Invalidate cache
        invalidate_null_distribution_cache(cache_file=cache_file)
        assert not cache_file.exists()
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# 8. Insufficient Statistical Evidence & State U Routing
# ---------------------------------------------------------------------------

def test_insufficient_null_pairs_returns_state_u():
    """Verify that when null pool has < 5 pairs, null_distribution_test returns None."""
    p_val = null_distribution_test(0.85, null_scores=[0.5, 0.6, 0.7])  # Only 3 pairs < 5
    assert p_val is None


# ---------------------------------------------------------------------------
# 9. Asynchronous 13-Stage Job Manager Tests
# ---------------------------------------------------------------------------

def test_job_manager_lifecycle():
    """Verify job manager creates, updates, and tracks 13-stage jobs."""
    job_id = job_manager.create_job(Path("mock.wav"), {"title": "Test Track"})
    assert job_id is not None

    status = job_manager.get_job(job_id)
    assert status["status"] == "queued"
    assert status["stage"] >= 1

    # Advance stage
    job_manager.update_job_progress(
        job_id,
        stage=4,
        stage_name="Cosine Prefilter",
        pct=30.0,
        cur=10,
        tot=100,
        msg="Prefiltered 10/100 candidates",
    )

    updated = job_manager.get_job(job_id)
    assert updated["stage"] == 4
    assert updated["progress_pct"] == 30.0
    assert len(updated["events"]) == 1

    # Complete job
    job_manager.complete_job(job_id, result=None)
    completed = job_manager.get_job(job_id)
    assert completed["status"] == "completed"
    assert completed["progress_pct"] == 100.0


# ---------------------------------------------------------------------------
# 10. API Index Status Endpoint Test
# ---------------------------------------------------------------------------

def test_api_index_status_endpoint():
    """Verify GET /api/index/status returns expected schema and fields."""
    client = TestClient(app)
    resp = client.get("/api/index/status")
    assert resp.status_code == 200
    data = resp.json()
    assert "total_indexed" in data
    assert "is_complete" in data
    assert "status" in data
    assert "human_tracks" in data
    assert "ai_tracks" in data
    assert "unresolved_records_count" in data


# ---------------------------------------------------------------------------
# 11. Dataset Scan API Endpoints & State Management
# ---------------------------------------------------------------------------

def test_dataset_scan_api_endpoints():
    """Verify dataset scan start, stop, and status endpoints operate cleanly."""
    client = TestClient(app)

    # 1. Check status
    stat_resp = client.get("/api/dataset/scan/status")
    assert stat_resp.status_code == 200
    stat_data = stat_resp.json()
    assert "records_scanned" in stat_data
    assert "audio_indexed" in stat_data
    assert "is_scanning" in stat_data

    # 2. Stop scan when not running
    stop_resp = client.post("/api/dataset/scan/stop")
    assert stop_resp.status_code == 200
    stop_data = stop_resp.json()
    assert stop_data["status"] in ("not_running", "stopping")


# ---------------------------------------------------------------------------
# 12. Creative Score Insufficient Reference Population Guard
# ---------------------------------------------------------------------------

def test_creative_score_insufficient_reference_population_tier():
    """When reference population has < 5 tracks, creative score tier reports insufficient reference population."""
    from src.creative_score import compute_creative_score

    # Pool with only 3 samples (< 5 required)
    res = compute_creative_score(
        observed_similarity=0.75,
        genre_pool_scores=[0.4, 0.45, 0.5],
        min_pool_size=5,
    )
    assert res.is_available is False
    assert res.broad_score_percent is None
    assert "Insufficient reference population" in res.score_tier
    assert "minimum 5 required" in res.score_tier


# ---------------------------------------------------------------------------
# 13. Null Distribution Human-Only Baseline Filtering
# ---------------------------------------------------------------------------

def test_null_distribution_human_only_baseline_filtering(tmp_path):
    """Verify get_or_build_cached_null_distribution filters to human tracks when origin labels exist."""
    candidates = {
        "human_1": np.ones((12, 10)) * 0.9,
        "human_2": np.ones((12, 10)) * 0.8,
        "ai_1": np.ones((12, 10)) * 0.5,
        "unknown_1": np.ones((12, 10)) * 0.3,
    }
    metadata = {
        "human_1": {"origin_label": "human", "genre": "Rock"},
        "human_2": {"origin_label": "human", "genre": "Rock"},
        "ai_1": {"origin_label": "ai_generated", "genre": "Rock"},
        "unknown_1": {"origin_label": "unknown", "genre": "Rock"},
    }

    evaluated_pairs = []
    def mock_sim(c1, c2):
        return 0.70

    cache_file = tmp_path / "null_dist_test.json"
    scores, genre_dict = get_or_build_cached_null_distribution(
        chroma_dict=candidates,
        similarity_func=mock_sim,
        cache_file=cache_file,
        candidate_metadata=metadata,
        force_rebuild=True,
    )

    # Only human_1 and human_2 should form a pair (1 pair)
    assert len(scores) == 1
    assert cache_file.is_file()
    with open(cache_file, "r", encoding="utf-8") as f:
        cache_data = json.load(f)
    assert "pairwise_dependency_note" in cache_data
    assert set(cache_data["source_track_ids"]) == {"human_1", "human_2"}

