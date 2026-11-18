"""
Tests for Strict Fresh-Production Mode Legacy-Artifact Isolation.

Verifies:
1. Legacy artifacts from the retired 261-record corpus are strictly rejected in fresh-production mode:
   - Historical origin classifier checkpoints (models/origin_classifier.joblib)
   - Historical reference indexes (features/remote_index.json)
   - Historical null distributions (features/null_distribution.json)
   - Historical feature/embedding caches (features/*_beat.npy, features/mert/*.npy)
2. Approved externally pretrained frozen encoders (m-a-p/MERT-v1-95M) remain loadable and exempt.
3. Newly generated artifacts with valid manifests load successfully in fresh-production mode.
4. Clearly identified test-only fixture models load successfully in fresh-production mode.
5. Historical artifacts remain accessible in explicitly designated legacy/historical mode.
6. Zero silent fallback to old artifacts occurs.
"""

import json
import os
import shutil
from pathlib import Path
import numpy as np
import pytest

from src.config import (
    REPO_ROOT,
    MODELS_DIR,
    FEATURES_DIR,
    REMOTE_INDEX_FILE,
    NULL_DIST_CACHE_FILE,
    LegacyArtifactError,
    LEGACY_ARTIFACT_HASHES,
    APPROVED_EXTERNAL_ENCODERS,
    is_fresh_production_mode_active,
    is_legacy_mode_active,
    get_system_mode,
)
from src.origin_classifier import (
    load_origin_classifier,
    predict_origin,
    create_test_fixture_origin_classifier,
    is_test_fixture_model,
)
from src.remote_dataset import (
    load_remote_index,
    validate_feature_cache_path,
)
from src.stats_test import (
    load_null_distribution,
    get_or_build_cached_null_distribution,
)
from src.mert_embedding import (
    is_approved_external_encoder,
    verify_mert_external_foundation_approval,
)


@pytest.fixture(autouse=True)
def ensure_test_fixture_exists():
    """Ensure the test fixture model exists for testing."""
    fixture_path = MODELS_DIR / "test_fixture_origin_classifier.joblib"
    if not fixture_path.exists():
        create_test_fixture_origin_classifier(fixture_path)
    return fixture_path


# ---------------------------------------------------------------------------
# Test 1: Historical Origin Classifier Checkpoint Rejection
# ---------------------------------------------------------------------------

def test_fresh_production_rejects_historical_classifier_checkpoint(monkeypatch):
    """In fresh-production mode, loading the legacy 261 checkpoint must be strictly rejected."""
    monkeypatch.setenv("MUSIC_SIMILARITY_MODE", "fresh_production")
    monkeypatch.delenv("MUSIC_ORIGIN_TEST_FIXTURE_PATH", raising=False)
    monkeypatch.delenv("MUSIC_ORIGIN_MODEL_PATH", raising=False)

    legacy_model_path = MODELS_DIR / "origin_classifier.joblib"
    assert legacy_model_path.is_file(), "Legacy checkpoint must exist for isolation test"

    # Strict load must raise LegacyArtifactError
    with pytest.raises(LegacyArtifactError, match="retired 261-record corpus"):
        load_origin_classifier(model_path=legacy_model_path, strict=True)

    # Non-strict load must return None (honest unconfigured state, no crash)
    model = load_origin_classifier(model_path=legacy_model_path, strict=False)
    assert model is None, "Non-strict load of legacy checkpoint must return None"


def test_fresh_production_predict_origin_returns_unconfigured_without_silent_fallback(monkeypatch, tmp_path):
    """In fresh-production mode, predict_origin must return Unconfigured rather than silently using legacy checkpoint."""
    monkeypatch.setenv("MUSIC_SIMILARITY_MODE", "fresh_production")
    monkeypatch.delenv("MUSIC_ORIGIN_TEST_FIXTURE_PATH", raising=False)
    monkeypatch.delenv("MUSIC_ORIGIN_MODEL_PATH", raising=False)

    # Create dummy audio file
    dummy_wav = tmp_path / "test_dummy.wav"
    import soundfile as sf
    sf.write(str(dummy_wav), np.zeros(22050, dtype=np.float32), 22050)

    result = predict_origin(dummy_wav)
    assert result.predicted_origin == "Unconfigured"
    assert result.model_status == "Unconfigured / Missing Labeled Data"
    assert result.confidence is None
    assert result.probability_ai is None


# ---------------------------------------------------------------------------
# Test 2: Historical Remote Index Rejection
# ---------------------------------------------------------------------------

def test_fresh_production_rejects_historical_remote_index(monkeypatch):
    """In fresh-production mode, loading the legacy 261-record remote_index.json must be strictly rejected."""
    monkeypatch.setenv("MUSIC_SIMILARITY_MODE", "fresh_production")

    # Strict load must raise LegacyArtifactError
    with pytest.raises(LegacyArtifactError, match="Legacy 261-record corpus index"):
        load_remote_index(index_file=REMOTE_INDEX_FILE, strict=True)

    # Non-strict load must return empty skeleton with status UNCONFIGURED_FRESH_PRODUCTION
    idx = load_remote_index(index_file=REMOTE_INDEX_FILE, strict=False)
    assert idx["total_indexed"] == 0
    assert idx["scan_status"] == "UNCONFIGURED_FRESH_PRODUCTION"
    assert len(idx["tracks"]) == 0


# ---------------------------------------------------------------------------
# Test 3: Historical Null Distribution Rejection
# ---------------------------------------------------------------------------

def test_fresh_production_rejects_historical_null_distribution(monkeypatch):
    """In fresh-production mode, loading the legacy 261 null distribution (6105 pairs) must be strictly rejected."""
    monkeypatch.setenv("MUSIC_SIMILARITY_MODE", "fresh_production")

    # Strict load must raise LegacyArtifactError
    with pytest.raises(LegacyArtifactError, match="retired 261-record corpus"):
        load_null_distribution(cache_file=NULL_DIST_CACHE_FILE, strict=True)

    # get_or_build_cached_null_distribution with strict=True must raise
    with pytest.raises(LegacyArtifactError, match="retired 261-record corpus"):
        get_or_build_cached_null_distribution(
            chroma_dict={},
            similarity_func=lambda a, b: 0.5,
            cache_file=NULL_DIST_CACHE_FILE,
            strict=True,
        )


def test_fresh_production_null_distribution_bypasses_legacy_cache(monkeypatch):
    """In fresh-production mode with strict=False, get_or_build computes dynamically without loading legacy scores."""
    monkeypatch.setenv("MUSIC_SIMILARITY_MODE", "fresh_production")

    # Candidate dictionary with only 2 items (< min_pairs=5)
    chroma_dict = {
        "track_a": np.random.randn(12, 10).astype(np.float32),
        "track_b": np.random.randn(12, 10).astype(np.float32),
    }

    scores, genre_dict = get_or_build_cached_null_distribution(
        chroma_dict=chroma_dict,
        similarity_func=lambda a, b: 0.5,
        cache_file=NULL_DIST_CACHE_FILE,
        strict=False,
    )
    # The legacy cache has 6,105 scores; because it is isolated, scores must NOT be 6,105
    assert len(scores) != 6105, "Must NOT return the 6,105 legacy scores from disk cache"
    assert len(scores) <= 1  # 2 tracks -> exactly 1 pair


# ---------------------------------------------------------------------------
# Test 4: Historical Feature and Embedding Cache Rejection
# ---------------------------------------------------------------------------

def test_fresh_production_rejects_legacy_feature_caches(monkeypatch):
    """In fresh-production mode, accessing legacy 261 feature files must be rejected."""
    monkeypatch.setenv("MUSIC_SIMILARITY_MODE", "fresh_production")

    with pytest.raises(LegacyArtifactError, match="retired 261 corpus"):
        validate_feature_cache_path(FEATURES_DIR / "haim_ai_lyria_pro3_00001_beat.npy")

    with pytest.raises(LegacyArtifactError, match="retired 261 corpus"):
        validate_feature_cache_path(FEATURES_DIR / "mert" / "haim_ai_lyria_pro3_00001_mert.npy")


# ---------------------------------------------------------------------------
# Test 5: Approved External Pretrained Foundation Model Exemption
# ---------------------------------------------------------------------------

def test_approved_external_pretrained_encoder_exemption(monkeypatch):
    """Approved external foundation encoders (m-a-p/MERT-v1-95M) are exempt from legacy isolation."""
    monkeypatch.setenv("MUSIC_SIMILARITY_MODE", "fresh_production")

    assert "m-a-p/MERT-v1-95M" in APPROVED_EXTERNAL_ENCODERS
    assert is_approved_external_encoder("m-a-p/MERT-v1-95M") is True

    approval = verify_mert_external_foundation_approval()
    assert approval["is_approved_external_encoder"] is True
    assert approval["exempt_from_legacy_isolation"] is True
    assert approval["fresh_production_allowed"] is True

    # Unapproved model IDs must be rejected
    assert is_approved_external_encoder("malicious_or_unapproved_model") is False


# ---------------------------------------------------------------------------
# Test 6: Newly Generated Artifacts Load Cleanly in Fresh-Production Mode
# ---------------------------------------------------------------------------

def test_fresh_production_allows_newly_generated_artifacts(monkeypatch, tmp_path):
    """Newly generated artifacts with valid fresh manifests load cleanly in fresh-production mode."""
    monkeypatch.setenv("MUSIC_SIMILARITY_MODE", "fresh_production")

    # 1. Fresh dataset index
    fresh_index_path = tmp_path / "fresh_production_index.json"
    fresh_index_data = {
        "version": "2.0",
        "dataset_name": "fresh_dataset_v2",
        "provenance": {
            "mode": "fresh_production",
            "manifest_hash": "a1b2c3d4e5f6",
            "created_at": "2026-11-02T00:00:00Z",
        },
        "total_indexed": 3,
        "scan_status": "COMPLETED",
        "is_complete": True,
        "tracks": {
            "fresh_track_001": {"origin_label": "human", "duration_sec": 30.0},
            "fresh_track_002": {"origin_label": "ai", "duration_sec": 30.0},
            "fresh_track_003": {"origin_label": "human", "duration_sec": 30.0},
        },
    }
    with open(fresh_index_path, "w", encoding="utf-8") as f:
        json.dump(fresh_index_data, f)

    loaded_index = load_remote_index(index_file=fresh_index_path, strict=True)
    assert loaded_index["total_indexed"] == 3
    assert loaded_index["dataset_name"] == "fresh_dataset_v2"

    # 2. Fresh null distribution
    fresh_null_path = tmp_path / "fresh_null_distribution.json"
    fresh_null_data = {
        "version": "2.0",
        "provenance": "fresh_production_v2",
        "source_track_ids": ["fresh_track_001", "fresh_track_003"],
        "sample_pairs_count": 1,
        "null_scores": [0.42],
    }
    with open(fresh_null_path, "w", encoding="utf-8") as f:
        json.dump(fresh_null_data, f)

    loaded_null = load_null_distribution(cache_file=fresh_null_path, strict=True)
    assert loaded_null["null_scores"] == [0.42]


# ---------------------------------------------------------------------------
# Test 7: Clearly Identified Test Fixture Model Loading
# ---------------------------------------------------------------------------

def test_fresh_production_allows_test_fixture_origin_classifier(monkeypatch, tmp_path):
    """Clearly identified test-only fixture models load and infer properly in fresh-production mode."""
    monkeypatch.setenv("MUSIC_SIMILARITY_MODE", "fresh_production")
    fixture_path = MODELS_DIR / "test_fixture_origin_classifier.joblib"
    assert fixture_path.is_file()

    model = load_origin_classifier(model_path=fixture_path, strict=True)
    assert model is not None
    assert is_test_fixture_model(model) is True

    # Dummy audio for inference
    dummy_wav = tmp_path / "test_audio.wav"
    import soundfile as sf
    sf.write(str(dummy_wav), np.random.randn(22050).astype(np.float32), 22050)

    result = predict_origin(dummy_wav, model_path=fixture_path)
    assert result.predicted_origin in ("Human", "AI")
    assert result.model_status == "Test Fixture Active"
    assert result.model_version == "test-fixture-1.0.0"
    assert result.confidence is not None
    assert result.probability_ai is not None


# ---------------------------------------------------------------------------
# Test 8: Historical Artifacts Accessible in Explicit Legacy Mode
# ---------------------------------------------------------------------------

def test_historical_artifacts_accessible_in_legacy_mode(monkeypatch):
    """In explicitly designated legacy mode, historical artifacts from the 261 corpus remain accessible."""
    monkeypatch.setenv("MUSIC_SIMILARITY_MODE", "legacy")

    assert is_legacy_mode_active() is True
    assert is_fresh_production_mode_active() is False

    # Historical classifier loads
    legacy_model = load_origin_classifier(strict=True)
    assert legacy_model is not None, "Historical classifier must load in legacy mode"

    # Historical remote index loads
    legacy_index = load_remote_index(strict=True)
    assert legacy_index["total_indexed"] == 261, "Historical index must load 261 tracks in legacy mode"

    # Historical null distribution loads
    legacy_null = load_null_distribution(strict=True)
    assert len(legacy_null.get("null_scores", [])) == 6105, "Historical null distribution must have 6,105 pairs in legacy mode"

    # Legacy feature path validation passes with allow_legacy=True
    validated_path = validate_feature_cache_path(FEATURES_DIR / "haim_ai_lyria_pro3_00001_beat.npy", allow_legacy=True)
    assert validated_path.name == "haim_ai_lyria_pro3_00001_beat.npy"


# ---------------------------------------------------------------------------
# Test 9: Zero Silent Fallback Guarantee
# ---------------------------------------------------------------------------

def test_zero_silent_fallback_guarantee(monkeypatch, tmp_path):
    """Verify that when no fresh artifact exists, no silent fallback to legacy 261 artifacts occurs."""
    monkeypatch.setenv("MUSIC_SIMILARITY_MODE", "fresh_production")
    monkeypatch.delenv("MUSIC_ORIGIN_TEST_FIXTURE_PATH", raising=False)
    monkeypatch.delenv("MUSIC_ORIGIN_MODEL_PATH", raising=False)

    # Classifier: load without path must return None, not the legacy 261 model
    clf = load_origin_classifier(model_path=None, strict=False)
    assert clf is None, "Must not silently fall back to legacy classifier"

    # Index: when no fresh v2 index exists, load default must return empty skeleton, not legacy 261 tracks
    monkeypatch.setattr("src.remote_dataset.V2_INDEX_FILE", tmp_path / "nonexistent_v2_index.json")
    idx = load_remote_index(index_file=None, strict=False)
    assert idx["total_indexed"] == 0
    assert len(idx["tracks"]) == 0

    # Null distribution: load default must return empty dict, not 6,105 pairs
    null_dist = load_null_distribution(cache_file=None, strict=False)
    assert null_dist == {}

