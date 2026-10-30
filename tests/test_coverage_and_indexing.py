"""
Unit tests for Authoritative Decision 3 (Stable Remote Indexing) and
Authoritative Decision 5 (Feature 6 Alignment Coverage & Candidate Sorting).
"""

import json
from pathlib import Path
import numpy as np
import pytest

from src.mimic import (
    build_mimic_analysis,
    calculate_alignment_coverage,
    get_coverage_interpretation,
)
from src.remote_dataset import (
    generate_stable_track_id,
    load_remote_index,
    save_remote_index,
)
from src.schemas import MetadataContext


# ---------------------------------------------------------------------------
# Decision 3: Stable Remote Indexing Tests
# ---------------------------------------------------------------------------

def test_identical_audio_input_produces_identical_id():
    """Identical audio payload produces the exact same track ID."""
    audio_bytes = b"RIFF" + b"\x00" * 500
    id1 = generate_stable_track_id(audio_bytes)
    id2 = generate_stable_track_id(audio_bytes)
    assert id1 == id2
    assert id1.startswith("track_")
    assert len(id1) == 18  # "track_" (6) + 12 hex chars = 18


def test_different_audio_input_produces_distinct_ids():
    """Different audio content produces distinct track IDs."""
    audio_a = b"RIFF_AUDIO_SAMPLE_A_12345"
    audio_b = b"RIFF_AUDIO_SAMPLE_B_67890"
    id_a = generate_stable_track_id(audio_a)
    id_b = generate_stable_track_id(audio_b)
    assert id_a != id_b
    assert id_a.startswith("track_")
    assert id_b.startswith("track_")


def test_reindexing_same_track_does_not_create_duplicates(tmp_path):
    """Re-indexing existing track reuses ID and avoids duplicates."""
    audio_bytes = b"RIFF_SAMPLE_DATA_X"
    existing_index = {
        "tracks": {
            "track_existing_01": {
                "track_id": "track_existing_01",
                "source_id": "track_01_src",
                "content_hash": "dummy",
            }
        }
    }
    # Track with identical source_id returns existing ID without collision extension
    derived_id = generate_stable_track_id(
        audio_bytes,
        existing_ids=existing_index["tracks"],
        provenance_id="track_01_src",
    )
    assert derived_id.startswith("track_")


def test_atomic_index_save_and_recovery(tmp_path):
    """save_remote_index writes atomically via .tmp and os.replace without corrupting target."""
    idx_file = tmp_path / "remote_index.json"
    data = {
        "total_indexed": 2,
        "tracks": {
            "t1": {"track_id": "t1", "origin_label": "human"},
            "t2": {"track_id": "t2", "origin_label": "ai"},
        },
    }
    save_remote_index(data, idx_file)
    assert idx_file.is_file()

    loaded = load_remote_index(idx_file)
    assert loaded["total_indexed"] == 2
    assert "t1" in loaded["tracks"]
    assert "t2" in loaded["tracks"]


# ---------------------------------------------------------------------------
# Decision 5: Alignment Coverage & Candidate Sorting Tests
# ---------------------------------------------------------------------------

def test_alignment_coverage_known_path_and_costs():
    """Alignment coverage calculates matched_query_frames / total_query_frames accurately."""
    # 10 query frames total (indices 0 to 9)
    # Alignment path: 10 steps, 1-to-1 mapping
    path = [(i, i) for i in range(10)]
    # Local costs: first 4 steps are very low cost (0.1), remaining are high (1.0)
    # Mean cost = (4*0.1 + 6*1.0) / 10 = 0.64
    # Qualifying threshold = 0.64 * 0.70 = 0.448
    # Steps 0, 1, 2, 3 qualify (cost 0.1 <= 0.448) -> 4 unique query frames matched out of 10
    costs = [0.1, 0.1, 0.1, 0.1, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

    cov, cov_pct, interp = calculate_alignment_coverage(path, costs, total_query_frames=10)
    assert cov is not None
    assert np.isclose(cov, 0.40)
    assert np.isclose(cov_pct, 40.0)
    assert interp == "Partial structural match"


def test_repeated_path_steps_do_not_inflate_coverage():
    """Multiple reference frames mapping to the same query frame must NOT inflate coverage."""
    # Query frame 0 maps to 5 reference frames with low cost (0.1)
    # Remaining query frames have high costs (1.0)
    # Total query frames = 10
    path = [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (1, 5), (2, 6)]
    costs = [0.1, 0.1, 0.1, 0.1, 0.1, 1.0, 1.0]

    cov, cov_pct, interp = calculate_alignment_coverage(path, costs, total_query_frames=10)
    # Even though 5 path steps qualified, they all correspond to query frame 0 -> only 1 unique frame matched!
    assert cov is not None
    assert np.isclose(cov, 0.10)
    assert np.isclose(cov_pct, 10.0)
    assert interp == "Isolated passage"


def test_coverage_bounds_and_interpretation_tiers():
    """Verify explicit coverage interpretation tiers and boundary behavior."""
    # Boundary tests
    assert get_coverage_interpretation(0.0) == "Isolated passage"
    assert get_coverage_interpretation(0.15) == "Isolated passage"
    assert get_coverage_interpretation(0.20) == "Isolated passage"
    assert get_coverage_interpretation(0.200001) == "Partial structural match"
    assert get_coverage_interpretation(0.35) == "Partial structural match"
    assert get_coverage_interpretation(0.50) == "Partial structural match"
    assert get_coverage_interpretation(0.500001) == "Broad structural overlap"
    assert get_coverage_interpretation(0.85) == "Broad structural overlap"
    assert get_coverage_interpretation(1.0) == "Broad structural overlap"
    assert get_coverage_interpretation(None) == "Unavailable"


def test_missing_alignment_data_returns_none():
    """Missing or empty path/costs returns (None, None, None) without fabricating values."""
    assert calculate_alignment_coverage(None, [0.1], 10) == (None, None, None)
    assert calculate_alignment_coverage([(0, 0)], None, 10) == (None, None, None)
    assert calculate_alignment_coverage([], [], 10) == (None, None, None)
    assert calculate_alignment_coverage([(0, 0)], [0.1], 0) == (None, None, None)


def test_primary_candidate_sorting_remains_p_value_based():
    """Feature 6 ranks candidates by ascending valid p-value, tiebreaker descending similarity."""
    query_ctx = MetadataContext(track_id="q", title="Query", duration_sec=60.0)
    candidates = [
        # Candidate A: high similarity 0.90, p_val 0.05
        {"candidate_id": "cand_a", "title": "A", "similarity": 0.90, "p_value": 0.05, "severity_label": "Advisory"},
        # Candidate B: lower similarity 0.82, but lower (more significant) p_val 0.005 -> should rank FIRST
        {"candidate_id": "cand_b", "title": "B", "similarity": 0.82, "p_value": 0.005, "severity_label": "Critical"},
        # Candidate C: same p_val as B (0.005), but higher similarity 0.88 -> should rank BEFORE B
        {"candidate_id": "cand_c", "title": "C", "similarity": 0.88, "p_value": 0.005, "severity_label": "Critical"},
        # Candidate D: p_value is None -> should rank at the END
        {"candidate_id": "cand_d", "title": "D", "similarity": 0.95, "p_value": None, "severity_label": "Incomplete"},
    ]

    analysis = build_mimic_analysis(query_ctx, candidates, max_mimic_candidates=10)
    ranked_ids = [c.candidate_id for c in analysis.top_mimic_candidates]

    # C has p=0.005, S=0.88; B has p=0.005, S=0.82; A has p=0.05; D has p=None
    assert ranked_ids == ["cand_c", "cand_b", "cand_a", "cand_d"]
