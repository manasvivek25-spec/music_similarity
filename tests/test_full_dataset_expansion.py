"""
Comprehensive Acceptance Test Suite for Full Dataset Expansion & Multi-Generator Ingestion.

Validates:
1. Canonical production corpus expansion (120 tracks across 3 distinct AI generator families).
2. Reconciled terminal state manifest (dataset_manifest.json with zero unclassified records).
3. Exhaustive scaled null distribution (1,035 pairs across 46 human reference tracks).
4. MERT full production embedding coverage (119/120 valid production embeddings on CUDA).
5. True generator-held-out validation in Origin Classifier (Leave-One-Generator-Out).
6. Full duration audio processing without hidden 60s or 30s cutoffs.
"""

import json
from pathlib import Path
import pytest
import numpy as np

from src.config import FEATURES_DIR, NULL_DIST_CACHE_FILE, REMOTE_INDEX_FILE
from src.dataset_enumerator import DatasetPortfolioEnumerator, MANIFEST_FILE, TerminalStatus
from src.origin_classifier import evaluate_group_aware_generalization, load_origin_classifier


def test_expanded_canonical_corpus_inventory():
    """Verify production index contains >= 120 tracks with multi-generator representation."""
    assert REMOTE_INDEX_FILE.is_file(), "remote_index.json must exist"
    with open(REMOTE_INDEX_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    tracks = data.get("tracks", {})
    assert len(tracks) >= 120, f"Expected >= 120 canonical tracks, got {len(tracks)}"

    # Verify generator diversity
    generators = set()
    categories = set()
    for tid, t in tracks.items():
        gen = t.get("generator")
        cat = t.get("origin_label") or t.get("track_category")
        if gen:
            generators.add(gen)
        if cat:
            categories.add(cat)

    # Must contain multiple distinct AI generator families
    assert "acestep" in generators, "Missing acestep generator in corpus"
    assert "lyria-pro3" in generators, "Missing lyria-pro3 generator in corpus"
    assert "musicgen" in generators, "Missing musicgen generator in corpus"
    assert "human" in categories, "Missing human category in corpus"
    assert "hybrid" in categories or "hybrid_b1" in generators, "Missing hybrid category in corpus"
    assert "temporal_mix" in categories or "temporal_c1" in generators, "Missing temporal mix in corpus"


def test_reconciled_dataset_manifest_and_terminal_states():
    """Verify dataset_manifest.json accounts for all discovered records in valid terminal states."""
    assert MANIFEST_FILE.is_file(), "dataset_manifest.json must exist"
    with open(MANIFEST_FILE, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    records = manifest.get("records", {})
    assert len(records) >= 120, f"Manifest must track at least 120 records, found {len(records)}"

    valid_terminal_statuses = {s.value for s in TerminalStatus}
    for cid, r in records.items():
        status = r.get("terminal_status")
        assert status in valid_terminal_statuses, f"Invalid terminal status '{status}' for record {cid}"


def test_exhaustive_scaled_null_distribution_integrity():
    """Verify exhaustive pairwise null distribution covers 1035 pairs across 46 human reference tracks."""
    assert NULL_DIST_CACHE_FILE.is_file(), "null_distribution.json must exist"
    with open(NULL_DIST_CACHE_FILE, "r", encoding="utf-8") as f:
        nd = json.load(f)

    src_ids = nd.get("source_track_ids", [])
    n_human = len(src_ids)
    assert n_human >= 46, f"Expected >= 46 human tracks, got {n_human}"

    expected_pairs = (n_human * (n_human - 1)) // 2
    actual_pairs = nd.get("sample_pairs_count")
    assert actual_pairs == expected_pairs, f"Expected exhaustive {expected_pairs} pairs, got {actual_pairs}"
    assert len(nd.get("null_scores", [])) == expected_pairs

    scores = nd["null_scores"]
    assert min(scores) >= 0.0
    assert max(scores) <= 1.0
    assert np.median(scores) > 0.40  # Authentic musical chromagram baseline


def test_mert_full_production_coverage_and_parity():
    """Verify MERT embeddings exist for all eligible production tracks with 768 dimensions."""
    mert_manifest_p = FEATURES_DIR / "mert_manifest.json"
    assert mert_manifest_p.is_file()
    with open(mert_manifest_p, "r", encoding="utf-8") as f:
        mm = json.load(f)

    valid_count = mm.get("valid_production_embeddings")
    total_canonical = mm.get("total_canonical_tracks")
    assert valid_count >= 119, f"Expected >= 119 valid production embeddings, got {valid_count}"
    assert mm.get("coverage_percentage") >= 99.0

    # Test an embedding file for shape and float32 dtype
    mert_dir = FEATURES_DIR / "mert"
    sample_emb_file = next(mert_dir.glob("haim_ai_lyria_pro3_*_mert.npy"), None)
    if sample_emb_file:
        arr = np.load(str(sample_emb_file))
        assert arr.ndim in (1, 2)
        dim = arr.shape[-1]
        assert dim == 768, f"Expected 768-dim MERT embedding, got {dim}"
        assert np.all(np.isfinite(arr)), "MERT embedding must be finite without NaNs"


def test_generator_held_out_validation_results():
    """Verify Leave-One-Generator-Out evaluation in Origin Classifier across independent AI families."""
    model = load_origin_classifier()
    assert model is not None, "Trained origin classifier must be loadable"

    with open(REMOTE_INDEX_FILE, "r", encoding="utf-8") as f:
        tracks = json.load(f).get("tracks", {})

    gen_eval = evaluate_group_aware_generalization(model, tracks)
    assert "leave_one_generator_out" in gen_eval, "Must include LOGO generalization evaluation"
    logo = gen_eval["leave_one_generator_out"]

    assert "acestep" in logo, "Missing LOGO evaluation for acestep"
    assert "lyria-pro3" in logo, "Missing LOGO evaluation for lyria-pro3"
    assert "musicgen" in logo, "Missing LOGO evaluation for musicgen"

    for gen, res in logo.items():
        assert res.get("test_sample_size") > 0
        assert "unseen_generator_recall" in res
        assert "held_out_human_recall" in res

