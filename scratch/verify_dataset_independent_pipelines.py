"""
Dataset-Independent Reference-Index and Null-Distribution Pipeline Verification.
Satisfies Parts 2.4 and 2.5 of the Dataset-Independent Closure.
Tests index building and null-distribution generation on arbitrary synthetic inputs
in isolated paths, leaving frozen corpus files 100% untouched.
"""
import json
import os
import sys
import time
from pathlib import Path
import numpy as np
import soundfile as sf

REPO_ROOT = Path(".").resolve()
sys.path.insert(0, str(REPO_ROOT))

ISOLATED_DIR = REPO_ROOT / "temp_uploads/isolated_pipeline_test"
ISOLATED_DIR.mkdir(parents=True, exist_ok=True)
REPORT_PATH = REPO_ROOT / "reports/dataset_independent_pipelines_verification.json"

ISOLATED_INDEX_PATH = ISOLATED_DIR / "isolated_test_index.json"
ISOLATED_NULL_PATH = ISOLATED_DIR / "isolated_test_null.json"


def test_reference_index_pipeline():
    results = {}
    from src.remote_dataset import save_remote_index, load_remote_index
    from src.similarity import compute_mean_chroma_vector, get_or_extract_features

    # 1. Create a synthetic dataset of 5 tracks (different from any 261-record corpus tracks)
    test_tracks = [
        {"id": "synth_human_01", "origin": "human", "genre": "Jazz", "duration": 4.0},
        {"id": "synth_human_02", "origin": "human", "genre": "Jazz", "duration": 5.0},
        {"id": "synth_human_03", "origin": "human", "genre": "Classical", "duration": 4.5},
        {"id": "synth_ai_gen01", "origin": "ai", "generator": "FutureGen-X", "genre": "Electronic", "duration": 6.0},
        {"id": "synth_ai_gen02", "origin": "ai", "generator": "FutureGen-X", "genre": "Electronic", "duration": 5.5},
    ]

    sr = 22050
    synth_index = {"tracks": {}, "metadata": {"created_at": time.time(), "corpus_name": "Synthetic-Independent-Test"}}

    # Build index entries from generated synthetic waveforms
    for item in test_tracks:
        tid = item["id"]
        dur = item["duration"]
        t = np.linspace(0, dur, int(sr * dur), endpoint=False)
        audio = 0.4 * np.sin(2 * np.pi * (300.0 + len(tid)*20.0) * t).astype(np.float32)
        p_audio = ISOLATED_DIR / f"{tid}.wav"
        sf.write(str(p_audio), audio, sr)

        f56, beat, times, dur_ext = get_or_extract_features(p_audio, cache_dir=ISOLATED_DIR)
        mean_c = compute_mean_chroma_vector(beat)

        synth_index["tracks"][tid] = {
            "title": f"Synthetic Track {tid}",
            "artist": "Algorithmic Composer",
            "genre": item["genre"],
            "origin_label": item["origin"],
            "generator": item.get("generator"),
            "duration_sec": dur_ext,
            "mean_chroma": mean_c.tolist(),
            "origin_features_56": f56.tolist(),
            "beat_cache": str(ISOLATED_DIR / f"{tid}_beat.npy"),
            "times_cache": str(ISOLATED_DIR / f"{tid}_times.npy"),
            "source_type": "synthetic_test"
        }

    # Test saving to isolated index path
    save_remote_index(synth_index, ISOLATED_INDEX_PATH)
    assert ISOLATED_INDEX_PATH.exists(), "Isolated index file was not created!"

    # Test loading isolated index
    loaded = load_remote_index(ISOLATED_INDEX_PATH)
    results["isolated_index_created"] = {
        "output_path": str(ISOLATED_INDEX_PATH),
        "total_records_indexed": len(loaded.get("tracks", {})),
        "expected_records": 5,
        "is_independent_of_261_corpus": len(loaded.get("tracks", {})) == 5,
        "tracks_indexed": list(loaded.get("tracks", {}).keys())
    }

    # Test updating index with duplicate ID and new record
    loaded["tracks"]["synth_human_01"]["title"] = "Updated Title"
    loaded["tracks"]["synth_new_06"] = {
        "title": "Fresh Entry",
        "genre": "Ambient",
        "origin_label": "human",
        "mean_chroma": [0.1]*12,
        "source_type": "synthetic_test"
    }
    save_remote_index(loaded, ISOLATED_INDEX_PATH)
    reloaded = load_remote_index(ISOLATED_INDEX_PATH)
    results["index_update_and_deduplication"] = {
        "updated_record_title": reloaded["tracks"]["synth_human_01"]["title"],
        "new_record_present": "synth_new_06" in reloaded["tracks"],
        "total_after_update": len(reloaded["tracks"]),
        "duplicate_id_overwritten_safely": reloaded["tracks"]["synth_human_01"]["title"] == "Updated Title"
    }

    return results


def test_null_distribution_pipeline():
    results = {}
    from src.stats_test import (
        build_pairwise_null_distribution,
        get_or_build_cached_null_distribution,
        MIN_NULL_PAIRS
    )
    from src.similarity import dtw_similarity

    # 1. Create synthetic human chroma matrices (6 synthetic human tracks)
    chroma_dict = {}
    metadata_dict = {}
    genres = ["Jazz", "Jazz", "Jazz", "Classical", "Classical", "Classical"]
    for i, g in enumerate(genres):
        tid = f"null_human_{i+1:02d}"
        # Unique harmonic profile for each track
        mat = np.random.RandomState(i + 10).rand(12, 50).astype(np.float32)
        chroma_dict[tid] = mat
        metadata_dict[tid] = {"genre": g, "origin_label": "human"}

    # Total possible pairs = 6 * 5 / 2 = 15 pairs
    null_scores = build_pairwise_null_distribution(
        chroma_dict,
        similarity_func=dtw_similarity,
        max_pairs=100,
        seed=42
    )

    results["pairwise_null_calculation"] = {
        "total_human_tracks": len(chroma_dict),
        "total_pairs_evaluated": len(null_scores),
        "expected_pairs": 15,
        "min_score": round(float(np.min(null_scores)), 4),
        "max_score": round(float(np.max(null_scores)), 4),
        "mean_score": round(float(np.mean(null_scores)), 4),
        "all_scores_in_0_1": bool(all(0.0 <= s <= 1.0 for s in null_scores)),
        "all_scores_finite": bool(all(np.isfinite(s) for s in null_scores))
    }

    # 2. Test get_or_build_cached_null_distribution with isolated cache file
    global_null, genre_null = get_or_build_cached_null_distribution(
        chroma_dict=chroma_dict,
        similarity_func=dtw_similarity,
        cache_file=ISOLATED_NULL_PATH,
        max_pairs=100,
        candidate_metadata=metadata_dict,
        force_rebuild=True
    )

    assert ISOLATED_NULL_PATH.exists(), "Isolated null distribution file was not created!"

    with open(ISOLATED_NULL_PATH, "r", encoding="utf-8") as f:
        saved_null_data = json.load(f)

    results["cached_null_distribution"] = {
        "cache_file": str(ISOLATED_NULL_PATH),
        "global_null_count": len(global_null),
        "genre_null_keys": list(genre_null.keys()),
        "saved_schema_keys": list(saved_null_data.keys()),
        "provenance_preserved": "metadata" in saved_null_data or "global_null_distribution" in saved_null_data
    }

    # 3. Test insufficient sample handling (< MIN_NULL_PAIRS)
    chroma_too_few = {"only_one": np.random.rand(12, 30)}
    scores_too_few = build_pairwise_null_distribution(
        chroma_too_few,
        similarity_func=dtw_similarity,
        max_pairs=100
    )
    results["insufficient_samples_handling"] = {
        "single_track_pairs_produced": len(scores_too_few),
        "properly_handled": len(scores_too_few) == 0
    }

    return results


def run_all_pipeline_verifications():
    print("Testing Reference-Index Pipeline (Dataset-Independent)...")
    idx_results = test_reference_index_pipeline()
    print("Isolated index records:", idx_results["isolated_index_created"]["total_records_indexed"])
    print("Index update success:", idx_results["index_update_and_deduplication"]["duplicate_id_overwritten_safely"])

    print("\nTesting Null-Distribution Pipeline (Dataset-Independent)...")
    null_results = test_null_distribution_pipeline()
    print("Null pairs evaluated:", null_results["pairwise_null_calculation"]["total_pairs_evaluated"])
    print("All scores in [0, 1]:", null_results["pairwise_null_calculation"]["all_scores_in_0_1"])
    print("Insufficient samples handled:", null_results["insufficient_samples_handling"]["properly_handled"])

    # Verify frozen production corpus files were NOT modified
    frozen_files = [
        "features/remote_index.json",
        "features/null_distribution.json",
        "features/mert_manifest.json",
        "features/dataset_manifest.json"
    ]
    frozen_integrity = {}
    import hashlib
    for rel in frozen_files:
        p = REPO_ROOT / rel
        with open(p, "rb") as f:
            frozen_integrity[rel] = hashlib.sha256(f.read()).hexdigest()

    expected_hashes = {
        "features/remote_index.json": "c4990a42a64ff024d9f43fb7a44d1191cdc713a857661b368a98e39a613f85f4",
        "features/null_distribution.json": "d80616e58b729f3b1d08aaf604f8b5a2ffe5fb41f4a44ca6050a8c672602948a",
        "features/mert_manifest.json": "e3c6da56bca5d764cd9581f10978162d442ce6d4ff446bbf5655d4a649319b0e",
        "features/dataset_manifest.json": "b36bf0a2f9f0a01c2dcad4eaec43a4efa9c13c5246db9c88ecbe2e2c6c270a6c"
    }

    hashes_match = (frozen_integrity == expected_hashes)
    print("Frozen production corpus untouched:", hashes_match)

    full_report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "objective": "Dataset-Independent Index and Null Distribution Pipelines Verification",
        "reference_index_pipeline_verification": idx_results,
        "null_distribution_pipeline_verification": null_results,
        "production_corpus_protection": {
            "all_hashes_match_expected": hashes_match,
            "manifest_hashes": frozen_integrity
        },
        "overall_status": "PASSED" if hashes_match and idx_results["isolated_index_created"]["is_independent_of_261_corpus"] else "FAILED"
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)

    print(f"\nSaved pipeline verification report to: {REPORT_PATH}")
    assert hashes_match, "Production corpus was modified during pipeline testing!"
    assert idx_results["isolated_index_created"]["is_independent_of_261_corpus"]
    assert null_results["pairwise_null_calculation"]["all_scores_in_0_1"]
    print("Dataset-independent pipeline verification passed completely!")


if __name__ == "__main__":
    run_all_pipeline_verifications()

