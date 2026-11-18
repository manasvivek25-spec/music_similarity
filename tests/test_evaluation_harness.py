"""
Unit and Integration Tests for the Offline Evaluation Harness.
Verifies held-out dataset isolation, controlled truth coverage,
statistical uncertainty math, baseline models, and reproducible artifact generation.
"""

import csv
import json
from pathlib import Path
import numpy as np
import pytest

from src.config import REPO_ROOT
from evaluation.manifest import MANIFEST_PATH, ROOT_MANIFEST_PATH, HELD_OUT_DIR
from evaluation.statistics import (
    compute_wilson_ci,
    compute_clopper_pearson_ci,
    compute_group_bootstrap_ci,
    compute_mcnemar_test,
    compute_paired_bootstrap,
    compute_roc_pr_and_eer,
    compute_ece,
    compute_mirex_key_score,
    compute_tempo_accuracies,
    compute_segment_iou_and_time_error,
)
from evaluation.baselines import (
    baseline_mean_chroma_cosine,
    baseline_dtw_no_shift,
    baseline_fixed_similarity_decision,
    baseline_majority_classifier,
    baseline_metadata_match,
    train_f4_baseline_models,
)


def test_evaluation_manifest_isolation_and_schema():
    """Verify evaluation_manifest.csv exists, has valid schema, and zero overlap with reference index."""
    assert MANIFEST_PATH.is_file(), f"Manifest file missing: {MANIFEST_PATH}"
    assert ROOT_MANIFEST_PATH.is_file(), f"Root manifest file missing: {ROOT_MANIFEST_PATH}"

    expected_cols = ["sha256", "label", "generator", "source", "group_id", "bpm", "key", "related_ids", "transform_params"]

    records = []
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == expected_cols, f"Invalid manifest header: {reader.fieldnames}"
        for row in reader:
            records.append(row)

    assert len(records) >= 50, f"Expected at least 50 held-out items, got {len(records)}"

    # Cryptographic Isolation Check
    manifest_hashes = {r["sha256"] for r in records}
    ref_index_path = REPO_ROOT / "features" / "remote_index.json"
    if ref_index_path.is_file():
        with open(ref_index_path, "r", encoding="utf-8") as f:
            ref_data = json.load(f)
            ref_tracks = ref_data.get("tracks", {})
            # Verify no hash collisions
            for tid, tinfo in ref_tracks.items():
                assert tid not in manifest_hashes, f"Collision: track ID {tid} in manifest!"


def test_controlled_truth_transformations_coverage():
    """Verify all required perturbations exist: transpositions +-1..6, time-stretches, noise, excerpts, spliced."""
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        records = list(reader)

    transform_types = set()
    transpositions = set()
    stretch_rates = set()

    for r in records:
        params = json.loads(r["transform_params"])
        ttype = params.get("type")
        transform_types.add(ttype)

        if ttype == "transposition":
            transpositions.add(params.get("k"))
        elif ttype == "time_stretch":
            stretch_rates.add(params.get("rate"))

    expected_k = {-6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6}
    assert expected_k.issubset(transpositions), f"Missing transposition shifts: {expected_k - transpositions}"
    assert {0.9, 1.1}.issubset(stretch_rates), f"Missing time stretches: {stretch_rates}"
    assert "additive_noise" in transform_types, "Missing additive noise perturbations"
    assert "excerpt" in transform_types, "Missing excerpt perturbations"
    assert "spliced" in transform_types, "Missing spliced query perturbations"


def test_statistical_uncertainty_bounds():
    """Verify Wilson, Clopper-Pearson, and group bootstrap CI properties."""
    # Wilson interval
    low_w, high_w = compute_wilson_ci(40, 50, confidence=0.95)
    assert 0.0 <= low_w <= 0.8 <= high_w <= 1.0

    # Clopper-Pearson exact
    low_cp, high_cp = compute_clopper_pearson_ci(40, 50, confidence=0.95)
    assert 0.0 <= low_cp <= 0.8 <= high_cp <= 1.0

    # Group bootstrap (>= 2000 iterations)
    groups = np.array(["g1", "g1", "g2", "g2", "g3", "g3", "g4", "g4"])
    y_true = np.array([1, 1, 0, 0, 1, 0, 1, 0])
    y_pred = np.array([0.9, 0.8, 0.2, 0.1, 0.7, 0.3, 0.6, 0.4])

    pt, (low_gb, high_gb) = compute_group_bootstrap_ci(
        groups, y_true, y_pred,
        metric_fn=lambda yt, yp: float(np.mean(yp)),
        n_bootstraps=2000,
        seed=42,
    )
    assert low_gb <= pt <= high_gb

    # McNemar test
    mcn = compute_mcnemar_test(
        np.array([1, 1, 1, 0, 0]),
        np.array([1, 1, 1, 0, 0]),
        np.array([0, 0, 1, 1, 0]),
    )
    assert "statistic" in mcn and "p_value" in mcn

    # Paired bootstrap
    paired = compute_paired_bootstrap(
        np.array([0.9, 0.8, 0.85]),
        np.array([0.5, 0.4, 0.45]),
        n_bootstraps=2000,
        seed=42,
    )
    assert paired["mean_delta"] > 0
    assert paired["ci"][0] <= paired["mean_delta"] <= paired["ci"][1]


def test_baseline_algorithms():
    """Verify comparative baseline models execution and constraints."""
    chroma_a = np.ones((12, 10))
    chroma_b = np.ones((12, 10))

    # Mean-chroma cosine
    sim_cos = baseline_mean_chroma_cosine(chroma_a, chroma_b)
    assert 0.0 <= sim_cos <= 1.0

    # DTW without 12-shift search
    dtw_base = baseline_dtw_no_shift(chroma_a, chroma_b)
    assert dtw_base.best_shift == 0
    assert dtw_base.similarity > 0.99

    # Fixed cutoff
    fix_dec = baseline_fixed_similarity_decision(0.80, threshold=0.65)
    assert fix_dec["is_significant"] is True

    # F4 Baselines
    X_synth = np.random.rand(20, 56)
    y_synth = np.array([0] * 10 + [1] * 10)
    fitted_f4 = train_f4_baseline_models(X_synth, y_synth, seed=42)
    assert "lr" in fitted_f4
    assert "linearsvc" in fitted_f4
    assert "rf" in fitted_f4


def test_evaluation_results_and_master_table_generated():
    """Verify reports/evaluation_results.json and evaluation_table.md are present and populated."""
    json_path = REPO_ROOT / "reports" / "evaluation_results.json"
    md_path = REPO_ROOT / "reports" / "evaluation_table.md"

    assert json_path.is_file(), f"evaluation_results.json missing at {json_path}"
    assert md_path.is_file(), f"evaluation_table.md missing at {md_path}"

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Check that all canonical features 1 through 7 are present
    assert "feature_1_similarity" in data
    assert "feature_2_significance" in data
    assert "feature_3_creative_novelty" in data
    assert "feature_4_origin" in data
    assert "feature_5_musical_context" in data
    assert "feature_6_mimic" in data
    assert "feature_7_timeline" in data
    assert "master_table" in data

    # Verify Markdown table contains all required column headers
    with open(md_path, "r", encoding="utf-8") as f:
        md_text = f.read()

    assert "| Feature | Metric | Ours (95% CI) | Baseline | n | Truth source |" in md_text
    assert "Feature 1: Similarity Engine" in md_text
    assert "Feature 2: Statistical Gateway" in md_text
    assert "Feature 3: Creative Novelty" in md_text
    assert "Feature 4: Origin Classifier" in md_text
    assert "Feature 5: Musical Context" in md_text
    assert "Feature 6: Candidate Identification" in md_text
    assert "Feature 7: Timeline Localization" in md_text
