"""
Unit tests for Feature 2 (Statistical Gateway), Feature 3 (Creative Score),
Case Routing, Mimic Identification, and Verdict Logic.
"""

import numpy as np
import pytest
from src.creative_score import (
    calculate_z_score_percent,
    compute_creative_score,
    get_creative_score_tier,
)
from src.mimic import build_mimic_analysis, evaluate_artist_relationship
from src.pipeline import evaluate_final_verdict
from src.schemas import MetadataContext
from src.stats_test import (
    compute_confidence_label,
    compute_severity_label,
    evaluate_statistical_gateway,
    null_distribution_test,
)


def test_null_distribution_test_calculation():
    """Verify empirical p-value calculation against known distributions."""
    null_scores = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    # Query = 0.85 -> 2 values (0.9, 1.0) >= 0.85 -> p = 0.20
    assert np.isclose(null_distribution_test(0.85, null_scores), 0.20)

    # Query = 1.0 -> 1 value (1.0) >= 1.0 -> p = 0.10
    assert np.isclose(null_distribution_test(1.0, null_scores), 0.10)

    # Query = 1.1 -> 0 values -> p = 0.0 (preserves 0.0 possibility)
    assert np.isclose(null_distribution_test(1.1, null_scores), 0.0)

    # Query = 0.05 -> all 10 >= 0.05 -> p = 1.0
    assert np.isclose(null_distribution_test(0.05, null_scores), 1.0)


def test_empty_or_non_finite_null_distribution_returns_none():
    """Empty or non-finite null distribution returns None (Decision 1 guard)."""
    assert null_distribution_test(0.5, []) is None
    assert null_distribution_test(0.5, [np.nan, np.inf]) is None


def test_genre_fallback_below_30_songs():
    """When a genre has fewer than 30 songs, statistical gateway falls back to global null."""
    # Genre with only 10 samples (< 30)
    genre_null = {"Jazz": [0.4] * 10}
    global_null = [0.3] * 50

    res = evaluate_statistical_gateway(
        query_score=0.5,
        genre="Jazz",
        genre_null_scores=genre_null,
        global_null_scores=global_null,
        min_genre_samples=30,
    )
    assert res.fallback_source == "global_null"

    # Genre with 35 samples (>= 30)
    genre_null_large = {"Rock": [0.4] * 35}
    res_genre = evaluate_statistical_gateway(
        query_score=0.5,
        genre="Rock",
        genre_null_scores=genre_null_large,
        global_null_scores=global_null,
        min_genre_samples=30,
    )
    assert res_genre.fallback_source == "genre_specific"


def test_confidence_thresholds():
    """Verify sample size to confidence level mapping: n>=1000 High, 300<=n<1000 Moderate, 5<=n<300 Low, n<5 insufficient."""
    assert compute_confidence_label(1200) == "High"
    assert compute_confidence_label(1000) == "High"
    assert compute_confidence_label(999) == "Moderate"
    assert compute_confidence_label(300) == "Moderate"
    assert compute_confidence_label(299) == "Low"
    assert compute_confidence_label(50) == "Low"
    assert compute_confidence_label(5) == "Low"
    assert compute_confidence_label(4) == "insufficient"
    assert compute_confidence_label(0) == "insufficient"


def test_creative_score_ranges_and_zero_variance():
    """Verify creative score calculation, range in [0, 100], and zero-variance safety."""
    # Normal distribution: mean 0.5, std 0.1
    # Query similarity = 0.5 (identical to mean) -> z = 0 -> creative_score = 50.0%
    score_mid = calculate_z_score_percent(0.5, mean=0.5, std=0.1)
    assert np.isclose(score_mid, 50.0, atol=1e-3)

    # Zero variance: std = 0.0
    score_zero_var_same = calculate_z_score_percent(0.5, mean=0.5, std=0.0)
    assert np.isclose(score_zero_var_same, 50.0)

    score_zero_var_high = calculate_z_score_percent(0.8, mean=0.5, std=0.0)
    assert score_zero_var_high < 1.0  # High similarity compared to population -> very low novelty

    score_zero_var_low = calculate_z_score_percent(0.2, mean=0.5, std=0.0)
    assert score_zero_var_low > 99.0  # Low similarity -> high novelty

    # Score tiers
    assert get_creative_score_tier(80.0) == "High Creative Novelty"
    assert get_creative_score_tier(60.0) == "Standard Compositional Variance"
    assert get_creative_score_tier(40.0) == "Algorithmic Saturation Risk"
    assert get_creative_score_tier(15.0) == "High Structural Imitation"


def test_narrow_score_requires_5_survivors():
    """Narrow creative score should only be computed if >= 5 survivors exist."""
    pool_scores = [0.2, 0.3, 0.4, 0.5, 0.6]

    # Only 3 survivors (< 5)
    res_few = compute_creative_score(
        observed_similarity=0.7,
        genre_pool_scores=pool_scores,
        survivor_scores=[0.6, 0.65, 0.7],
        min_survivors=5,
    )
    assert res_few.is_available is True
    assert res_few.broad_score_percent is not None
    assert res_few.narrow_score_percent is None

    # 6 survivors (>= 5)
    res_many = compute_creative_score(
        observed_similarity=0.7,
        genre_pool_scores=pool_scores,
        survivor_scores=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75],
        min_survivors=5,
    )
    assert res_many.narrow_score_percent is not None


def test_same_artist_warning_suppression():
    """When query and candidate artists match, suppress warning and flag as 'Consistent Artist Style'."""
    query_ctx = MetadataContext(
        track_id="q1",
        title="Song A",
        artist="Radiohead",
        genre="Alternative",
        duration_sec=120.0,
    )

    candidates = [
        # Same artist
        {
            "candidate_id": "c1",
            "title": "Song B",
            "artist": "Radiohead",
            "similarity": 0.90,
            "p_value": 0.005,
            "severity_label": "Critical Anomaly",
        },
        # Different artist
        {
            "candidate_id": "c2",
            "title": "Song C",
            "artist": "Coldplay",
            "similarity": 0.88,
            "p_value": 0.008,
            "severity_label": "Critical Anomaly",
        },
    ]

    resp = build_mimic_analysis(query_ctx, candidates, max_mimic_candidates=3)
    c1 = next(c for c in resp.top_mimic_candidates if c.candidate_id == "c1")
    c2 = next(c for c in resp.top_mimic_candidates if c.candidate_id == "c2")

    # Same artist: warning suppressed
    assert c1.artist_relationship == "Consistent Artist Style"
    assert c1.mimic_warning_active is False
    # P-value and statistical evidence preserved
    assert c1.p_value == 0.005

    # Different artist: warning active
    assert c2.artist_relationship == "Different Artist"
    assert c2.mimic_warning_active is True


def test_final_verdict_three_indicators():
    """Test 3-indicator verdict logic (Similarity>=0.85, P-val<=0.01, Creative<=0.24)."""
    # 3 true -> Critical
    v3 = evaluate_final_verdict(best_similarity=0.90, best_p_value=0.005, normalized_creative_score=0.15)
    assert v3.verdict_code == "critical"
    assert v3.true_indicators_count == 3

    # 2 true -> Warning
    v2 = evaluate_final_verdict(best_similarity=0.90, best_p_value=0.005, normalized_creative_score=0.50)
    assert v2.verdict_code == "warning"
    assert v2.true_indicators_count == 2

    # 1 true -> Advisory
    v1 = evaluate_final_verdict(best_similarity=0.90, best_p_value=0.08, normalized_creative_score=0.60)
    assert v1.verdict_code == "advisory"
    assert v1.true_indicators_count == 1

    # 0 true -> Clear
    v0 = evaluate_final_verdict(best_similarity=0.40, best_p_value=0.30, normalized_creative_score=0.80)
    assert v0.verdict_code == "clear"
    assert v0.true_indicators_count == 0

    # Missing evidence -> Incomplete
    v_inc = evaluate_final_verdict(best_similarity=None, best_p_value=None, normalized_creative_score=None)
    assert v_inc.verdict_code == "incomplete"


def test_empirical_p_value_authoritative_convention():
    """Verify authoritative empirical p-value convention: count(null >= query) / N when N >= 5."""
    # When N >= 5, exact k/N formula applies
    null_five = [0.10, 0.20, 0.30, 0.40, 0.50]

    # Query strictly exceeds all null scores -> p = 0.0
    p_high = null_distribution_test(0.60, null_five)
    assert p_high == 0.0

    # Query is below all null scores -> p = 1.0 (5/5)
    p_low = null_distribution_test(0.05, null_five)
    assert p_low == 1.0

    # Query matches 2 scores (0.40, 0.50) -> p = 2/5 = 0.40
    p_mid = null_distribution_test(0.35, null_five)
    assert np.isclose(p_mid, 0.40)

    # N >= 5 and N < 300 gives Low confidence
    assert compute_confidence_label(len(null_five)) == "Low"


def test_null_distribution_minimum_pair_guard():
    """Verify that N < 5 returns p_value=None and confidence='insufficient' (Decision 1)."""
    # N = 0, 1, 2, 3, 4 must all return None
    for n in range(5):
        sample = [0.5] * n
        p_val = null_distribution_test(0.5, sample)
        assert p_val is None, f"Expected None for N={n}, got {p_val}"
        assert compute_confidence_label(n) == "insufficient"

    # Gateway returns explicit StatisticalGatewayResult with explanation
    gw_result = evaluate_statistical_gateway(
        query_score=0.7,
        genre="classical",
        genre_null_scores={},
        global_null_scores=[0.3, 0.4],  # N = 2 < 5
        min_genre_samples=30,
    )
    assert gw_result.p_value is None
    assert gw_result.confidence == "insufficient"
    assert gw_result.null_sample_count == 2
    assert gw_result.min_required == 5
    assert "Insufficient null population" in gw_result.explanation


def test_non_finite_null_scores_filtering():
    """Non-finite scores (NaN, inf) are filtered before evaluating sample size."""
    # 5 finite scores + 2 non-finite -> N = 5 -> valid calculation
    scores_with_nan = [np.nan, 0.1, 0.2, np.inf, 0.3, 0.4, 0.5]
    p_val = null_distribution_test(0.25, scores_with_nan)
    # 3 finite scores >= 0.25 (0.3, 0.4, 0.5) out of 5 valid -> p = 3/5 = 0.60
    assert np.isclose(p_val, 0.60)

    # 4 finite scores + 2 non-finite -> N = 4 < 5 -> returns None
    scores_under_min = [np.nan, 0.1, 0.2, 0.3, 0.4, np.inf]
    p_val_under = null_distribution_test(0.25, scores_under_min)
    assert p_val_under is None


def test_unavailable_p_value_produces_incomplete_verdict():
    """When p_value is None, verdict is incomplete — insufficient null population (Decision 1)."""
    v = evaluate_final_verdict(
        best_similarity=0.90,  # similarity indicator would be True (>= 0.85)
        best_p_value=None,     # p-value unavailable
        normalized_creative_score=0.15,  # creative indicator would be True (<= 0.24)
        confidence="insufficient",
        candidates_evaluated=2,
        null_sample_count=2,
    )
    # Must NOT produce a completed Critical or Warning verdict
    assert v.verdict_code == "incomplete"
    assert "Incomplete" in v.verdict_label
    assert "Insufficient Null Population" in v.verdict_label
    assert v.indicator_p_value is None
    assert v.indicator_similarity is True
    assert v.indicator_creative is True
    assert "provisional" in v.explanation.lower()
    assert "minimum 5 required" in v.explanation.lower()


def test_normalized_distance_and_similarity_consistency():
    """Verify S = 1 / (1 + D) and D = (1 / S) - 1 relationship."""
    similarity = 0.547
    derived_dist = (1.0 / similarity) - 1.0
    assert np.isclose(derived_dist, 0.82815356)
    assert round(derived_dist, 3) == 0.828

    # Recover similarity from distance
    recovered_sim = 1.0 / (1.0 + derived_dist)
    assert np.isclose(recovered_sim, similarity)

    # Test MimicCandidate populates and preserves normalized_distance
    query_ctx = MetadataContext(
        track_id="q1",
        title="Query Track",
        duration_sec=60.0,
    )
    cand_records = [{
        "candidate_id": "cand_1",
        "title": "Match Track",
        "similarity": 0.547,
        "normalized_distance": 0.8282,
        "best_shift": 6,
        "p_value": 0.0,
        "severity_label": "Critical Anomaly",
    }]
    resp = build_mimic_analysis(query_ctx, cand_records)
    cand = resp.top_mimic_candidates[0]
    assert cand.normalized_distance == 0.8282
    assert cand.similarity == 0.547


def test_prefilter_percentage_clamping_no_negative_zero():
    """Verify coarse cosine prefilter percentage removed is clamped and never returns negative zero."""
    from src.similarity import coarse_cosine_prefilter
    query_chroma = np.ones((12, 10))
    # Identical candidates -> 0% removed
    candidates = {
        "c1": np.ones((12, 10)),
        "c2": np.ones((12, 10)),
    }
    _, stats = coarse_cosine_prefilter(query_chroma, candidates, threshold=0.30)
    assert stats.before_count == 2
    assert stats.after_count == 2
    assert stats.percentage_removed == 0.0
    assert str(stats.percentage_removed) == "0.0"


def test_creative_score_precision_and_tier_units():
    """Verify creative score percent (0-100) and normalized (0-1) bounds and tiers."""
    scores = [0.547, 0.35]
    res = compute_creative_score(0.547, scores)
    assert res.is_available is True
    assert 0.0 <= res.broad_score_percent <= 100.0
    assert 0.0 <= res.broad_score_normalized <= 1.0
    assert np.isclose(res.broad_score_percent / 100.0, res.broad_score_normalized, atol=1e-3)
    # 24.0% falls into High Structural Imitation (< 25%)
    assert res.score_tier == "High Structural Imitation"
    # Normalized creative score 0.2398 triggers indicator (<= 0.24)
    assert res.broad_score_normalized <= 0.24

