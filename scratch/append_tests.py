import os

new_tests = """

def test_creative_score_tier_boundaries_and_scaling():
    \"\"\"Verify exact tier boundaries and strict percent-to-normalized scaling invariant.\"\"\"
    # Boundary 1: [0.0, 25.0) -> High Structural Imitation
    assert get_creative_score_tier(0.0) == "High Structural Imitation"
    assert get_creative_score_tier(10.5) == "High Structural Imitation"
    assert get_creative_score_tier(24.99) == "High Structural Imitation"

    # Boundary 2: [25.0, 50.0) -> Algorithmic Saturation Risk
    assert get_creative_score_tier(25.0) == "Algorithmic Saturation Risk"
    assert get_creative_score_tier(37.5) == "Algorithmic Saturation Risk"
    assert get_creative_score_tier(49.99) == "Algorithmic Saturation Risk"

    # Boundary 3: [50.0, 75.0) -> Standard Compositional Variance
    assert get_creative_score_tier(50.0) == "Standard Compositional Variance"
    assert get_creative_score_tier(62.5) == "Standard Compositional Variance"
    assert get_creative_score_tier(74.99) == "Standard Compositional Variance"

    # Boundary 4: [75.0, 100.0] -> High Creative Novelty
    assert get_creative_score_tier(75.0) == "High Creative Novelty"
    assert get_creative_score_tier(88.0) == "High Creative Novelty"
    assert get_creative_score_tier(100.0) == "High Creative Novelty"

    # Scaling invariant: broad_score_normalized == broad_score_percent / 100.0
    pool = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    for sim in [0.10, 0.25, 0.35, 0.45, 0.55, 0.70, 0.90]:
        res = compute_creative_score(observed_similarity=sim, genre_pool_scores=pool)
        assert res.is_available is True
        assert 0.0 <= res.broad_score_percent <= 100.0
        assert 0.0 <= res.broad_score_normalized <= 1.0
        assert np.isclose(res.broad_score_normalized, round(res.broad_score_percent / 100.0, 4), atol=1e-4)
        assert res.score_tier == get_creative_score_tier(res.broad_score_percent)


def test_creative_score_real_corpus_query():
    \"\"\"Verify Feature 3 against real frozen corpus null distribution.\"\"\"
    import json
    from pathlib import Path
    null_path = Path("features/null_distribution.json")
    if null_path.exists():
        with open(null_path, "r", encoding="utf-8") as f:
            null_data = json.load(f)
        pool = null_data.get("global_null_distribution", [])
        if len(pool) >= 10:
            # High similarity -> low novelty (< 25%)
            res_high_sim = compute_creative_score(observed_similarity=0.85, genre_pool_scores=pool)
            assert res_high_sim.is_available is True
            assert res_high_sim.broad_score_percent < 25.0
            assert res_high_sim.score_tier == "High Structural Imitation"
            assert res_high_sim.broad_score_normalized == round(res_high_sim.broad_score_percent / 100.0, 4)

            # Low similarity -> high novelty (> 75%)
            res_low_sim = compute_creative_score(observed_similarity=0.10, genre_pool_scores=pool)
            assert res_low_sim.is_available is True
            assert res_low_sim.broad_score_percent > 75.0
            assert res_low_sim.score_tier == "High Creative Novelty"
            assert res_low_sim.broad_score_normalized == round(res_low_sim.broad_score_percent / 100.0, 4)
"""

with open('tests/test_stats.py', 'a', encoding='utf-8', newline='\r\n') as f:
    f.write(new_tests)
print("Successfully appended new tests to tests/test_stats.py")

