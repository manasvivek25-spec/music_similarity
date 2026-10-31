"""
Feature 3: Creative Score Module.

Computes broad novelty scores (against full genre population) and narrow novelty scores
(against statistical survivors), implements zero-variance safety, assigns score tiers,
and provides case-aware routing across target genres.
"""

from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
from scipy.stats import norm

from src.config import NARROW_CREATIVE_MIN_SURVIVORS
from src.schemas import CreativeScoreResult


def get_creative_score_tier(score_percent: Optional[float]) -> str:
    """
    Map a creative score percentage (0-100) to its scientific novelty tier.

    Tiers:
    - [75.0, 100.0]: High Creative Novelty
    - [50.0, 75.0): Standard Compositional Variance
    - [25.0, 50.0): Algorithmic Saturation Risk
    - [0.0, 25.0): High Structural Imitation
    - None: Unavailable
    """
    if score_percent is None or not np.isfinite(score_percent):
        return "Unavailable"

    if score_percent >= 75.0:
        return "High Creative Novelty"
    elif score_percent >= 50.0:
        return "Standard Compositional Variance"
    elif score_percent >= 25.0:
        return "Algorithmic Saturation Risk"
    else:
        return "High Structural Imitation"


def calculate_z_score_percent(similarity: float, mean: float, std: float) -> float:
    """
    Compute creative score percentage from similarity and population distribution.

    Formula:
        z = (similarity - mean) / std
        creative_score_percent = (1 - norm.cdf(z)) * 100

    Handles zero-variance and non-finite cases safely without throwing division-by-zero errors.
    """
    if not (np.isfinite(similarity) and np.isfinite(mean) and np.isfinite(std)):
        return 50.0

    if std <= 1e-12:
        # Zero variance in reference pool
        if abs(similarity - mean) < 1e-9:
            z = 0.0
        elif similarity > mean:
            z = 10.0  # High similarity compared to degenerate population -> imitation
        else:
            z = -10.0  # Lower similarity -> high novelty
    else:
        z = (similarity - mean) / std

    # Use survival function norm.sf(z) for upper tail 1 - Phi(z) to avoid floating-point underflow at large z
    score_pct = float(norm.sf(z)) * 100.0
    return float(np.clip(score_pct, 0.0, 100.0))


def compute_creative_score(
    observed_similarity: float,
    genre_pool_scores: Sequence[float],
    survivor_scores: Optional[Sequence[float]] = None,
    min_survivors: int = NARROW_CREATIVE_MIN_SURVIVORS,
    evaluated_genres: Optional[List[str]] = None,
    min_pool_size: int = 2,
) -> CreativeScoreResult:
    """
    Compute broad and narrow creative scores with validation and tier assignment.

    Parameters
    ----------
    observed_similarity : float
        Observed similarity score in [0, 1].
    genre_pool_scores : Sequence[float]
        Similarity distribution of the relevant genre reference population.
    survivor_scores : Optional[Sequence[float]]
        Similarity scores of surviving candidates for narrow scoring.
    min_survivors : int, optional
        Minimum survivors required to compute narrow score (default 5).
    evaluated_genres : Optional[List[str]]
        Genres evaluated in this calculation.
    min_pool_size : int, optional
        Minimum reference population samples required to compute broad score (default 5).

    Returns
    -------
    CreativeScoreResult
        Contains broad and narrow percentages, normalized scores [0, 1],
        score tier, and availability status.
    """
    valid_pool = [s for s in genre_pool_scores if np.isfinite(s)]
    genres_list = evaluated_genres or []

    if len(valid_pool) < min_pool_size:
        return CreativeScoreResult(
            broad_score_percent=None,
            broad_score_normalized=None,
            narrow_score_percent=None,
            narrow_score_normalized=None,
            score_tier=f"Insufficient reference population (N={len(valid_pool)}, minimum {min_pool_size} required)" if min_pool_size > 0 else "Unavailable",
            is_available=False,
            evaluated_genres=genres_list,
        )

    pool_mean = float(np.mean(valid_pool))
    pool_std = float(np.std(valid_pool, ddof=1)) if len(valid_pool) > 1 else 0.0

    broad_pct = calculate_z_score_percent(observed_similarity, pool_mean, pool_std)
    broad_norm = broad_pct / 100.0

    # Narrow score: only when at least `min_survivors` (5) survivors are available
    narrow_pct: Optional[float] = None
    narrow_norm: Optional[float] = None

    if survivor_scores is not None:
        valid_survivors = [s for s in survivor_scores if np.isfinite(s)]
        if len(valid_survivors) >= min_survivors:
            surv_mean = float(np.mean(valid_survivors))
            surv_std = float(np.std(valid_survivors, ddof=1)) if len(valid_survivors) > 1 else 0.0
            narrow_pct = calculate_z_score_percent(observed_similarity, surv_mean, surv_std)
            narrow_norm = narrow_pct / 100.0

    tier = get_creative_score_tier(broad_pct)

    # Compute bootstrap confidence interval if reference pool has sufficient samples
    broad_ci: Optional[Tuple[float, float]] = None
    if len(valid_pool) >= 5:
        try:
            from src.stats_test import compute_novelty_bootstrap_ci
            ci_dict = compute_novelty_bootstrap_ci(observed_similarity, valid_pool, n_bootstrap=1000)
            if ci_dict:
                broad_ci = (ci_dict["ci_lower"], ci_dict["ci_upper"])
        except Exception:
            pass

    return CreativeScoreResult(
        broad_score_percent=round(broad_pct, 2),
        broad_score_normalized=round(broad_norm, 4),
        narrow_score_percent=round(narrow_pct, 2) if narrow_pct is not None else None,
        narrow_score_normalized=round(narrow_norm, 4) if narrow_norm is not None else None,
        score_tier=tier,
        is_available=True,
        evaluated_genres=genres_list,
        broad_score_ci=broad_ci,
    )
