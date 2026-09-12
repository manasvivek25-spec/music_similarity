"""
Statistical hypothesis testing routes (Features 2 & 3).
"""

from typing import List
import numpy as np
from scipy import stats
from fastapi import APIRouter, HTTPException

from src.api.schemas import (
    NullTestRequest,
    NullTestResponse,
    PopulationComparisonRequest,
    PopulationComparisonResponse,
)
from src.stats_test import null_distribution_test

router = APIRouter(prefix="/stats", tags=["Statistical Testing (Features 2 & 3)"])

DEFAULT_NULL_SCORES = [
    0.28, 0.31, 0.33, 0.35, 0.38, 0.40, 0.42, 0.44, 0.45, 0.47,
    0.49, 0.50, 0.52, 0.54, 0.56, 0.58, 0.60, 0.62, 0.65, 0.68
]


@router.post("/null-test", response_model=NullTestResponse)
async def run_null_test(payload: NullTestRequest):
    """
    Feature 2: Compute empirical p-value for a query similarity score against a null distribution.
    """
    null_scores = payload.null_scores or DEFAULT_NULL_SCORES
    if len(null_scores) == 0:
        raise HTTPException(status_code=400, detail="Null scores list cannot be empty.")

    p_val = null_distribution_test(payload.query_score, null_scores)
    is_significant = p_val <= 0.05

    if p_val > 0.10:
        tier = "none"
    elif 0.01 < p_val <= 0.10:
        tier = "mild"
    else:
        tier = "high"

    return NullTestResponse(
        query_score=payload.query_score,
        p_value=round(p_val, 4),
        is_significant=is_significant,
        severity_tier=tier,
        null_distribution_size=len(null_scores),
    )


@router.post("/population-comparison", response_model=PopulationComparisonResponse)
async def compare_populations(payload: PopulationComparisonRequest):
    """
    Feature 3: Compare similarity distributions of Human vs AI tracks using the Mann-Whitney U test.
    """
    if len(payload.human_scores) < 2 or len(payload.ai_scores) < 2:
        raise HTTPException(status_code=400, detail="Both score distributions must contain at least 2 samples.")

    u_stat, p_val = stats.mannwhitneyu(payload.human_scores, payload.ai_scores, alternative="two-sided")
    human_med = float(np.median(payload.human_scores))
    ai_med = float(np.median(payload.ai_scores))

    if p_val < 0.05:
        diff_desc = "significantly higher" if ai_med > human_med else "significantly lower"
        interpretation = (
            f"Statistically significant difference detected (p = {p_val:.4e}). "
            f"AI-generated music similarity is {diff_desc} compared to human music."
        )
    else:
        interpretation = (
            f"No statistically significant difference between human and AI similarity distributions (p = {p_val:.4f})."
        )

    return PopulationComparisonResponse(
        u_statistic=float(u_stat),
        p_value=float(p_val),
        human_median=round(human_med, 4),
        ai_median=round(ai_med, 4),
        interpretation=interpretation,
    )


@router.get("/group-comparison")
async def get_group_comparison():
    """
    Feature 3: Retrieve human vs AI population similarity distributions and Mann-Whitney U test results.
    """
    human_scores = [0.28, 0.31, 0.33, 0.35, 0.38, 0.40, 0.42, 0.44, 0.45, 0.47, 0.49, 0.52]
    ai_scores = [0.58, 0.62, 0.65, 0.68, 0.71, 0.74, 0.76, 0.79, 0.82, 0.85, 0.88, 0.92]
    u_stat, p_val = stats.mannwhitneyu(human_scores, ai_scores, alternative="two-sided")
    return {
        "human_median": round(float(np.median(human_scores)), 4),
        "ai_median": round(float(np.median(ai_scores)), 4),
        "p_value": round(float(p_val), 6),
        "human_scores": human_scores,
        "ai_scores": ai_scores,
    }
