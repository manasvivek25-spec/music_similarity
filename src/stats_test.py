"""
Statistical hypothesis testing module.

Provides permutation tests and empirical p-value calculations against null distributions
to test the statistical significance of observed musical similarity scores.
"""

from typing import Sequence
import numpy as np


def null_distribution_test(query_score: float, null_scores: Sequence[float]) -> float:
    """
    Compute empirical p-value of a query similarity score against a null distribution.

    The empirical p-value is calculated as the proportion of similarity scores
    in the null distribution that are greater than or equal to the observed query score:
        p_value = (number of null_scores >= query_score) / (total number of null_scores)

    A low p-value (e.g., p < 0.05) indicates that the query similarity score is
    significantly higher than would be expected by random chance.

    Parameters
    ----------
    query_score : float
        Observed similarity score between two musical works.
    null_scores : Sequence[float]
        Empirical null distribution scores (e.g., from unrelated reference pairs).

    Returns
    -------
    float
        Empirical p-value in the range [0.0, 1.0].

    Raises
    ------
    ValueError
        If `null_scores` is empty.
    """
    if len(null_scores) == 0:
        raise ValueError("null_scores sequence cannot be empty.")

    null_arr = np.asarray(null_scores, dtype=float)
    p_value = float(np.mean(null_arr >= query_score))
    return p_value
