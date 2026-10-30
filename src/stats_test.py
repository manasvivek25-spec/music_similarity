"""
Statistical hypothesis testing module.

Provides permutation tests, null distribution construction from reference audio pools,
empirical p-value calculations, and gateway filtering according to defined confidence
and severity criteria.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np

from src.config import (
    GENRE_MIN_SAMPLE_SIZE,
    MAX_SURVIVORS,
    MIN_NULL_PAIRS,
    NULL_DIST_SAMPLE_PAIRS,
    P_VALUE_BORDERLINE_THRESHOLD,
    P_VALUE_CRITICAL_THRESHOLD,
)
from src.schemas import StatisticalGatewayResult


def null_distribution_test(
    query_score: float,
    null_scores: Sequence[float],
    min_pairs: int = MIN_NULL_PAIRS,
) -> Optional[float]:
    """
    Compute empirical p-value of a query similarity score against a null distribution.

    Authoritative Convention (Authoritative Decision 1):
        Retains the unsmoothed empirical formula:
            p = k / N
        where:
            k = count(valid_null_score >= query_score)
            N = count of valid, finite null scores actually used.

    Guard:
        When the number of valid finite null scores is less than min_pairs (default 5),
        returns None to indicate insufficient null population for statistical evaluation.

    Parameters
    ----------
    query_score : float
        Observed similarity score between two musical tracks.
    null_scores : Sequence[float]
        Empirical null distribution scores from reference comparisons.
    min_pairs : int, optional
        Minimum number of valid null pairs required (default: MIN_NULL_PAIRS = 5).

    Returns
    -------
    Optional[float]
        Empirical p-value in [0.0, 1.0], or None if valid null sample count < min_pairs.

    Raises
    ------
    ValueError
        If null_scores is empty or contains no finite values.
    """
    if null_scores is None or len(null_scores) == 0:
        return None

    null_arr = np.asarray(null_scores, dtype=float)
    valid_mask = np.isfinite(null_arr)
    valid_scores = null_arr[valid_mask]

    if len(valid_scores) < min_pairs:
        return None

    p_value = float(np.mean(valid_scores >= query_score))
    return float(np.clip(p_value, 0.0, 1.0))


def benjamini_hochberg_correction(
    p_values: Sequence[Optional[float]],
) -> List[Optional[float]]:
    """
    Apply Benjamini-Hochberg (BH) False Discovery Rate (FDR) procedure.

    Calculates adjusted p-values (q-values) across the evaluated candidate pool:
        p_adj(i) = min_{j >= i} min(1.0, (m / j) * p_(j))
    where:
        m = total valid non-null p-values
        p_(1) <= p_(2) <= ... <= p_(m) are sorted p-values.

    Preserves None for candidates where p-value was unavailable.
    Does NOT replace raw p-values or alter locked primary thresholds.

    Parameters
    ----------
    p_values : Sequence[Optional[float]]
        Collection of raw empirical p-values (may contain None).

    Returns
    -------
    List[Optional[float]]
        Adjusted p-values matching original input ordering.
    """
    if not p_values:
        return []

    # Filter valid p-values with original indices
    indexed_p = [(idx, float(p)) for idx, p in enumerate(p_values) if p is not None and np.isfinite(p)]
    m = len(indexed_p)

    adjusted_results: List[Optional[float]] = [None] * len(p_values)
    if m == 0:
        return adjusted_results

    # Sort ascending by p-value: rank from 1 to m
    indexed_p.sort(key=lambda x: x[1])

    # Compute raw step values: (m / rank) * p
    raw_adjusted = []
    for rank_1, (orig_idx, p_val) in enumerate(indexed_p, start=1):
        q = (m / rank_1) * p_val
        raw_adjusted.append((orig_idx, min(1.0, q)))

    # Ensure monotonicity from right to left: p_adj[i] = min(p_adj[i], p_adj[i+1])
    monotone_q = [0.0] * m
    min_so_far = 1.0
    for i in range(m - 1, -1, -1):
        orig_idx, q_val = raw_adjusted[i]
        min_so_far = min(min_so_far, q_val)
        monotone_q[i] = (orig_idx, round(float(np.clip(min_so_far, 0.0, 1.0)), 4))

    for orig_idx, q_final in monotone_q:
        adjusted_results[orig_idx] = q_final

    return adjusted_results


def compute_confidence_label(n_samples: int) -> str:
    """
    Determine operational sample-size confidence tier based on the number of valid null pairs:
    - n >= 1000: High
    - 300 <= n < 1000: Moderate
    - 5 <= n < 300: Low
    - n < 5: insufficient

    Note: These operational sample-size labels categorize sample availability and
    do not constitute formal mathematical guarantees of statistical confidence or independence.
    """
    if n_samples >= 1000:
        return "High"
    elif n_samples >= 300:
        return "Moderate"
    elif n_samples >= 5:
        return "Low"
    else:
        return "insufficient"


def compute_severity_label(p_value: Optional[float]) -> str:
    """
    Determine statistical gateway severity label based on empirical p-value:

    - p > 0.10: Valid Coincidence (drop)
    - 0.01 < p <= 0.10: Borderline Indicator (retain)
    - p <= 0.01: Critical Anomaly (retain)
    """
    if p_value is None:
        return "Indeterminate"

    if p_value <= P_VALUE_CRITICAL_THRESHOLD:
        return "Critical Anomaly"
    elif p_value <= P_VALUE_BORDERLINE_THRESHOLD:
        return "Borderline Indicator"
    else:
        return "Valid Coincidence"


def build_pairwise_null_distribution(
    chroma_dict: Dict[str, np.ndarray],
    similarity_func,
    max_pairs: int = NULL_DIST_SAMPLE_PAIRS,
    seed: int = 42
) -> List[float]:
    """
    Construct empirical null distribution scores from a pool of reference tracks.

    Avoids self-comparisons and duplicate-pair inflation by considering only (i < j).
    Uses deterministic pair sampling when total unique pairs exceed max_pairs.

    Parameters
    ----------
    chroma_dict : Dict[str, np.ndarray]
        Mapping of track_id -> beat-synchronized chroma matrix.
    similarity_func : callable
        Function accepting (chroma_a, chroma_b) and returning float similarity.
    max_pairs : int, optional
        Maximum pairs to evaluate, default 2000.
    seed : int, optional
        Random seed for deterministic pair selection.

    Returns
    -------
    List[float]
        Computed null distribution similarity scores.
    """
    track_ids = list(chroma_dict.keys())
    n = len(track_ids)
    if n < 2:
        return []

    # Generate all unique (i, j) where i < j
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]

    # Deterministic sampling if pairs exceed cap
    if len(pairs) > max_pairs:
        rng = np.random.default_rng(seed)
        sampled_indices = rng.choice(len(pairs), size=max_pairs, replace=False)
        pairs = [pairs[idx] for idx in sampled_indices]

    null_scores: List[float] = []
    for i, j in pairs:
        score = similarity_func(chroma_dict[track_ids[i]], chroma_dict[track_ids[j]])
        if np.isfinite(score):
            null_scores.append(float(score))

    return null_scores


def evaluate_statistical_gateway(
    query_score: float,
    genre: Optional[str],
    genre_null_scores: Dict[str, List[float]],
    global_null_scores: List[float],
    min_genre_samples: int = GENRE_MIN_SAMPLE_SIZE,
    min_null_pairs: int = MIN_NULL_PAIRS,
) -> StatisticalGatewayResult:
    """
    Evaluate observed query score against appropriate null distribution.

    Selects genre-specific null distribution if genre is provided and has at least
    `min_genre_samples` items. Otherwise, falls back to the global null distribution.

    Parameters
    ----------
    query_score : float
        Observed similarity score.
    genre : Optional[str]
        Target genre for query or candidate match.
    genre_null_scores : Dict[str, List[float]]
        Precomputed null distributions keyed by genre name.
    global_null_scores : List[float]
        Global reference null distribution.
    min_genre_samples : int, optional
        Threshold to use genre-specific distribution (default 30).
    min_null_pairs : int, optional
        Minimum valid null pairs required (default: MIN_NULL_PAIRS = 5).

    Returns
    -------
    StatisticalGatewayResult
        Contains empirical p-value, sample count, confidence, severity, and fallback source.
    """
    fallback_source = "none"
    chosen_scores: List[float] = []

    # Check for genre-specific distribution
    if genre and genre in genre_null_scores and len(genre_null_scores[genre]) >= min_genre_samples:
        chosen_scores = genre_null_scores[genre]
        fallback_source = "genre_specific"
    elif len(global_null_scores) > 0:
        chosen_scores = global_null_scores
        fallback_source = "global_null"

    if len(chosen_scores) == 0:
        return StatisticalGatewayResult(
            p_value=None,
            null_sample_count=0,
            confidence="insufficient",
            severity_label="Indeterminate",
            fallback_source="none",
            min_required=min_null_pairs,
            explanation=f"Insufficient null population (N=0, minimum {min_null_pairs} required).",
        )

    null_arr = np.asarray(chosen_scores, dtype=float)
    valid_scores = null_arr[np.isfinite(null_arr)]
    sample_count = int(len(valid_scores))

    if sample_count < min_null_pairs:
        return StatisticalGatewayResult(
            p_value=None,
            null_sample_count=sample_count,
            confidence="insufficient",
            severity_label="Indeterminate",
            fallback_source=fallback_source,
            min_required=min_null_pairs,
            explanation=f"Insufficient null population (N={sample_count}, minimum {min_null_pairs} required).",
        )

    p_val = null_distribution_test(query_score, valid_scores, min_pairs=min_null_pairs)
    confidence = compute_confidence_label(sample_count)
    severity = compute_severity_label(p_val)

    return StatisticalGatewayResult(
        p_value=p_val,
        null_sample_count=sample_count,
        confidence=confidence,
        severity_label=severity,
        fallback_source=fallback_source,
        min_required=min_null_pairs,
        explanation=None,
    )


def invalidate_null_distribution_cache(cache_file: Optional[Path] = None) -> bool:
    """
    Safely delete or invalidate the cached null distribution when reference corpus changes.
    """
    from src.config import NULL_DIST_CACHE_FILE
    target = cache_file or NULL_DIST_CACHE_FILE
    try:
        if target.is_file():
            target.unlink()
            return True
    except Exception:
        pass
    return False


def get_or_build_cached_null_distribution(
    chroma_dict: Dict[str, np.ndarray],
    similarity_func,
    cache_file: Optional[Path] = None,
    max_pairs: int = NULL_DIST_SAMPLE_PAIRS,
    candidate_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
    force_rebuild: bool = False,
) -> Tuple[List[float], Dict[str, List[float]]]:
    """
    Load precomputed pairwise null distribution from disk cache, or compute and cache it.

    Avoids repeated pairwise DTW computations across queries. Caches are invalidated
    automatically if source candidate IDs change or force_rebuild=True.

    Parameters
    ----------
    chroma_dict : Dict[str, np.ndarray]
        Mapping of candidate track_id -> beat-synchronized chroma matrix.
    similarity_func : callable
        Function accepting (chroma_a, chroma_b) -> float.
    cache_file : Optional[Path]
        Location to save/load null distribution JSON.
    max_pairs : int
        Maximum pairwise combinations to evaluate.
    candidate_metadata : Optional[Dict[str, Dict[str, Any]]]
        Candidate metadata (for per-genre null distributions).
    force_rebuild : bool
        If True, ignores existing cache and recomputes.

    Returns
    -------
    Tuple[List[float], Dict[str, List[float]]]
        (global_null_scores, genre_null_scores_dict)
    """
    import json
    import os
    import time
    from src.config import NULL_DIST_CACHE_FILE

    target_file = cache_file or NULL_DIST_CACHE_FILE

    # Determine baseline tracks: filter strictly to human tracks if origin metadata exists
    has_origin = False
    eval_chroma = chroma_dict
    if candidate_metadata:
        has_origin = any("origin_label" in meta for meta in candidate_metadata.values())
        if has_origin:
            eval_chroma = {
                cid: c for cid, c in chroma_dict.items()
                if str(candidate_metadata.get(cid, {}).get("origin_label", "")).lower() in ("human", "0", "false")
            }

    eval_ids = sorted(list(eval_chroma.keys()))

    if not force_rebuild and target_file.is_file():
        try:
            with open(target_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            cached_ids = data.get("source_track_ids", [])
            # Valid if cached track IDs match current baseline tracks and has null scores
            if set(cached_ids) == set(eval_ids) and "null_scores" in data:
                return data["null_scores"], data.get("genre_null_scores", {})
        except Exception:
            pass  # Fall through to rebuild on parse/read error

    # Compute pairwise global null distribution
    global_null = build_pairwise_null_distribution(
        eval_chroma, similarity_func, max_pairs=max_pairs
    )

    # Compute genre-specific null distributions if metadata provided (also using human baseline)
    genre_null: Dict[str, List[float]] = {}
    if candidate_metadata:
        genre_groups: Dict[str, Dict[str, np.ndarray]] = {}
        for cid, meta in candidate_metadata.items():
            g = meta.get("genre")
            if g and cid in eval_chroma:
                genre_groups.setdefault(g, {})[cid] = eval_chroma[cid]
        for g, g_chroma in genre_groups.items():
            if len(g_chroma) >= 2:
                g_scores = build_pairwise_null_distribution(
                    g_chroma, similarity_func, max_pairs=min(max_pairs, 500)
                )
                if g_scores:
                    genre_null[g] = g_scores

    # Persist cache atomically
    try:
        target_file.parent.mkdir(parents=True, exist_ok=True)
        tmp_target = target_file.with_suffix(".tmp")
        payload = {
            "version": "1.0",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "source_track_ids": eval_ids,
            "sample_pairs_count": len(global_null),
            "null_scores": [round(float(s), 5) for s in global_null],
            "genre_null_scores": {
                g: [round(float(s), 5) for s in scores] for g, scores in genre_null.items()
            },
            "pairwise_dependency_note": (
                "Pairwise null observations are formed from pairs within the reference pool "
                "and share underlying tracks. While k/N provides the empirical p-value, "
                "observations are not strictly independent."
            ),
        }
        with open(tmp_target, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp_target, target_file)
    except Exception:
        pass

    return global_null, genre_null

