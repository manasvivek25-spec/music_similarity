"""
Statistical hypothesis testing module.

Provides permutation tests, null distribution construction from reference audio pools,
empirical p-value calculations, and gateway filtering according to defined confidence
and severity criteria.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
from scipy import stats
from scipy.stats import levene, mannwhitneyu, norm, shapiro, ttest_ind

from src.config import (
    GENRE_MIN_SAMPLE_SIZE,
    MAX_SURVIVORS,
    MIN_NULL_PAIRS,
    NULL_DIST_SAMPLE_PAIRS,
    P_VALUE_BORDERLINE_THRESHOLD,
    P_VALUE_CRITICAL_THRESHOLD,
)
from src.schemas import StatisticalGatewayResult


def is_fixture_identifier(identifier: str, file_path: Optional[str] = None) -> bool:
    """
    Check if a track ID or path represents a local test fixture / synthetic audio.
    Strictly prevents artificial test fixtures (e.g. test_65s.wav, short_clip_7s.wav,
    test_transformed_3semitones.wav) from contaminating production candidate catalogs
    or reference null distributions.
    """
    ident = str(identifier).lower()
    if any(ident.startswith(pfx) for pfx in ("test_", "fixture_", "short_clip_", "synth_", "drone_", "tmp")):
        return True
    if file_path:
        fp = str(file_path).lower().replace("\\", "/")
        if any(seg in fp for seg in ("temp_uploads", "fixtures", "test_", "short_clip")):
            return True
    return False


def null_distribution_test(
    query_score: float,
    null_scores: Sequence[float],
    min_pairs: int = MIN_NULL_PAIRS,
) -> Optional[float]:
    """
    Compute empirical p-value of a query similarity score against a null distribution.

    Authoritative Convention (Authoritative Decision 1):
        Retains the unsmoothed empirical formula:
            p = k / N = sum(I[S_i >= S_q]) / N
        where:
            k = count(valid_null_score >= query_score)
            N = count of valid, finite null scores actually used.
        No plus-one smoothing (+1) is applied: when k = 0, p = 0.0.

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
    """
    if null_scores is None or len(null_scores) == 0:
        return None

    null_arr = np.asarray(null_scores, dtype=float)
    valid_mask = np.isfinite(null_arr)
    valid_scores = null_arr[valid_mask]
    N = int(len(valid_scores))

    if N < min_pairs:
        return None

    k = int(np.sum(valid_scores >= query_score))
    p_value = float(k / N)
    return float(np.clip(p_value, 0.0, 1.0))


def null_distribution_test_detailed(
    query_score: float,
    null_scores: Sequence[float],
    min_pairs: int = MIN_NULL_PAIRS,
) -> Tuple[Optional[float], Optional[int], int]:
    """
    Compute empirical p-value along with exceedance count (k) and valid null size (N).

    Formula:
        p = k / N = sum(I[S_i >= S_q]) / N
    Strictly unsmoothed: when k = 0, p = 0.0.

    Returns
    -------
    Tuple[Optional[float], Optional[int], int]
        (p_value, exceedance_count k, null_sample_size N)
    """
    if null_scores is None or len(null_scores) == 0:
        return (None, None, 0)

    null_arr = np.asarray(null_scores, dtype=float)
    valid_mask = np.isfinite(null_arr)
    valid_scores = null_arr[valid_mask]
    N = int(len(valid_scores))

    if N < min_pairs:
        return (None, None, N)

    k = int(np.sum(valid_scores >= query_score))
    p_value = float(np.clip(float(k / N), 0.0, 1.0))
    return (p_value, k, N)


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
        c_i = chroma_dict[track_ids[i]]
        c_j = chroma_dict[track_ids[j]]
        c_i_eval = c_i[:, :300] if (c_i.ndim == 2 and c_i.shape[1] > 300) else c_i
        c_j_eval = c_j[:, :300] if (c_j.ndim == 2 and c_j.shape[1] > 300) else c_j
        score = similarity_func(c_i_eval, c_j_eval)
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
            exceedance_count=None,
            null_sample_size=0,
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
            exceedance_count=None,
            null_sample_size=sample_count,
            confidence="insufficient",
            severity_label="Indeterminate",
            fallback_source=fallback_source,
            min_required=min_null_pairs,
            explanation=f"Insufficient null population (N={sample_count}, minimum {min_null_pairs} required).",
        )

    p_val, k_val, N_val = null_distribution_test_detailed(query_score, valid_scores, min_pairs=min_null_pairs)
    confidence = compute_confidence_label(sample_count)
    severity = compute_severity_label(p_val)

    explanation = None
    if p_val == 0.0:
        explanation = (
            f"Empirical p=0.0 (k=0 of {sample_count} null comparisons exceeded query score {query_score:.4f}; "
            f"bound p < 1/{sample_count} = {1.0/sample_count:.4f} due to finite reference null sample size)."
        )

    return StatisticalGatewayResult(
        p_value=p_val,
        null_sample_count=sample_count,
        exceedance_count=k_val,
        null_sample_size=N_val,
        confidence=confidence,
        severity_label=severity,
        fallback_source=fallback_source,
        min_required=min_null_pairs,
        explanation=explanation,
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

    # Determine baseline tracks: filter strictly to non-fixture human tracks if origin metadata exists
    has_origin = False
    eval_chroma = {}
    for cid, c in chroma_dict.items():
        if is_fixture_identifier(cid):
            continue
        if candidate_metadata:
            meta = candidate_metadata.get(cid, {})
            if meta.get("is_fixture", False):
                continue
            origin = str(meta.get("origin_label", "")).lower()
            if origin in ("human", "0", "false"):
                eval_chroma[cid] = c
        else:
            eval_chroma[cid] = c

    eval_ids = sorted(list(eval_chroma.keys()))

    if not force_rebuild and target_file.is_file():
        try:
            with open(target_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            cached_ids = data.get("source_track_ids", [])
            cached_scores = data.get("null_scores", [])
            # Return cached null distribution if it has cached scores and covers the requested tracks
            if len(cached_scores) > 0:
                if (not eval_ids) or set(eval_ids).issubset(set(cached_ids)) or set(cached_ids) == set(eval_ids):
                    return cached_scores, data.get("genre_null_scores", {})
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
    if Path(target_file).resolve() == Path(NULL_DIST_CACHE_FILE).resolve() and len(eval_ids) < 30:
        return global_null, genre_null
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


def compute_novelty_bootstrap_ci(
    observed_similarity: float,
    reference_scores: Sequence[float],
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> Optional[Dict[str, float]]:
    """
    Compute bootstrap confidence intervals for Feature 3 relative structural novelty score.

    Resamples reference population with replacement B times, recalculating the novelty
    score across each replicate to estimate empirical sampling variance and confidence bounds.

    Parameters
    ----------
    observed_similarity : float
        Observed query similarity score in [0, 1].
    reference_scores : Sequence[float]
        Empirical reference similarity scores from the eligible population.
    n_bootstrap : int
        Number of bootstrap resamples (default 1000).
    ci_level : float
        Confidence level for interval, e.g. 0.95 for 95% CI.
    seed : int
        Random seed for deterministic reproducibility.

    Returns
    -------
    Optional[Dict[str, float]]
        Contains point estimate, lower and upper confidence limits, or None if insufficient samples (< 5).
    """
    valid = np.asarray([s for s in reference_scores if np.isfinite(s)], dtype=float)
    if len(valid) < 5:
        return None

    # Point estimate
    mu_pt = float(np.mean(valid))
    sigma_pt = float(np.std(valid, ddof=1)) if len(valid) > 1 else 0.0
    if sigma_pt <= 1e-12:
        z_pt = 0.0 if abs(observed_similarity - mu_pt) < 1e-9 else (10.0 if observed_similarity > mu_pt else -10.0)
    else:
        z_pt = (observed_similarity - mu_pt) / sigma_pt
    pt_score = float(np.clip(float(norm.sf(z_pt)) * 100.0, 0.0, 100.0))

    rng = np.random.default_rng(seed)
    n = len(valid)
    bootstrap_scores = np.empty(n_bootstrap, dtype=float)

    for b in range(n_bootstrap):
        sample = rng.choice(valid, size=n, replace=True)
        mu_b = float(np.mean(sample))
        sigma_b = float(np.std(sample, ddof=1)) if n > 1 else 0.0
        if sigma_b <= 1e-12:
            z_b = 0.0 if abs(observed_similarity - mu_b) < 1e-9 else (10.0 if observed_similarity > mu_b else -10.0)
        else:
            z_b = (observed_similarity - mu_b) / sigma_b
        bootstrap_scores[b] = np.clip(float(norm.sf(z_b)) * 100.0, 0.0, 100.0)

    alpha = 1.0 - ci_level
    ci_lower = float(np.percentile(bootstrap_scores, 100.0 * (alpha / 2.0)))
    ci_upper = float(np.percentile(bootstrap_scores, 100.0 * (1.0 - alpha / 2.0)))

    return {
        "point_estimate": round(pt_score, 2),
        "ci_lower": round(ci_lower, 2),
        "ci_upper": round(ci_upper, 2),
        "ci_level": ci_level,
        "n_bootstrap": n_bootstrap,
        "sample_size": n,
        "reference_mean": round(mu_pt, 4),
        "reference_std": round(sigma_pt, 4),
    }


def two_sample_hypothesis_test(
    sample_a: Sequence[float],
    sample_b: Sequence[float],
    name_a: str = "Human-Human",
    name_b: str = "Human-AI",
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Perform rigorous two-sample inferential hypothesis testing between two similarity or acoustic distributions.

    Evaluates:
    - Sample counts, means, medians, variances, and IQRs.
    - Normality via Shapiro-Wilk (or D'Agostino-Pearson for larger samples).
    - Variance homogeneity via Levene's test.
    - Welch's two-sample t-test (unequal variances assumed).
    - Nonparametric Mann-Whitney U test (robust to distribution skew).
    - Effect sizes: Cohen's d and Rank-biserial correlation.
    - Full academic disclaimers regarding pairwise track dependence.

    Parameters
    ----------
    sample_a : Sequence[float]
        Distribution observations for group A.
    sample_b : Sequence[float]
        Distribution observations for group B.
    name_a : str
        Label for group A.
    name_b : str
        Label for group B.
    alpha : float
        Significance threshold (default 0.05).

    Returns
    -------
    Dict[str, Any]
        Structured test statistics, p-values, effect sizes, and interpretative conclusions.
    """
    arr_a = np.asarray([x for x in sample_a if np.isfinite(x)], dtype=float)
    arr_b = np.asarray([x for x in sample_b if np.isfinite(x)], dtype=float)

    n_a = len(arr_a)
    n_b = len(arr_b)

    if n_a < 2 or n_b < 2:
        return {
            "status": "insufficient_data",
            "message": f"Insufficient observations for hypothesis testing (N_A={n_a}, N_B={n_b}; minimum 2 per group required).",
            "group_a": {"name": name_a, "n": n_a},
            "group_b": {"name": name_b, "n": n_b},
        }

    # Descriptive statistics
    mean_a = float(np.mean(arr_a))
    mean_b = float(np.mean(arr_b))
    std_a = float(np.std(arr_a, ddof=1)) if n_a > 1 else 0.0
    std_b = float(np.std(arr_b, ddof=1)) if n_b > 1 else 0.0
    med_a = float(np.median(arr_a))
    med_b = float(np.median(arr_b))
    iqr_a = float(np.percentile(arr_a, 75) - np.percentile(arr_a, 25))
    iqr_b = float(np.percentile(arr_b, 75) - np.percentile(arr_b, 25))

    # Normality testing
    norm_a_p: Optional[float] = None
    norm_b_p: Optional[float] = None
    if 3 <= n_a <= 5000:
        norm_a_p = float(shapiro(arr_a).pvalue)
    if 3 <= n_b <= 5000:
        norm_b_p = float(shapiro(arr_b).pvalue)

    # Variance homogeneity (Levene's test)
    levene_stat, levene_p = levene(arr_a, arr_b)

    # Welch's two-sample t-test (robust to unequal variances)
    t_res = ttest_ind(arr_a, arr_b, equal_var=False)
    t_stat = float(t_res.statistic)
    t_p = float(t_res.pvalue)

    # Degrees of freedom (Welch-Satterthwaite equation)
    v_a = (std_a ** 2) / max(n_a, 1)
    v_b = (std_b ** 2) / max(n_b, 1)
    denom = ((v_a ** 2) / max(n_a - 1, 1)) + ((v_b ** 2) / max(n_b - 1, 1))
    df_welch = float(((v_a + v_b) ** 2) / denom) if denom > 1e-12 else float(n_a + n_b - 2)

    # Mann-Whitney U test (non-parametric alternative)
    u_res = mannwhitneyu(arr_a, arr_b, alternative="two-sided")
    u_stat = float(u_res.statistic)
    u_p = float(u_res.pvalue)

    # Rank-biserial correlation: r = 1 - (2*U / (n_a * n_b))
    rank_biserial = float(1.0 - (2.0 * u_stat / max(n_a * n_b, 1)))

    # Cohen's d effect size
    pooled_sd = float(np.sqrt(((n_a - 1) * (std_a ** 2) + (n_b - 1) * (std_b ** 2)) / max(n_a + n_b - 2, 1)))
    cohen_d = float((mean_a - mean_b) / pooled_sd) if pooled_sd > 1e-9 else 0.0

    # Interpretations
    welch_sig = bool(t_p < alpha)
    mwu_sig = bool(u_p < alpha)

    h0_statement = f"The similarity distributions of {name_a} and {name_b} are identical."
    h1_statement = f"The similarity distributions of {name_a} and {name_b} differ significantly."

    if mwu_sig:
        interp = (
            f"Statistically significant difference observed between {name_a} (median={med_a:.3f}) "
            f"and {name_b} (median={med_b:.3f}) under Mann-Whitney U (U={u_stat:.1f}, p={u_p:.4e}, "
            f"rank-biserial r={rank_biserial:.3f}). H0 is rejected at alpha={alpha}."
        )
    else:
        interp = (
            f"No statistically significant difference detected between {name_a} (median={med_a:.3f}) "
            f"and {name_b} (median={med_b:.3f}) under Mann-Whitney U (U={u_stat:.1f}, p={u_p:.4e}). "
            f"Fail to reject H0 at alpha={alpha}."
        )

    return {
        "status": "success",
        "hypotheses": {
            "null_hypothesis_H0": h0_statement,
            "alternative_hypothesis_H1": h1_statement,
        },
        "group_a": {
            "name": name_a,
            "sample_size": n_a,
            "mean": round(mean_a, 4),
            "median": round(med_a, 4),
            "std": round(std_a, 4),
            "iqr": round(iqr_a, 4),
            "shapiro_normality_p": round(norm_a_p, 4) if norm_a_p is not None else None,
        },
        "group_b": {
            "name": name_b,
            "sample_size": n_b,
            "mean": round(mean_b, 4),
            "median": round(med_b, 4),
            "std": round(std_b, 4),
            "iqr": round(iqr_b, 4),
            "shapiro_normality_p": round(norm_b_p, 4) if norm_b_p is not None else None,
        },
        "tests": {
            "levene_homogeneity": {
                "statistic": round(float(levene_stat), 4),
                "p_value": round(float(levene_p), 4),
                "equal_variance_assumed": bool(levene_p >= alpha),
            },
            "welch_t_test": {
                "t_statistic": round(t_stat, 4),
                "p_value": round(t_p, 4),
                "degrees_of_freedom": round(df_welch, 2),
                "significant": welch_sig,
                "effect_size_cohen_d": round(cohen_d, 4),
            },
            "mann_whitney_u_test": {
                "u_statistic": round(u_stat, 2),
                "p_value": round(u_p, 4),
                "rank_biserial_r": round(rank_biserial, 4),
                "significant": mwu_sig,
            },
        },
        "conclusion": interp,
        "limitations": (
            "Pairwise comparisons sharing common underlying tracks violate independent observation assumptions; "
            "results should be evaluated as empirical distribution divergence rather than independent random trials."
        ),
    }


