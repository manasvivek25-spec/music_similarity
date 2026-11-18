"""
Statistical Uncertainty and Metrics Computation Module.
Provides Wilson, Clopper-Pearson, group bootstrap, McNemar, paired bootstrap,
and complete scientific metric calculations for the evaluation harness.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import scipy.stats as stats
from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    balanced_accuracy_score,
    confusion_matrix,
    brier_score_loss,
)


def compute_wilson_ci(k: int, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    """
    Compute Wilson score interval for a binomial proportion.
    """
    if n <= 0:
        return 0.0, 0.0
    alpha = 1.0 - confidence
    z = stats.norm.ppf(1.0 - alpha / 2.0)
    p = float(k) / float(n)
    denominator = 1.0 + (z**2) / float(n)
    center = (p + (z**2) / (2.0 * float(n))) / denominator
    margin = (z / denominator) * np.sqrt(
        (p * (1.0 - p)) / float(n) + (z**2) / (4.0 * (float(n) ** 2))
    )
    lower = max(0.0, float(center - margin))
    upper = min(1.0, float(center + margin))
    return round(lower, 4), round(upper, 4)


def compute_clopper_pearson_ci(k: int, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    """
    Compute Clopper-Pearson exact confidence interval for a binomial proportion.
    """
    if n <= 0:
        return 0.0, 0.0
    alpha = 1.0 - confidence
    lower = float(stats.beta.ppf(alpha / 2.0, k, n - k + 1)) if k > 0 else 0.0
    upper = float(stats.beta.ppf(1.0 - alpha / 2.0, k + 1, n - k)) if k < n else 1.0
    return round(max(0.0, lower), 4), round(min(1.0, upper), 4)


def compute_group_bootstrap_ci(
    groups: np.ndarray,
    y_true: np.ndarray,
    y_score_or_pred: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_bootstraps: int = 2000,
    confidence: float = 0.95,
    seed: int = 42,
) -> Tuple[float, Tuple[float, float]]:
    """
    Compute group bootstrap confidence interval resampling whole groups with replacement.
    Required for clustered / family-correlated items (>= 2000 iterations).
    """
    groups = np.asarray(groups)
    y_true = np.asarray(y_true)
    y_score_or_pred = np.asarray(y_score_or_pred)

    point_estimate = float(metric_fn(y_true, y_score_or_pred))

    unique_groups = np.unique(groups)
    n_groups = len(unique_groups)
    if n_groups < 2:
        return point_estimate, (point_estimate, point_estimate)

    group_to_indices = {g: np.where(groups == g)[0] for g in unique_groups}

    rng = np.random.default_rng(seed)
    boot_estimates = []

    for _ in range(n_bootstraps):
        sampled_groups = rng.choice(unique_groups, size=n_groups, replace=True)
        sampled_indices = np.concatenate([group_to_indices[g] for g in sampled_groups])

        sample_true = y_true[sampled_indices]
        sample_score = y_score_or_pred[sampled_indices]

        # Ensure both classes present if metric requires classification
        if len(np.unique(y_true)) >= 2 and len(np.unique(sample_true)) < 2:
            continue

        try:
            val = float(metric_fn(sample_true, sample_score))
            if np.isfinite(val):
                boot_estimates.append(val)
        except Exception:
            continue

    if len(boot_estimates) < 100:
        return point_estimate, (point_estimate, point_estimate)

    alpha = 1.0 - confidence
    lower = float(np.percentile(boot_estimates, (alpha / 2.0) * 100.0))
    upper = float(np.percentile(boot_estimates, (1.0 - alpha / 2.0) * 100.0))

    return round(point_estimate, 4), (round(lower, 4), round(upper, 4))


def compute_mcnemar_test(
    y_true: np.ndarray,
    y_pred_ours: np.ndarray,
    y_pred_base: np.ndarray,
) -> Dict[str, Any]:
    """
    Compute McNemar's test for paired binary classification comparisons.
    Contingency table:
      b: Ours correct, Baseline incorrect
      c: Ours incorrect, Baseline correct
    """
    y_true = np.asarray(y_true)
    y_pred_ours = np.asarray(y_pred_ours)
    y_pred_base = np.asarray(y_pred_base)

    ours_correct = (y_pred_ours == y_true)
    base_correct = (y_pred_base == y_true)

    b = int(np.sum(ours_correct & (~base_correct)))
    c = int(np.sum((~ours_correct) & base_correct))

    if (b + c) == 0:
        return {
            "statistic": 0.0,
            "p_value": 1.0,
            "b_ours_only_correct": 0,
            "c_base_only_correct": 0,
            "significant_05": False,
        }

    # Edwards continuity correction
    stat = ((abs(b - c) - 1.0) ** 2) / float(b + c) if (b + c) >= 25 else float(abs(b - c))
    if (b + c) < 25:
        # Exact two-sided binomial test under p=0.5
        p_val = 2.0 * min(
            stats.binom.cdf(min(b, c), b + c, 0.5),
            1.0 - stats.binom.cdf(max(b, c) - 1, b + c, 0.5),
        )
        p_val = min(1.0, max(0.0, float(p_val)))
    else:
        p_val = float(stats.chi2.sf(stat, df=1))

    return {
        "statistic": round(stat, 4),
        "p_value": round(p_val, 6),
        "b_ours_only_correct": b,
        "c_base_only_correct": c,
        "significant_05": p_val < 0.05,
    }


def compute_paired_bootstrap(
    vals_ours: np.ndarray,
    vals_base: np.ndarray,
    n_bootstraps: int = 2000,
    confidence: float = 0.95,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Paired bootstrap test for continuous metrics (MAE, IoU, etc.).
    Evaluates delta = ours - base with 95% CI and two-sided p-value.
    """
    vals_ours = np.asarray(vals_ours)
    vals_base = np.asarray(vals_base)
    deltas = vals_ours - vals_base
    n = len(deltas)
    if n == 0:
        return {"mean_delta": 0.0, "ci": (0.0, 0.0), "p_value": 1.0}

    mean_delta = float(np.mean(deltas))
    rng = np.random.default_rng(seed)

    boot_means = []
    for _ in range(n_bootstraps):
        sample = rng.choice(deltas, size=n, replace=True)
        boot_means.append(float(np.mean(sample)))

    alpha = 1.0 - confidence
    lower = float(np.percentile(boot_means, (alpha / 2.0) * 100.0))
    upper = float(np.percentile(boot_means, (1.0 - alpha / 2.0) * 100.0))

    # Empirical two-sided p-value against null hypothesis mean == 0
    p_val = 2.0 * min(
        np.mean([m <= 0.0 for m in boot_means]),
        np.mean([m >= 0.0 for m in boot_means]),
    )
    p_val = min(1.0, max(0.0, float(p_val)))

    return {
        "mean_delta": round(mean_delta, 4),
        "ci": (round(lower, 4), round(upper, 4)),
        "p_value": round(p_val, 6),
        "significant_05": p_val < 0.05,
    }


def compute_roc_pr_and_eer(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> Dict[str, float]:
    """
    Compute ROC AUC, PR AUC, Equal Error Rate (EER), and TPR at 1% and 5% FPR.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)

    # Equal Error Rate (EER): point where FPR == FNR = 1 - TPR
    fnr = 1.0 - tpr
    eer_idx = int(np.argmin(np.abs(fpr - fnr)))
    eer = float((fpr[eer_idx] + fnr[eer_idx]) / 2.0)

    # TPR at 1% and 5% FPR
    tpr_at_1pct_fpr = float(np.max(tpr[fpr <= 0.01])) if np.any(fpr <= 0.01) else 0.0
    tpr_at_5pct_fpr = float(np.max(tpr[fpr <= 0.05])) if np.any(fpr <= 0.05) else 0.0

    return {
        "roc_auc": round(float(roc_auc), 4),
        "pr_auc": round(float(pr_auc), 4),
        "eer": round(eer, 4),
        "tpr_at_1pct_fpr": round(tpr_at_1pct_fpr, 4),
        "tpr_at_5pct_fpr": round(tpr_at_5pct_fpr, 4),
    }


def compute_ece(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Compute Expected Calibration Error (ECE) with equal-width probability bins.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    bin_limits = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    if n == 0:
        return 0.0

    for i in range(n_bins):
        low, high = bin_limits[i], bin_limits[i + 1]
        mask = (y_prob >= low) & (y_prob <= high if i == n_bins - 1 else y_prob < high)
        bin_count = int(np.sum(mask))
        if bin_count > 0:
            bin_acc = float(np.mean(y_true[mask]))
            bin_conf = float(np.mean(y_prob[mask]))
            ece += (bin_count / float(n)) * abs(bin_acc - bin_conf)

    return round(float(ece), 4)


def compute_mirex_key_score(true_key: str, est_key: str) -> float:
    """
    Compute standard MIREX Key Score:
    1.0: Exact match (same tonic and mode)
    0.5: Dominant / fifth (7 semitones shift, same mode)
    0.3: Relative major / minor (3 semitones shift, opposite mode)
    0.2: Parallel major / minor (same tonic, opposite mode)
    0.0: Other
    """
    if not true_key or not est_key or "Unknown" in (true_key, est_key):
        return 0.0

    pitch_classes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

    def parse_key(k: str) -> Tuple[Optional[int], Optional[str]]:
        parts = k.strip().split()
        if len(parts) < 2:
            return None, None
        tonic, mode = parts[0], parts[1].capitalize()
        if tonic in pitch_classes:
            return pitch_classes.index(tonic), mode
        return None, None

    t_idx, t_mode = parse_key(true_key)
    e_idx, e_mode = parse_key(est_key)

    if t_idx is None or e_idx is None:
        return 0.0

    if t_idx == e_idx and t_mode == e_mode:
        return 1.0

    shift = (e_idx - t_idx) % 12

    # Dominant / Fifth (same mode, 7 semitones)
    if shift == 7 and t_mode == e_mode:
        return 0.5

    # Relative:
    # Major true -> Minor est with shift = 9 (i.e. C Major -> A Minor)
    if t_mode == "Major" and e_mode == "Minor" and shift == 9:
        return 0.3
    # Minor true -> Major est with shift = 3 (i.e. A Minor -> C Major)
    if t_mode == "Minor" and e_mode == "Major" and shift == 3:
        return 0.3

    # Parallel: same tonic, different mode
    if t_idx == e_idx and t_mode != e_mode:
        return 0.2

    return 0.0


def compute_tempo_accuracies(
    true_bpm: float,
    est_bpm: Optional[float],
    tolerance: float = 0.08,
) -> Tuple[bool, bool, float]:
    """
    Compute MIREX Tempo Acc1, Acc2, and BPM Absolute Error.
    Acc1: Within tolerance (default 8%) of true BPM.
    Acc2: Within tolerance of true BPM or harmonic multiples (0.333, 0.5, 2.0, 3.0).
    BPM MAE: |est_bpm - true_bpm|.
    """
    if est_bpm is None or not np.isfinite(est_bpm) or est_bpm <= 0:
        return False, False, float(true_bpm)

    error_raw = abs(est_bpm - true_bpm)
    ratio = est_bpm / true_bpm

    acc1 = abs(1.0 - ratio) <= tolerance

    harmonic_multipliers = [1.0 / 3.0, 0.5, 1.0, 2.0, 3.0]
    acc2 = any(abs(1.0 - (ratio / m)) <= tolerance for m in harmonic_multipliers)

    return bool(acc1), bool(acc2), round(float(error_raw), 2)


def compute_segment_iou_and_time_error(
    true_interval: Tuple[float, float],
    pred_interval: Optional[Tuple[float, float]],
) -> Tuple[float, float]:
    """
    Compute Intersection over Union (IoU) and mean timestamp error for a temporal segment.
    """
    t_start, t_end = true_interval
    if pred_interval is None:
        return 0.0, float(t_end - t_start)

    p_start, p_end = pred_interval

    overlap_start = max(t_start, p_start)
    overlap_end = min(t_end, p_end)
    overlap = max(0.0, overlap_end - overlap_start)

    union_start = min(t_start, p_start)
    union_end = max(t_end, p_end)
    union = max(1e-6, union_end - union_start)

    iou = overlap / union
    time_err = (abs(p_start - t_start) + abs(p_end - t_end)) / 2.0

    return round(float(iou), 4), round(float(time_err), 4)
