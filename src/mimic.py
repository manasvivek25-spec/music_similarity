"""
Feature 6: Mimic / Fraud Identification Module.

Ranks candidates by ascending empirical p-value (statistical anomalousness),
applies same-artist warning suppression ('Consistent Artist Style'), preserves
statistical evidence independently of warnings, computes alignment coverage
from retained DTW paths and local costs, and surfaces independent origin flags.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
from src.schemas import MetadataContext, MimicCandidate, MimicResponse


def normalize_artist_string(artist: Optional[str]) -> Optional[str]:
    """Normalize artist name for reliable string comparison."""
    if not artist:
        return None
    cleaned = artist.strip().lower()
    return cleaned if cleaned else None


def evaluate_artist_relationship(
    query_artist: Optional[str],
    candidate_artist: Optional[str]
) -> str:
    """
    Determine artist relationship between query track and candidate track.

    Returns:
    - 'Consistent Artist Style': both known and match.
    - 'Different Artist': both known and differ.
    - 'Unknown Artist': one or both unknown.
    """
    norm_q = normalize_artist_string(query_artist)
    norm_c = normalize_artist_string(candidate_artist)

    if norm_q and norm_c:
        if norm_q == norm_c:
            return "Consistent Artist Style"
        else:
            return "Different Artist"
    else:
        return "Unknown Artist"


def get_coverage_interpretation(coverage: Optional[float]) -> Optional[str]:
    """
    Classify alignment coverage into predefined tiers using explicit boundary comparisons:
    - 0% - 20%: Isolated passage
    - >20% - 50%: Partial structural match
    - >50% - 100%: Broad structural overlap

    Coverage is an analytical localization metric, not evidence of intentional copying.
    """
    if coverage is None or not np.isfinite(coverage):
        return "Unavailable"
    cov = float(coverage)
    if cov <= 0.20:
        return "Isolated passage"
    elif cov <= 0.50:
        return "Partial structural match"
    else:
        return "Broad structural overlap"


def calculate_alignment_coverage(
    alignment_path: Optional[Sequence[Tuple[int, int]]],
    local_costs: Optional[Sequence[float]],
    total_query_frames: Optional[int] = None,
    cost_threshold_ratio: float = 0.70,
) -> Tuple[Optional[float], Optional[float], Optional[str]]:
    """
    Compute Feature 6 alignment coverage ratio, percentage, and interpretation.

    Reuses Feature 1's retained DTW path and local alignment costs without rerunning DTW.
    Formula:
        alignment_coverage = matched_query_frames / total_query_frames

    Important Rules:
        - Global DTW aligns sequences even when local costs are poor; do NOT count every step.
        - matched_query_frames is the count of unique query frames covered by qualifying
          strong-local-similarity regions (cost <= mean_cost * cost_threshold_ratio).
        - Exclude repeated path steps from inflating coverage: each query frame is added to
          a set once, even if the DTW path maps it to multiple reference frames.
        - If necessary alignment data is missing, returns (None, None, None) rather than fabricating.
        - Every returned coverage value is finite and lies in [0.0, 1.0].

    Parameters
    ----------
    alignment_path : Optional[Sequence[Tuple[int, int]]]
        List of (query_frame, candidate_frame) index pairs along the optimal DTW path.
    local_costs : Optional[Sequence[float]]
        Local step costs corresponding to each step in alignment_path.
    total_query_frames : Optional[int], optional
        Total frames in the query track. If omitted, deduced from the maximum unique query frames in path.
    cost_threshold_ratio : float, optional
        Fraction of mean step cost below which a step qualifies as a strong match (default 0.70).

    Returns
    -------
    Tuple[Optional[float], Optional[float], Optional[str]]
        (alignment_coverage [0, 1], coverage_percent [0, 100], coverage_interpretation)
    """
    if alignment_path is None or local_costs is None:
        return (None, None, None)

    if len(alignment_path) == 0 or len(local_costs) == 0:
        return (None, None, None)

    if len(alignment_path) != len(local_costs):
        return (None, None, None)

    costs_arr = np.asarray(local_costs, dtype=float)
    finite_mask = np.isfinite(costs_arr)
    if not np.any(finite_mask):
        return (None, None, None)

    mean_cost = float(np.mean(costs_arr[finite_mask]))
    cutoff = mean_cost * cost_threshold_ratio

    matched_query_frames: set = set()
    all_query_frames: set = set()

    for (fa, fb), cost in zip(alignment_path, costs_arr):
        all_query_frames.add(fa)
        if np.isfinite(cost) and cost <= cutoff:
            matched_query_frames.add(fa)

    if total_query_frames is not None:
        if total_query_frames <= 0:
            return (None, None, None)
        tot_q = total_query_frames
    else:
        tot_q = len(all_query_frames)

    if tot_q <= 0:
        return (None, None, None)

    coverage = float(np.clip(len(matched_query_frames) / tot_q, 0.0, 1.0))
    coverage_pct = round(coverage * 100.0, 1)
    interpretation = get_coverage_interpretation(coverage)

    return (coverage, coverage_pct, interpretation)


def build_mimic_analysis(
    query_context: MetadataContext,
    candidate_records: List[Dict[str, Any]],
    max_mimic_candidates: int = 3
) -> MimicResponse:
    """
    Construct Feature 6 Mimic Identification results.

    Parameters
    ----------
    query_context : MetadataContext
        Metadata and musical context of query track.
    candidate_records : List[Dict[str, Any]]
        List of candidate evaluation dictionaries containing:
        candidate_id, title, artist, genre, similarity, similarity_tier,
        p_value, confidence, severity_label, best_shift, origin_prediction,
        and optionally alignment_coverage, alignment_path, local_costs.
    max_mimic_candidates : int, optional
        Maximum top candidates to return (default 3).

    Returns
    -------
    MimicResponse
        Ranked mimic candidates with artist relationship checks, alignment coverage,
        and warning flags.
    """
    evaluated_count = len(candidate_records)

    # Primary candidate sorting remains p-value based:
    # 1. Valid finite p-values first, sorted ascending (smallest p-value = most anomalous).
    # 2. Unavailable p-values (None) sorted after valid p-values.
    # 3. Tie-breaker: descending similarity score.
    def sort_key(rec: Dict[str, Any]) -> tuple:
        pval = rec.get("p_value")
        sim = float(rec.get("similarity", 0.0))
        if pval is not None and np.isfinite(pval):
            return (0, float(pval), -sim)
        else:
            return (1, 1.0, -sim)

    sorted_records = sorted(candidate_records, key=sort_key)

    candidates_out: List[MimicCandidate] = []
    survivors_count = 0

    for rec in sorted_records:
        cand_artist = rec.get("artist")
        relationship = evaluate_artist_relationship(query_context.artist, cand_artist)

        pval = rec.get("p_value")
        severity = rec.get("severity_label", "Indeterminate")

        # Determine if candidate is a statistical survivor (valid p <= 0.10)
        is_survivor = pval is not None and pval <= 0.10
        if is_survivor:
            survivors_count += 1

        # Authoritative Rule: Do not produce statistically qualified mimic warnings
        # from candidates that have no valid statistical-gateway result.
        if pval is None:
            mimic_warning = False
        elif relationship == "Consistent Artist Style":
            # Same known artist detected: suppress warning, label Consistent Artist Style
            mimic_warning = False
        else:
            mimic_warning = is_survivor

        norm_dist = rec.get("normalized_distance")
        sim_val = float(rec.get("similarity", 0.0))
        if norm_dist is None and sim_val > 0.0:
            norm_dist = (1.0 / sim_val) - 1.0
        if norm_dist is not None:
            norm_dist = round(float(norm_dist), 4)

        # Alignment coverage metric
        cov = rec.get("alignment_coverage")
        cov_pct = rec.get("coverage_percent")
        cov_interp = rec.get("coverage_interpretation")

        if cov is None and ("alignment_path" in rec and "local_costs" in rec):
            cov, cov_pct, cov_interp = calculate_alignment_coverage(
                rec.get("alignment_path"),
                rec.get("local_costs"),
                total_query_frames=rec.get("total_query_frames")
            )

        cand_obj = MimicCandidate(
            candidate_id=str(rec.get("candidate_id", "")),
            title=rec.get("title"),
            artist=cand_artist,
            genre=rec.get("genre"),
            similarity=sim_val,
            normalized_distance=norm_dist,
            similarity_tier=str(rec.get("similarity_tier", "Low Similarity")),
            p_value=pval,
            adjusted_p_value=rec.get("adjusted_p_value"),
            confidence=str(rec.get("confidence", "Low")),
            severity_label=severity,
            best_shift=int(rec.get("best_shift", 0)),
            artist_relationship=relationship,
            mimic_warning_active=mimic_warning,
            origin_prediction=rec.get("origin_prediction"),
            alignment_coverage=cov,
            coverage_percent=cov_pct,
            coverage_interpretation=cov_interp,
            duration_sec=rec.get("duration_sec"),
            duration_disparity_warning=bool(rec.get("duration_disparity_warning", False)),
            disparity_explanation=rec.get("disparity_explanation"),
        )
        candidates_out.append(cand_obj)

    return MimicResponse(
        query_track=query_context,
        evaluated_count=evaluated_count,
        survivors_count=survivors_count,
        top_mimic_candidates=candidates_out[:max_mimic_candidates],
    )
