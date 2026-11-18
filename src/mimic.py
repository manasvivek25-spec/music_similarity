"""
Feature 6: Mimic Evidence & Candidate Identification Module.

Ranks candidates by ascending empirical p-value (statistical anomalousness),
applies same-artist warning suppression ('Consistent Artist Style'), preserves
statistical evidence independently of warnings, computes alignment coverage and
path-level structural mimic evidence (region count, continuity, sequential
consistency, local-similarity concentration, and Mimic Evidence level) from
retained DTW paths and local costs without re-running DTW, and surfaces
independent origin flags.
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


def calculate_path_mimic_evidence(
    alignment_path: Optional[Sequence[Tuple[int, int]]],
    local_costs: Optional[Sequence[float]],
    cost_threshold_ratio: float = 0.70,
    min_region_steps: int = 4,
) -> Dict[str, Optional[float]]:
    """
    Compute Feature 6 structural mimic evidence fields directly from Feature 1's
    reused DTW alignment path and local costs (zero DTW recalculation).

    See docs/F6_mimic_evidence.md for exact mathematical definitions:
    - significant_region_count: distinct maximal contiguous qualifying runs with length >= min_region_steps
    - region_continuity: fraction of qualifying steps belonging to coherent significant regions
    - sequential_consistency: monotonic temporal progression across (or within) significant regions
    - local_similarity_concentration: fraction of total path local similarity mass inside qualifying steps
    """
    empty: Dict[str, Optional[float]] = {
        "significant_region_count": None,
        "region_continuity": None,
        "sequential_consistency": None,
        "local_similarity_concentration": None,
    }
    if alignment_path is None or local_costs is None:
        return empty
    if len(alignment_path) == 0 or len(local_costs) == 0 or len(alignment_path) != len(local_costs):
        return empty

    costs_arr = np.asarray(local_costs, dtype=float)
    finite_mask = np.isfinite(costs_arr)
    if not np.any(finite_mask):
        return empty

    mean_cost = float(np.mean(costs_arr[finite_mask]))
    cutoff = mean_cost * cost_threshold_ratio

    qualifying_indices: List[int] = [
        idx for idx, c in enumerate(costs_arr) if np.isfinite(c) and c <= cutoff
    ]

    # Partition qualifying indices into maximal contiguous runs
    runs: List[List[int]] = []
    cur: List[int] = []
    for idx in qualifying_indices:
        if not cur or idx == cur[-1] + 1:
            cur.append(idx)
        else:
            runs.append(cur)
            cur = [idx]
    if cur:
        runs.append(cur)

    sig_regions = [r for r in runs if len(r) >= min_region_steps]
    region_count = len(sig_regions)

    # 1. Region continuity / coherence
    total_q = len(qualifying_indices)
    if total_q == 0:
        continuity = 0.0
    else:
        coherent_steps = sum(len(r) for r in sig_regions)
        continuity = float(np.clip(coherent_steps / float(total_q), 0.0, 1.0))

    # 2. Sequential consistency
    if region_count >= 2:
        # Sort regions by query start frame and measure monotonic advance in both query & candidate
        starts = [alignment_path[r[0]] for r in sig_regions]
        starts.sort(key=lambda pair: (pair[0], pair[1]))
        consistent_pairs = sum(
            1
            for j in range(len(starts) - 1)
            if starts[j + 1][0] >= starts[j][0] and starts[j + 1][1] >= starts[j][1]
        )
        seq_consistency = float(np.clip(consistent_pairs / float(len(starts) - 1), 0.0, 1.0))
    elif region_count == 1:
        r0 = sig_regions[0]
        if len(r0) <= 1:
            seq_consistency = 1.0
        else:
            advances = 0
            for k in range(len(r0) - 1):
                a0, b0 = alignment_path[r0[k]]
                a1, b1 = alignment_path[r0[k + 1]]
                if (a1 >= a0 and b1 >= b0) and (a1 > a0 or b1 > b0):
                    advances += 1
            seq_consistency = float(np.clip(advances / float(len(r0) - 1), 0.0, 1.0))
    else:
        seq_consistency = 0.0

    # 3. Local-similarity concentration
    local_sims = np.where(finite_mask, 1.0 / (1.0 + np.maximum(costs_arr, 0.0)), 0.0)
    total_sim_mass = float(np.sum(local_sims))
    if total_sim_mass <= 1e-12 or total_q == 0:
        concentration = 0.0
    else:
        q_mass = float(np.sum(local_sims[qualifying_indices]))
        concentration = float(np.clip(q_mass / total_sim_mass, 0.0, 1.0))

    return {
        "significant_region_count": int(region_count),
        "region_continuity": round(continuity, 4),
        "sequential_consistency": round(seq_consistency, 4),
        "local_similarity_concentration": round(concentration, 4),
    }


def classify_mimic_evidence_level(
    similarity: float,
    p_value: Optional[float],
    artist_relationship: str,
    alignment_coverage: Optional[float],
    significant_region_count: Optional[int],
    region_continuity: Optional[float],
    sequential_consistency: Optional[float],
    local_similarity_concentration: Optional[float],
    case_type: Optional[str] = None,
) -> str:
    """
    Determine Feature 6 Mimic Evidence level ('High', 'Moderate', 'Low', or 'Not applicable')
    combining path-derived structural evidence with empirical p-value as supporting evidence only.
    Exact thresholds documented in docs/F6_mimic_evidence.md.
    """
    if case_type in ("Case C", "Case U"):
        return "Not applicable"
    if p_value is None or not np.isfinite(p_value):
        return "Not applicable"
    if similarity < 0.45:
        return "Not applicable"
    if artist_relationship == "Consistent Artist Style":
        return "Not applicable"

    # When DTW path evidence is available, require structural evidence (never reduce to p-sorting)
    has_path_metrics = (
        significant_region_count is not None
        and region_continuity is not None
        and sequential_consistency is not None
        and local_similarity_concentration is not None
    )

    if has_path_metrics:
        if significant_region_count == 0 and p_value > 0.10:
            return "Not applicable"
        cov = alignment_coverage if alignment_coverage is not None else 0.0
        if (
            p_value <= 0.01
            and significant_region_count >= 1
            and cov >= 0.30
            and region_continuity >= 0.45
            and sequential_consistency >= 0.70
            and local_similarity_concentration >= 0.35
        ):
            return "High"
        if (
            p_value <= 0.10
            and significant_region_count >= 1
            and cov >= 0.15
            and sequential_consistency >= 0.50
        ):
            return "Moderate"
        return "Low"

    # Degraded fallback when caller omitted DTW path arrays
    if p_value > 0.10:
        return "Low"
    cov = alignment_coverage if alignment_coverage is not None else 0.0
    if p_value <= 0.01 and similarity >= 0.85 and cov >= 0.30:
        return "High"
    if p_value <= 0.10 and similarity >= 0.65:
        return "Moderate"
    return "Low"


def build_mimic_analysis(
    query_context: MetadataContext,
    candidate_records: List[Dict[str, Any]],
    max_mimic_candidates: int = 3,
    case_type: Optional[str] = None,
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

        align_path = rec.get("alignment_path")
        local_costs = rec.get("local_costs")
        if (align_path is None or local_costs is None) and rec.get("dtw_result") is not None:
            dtw_obj = rec.get("dtw_result")
            align_path = getattr(dtw_obj, "alignment_path", None)
            local_costs = getattr(dtw_obj, "local_costs", None)

        if cov is None and (align_path is not None and local_costs is not None):
            cov, cov_pct, cov_interp = calculate_alignment_coverage(
                align_path,
                local_costs,
                total_query_frames=rec.get("total_query_frames")
            )

        # Path-derived structural mimic evidence (no DTW recalculation)
        path_ev = calculate_path_mimic_evidence(align_path, local_costs)
        sig_regions = rec.get("significant_region_count", path_ev["significant_region_count"])
        continuity = rec.get("region_continuity", path_ev["region_continuity"])
        seq_cons = rec.get("sequential_consistency", path_ev["sequential_consistency"])
        local_conc = rec.get("local_similarity_concentration", path_ev["local_similarity_concentration"])

        evidence_level = classify_mimic_evidence_level(
            similarity=sim_val,
            p_value=pval,
            artist_relationship=relationship,
            alignment_coverage=cov,
            significant_region_count=sig_regions,
            region_continuity=continuity,
            sequential_consistency=seq_cons,
            local_similarity_concentration=local_conc,
            case_type=case_type,
        )

        # Compute research-only smoothed p-value (k+1)/(N+1) if exceedance_count and null_sample_size present
        exc_k = rec.get("exceedance_count")
        null_n = rec.get("null_sample_size")
        smoothed_p = rec.get("smoothed_p_value")
        if smoothed_p is None and exc_k is not None and null_n is not None and int(null_n) > 0:
            smoothed_p = round(float((int(exc_k) + 1) / (int(null_n) + 1)), 6)

        cand_obj = MimicCandidate(
            candidate_id=str(rec.get("candidate_id", "")),
            title=rec.get("title"),
            artist=cand_artist,
            genre=rec.get("genre"),
            similarity=sim_val,
            normalized_distance=norm_dist,
            similarity_tier=str(rec.get("similarity_tier", "Low Similarity")),
            p_value=pval,
            smoothed_p_value=smoothed_p,
            adjusted_p_value=rec.get("adjusted_p_value"),
            exceedance_count=exc_k,
            null_sample_size=null_n,
            confidence=str(rec.get("confidence", "Low")),
            severity_label=severity,
            best_shift=int(rec.get("best_shift", 0)),
            artist_relationship=relationship,
            mimic_warning_active=mimic_warning,
            origin_prediction=rec.get("origin_prediction"),
            alignment_coverage=cov,
            coverage_percent=cov_pct,
            coverage_interpretation=cov_interp,
            significant_region_count=sig_regions,
            region_continuity=continuity,
            sequential_consistency=seq_cons,
            local_similarity_concentration=local_conc,
            mimic_evidence_level=evidence_level,
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
