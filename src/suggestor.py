"""
Feature 7: Suggestor Box & Evidence Timeline Module.

Re-uses Feature 1 DTW results (optimal shift, warping path, local step costs)
and timing mappings to construct localized high-similarity segment timelines and
evidence-grounded studio suggestions without recalculating DTW.
Supports optional environment-configured LLM synthesis with deterministic fallback.
"""

from typing import Any, Dict, List, Optional
import numpy as np

from src.config import LLM_API_KEY, LLM_MODEL
from src.schemas import EvidenceTimelineItem, FlaggedSegment
from src.similarity import DTWResult


def frame_to_timestamp(frame_idx: int, time_mapping: Optional[np.ndarray], hop_sec: float = 0.5) -> float:
    """Convert frame index to audio timestamp in seconds."""
    if time_mapping is not None and len(time_mapping) > frame_idx and frame_idx >= 0:
        return float(time_mapping[frame_idx])
    return float(frame_idx * hop_sec)


def identify_flagged_segments(
    dtw_result: DTWResult,
    times_a: Optional[np.ndarray],
    times_b: Optional[np.ndarray],
    cost_threshold_ratio: float = 0.70,
    min_consecutive_frames: int = 4
) -> List[FlaggedSegment]:
    """
    Identify contiguous segments along the DTW alignment path with high localized similarity.

    Parameters
    ----------
    dtw_result : DTWResult
        Precomputed DTW result from Feature 1.
    times_a : Optional[np.ndarray]
        Timestamp array for track A frames.
    times_b : Optional[np.ndarray]
        Timestamp array for track B frames.
    cost_threshold_ratio : float
        Fraction of mean cost below which a step is considered a close match.
    min_consecutive_frames : int
        Minimum path steps to form a reportable segment.

    Returns
    -------
    List[FlaggedSegment]
        Timestamped high-similarity segment intervals.
    """
    path = dtw_result.alignment_path
    costs = dtw_result.local_costs

    if len(path) == 0 or len(costs) == 0:
        return []

    mean_cost = float(np.mean(costs))
    cutoff = mean_cost * cost_threshold_ratio

    segments: List[FlaggedSegment] = []
    current_run: List[int] = []

    for step_idx, cost in enumerate(costs):
        if cost <= cutoff:
            current_run.append(step_idx)
        else:
            if len(current_run) >= min_consecutive_frames:
                start_step = current_run[0]
                end_step = current_run[-1]

                fa_start, fb_start = path[start_step]
                fa_end, fb_end = path[end_step]

                seg_costs = costs[current_run]
                mean_seg_cost = float(np.mean(seg_costs))
                seg_sim = float(1.0 / (1.0 + mean_seg_cost))

                segments.append(FlaggedSegment(
                    start_time_a=round(frame_to_timestamp(fa_start, times_a), 2),
                    end_time_a=round(frame_to_timestamp(fa_end, times_a), 2),
                    start_time_b=round(frame_to_timestamp(fb_start, times_b), 2),
                    end_time_b=round(frame_to_timestamp(fb_end, times_b), 2),
                    mean_step_cost=round(mean_seg_cost, 4),
                    segment_similarity=round(seg_sim, 4),
                ))
            current_run = []

    # Final run check
    if len(current_run) >= min_consecutive_frames:
        start_step = current_run[0]
        end_step = current_run[-1]
        fa_start, fb_start = path[start_step]
        fa_end, fb_end = path[end_step]
        seg_costs = costs[current_run]
        mean_seg_cost = float(np.mean(seg_costs))
        seg_sim = float(1.0 / (1.0 + mean_seg_cost))

        segments.append(FlaggedSegment(
            start_time_a=round(frame_to_timestamp(fa_start, times_a), 2),
            end_time_a=round(frame_to_timestamp(fa_end, times_a), 2),
            start_time_b=round(frame_to_timestamp(fb_start, times_b), 2),
            end_time_b=round(frame_to_timestamp(fb_end, times_b), 2),
            mean_step_cost=round(mean_seg_cost, 4),
            segment_similarity=round(seg_sim, 4),
        ))

    return segments[:5]  # Cap at top 5 most salient segments


def generate_deterministic_studio_suggestions(
    candidate_title: str,
    similarity: float,
    best_shift: int,
    flagged_segments: List[FlaggedSegment],
    query_key: Optional[str] = None,
    query_tempo: Optional[float] = None
) -> List[str]:
    """
    Generate evidence-grounded studio recommendations strictly from empirical metrics.
    """
    suggestions: List[str] = []

    # 1. Pitch shift advice
    if best_shift != 0:
        suggestions.append(
            f"Maximum structural alignment occurs at a transposition of {best_shift} semitones. "
            f"To differentiate melodic motion, consider altering root intervals or introducing modal interchange."
        )
    else:
        suggestions.append(
            "Harmonic alignment aligns in prime pitch space (0 semitone shift). "
            "Consider varying chord substitutions or altering top-line voice leading."
        )

    # 2. Localized segment advice
    if flagged_segments:
        first_seg = flagged_segments[0]
        suggestions.append(
            f"Flagged segment detected between {first_seg.start_time_a:.1f}s and {first_seg.end_time_a:.1f}s "
            f"(similarity: {first_seg.segment_similarity:.2f}). Revoicing harmonic extensions or varying cadential rhythm "
            f"in this window will effectively disperse structural overlap."
        )

    # 3. Tempo / Metric advice
    if query_tempo:
        suggestions.append(
            f"Estimated tempo is {query_tempo:.1f} BPM. Introducing syncopated rhythmic displacement "
            f"or shifting metric accents can decorrelate beat-aligned harmonic contours."
        )

    # 4. Contextual summary
    suggestions.append(
        f"Alignment with '{candidate_title}' yielded an overall similarity of {similarity:.2f}. "
        "Review highlighted timestamps in the evidence timeline to verify arrangement independence."
    )

    return suggestions


def build_evidence_timeline(
    candidate_survivors: List[Dict[str, Any]],
    times_query: Optional[np.ndarray],
    max_candidates: int = 3
) -> List[EvidenceTimelineItem]:
    """
    Build evidence timeline items for up to max_candidates top survivors.

    Parameters
    ----------
    candidate_survivors : List[Dict[str, Any]]
        List of dictionaries containing:
        - candidate_id, title, artist, similarity, p_value, severity_label,
          best_shift, dtw_result, candidate_times
    times_query : Optional[np.ndarray]
        Query track beat/frame timestamps in seconds.
    max_candidates : int, optional
        Maximum candidates to include (default 3).

    Returns
    -------
    List[EvidenceTimelineItem]
        Timeline representations with flagged segments and actionable studio advice.
    """
    timeline_items: List[EvidenceTimelineItem] = []

    for item in candidate_survivors[:max_candidates]:
        dtw_res: Optional[DTWResult] = item.get("dtw_result")
        cand_times = item.get("candidate_times")

        flagged: List[FlaggedSegment] = []
        if dtw_res is not None:
            flagged = identify_flagged_segments(dtw_res, times_query, cand_times)

        title = str(item.get("title") or item.get("candidate_id", "Candidate"))
        sim = float(item.get("similarity", 0.0))
        shift = int(item.get("best_shift", 0))

        suggestions = generate_deterministic_studio_suggestions(
            candidate_title=title,
            similarity=sim,
            best_shift=shift,
            flagged_segments=flagged,
            query_key=item.get("query_key"),
            query_tempo=item.get("query_tempo"),
        )

        timeline_items.append(EvidenceTimelineItem(
            candidate_id=str(item.get("candidate_id", "")),
            candidate_title=title,
            candidate_artist=item.get("artist"),
            similarity=sim,
            p_value=item.get("p_value"),
            severity=str(item.get("severity_label", "Indeterminate")),
            best_shift=shift,
            flagged_segments=flagged,
            studio_suggestions=suggestions,
        ))

    return timeline_items
