"""
Similarity calculation and Suggestor Box routes (Features 1 & 7).
"""

import io
import tempfile
from pathlib import Path
from typing import List, Optional
import numpy as np
from fastapi import APIRouter, File, HTTPException, Query, UploadFile

from src.api.schemas import (
    DetailedAlignmentResponse,
    FlaggedSegment,
    MatrixPairRequest,
    PairwiseSimilarityResponse,
    SuggestorResponse,
)
from src.extract import extract_chroma
from src.similarity import DTWResult, dtw_align_detailed, dtw_similarity
from src.stats_test import null_distribution_test

router = APIRouter(prefix="/similarity", tags=["Similarity Engine (Features 1 & 7)"])

# Standard hop length & sample rate defaults for timestamp conversion
DEFAULT_SR = 22050
DEFAULT_HOP_LENGTH = 512
FRAME_DURATION_SEC = DEFAULT_HOP_LENGTH / DEFAULT_SR

# Default empirical null reference distribution (from unrelated songs)
DEFAULT_NULL_DISTRIBUTION = [
    0.28, 0.31, 0.33, 0.35, 0.38, 0.40, 0.42, 0.44, 0.45, 0.47,
    0.49, 0.50, 0.52, 0.54, 0.56, 0.58, 0.60, 0.62, 0.65, 0.68
]


def _save_upload_to_temp(uploaded_file: UploadFile) -> Path:
    """Safely stream uploaded audio file to a temporary file on disk."""
    suffix = Path(uploaded_file.filename or "audio.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = uploaded_file.file.read()
        tmp.write(content)
        tmp_path = Path(tmp.name)
    return tmp_path


def _localize_flagged_segments(
    result: DTWResult,
    threshold_cost: float = 0.5,
    min_frames: int = 20,
) -> List[FlaggedSegment]:
    """
    Reuses DTW alignment path and local step costs to detect contiguous
    high-similarity audio segments (local cost < threshold_cost).
    """
    segments: List[FlaggedSegment] = []
    current_segment_steps: List[int] = []

    for idx, cost in enumerate(result.local_costs):
        if cost <= threshold_cost:
            current_segment_steps.append(idx)
        else:
            if len(current_segment_steps) >= min_frames:
                first_step = current_segment_steps[0]
                last_step = current_segment_steps[-1]
                frame_a_start, frame_b_start = result.alignment_path[first_step]
                frame_a_end, frame_b_end = result.alignment_path[last_step]
                avg_cost = float(np.mean([result.local_costs[k] for k in current_segment_steps]))
                seg_sim = float(1.0 / (1.0 + avg_cost))

                segments.append(
                    FlaggedSegment(
                        start_sec=round(frame_a_start * FRAME_DURATION_SEC, 2),
                        end_sec=round(frame_a_end * FRAME_DURATION_SEC, 2),
                        matched_start_sec=round(frame_b_start * FRAME_DURATION_SEC, 2),
                        matched_end_sec=round(frame_b_end * FRAME_DURATION_SEC, 2),
                        mean_similarity=round(seg_sim, 4),
                    )
                )
            current_segment_steps = []

    # Final segment flush if ending on low cost
    if len(current_segment_steps) >= min_frames:
        first_step = current_segment_steps[0]
        last_step = current_segment_steps[-1]
        frame_a_start, frame_b_start = result.alignment_path[first_step]
        frame_a_end, frame_b_end = result.alignment_path[last_step]
        avg_cost = float(np.mean([result.local_costs[k] for k in current_segment_steps]))
        seg_sim = float(1.0 / (1.0 + avg_cost))

        segments.append(
            FlaggedSegment(
                start_sec=round(frame_a_start * FRAME_DURATION_SEC, 2),
                end_sec=round(frame_a_end * FRAME_DURATION_SEC, 2),
                matched_start_sec=round(frame_b_start * FRAME_DURATION_SEC, 2),
                matched_end_sec=round(frame_b_end * FRAME_DURATION_SEC, 2),
                mean_similarity=round(seg_sim, 4),
            )
        )

    return segments


@router.post("/pairwise/matrix", response_model=PairwiseSimilarityResponse)
async def pairwise_similarity_matrix(payload: MatrixPairRequest):
    """
    Calculate key- and tempo-invariant DTW similarity between two pre-extracted chroma matrices.
    """
    try:
        chroma_a = np.array(payload.chroma_a, dtype=float)
        chroma_b = np.array(payload.chroma_b, dtype=float)
        result = dtw_align_detailed(chroma_a, chroma_b)
        return PairwiseSimilarityResponse(
            similarity=result.similarity,
            normalized_distance=result.normalized_distance,
            raw_distance=result.raw_distance,
            best_shift=result.best_shift,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/pairwise/audio", response_model=PairwiseSimilarityResponse)
async def pairwise_similarity_audio(
    file_a: UploadFile = File(..., description="Query audio file (.wav, .mp3, etc.)"),
    file_b: UploadFile = File(..., description="Target audio file (.wav, .mp3, etc.)"),
):
    """
    Extract chroma features from two uploaded audio files and compute DTW similarity.
    """
    tmp_a, tmp_b = None, None
    try:
        tmp_a = _save_upload_to_temp(file_a)
        tmp_b = _save_upload_to_temp(file_b)

        chroma_a = extract_chroma(tmp_a)
        chroma_b = extract_chroma(tmp_b)

        result = dtw_align_detailed(chroma_a, chroma_b)
        return PairwiseSimilarityResponse(
            similarity=result.similarity,
            normalized_distance=result.normalized_distance,
            raw_distance=result.raw_distance,
            best_shift=result.best_shift,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Audio processing error: {exc}")
    finally:
        if tmp_a and tmp_a.exists():
            tmp_a.unlink(missing_ok=True)
        if tmp_b and tmp_b.exists():
            tmp_b.unlink(missing_ok=True)


@router.post("/detailed/matrix", response_model=DetailedAlignmentResponse)
async def detailed_alignment_matrix(payload: MatrixPairRequest):
    """
    Return full DTW alignment details including warping path, local costs, and transposition shift.
    """
    try:
        chroma_a = np.array(payload.chroma_a, dtype=float)
        chroma_b = np.array(payload.chroma_b, dtype=float)
        res = dtw_align_detailed(chroma_a, chroma_b)

        return DetailedAlignmentResponse(
            similarity=res.similarity,
            normalized_distance=res.normalized_distance,
            raw_distance=res.raw_distance,
            best_shift=res.best_shift,
            total_steps=len(res.alignment_path),
            alignment_path=res.alignment_path,
            local_costs=res.local_costs.tolist(),
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/suggest/matrix", response_model=SuggestorResponse)
async def suggestor_box_matrix(
    payload: MatrixPairRequest,
    p_value_override: Optional[float] = Query(None, ge=0.0, le=1.0),
):
    """
    Feature 7 (Suggestor Box): Localizes similar segments and provides tiered recommendations.

    Tiered Severity:
    - p > 0.10: No warning.
    - 0.01 < p <= 0.10: Mild indicator + suggestion (alter instrumentation/timbre).
    - p <= 0.01: High warning + specific suggestions (alter progression, rewrite section).
    """
    try:
        chroma_a = np.array(payload.chroma_a, dtype=float)
        chroma_b = np.array(payload.chroma_b, dtype=float)
        result = dtw_align_detailed(chroma_a, chroma_b)

        # Compute or use p-value
        if p_value_override is not None:
            p_val = p_value_override
        else:
            p_val = null_distribution_test(result.similarity, DEFAULT_NULL_DISTRIBUTION)

        flagged_segments = _localize_flagged_segments(result)

        if p_val > 0.10:
            severity = "none"
            suggestions = [
                "No statistically significant similarity observed (p > 0.10). No composition revision required."
            ]
        elif 0.01 < p_val <= 0.10:
            severity = "mild"
            suggestions = [
                "Moderate statistical similarity (0.01 < p <= 0.10).",
                "Consider altering melodic ornamentation, chord voicings, or instrumentation in flagged segments.",
            ]
        else:
            severity = "high"
            suggestions = [
                "High statistical similarity observed (p <= 0.01). Potential mimicry or dense harmonic overlap.",
                "Strongly recommend altering melodic contours, changing key modulations, or restructuring the flagged sections.",
                f"Matched at best transposition: {result.best_shift} semitones.",
            ]

        return SuggestorResponse(
            similarity=round(result.similarity, 4),
            p_value=round(p_val, 4),
            severity_tier=severity,
            suggestions=suggestions,
            flagged_segments=flagged_segments,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
