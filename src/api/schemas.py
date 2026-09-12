"""
Pydantic data schemas for API request and response validation.
"""

from typing import Dict, List, Optional, Tuple
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Health & Status Schemas
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str = Field(..., examples=["ok"])
    version: str = Field(..., examples=["0.1.0"])
    engine: str = Field(..., examples=["Similarity & Research Pipeline"])


# ---------------------------------------------------------------------------
# Feature 1: Similarity & DTW Schemas
# ---------------------------------------------------------------------------

class MatrixPairRequest(BaseModel):
    """Payload for comparing two already extracted (12, T) chroma matrices."""
    chroma_a: List[List[float]] = Field(..., description="Chroma matrix for track A of shape (12, T_a)")
    chroma_b: List[List[float]] = Field(..., description="Chroma matrix for track B of shape (12, T_b)")


class PairwiseSimilarityResponse(BaseModel):
    similarity: float = Field(..., ge=0.0, le=1.0, description="Normalized similarity score in [0, 1]")
    normalized_distance: float = Field(..., ge=0.0, description="DTW min-distance divided by path length")
    raw_distance: float = Field(..., ge=0.0, description="Raw DTW accumulated distance")
    best_shift: int = Field(..., ge=0, le=11, description="Optimal semitone shift (0..11) for key invariance")


class DetailedAlignmentResponse(PairwiseSimilarityResponse):
    total_steps: int = Field(..., description="Total points in alignment warping path")
    alignment_path: List[Tuple[int, int]] = Field(..., description="List of (frame_a, frame_b) index pairs")
    local_costs: List[float] = Field(..., description="Euclidean step distances along alignment path")


# ---------------------------------------------------------------------------
# Feature 7: Suggestor Box Schemas
# ---------------------------------------------------------------------------

class FlaggedSegment(BaseModel):
    start_sec: float = Field(..., description="Start timestamp in track A (seconds)")
    end_sec: float = Field(..., description="End timestamp in track A (seconds)")
    matched_start_sec: float = Field(..., description="Matched start timestamp in track B (seconds)")
    matched_end_sec: float = Field(..., description="Matched end timestamp in track B (seconds)")
    mean_similarity: float = Field(..., description="Average similarity in this localized segment")


class SuggestorResponse(BaseModel):
    similarity: float = Field(..., ge=0.0, le=1.0)
    p_value: Optional[float] = Field(None, description="Significance p-value against null distribution")
    severity_tier: str = Field(..., description="'none' (p > 0.10), 'mild' (0.01 < p <= 0.10), or 'high' (p <= 0.01)")
    suggestions: List[str] = Field(..., description="Actionable recommendations to differentiate flagged sections")
    flagged_segments: List[FlaggedSegment] = Field(default_factory=list, description="Timestamped high-similarity segments")


# ---------------------------------------------------------------------------
# Feature 2 & 3: Statistical Significance & Population Schemas
# ---------------------------------------------------------------------------

class NullTestRequest(BaseModel):
    query_score: float = Field(..., ge=0.0, le=1.0, description="Observed query similarity score")
    null_scores: Optional[List[float]] = Field(
        None,
        description="Optional empirical null distribution. If omitted, default reference pool distribution is used."
    )


class NullTestResponse(BaseModel):
    query_score: float
    p_value: float = Field(..., ge=0.0, le=1.0)
    is_significant: bool = Field(..., description="True if p_value <= 0.05")
    severity_tier: str = Field(..., description="Severity category based on empirical p-value")
    null_distribution_size: int


class PopulationComparisonRequest(BaseModel):
    human_scores: List[float] = Field(..., description="Pairwise similarity scores within human reference set")
    ai_scores: List[float] = Field(..., description="Pairwise similarity scores between AI tracks and human reference set")


class PopulationComparisonResponse(BaseModel):
    u_statistic: float
    p_value: float
    human_median: float
    ai_median: float
    interpretation: str


# ---------------------------------------------------------------------------
# Feature 6: Mimic / Candidate Ranking Schemas
# ---------------------------------------------------------------------------

class CandidateScore(BaseModel):
    candidate_id: str
    similarity: float
    normalized_distance: float
    best_shift: int
    p_value: Optional[float] = None
    is_top_match: bool = False
    ai_generated: Optional[bool] = None
    ai_confidence: Optional[float] = None
    genre: Optional[str] = None
    key: Optional[str] = None
    tempo_bpm: Optional[float] = None
    segments: Optional[List[Dict]] = None
    duration: Optional[float] = None


class CandidateRankResponse(BaseModel):
    query_id: str
    candidates: List[CandidateScore]
    top_match: Optional[CandidateScore] = None


# ---------------------------------------------------------------------------
# Feature 4 & 5: AI Classifier & Metadata Schemas
# ---------------------------------------------------------------------------

class OriginClassifyResponse(BaseModel):
    prediction: str = Field(..., examples=["ai"])
    confidence: float = Field(..., ge=0.0, le=1.0)
    spectral_flatness_mean: float
    notes: str


class MetadataExtractResponse(BaseModel):
    tempo_bpm: float = Field(..., examples=[120.0])
    estimated_key: str = Field(..., examples=["C Major"])
    duration_sec: float = Field(..., examples=[180.5])
    spectral_centroid_hz: float = Field(..., examples=[2400.0])
