"""
Pydantic data schemas for API request and response validation.

Defines typed interfaces for all stages of the music similarity research pipeline.
"""

from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Health & Runtime Configuration
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str = Field(..., examples=["ok"])
    version: str = Field(..., examples=["1.0.0"])
    service_available: bool = Field(True, description="API server is alive and accepting requests")
    analysis_ready: bool = Field(..., description="Sufficient audio dataset is indexed for analysis")
    total_indexed_tracks: int = Field(0, description="Total unique audio tracks indexed across local and remote sources")
    human_tracks: int = Field(0, description="Total tracks verified or labeled as Human")
    ai_tracks: int = Field(0, description="Total tracks verified or labeled as AI-generated")
    tracks_with_estimated_genre: int = Field(0, description="Total tracks with a model-estimated genre")
    null_pairs_global: Optional[int] = Field(None, description="Available precomputed global null pairs, or null if unbuilt")
    reference_track_count: int = Field(0, description="Number of reference pool audio files found")
    human_track_count: int = Field(0, description="Number of human songs found in local directory")
    ai_track_count: int = Field(0, description="Number of AI-generated songs found in local directory")
    remote_track_count: int = Field(0, description="Number of remote dataset tracks indexed")
    origin_classifier_status: str = Field(..., description="Status of independent origin classifier")
    llm_enabled: bool = Field(False, description="Whether optional LLM integration is active")


class ConfigResponse(BaseModel):
    sample_rate: int
    min_upload_duration_sec: float
    max_upload_size_bytes: int
    cosine_prefilter_threshold: float
    similarity_floor: float
    min_null_pairs: int = Field(5, description="Minimum null pairs required for empirical p-value")
    genre_model_available: bool = Field(False, description="Whether trained genre classifier model is loaded")
    genre_model_path: str = Field(..., description="Path to genre classifier model artifact")
    p_value_thresholds: Dict[str, float]
    creative_score_verdict_threshold: float
    max_survivors: int
    genre_min_sample_size: int
    narrow_creative_min_survivors: int
    llm_enabled: bool


# ---------------------------------------------------------------------------
# Stage 1: Coarse Cosine Prefilter Schemas
# ---------------------------------------------------------------------------

class CosinePrefilterStats(BaseModel):
    before_count: int = Field(..., description="Candidate count before cosine prefilter")
    after_count: int = Field(..., description="Candidate count surviving cosine prefilter")
    percentage_removed: float = Field(..., description="Percentage of candidate pool filtered out")


# ---------------------------------------------------------------------------
# Feature 1: DTW & Similarity Schemas
# ---------------------------------------------------------------------------

class MatrixPairRequest(BaseModel):
    """Payload for comparing two extracted (12, T) chroma matrices."""
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
    boundary_verified: bool = Field(True, description="Whether +-1 semitone boundary check confirmed minimum")


# ---------------------------------------------------------------------------
# Stage 3: Genre Influence Map Schemas
# ---------------------------------------------------------------------------

class GenreContribution(BaseModel):
    genre: str = Field(..., description="Genre name or 'Unknown'")
    count: int = Field(..., description="Number of candidates in this genre")
    percentage: float = Field(..., description="Percentage of candidate pool belonging to this genre")


class GenreMapResult(BaseModel):
    distribution: List[GenreContribution] = Field(..., description="Breakdown of candidate pool by genre")
    is_cross_genre: bool = Field(..., description="Whether candidates span genres outside query's native genre")
    native_genre: Optional[str] = Field(None, description="Query track's declared or detected native genre")
    matched_genres: List[str] = Field(default_factory=list, description="Unique genres present among candidates")


# ---------------------------------------------------------------------------
# Feature 2: Statistical Gateway Schemas
# ---------------------------------------------------------------------------

class StatisticalGatewayResult(BaseModel):
    p_value: Optional[float] = Field(None, description="Empirical p-value against null distribution")
    adjusted_p_value: Optional[float] = Field(None, description="Benjamini-Hochberg FDR adjusted p-value (q-value)")
    null_sample_count: int = Field(..., description="Number of pairwise comparisons in null distribution")
    exceedance_count: Optional[int] = Field(None, description="Count of null distribution scores >= query score (k)")
    null_sample_size: Optional[int] = Field(None, description="Total valid, finite null comparisons evaluated (N)")
    confidence: str = Field(..., description="'High' (>=1000), 'Moderate' (300-999), 'Low' (5-299), or 'insufficient' (<5)")
    severity_label: str = Field(..., description="'Valid Coincidence', 'Borderline Indicator', 'Critical Anomaly', or 'Indeterminate'")
    fallback_source: str = Field(..., description="'genre_specific', 'global_null', or 'none'")
    min_required: int = Field(5, description="Minimum null pairs required for empirical p-value")
    explanation: Optional[str] = Field(None, description="Explanation when null population is insufficient or finite sample note")


class NullTestRequest(BaseModel):
    query_score: float = Field(..., ge=0.0, le=1.0, description="Observed query similarity score")
    null_scores: Optional[List[float]] = Field(
        None,
        description="Optional empirical null distribution. If omitted, default reference pool distribution is used."
    )


class NullTestResponse(BaseModel):
    query_score: float
    p_value: Optional[float] = Field(None, ge=0.0, le=1.0)
    is_significant: bool = Field(False, description="True if p_value <= 0.05 and available")
    severity_tier: str = Field(..., description="Severity category based on empirical p-value")
    null_distribution_size: int
    exceedance_count: Optional[int] = Field(None, description="Count of null scores >= query score (k)")
    null_sample_size: Optional[int] = Field(None, description="Total valid null scores evaluated (N)")
    confidence: str = Field("Low", description="Confidence tier")
    min_required: int = Field(5, description="Minimum null pairs required")
    explanation: Optional[str] = Field(None, description="Explanation when null population is insufficient")


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
# Feature 3: Creative Score Schemas
# ---------------------------------------------------------------------------

class CreativeScoreResult(BaseModel):
    broad_score_percent: Optional[float] = Field(None, description="Relative novelty against full genre pool (0-100)")
    broad_score_normalized: Optional[float] = Field(None, description="Normalized broad score in [0, 1]")
    narrow_score_percent: Optional[float] = Field(None, description="Relative novelty against survivors (0-100)")
    narrow_score_normalized: Optional[float] = Field(None, description="Normalized narrow score in [0, 1]")
    score_tier: str = Field(..., description="'High Creative Novelty', 'Standard Compositional Variance', etc.")
    is_available: bool = Field(..., description="Whether sufficient reference population existed to calculate")
    evaluated_genres: List[str] = Field(default_factory=list, description="Target genres evaluated")
    broad_score_ci: Optional[Tuple[float, float]] = Field(None, description="Bootstrap 95% confidence interval (lower, upper)")


# ---------------------------------------------------------------------------
# Feature 4: Origin Classification Schemas
# ---------------------------------------------------------------------------

class OriginClassifierResult(BaseModel):
    predicted_origin: str = Field(..., description="'Human', 'AI', or 'Unconfigured'")
    confidence: Optional[float] = Field(None, description="Confidence in predicted origin [0, 1]")
    probability_ai: Optional[float] = Field(None, description="Model probability that track is AI-generated")
    model_status: str = Field(..., description="'Trained & Active' or 'Unconfigured / Missing Labeled Data'")
    model_version: Optional[str] = Field(None, description="Model version string if active")


class OriginClassifyResponse(BaseModel):
    prediction: str = Field(..., examples=["ai"])
    confidence: float = Field(..., ge=0.0, le=1.0)
    spectral_flatness_mean: float
    notes: str


# ---------------------------------------------------------------------------
# Feature 5: Metadata Context Schemas
# ---------------------------------------------------------------------------

class MetadataContext(BaseModel):
    track_id: str
    file_path: Optional[str] = None
    title: Optional[str] = None
    artist: Optional[str] = None
    genre: Optional[str] = None
    provided_key: Optional[str] = None
    estimated_key: Optional[str] = None
    provided_tempo: Optional[float] = None
    estimated_tempo: Optional[float] = None
    duration_sec: float = Field(..., description="Duration of audio file in seconds")


class MetadataExtractResponse(BaseModel):
    tempo_bpm: float = Field(..., examples=[120.0])
    estimated_key: str = Field(..., examples=["C Major"])
    duration_sec: float = Field(..., examples=[180.5])
    spectral_centroid_hz: float = Field(..., examples=[2400.0])


# ---------------------------------------------------------------------------
# Feature 6: Mimic Identification Schemas
# ---------------------------------------------------------------------------

class MimicCandidate(BaseModel):
    candidate_id: str
    title: Optional[str] = None
    artist: Optional[str] = None
    genre: Optional[str] = None
    similarity: float = Field(..., ge=0.0, le=1.0)
    normalized_distance: Optional[float] = Field(None, ge=0.0, description="Normalized DTW distance")
    similarity_tier: str
    p_value: Optional[float] = None
    adjusted_p_value: Optional[float] = Field(None, description="Benjamini-Hochberg FDR adjusted p-value")
    exceedance_count: Optional[int] = Field(None, description="Count of null distribution scores >= query score (k)")
    null_sample_size: Optional[int] = Field(None, description="Total valid null scores evaluated (N)")
    confidence: str
    severity_label: str
    best_shift: int
    artist_relationship: str = Field(..., description="'Consistent Artist Style', 'Different Artist', or 'Unknown Artist'")
    mimic_warning_active: bool = Field(..., description="True if potential mimic; suppressed if consistent artist style")
    origin_prediction: Optional[str] = None
    alignment_coverage: Optional[float] = Field(None, ge=0.0, le=1.0, description="Unique query frames aligned at strong similarity / total query frames")
    coverage_percent: Optional[float] = Field(None, ge=0.0, le=100.0, description="Alignment coverage percentage")
    coverage_interpretation: Optional[str] = Field(None, description="'Isolated passage' (0-20%), 'Partial structural match' (>20-50%), or 'Broad structural overlap' (>50-100%)")
    duration_sec: Optional[float] = Field(None, description="Candidate track duration in seconds")
    duration_disparity_warning: bool = Field(False, description="True if candidate duration is drastically different from query")
    disparity_explanation: Optional[str] = Field(None, description="Explanation of duration disparity and DTW path stretching")


class MimicResponse(BaseModel):
    query_track: MetadataContext
    evaluated_count: int
    survivors_count: int
    top_mimic_candidates: List[MimicCandidate] = Field(default_factory=list, description="Up to 3 top candidates")


class CandidateScore(BaseModel):
    candidate_id: str
    similarity: float
    normalized_distance: float
    best_shift: int
    p_value: Optional[float] = None
    exceedance_count: Optional[int] = None
    null_sample_size: Optional[int] = None
    is_top_match: bool = False
    ai_generated: Optional[bool] = None
    ai_confidence: Optional[float] = None
    genre: Optional[str] = None
    key: Optional[str] = None
    tempo_bpm: Optional[float] = None
    segments: Optional[List[Dict[str, Any]]] = None
    duration: Optional[float] = None
    alignment_coverage: Optional[float] = None
    coverage_percent: Optional[float] = None
    coverage_interpretation: Optional[str] = None


class CandidateRankResponse(BaseModel):
    query_id: str
    candidates: List[CandidateScore]
    top_match: Optional[CandidateScore] = None


# ---------------------------------------------------------------------------
# Feature 7: Suggestor Box & Evidence Timeline Schemas
# ---------------------------------------------------------------------------

class FlaggedSegment(BaseModel):
    start_time_a: float = Field(..., description="Start timestamp in query track A (seconds)")
    end_time_a: float = Field(..., description="End timestamp in query track A (seconds)")
    start_time_b: float = Field(..., description="Start timestamp in candidate track B (seconds)")
    end_time_b: float = Field(..., description="End timestamp in candidate track B (seconds)")
    mean_step_cost: float = Field(..., description="Mean alignment distance in this section")
    segment_similarity: float = Field(..., description="Normalized similarity for this localized section")


class SuggestorResponse(BaseModel):
    similarity: float = Field(..., ge=0.0, le=1.0)
    p_value: Optional[float] = Field(None, description="Significance p-value against null distribution")
    severity_tier: str = Field(..., description="'none' (p > 0.10), 'mild' (0.01 < p <= 0.10), or 'high' (p <= 0.01)")
    suggestions: List[str] = Field(..., description="Actionable recommendations to differentiate flagged sections")
    flagged_segments: List[FlaggedSegment] = Field(default_factory=list, description="Timestamped high-similarity segments")


class EvidenceTimelineItem(BaseModel):
    candidate_id: str
    candidate_title: Optional[str] = None
    candidate_artist: Optional[str] = None
    similarity: float
    p_value: Optional[float] = None
    severity: str
    best_shift: int
    flagged_segments: List[FlaggedSegment] = Field(default_factory=list)
    studio_suggestions: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Final Verdict & Case Routing Schemas
# ---------------------------------------------------------------------------

class FinalVerdict(BaseModel):
    verdict_code: str = Field(..., description="'critical', 'warning', 'advisory', 'clear', or 'incomplete'")
    verdict_label: str = Field(..., description="Human-readable verdict description")
    indicator_similarity: Optional[bool] = Field(None, description="Similarity >= 0.85")
    indicator_p_value: Optional[bool] = Field(None, description="P-value <= 0.01")
    indicator_creative: Optional[bool] = Field(None, description="Normalized Creative Score <= 0.24")
    true_indicators_count: int = Field(..., description="Number of true indicators (0 to 3)")
    explanation: str


class CaseRoutingResult(BaseModel):
    case_type: str = Field(..., description="'Case A', 'Case B', 'Case C', or 'Case U'")
    case_name: str = Field(..., description="'Cross-Genre', 'Same-Genre', 'No Match', 'Unresolved', or 'Undetermined Genre'")
    description: str


# ---------------------------------------------------------------------------
# Complete Pipeline Analysis Response
# ---------------------------------------------------------------------------

class PipelineAnalysisResponse(BaseModel):
    query_track: MetadataContext
    cosine_prefilter: CosinePrefilterStats
    genre_map: GenreMapResult
    total_corpus_size: int = Field(0, description="Total tracks indexed in the catalog and remote dataset")
    candidates_considered: int = Field(0, description="Total candidates considered before cosine prefilter")
    candidates_retained_prefilter: int = Field(0, description="Candidates surviving coarse cosine prefilter")
    candidates_discarded_prefilter: int = Field(0, description="Candidates excluded by coarse cosine prefilter")
    candidates_evaluated: int = Field(..., description="Candidates evaluated by 12-shift DTW alignment")
    candidates_surviving_floor: int = Field(..., description="Candidates exceeding the 0.45 similarity floor")
    candidates_surviving_pvalue: int = Field(..., description="Candidates surviving Feature 2 statistical gateway (p <= 0.10)")
    candidates_failed_to_load: int = Field(0, description="Candidates whose cache file was missing or corrupted")
    failed_candidate_details: List[Dict[str, str]] = Field(default_factory=list, description="Details of missing or unreadable candidates")
    null_distribution_sample_count: int = Field(0, description="Number of pairwise comparisons in null distribution")
    confidence_note: Optional[str] = Field(None, description="Statistical confidence warning when sample size is small")
    is_index_complete: bool = Field(False, description="Whether the analyzed dataset corpus has been fully indexed")
    corpus_status: str = Field("Partial", description="Description of corpus completeness")
    duration_disparity_detected: bool = Field(False, description="True if any top candidate has severe duration disparity with query")
    case_routing: CaseRoutingResult
    creative_score: CreativeScoreResult
    origin_classifier: OriginClassifierResult
    mimic_analysis: MimicResponse
    evidence_timeline: List[EvidenceTimelineItem] = Field(default_factory=list)
    extra_features: Optional[Dict[str, Any]] = Field(None, description="Independent acoustic and rhythmic descriptors (MFCC, HPSS, spectral, rhythm)")
    mert_status: Optional[Dict[str, Any]] = Field(None, description="MERT representation readiness and hardware status")
    final_verdict: FinalVerdict
    disclaimer: str = Field(
        "System parameters focus exclusively on harmonic and melodic pitch space. "
        "Rhythmic-only mimicry, timbral similarity, production style, and lyrical content "
        "fall outside active analytical boundaries. Results indicate statistical similarity — "
        "not legal plagiarism, copyright infringement, or AI training-data memorization."
    )


# ---------------------------------------------------------------------------
# Dataset Indexing and Real-Time Job Progress Schemas
# ---------------------------------------------------------------------------

class IndexStatusResponse(BaseModel):
    total_indexed: int
    is_complete: bool
    status: str  # "Not Started", "Running", "Partial", "Complete", "Completed with Errors"
    dataset_name: str
    split: str
    records_scanned: int = 0
    audio_indexed: int = 0
    existing_reused: int = 0
    duplicate_records: int = 0
    failed_records: int = 0
    records_awaiting_retry: int = 0
    human_tracks: int
    ai_tracks: int
    unknown_origin_tracks: int
    verified_genre_tracks: int
    estimated_genre_tracks: int
    unknown_genre_tracks: int
    unresolved_records_count: int
    unresolved_records: List[Dict[str, Any]] = Field(default_factory=list)
    null_pairs_cached: Optional[int] = None
    current_operation: Optional[str] = None
    elapsed_sec: float = 0.0
    downloaded_bytes: int = 0
    is_scanning: bool = False
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class DatasetScanStartRequest(BaseModel):
    full: bool = True
    sample_size: int = 10
    resume: bool = True
    force_reindex: bool = False


class JobStartResponse(BaseModel):
    job_id: str
    status: str
    message: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: str  # "queued", "running", "completed", "failed"
    stage: int  # 1 to 13
    stage_name: str
    progress_pct: float
    processed_count: int
    total_count: int
    current_operation: str
    message: str
    error: Optional[str] = None
    result: Optional[PipelineAnalysisResponse] = None

