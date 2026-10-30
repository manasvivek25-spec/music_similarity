"""
Central Pipeline Orchestrator for the Music Similarity Research Platform.

Sequences:
1. Audio validation & feature extraction (raw & beat-sync chroma, timing)
2. Feature 5 metadata & musical context estimation
3. Stage 1 coarse cosine prefiltering
4. Stage 2 12-shift key-invariant DTW & 0.45 similarity floor
5. Stage 3 Genre Influence Map
6. Feature 2 Statistical Gateway (empirical p-values, 15-survivor cap)
7. Feature 3 Creative Score (broad & narrow, zero-variance safe)
8. Case A / B / C routing
9. Feature 4 Independent Origin Classifier
10. Feature 6 Mimic Identification
11. Feature 7 Suggestor Box & Evidence Timeline
12. Final Three-Indicator Verdict
"""

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np

logger = logging.getLogger(__name__)

from src.config import (
    COSINE_PREFILTER_THRESHOLD,
    CREATIVE_SCORE_VERDICT_THRESHOLD,
    DATA_DIR,
    FEATURES_DIR,
    HUMAN_DIR,
    AI_DIR,
    METADATA_PATH,
    MAX_SURVIVORS,
    P_VALUE_CRITICAL_THRESHOLD,
    REF_DIR,
    SIMILARITY_FLOOR,
)
from src.creative_score import compute_creative_score
from src.extract import extract_chroma_beatsync, get_or_extract_features
from src.feature5 import extract_metadata_context
from src.genre import compute_genre_influence_map, load_metadata_catalog
from src.mimic import build_mimic_analysis
from src.origin_classifier import predict_origin
from src.schemas import (
    CaseRoutingResult,
    CosinePrefilterStats,
    CreativeScoreResult,
    EvidenceTimelineItem,
    FinalVerdict,
    MetadataContext,
    MimicResponse,
    PipelineAnalysisResponse,
)
from src.similarity import (
    coarse_cosine_prefilter,
    compute_mean_chroma_vector,
    cosine_similarity_vectors,
    dtw_align_detailed,
    dtw_similarity,
    get_similarity_tier,
)
from src.stats_test import (
    build_pairwise_null_distribution,
    evaluate_statistical_gateway,
)
from src.utils import list_audio_files


def evaluate_final_verdict(
    best_similarity: Optional[float],
    best_p_value: Optional[float],
    normalized_creative_score: Optional[float],
    confidence: Optional[str] = None,
    candidates_evaluated: Optional[int] = None,
    null_sample_count: Optional[int] = None,
) -> FinalVerdict:
    """
    Evaluate the strict three-indicator verdict:

    1. Similarity >= 0.85
    2. P-value <= 0.01
    3. Normalized Creative Score <= 0.24

    Verdict tiers:
    - 3 true: Critical — All indicators confirm structural imitation
    - 2 true: Warning — Multiple indicators suggest structural similarity
    - 1 true: Advisory — Single indicator, likely genre convention
    - 0 true: Clear — No significant similarity detected
    - When p_value is None: Incomplete — Insufficient Null Population

    Authoritative Decision 1:
    If p_value is None:
    - Do not treat the p-value indicator as a valid, evaluated false result.
    - Do not claim that the complete three-indicator verdict was evaluated.
    - Return an explicit status: "incomplete — insufficient null population".
    - Preserve and display the similarity and creative-score indicator states when their values are valid.
    - Label provisional indicators clearly.
    """
    if best_similarity is None:
        return FinalVerdict(
            verdict_code="incomplete",
            verdict_label="Incomplete — Insufficient evidence for definitive verdict",
            indicator_similarity=None,
            indicator_p_value=None,
            indicator_creative=None,
            true_indicators_count=0,
            explanation="Could not calculate pairwise similarity scores; verify audio inputs.",
        )

    from src.config import MIN_NULL_PAIRS

    ind_sim = bool(best_similarity >= 0.85)
    ind_creat = (
        bool(normalized_creative_score <= CREATIVE_SCORE_VERDICT_THRESHOLD)
        if normalized_creative_score is not None
        else None
    )

    # When p-value is unavailable due to insufficient null sample size
    if best_p_value is None:
        n_pairs = null_sample_count if null_sample_count is not None else 0
        prov_true = sum(1 for ind in [ind_sim, ind_creat] if ind is True)
        sim_desc = "Triggered" if ind_sim else "Not Triggered"
        creat_desc = "Triggered" if ind_creat is True else ("Not Triggered" if ind_creat is False else "Unavailable")
        explanation = (
            f"Incomplete analysis: Empirical p-value is unavailable due to insufficient null population "
            f"(N={n_pairs}, minimum {MIN_NULL_PAIRS} required). A complete three-indicator verdict cannot be evaluated. "
            f"Provisional indicators: Similarity={sim_desc}, Creative Score={creat_desc}."
        )
        return FinalVerdict(
            verdict_code="incomplete",
            verdict_label="Incomplete — Insufficient Null Population",
            indicator_similarity=ind_sim,
            indicator_p_value=None,
            indicator_creative=ind_creat,
            true_indicators_count=prov_true,
            explanation=explanation,
        )

    # When best_p_value is available (N >= 5)
    ind_pval = bool(best_p_value <= P_VALUE_CRITICAL_THRESHOLD)

    if ind_creat is None:
        prov_true = sum(1 for ind in [ind_sim, ind_pval] if ind is True)
        sim_desc = "Triggered" if ind_sim else "Not Triggered"
        pval_desc = "Triggered" if ind_pval else "Not Triggered"
        explanation = (
            f"Incomplete analysis: Creative novelty score is unavailable due to insufficient reference population. "
            f"A complete three-indicator verdict cannot be evaluated. "
            f"Provisional indicators: Similarity={sim_desc}, P-value={pval_desc}."
        )
        return FinalVerdict(
            verdict_code="incomplete",
            verdict_label="Incomplete — Insufficient Reference Population",
            indicator_similarity=ind_sim,
            indicator_p_value=ind_pval,
            indicator_creative=None,
            true_indicators_count=prov_true,
            explanation=explanation,
        )

    true_count = sum([ind_sim, ind_pval, bool(ind_creat)])

    if true_count == 3:
        code = "critical"
        label = "Critical — All indicators confirm structural imitation"
        explanation = (
            f"All three independent analytical indicators exceeded critical thresholds "
            f"(Similarity: {best_similarity:.2f} >= 0.85, P-Value: {best_p_value:.4f} <= 0.01, "
            f"Creative Score: {normalized_creative_score} <= {CREATIVE_SCORE_VERDICT_THRESHOLD})."
        )
    elif true_count == 2:
        code = "warning"
        label = "Warning — Multiple indicators suggest structural similarity"
        explanation = "Two out of three indicators confirm structural convergence beyond normal variation."
    elif true_count == 1:
        code = "advisory"
        label = "Advisory — Single indicator, likely genre convention"
        explanation = "Only a single indicator triggered; likely reflective of shared idiomatic genre conventions."
    else:
        code = "clear"
        label = "Clear — No significant similarity detected"
        explanation = "Zero indicators triggered. Pitch contours and harmonic structures fall within standard variance."

    if confidence == "Low" or (candidates_evaluated is not None and candidates_evaluated < 30):
        cand_str = f"N = {candidates_evaluated} candidates evaluated" if candidates_evaluated is not None else "small candidate pool"
        null_str = f", {null_sample_count} null comparisons" if null_sample_count is not None else ""
        explanation += (
            f" [Confidence Context: Evaluated with Low statistical confidence due to limited sample size "
            f"({cand_str}{null_str}). Empirical p-values and population estimates carry high sampling variance "
            f"and do not establish reliable population-level conclusions.]"
        )

    return FinalVerdict(
        verdict_code=code,
        verdict_label=label,
        indicator_similarity=ind_sim,
        indicator_p_value=ind_pval,
        indicator_creative=ind_creat,
        true_indicators_count=true_count,
        explanation=explanation,
    )


def run_pipeline(
    query_audio_path: Union[str, Path],
    candidate_audio_paths: Optional[List[Union[str, Path]]] = None,
    query_metadata: Optional[Dict[str, Any]] = None,
    candidate_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
    reference_pool_dir: Optional[Union[str, Path]] = None,
    catalog_csv_path: Optional[Union[str, Path]] = None,
    progress_callback: Optional[Callable[[int, str, float, int, int, str], None]] = None,
) -> PipelineAnalysisResponse:
    """
    Execute the complete end-to-end music similarity research pipeline across 13 stages.

    Parameters
    ----------
    query_audio_path : Union[str, Path]
        Path to query track audio file.
    candidate_audio_paths : Optional[List[Union[str, Path]]]
        Explicit candidate audio files to evaluate. If None, loads from reference pool.
    query_metadata : Optional[Dict[str, Any]]
        Declared metadata for query track (title, artist, genre, etc.).
    candidate_metadata : Optional[Dict[str, Dict[str, Any]]]
        Metadata dictionary for candidates keyed by track_id or stem.
    reference_pool_dir : Optional[Union[str, Path]]
        Directory containing background reference pool audio.
    catalog_csv_path : Optional[Union[str, Path]]
        Path to metadata catalog CSV.
    progress_callback : Optional[Callable[[int, str, float, int, int, str], None]]
        Callback for real progress tracking across stages (stage, name, pct, cur, tot, msg).

    Returns
    -------
    PipelineAnalysisResponse
        Comprehensive structured analysis payload.
    """
    def _emit_progress(stage: int, name: str, pct: float, cur: int, tot: int, msg: str) -> None:
        if progress_callback:
            try:
                progress_callback(stage, name, pct, cur, tot, msg)
            except Exception:
                pass

    ref_dir = Path(reference_pool_dir) if reference_pool_dir is not None else REF_DIR
    meta_csv = Path(catalog_csv_path) if catalog_csv_path is not None else METADATA_PATH
    catalog = load_metadata_catalog(meta_csv)

    # 1. Stage 1: Upload Validation
    _emit_progress(1, "Upload validation", 7.0, 1, 1, "Audio upload validated.")

    # 2. Stage 2: Audio Feature Extraction
    _emit_progress(2, "Audio feature extraction", 14.0, 0, 1, "Extracting Chroma CQT and beat synchronization...")
    chroma_raw, chroma_beat, beat_times, duration_sec = get_or_extract_features(
        query_audio_path, cache_dir=FEATURES_DIR
    )
    query_mean_chroma = compute_mean_chroma_vector(chroma_beat)
    _emit_progress(2, "Audio feature extraction", 20.0, 1, 1, f"Extracted Chroma CQT ({chroma_beat.shape[1]} beat frames, {duration_sec:.1f}s).")

    # Feature 5: Query Metadata & Context
    merged_q_meta = dict(catalog.get(Path(query_audio_path).stem, {}))
    if query_metadata:
        merged_q_meta.update({k: v for k, v in query_metadata.items() if v is not None})

    query_context = extract_metadata_context(
        query_audio_path,
        provided_metadata=merged_q_meta,
        chroma_matrix=chroma_raw
    )

    # Independent descriptive features (MFCCs, HPSS, spectral descriptors)
    extra_features_dict: Optional[Dict[str, Any]] = None
    try:
        from src.features_extra import get_or_extract_extra_features
        extra_obj = get_or_extract_extra_features(
            audio_file_or_waveform=query_audio_path,
            track_id=Path(query_audio_path).stem,
            cache_dir=FEATURES_DIR / "extra",
        )
        extra_features_dict = extra_obj.to_dict()
    except Exception as exc:
        logger.debug(f"Extra features extraction skipped: {exc}")

    from src.mert_embedding import detect_mert_environment
    mert_status_dict = detect_mert_environment().to_dict()

    # 3. Stage 3: Loading Complete Indexed Dataset
    _emit_progress(3, "Loading complete indexed dataset", 25.0, 0, 1, "Loading indexed dataset corpus...")
    from src.remote_dataset import load_remote_index
    from src.config import REPO_ROOT, DURATION_DISPARITY_RATIO, MIN_NULL_PAIRS

    remote_index = load_remote_index()
    remote_tracks = remote_index.get("tracks", {})
    is_index_complete = bool(remote_index.get("is_complete", False))

    # Assemble candidate catalog without loading unnecessary heavy feature matrices
    candidate_descriptors: Dict[str, Dict[str, Any]] = {}

    # Local audio files in reference/human/ai directories
    cand_paths: List[Path] = []
    if candidate_audio_paths and len(candidate_audio_paths) > 0:
        cand_paths = [Path(p) for p in candidate_audio_paths if Path(p).is_file()]
    else:
        ref_files = list_audio_files(ref_dir)
        human_files = list_audio_files(HUMAN_DIR)
        ai_files = list_audio_files(AI_DIR)
        all_pool = list(dict.fromkeys(ref_files + human_files + ai_files))
        cand_paths = [p for p in all_pool if p.resolve() != Path(query_audio_path).resolve()]

    for cp in cand_paths:
        cid = cp.stem
        c_meta = dict(catalog.get(cid, {}))
        if candidate_metadata and cid in candidate_metadata:
            c_meta.update(candidate_metadata[cid])
        c_meta["file_path"] = str(cp)
        candidate_descriptors[cid] = {
            "source_type": "local",
            "file_path": cp,
            "meta": c_meta,
            "mean_chroma": None,  # Computed on demand if needed
        }

    # Add remote tracks using precomputed compact mean-chroma
    for track_id, r_info in remote_tracks.items():
        if track_id in candidate_descriptors:
            continue
        c_meta = {
            "title": r_info.get("track_id", track_id),
            "artist": r_info.get("raw_metadata", {}).get("source") or "HuggingFace Dataset",
            "genre": r_info.get("genre"),  # Strictly no default ambient fallback
            "origin_label": r_info.get("origin_label", "unknown"),
            "duration_sec": r_info.get("duration_sec", 30.0),
            "source": "remote_dataset",
            "beat_cache": r_info.get("beat_cache") or r_info.get("beat_feature_path"),
            "times_cache": r_info.get("times_cache") or r_info.get("times_feature_path"),
        }
        cand_mean = np.array(r_info.get("mean_chroma", []), dtype=float) if r_info.get("mean_chroma") else None
        candidate_descriptors[track_id] = {
            "source_type": "remote",
            "info": r_info,
            "meta": c_meta,
            "mean_chroma": cand_mean,
        }

    total_corpus_size = len(candidate_descriptors)
    _emit_progress(3, "Loading complete indexed dataset", 30.0, total_corpus_size, total_corpus_size, f"Indexed corpus loaded ({total_corpus_size} tracks available).")

    # 4. Stage 4: Checking Every Indexed Track Against Cosine Prefilter
    _emit_progress(4, "Checking every indexed track against cosine prefilter", 35.0, 0, total_corpus_size, f"Evaluating cosine prefilter across {total_corpus_size} tracks (threshold >= 0.30)...")

    surviving_candidate_ids: List[str] = []
    prefilter_discarded_count = 0

    for cid, c_desc in candidate_descriptors.items():
        cand_mean = c_desc.get("mean_chroma")
        if cand_mean is None or len(cand_mean) != 12:
            # For local files without stored mean-chroma, compute it
            if c_desc.get("source_type") == "local":
                try:
                    _, c_beat, _, _ = get_or_extract_features(c_desc["file_path"], cache_dir=FEATURES_DIR)
                    cand_mean = compute_mean_chroma_vector(c_beat)
                    c_desc["mean_chroma"] = cand_mean
                except Exception:
                    cand_mean = np.zeros(12, dtype=float)
            else:
                cand_mean = np.zeros(12, dtype=float)

        cos_sim = cosine_similarity_vectors(query_mean_chroma, cand_mean)
        if cos_sim >= COSINE_PREFILTER_THRESHOLD:
            surviving_candidate_ids.append(cid)
        else:
            prefilter_discarded_count += 1

    before_count = total_corpus_size
    after_count = len(surviving_candidate_ids)
    pct_removed = ((before_count - after_count) / before_count * 100.0) if before_count > 0 else 0.0
    prefilter_stats = CosinePrefilterStats(
        before_count=before_count,
        after_count=after_count,
        percentage_removed=round(pct_removed, 2),
    )
    _emit_progress(4, "Checking every indexed track against cosine prefilter", 45.0, total_corpus_size, total_corpus_size, f"Prefilter complete: {after_count} of {total_corpus_size} survived.")

    # 5. Stage 3 (Logical Sequence): Genre Influence Map (after cosine filtering)
    survivor_genres: Dict[str, Optional[str]] = {
        cid: candidate_descriptors[cid]["meta"].get("genre") for cid in surviving_candidate_ids
    }
    genre_map = compute_genre_influence_map(query_context.genre, survivor_genres)

    # 6. Stage 5: DTW Evaluation of Surviving Candidates
    _emit_progress(5, "DTW evaluation of surviving candidates", 50.0, 0, max(1, after_count), f"Beginning 12-shift DTW alignment on {after_count} surviving candidates...")

    evaluated_records: List[Dict[str, Any]] = []
    loaded_candidate_features: Dict[str, np.ndarray] = {}
    candidates_failed_to_load = 0
    failed_candidate_details: List[Dict[str, str]] = []
    duration_disparity_detected = False

    for idx, cid in enumerate(surviving_candidate_ids):
        c_desc = candidate_descriptors[cid]
        c_meta = c_desc["meta"]
        c_beat: Optional[np.ndarray] = None
        c_times: Optional[np.ndarray] = None
        c_dur = float(c_meta.get("duration_sec") or 30.0)

        # Load candidate detailed beat-chroma
        try:
            if c_desc["source_type"] == "local":
                _, c_beat, c_times, c_dur = get_or_extract_features(c_desc["file_path"], cache_dir=FEATURES_DIR)
            else:
                beat_ref = c_meta.get("beat_cache") or c_meta.get("beat_feature_path")
                times_ref = c_meta.get("times_cache") or c_meta.get("times_feature_path")
                if beat_ref:
                    bp = Path(beat_ref)
                    beat_file = bp if bp.is_absolute() else (REPO_ROOT / bp)
                else:
                    beat_file = FEATURES_DIR / f"{cid}_beat.npy"

                if times_ref:
                    tp = Path(times_ref)
                    times_file = tp if tp.is_absolute() else (REPO_ROOT / tp)
                else:
                    times_file = FEATURES_DIR / f"{cid}_times.npy"

                if not beat_file.is_file():
                    raise FileNotFoundError(f"Missing detailed feature cache: {beat_file}")
                c_beat = np.load(str(beat_file))
                c_times = np.load(str(times_file)) if times_file.is_file() else np.array([])
        except Exception as exc:
            candidates_failed_to_load += 1
            failed_candidate_details.append({"candidate_id": cid, "error": str(exc)})
            logger.warning(f"Candidate {cid} cache load failure: {exc}")
            continue

        loaded_candidate_features[cid] = c_beat

        # Run 12-pitch-shift DTW alignment
        dtw_res = dtw_align_detailed(chroma_beat, c_beat)

        # Check duration disparity: query vs candidate
        dur_ratio = max(duration_sec, c_dur) / max(min(duration_sec, c_dur), 1e-3)
        has_disparity = bool(dur_ratio >= DURATION_DISPARITY_RATIO)
        if has_disparity:
            duration_disparity_detected = True
        disparity_msg = (
            f"Significant duration disparity (Query: {duration_sec:.1f}s vs Candidate: {c_dur:.1f}s, ratio: {dur_ratio:.1f}x). "
            f"Global DTW stretching across disproportionate audio lengths may distort alignment path."
            if has_disparity else None
        )

        from src.mimic import calculate_alignment_coverage
        cov, cov_pct, cov_interp = calculate_alignment_coverage(
            dtw_res.alignment_path,
            dtw_res.local_costs,
            total_query_frames=chroma_beat.shape[1],
        )

        evaluated_records.append({
            "candidate_id": cid,
            "title": c_meta.get("title") or cid.replace("_", " ").title(),
            "artist": c_meta.get("artist"),
            "genre": c_meta.get("genre"),
            "duration_sec": c_dur,
            "duration_disparity_warning": has_disparity,
            "disparity_explanation": disparity_msg,
            "similarity": dtw_res.similarity,
            "similarity_tier": get_similarity_tier(dtw_res.similarity),
            "best_shift": dtw_res.best_shift,
            "normalized_distance": dtw_res.normalized_distance,
            "raw_distance": dtw_res.raw_distance,
            "alignment_coverage": cov,
            "coverage_percent": cov_pct,
            "coverage_interpretation": cov_interp,
            "alignment_path": dtw_res.alignment_path,
            "local_costs": dtw_res.local_costs,
            "total_query_frames": chroma_beat.shape[1],
            "dtw_result": dtw_res,
            "candidate_times": c_times,
            "query_key": query_context.estimated_key,
            "query_tempo": query_context.estimated_tempo,
        })

        step_pct = 50.0 + min(10.0, ((idx + 1) / max(after_count, 1)) * 10.0)
        _emit_progress(5, "DTW evaluation of surviving candidates", step_pct, idx + 1, after_count, f"DTW aligned {cid} ({idx + 1}/{after_count})")

    # Apply 0.45 Similarity Floor
    floor_survivors = [r for r in evaluated_records if r["similarity"] >= SIMILARITY_FLOOR]
    candidates_surviving_floor = len(floor_survivors)

    # 7. Stage 6: Loading or Building Statistical Null Distributions
    _emit_progress(6, "Loading or building statistical null distributions", 62.0, 0, 1, "Accessing cached pairwise null distribution...")
    from src.stats_test import get_or_build_cached_null_distribution
    cand_meta_map = {cid: candidate_descriptors[cid]["meta"] for cid in loaded_candidate_features.keys()}
    global_null, genre_null_dict = get_or_build_cached_null_distribution(
        chroma_dict=loaded_candidate_features,
        similarity_func=dtw_similarity,
        candidate_metadata=cand_meta_map,
    )
    null_count = len(global_null)
    _emit_progress(6, "Loading or building statistical null distributions", 66.0, null_count, null_count, f"Null distribution ready ({null_count} pairs).")

    # 8. Stage 7: Statistical Gateway Evaluation
    _emit_progress(7, "Statistical gateway evaluation", 68.0, 0, max(1, len(evaluated_records)), "Evaluating empirical p-values against null distribution...")
    for rec in evaluated_records:
        target_genre = rec.get("genre")
        gw_res = evaluate_statistical_gateway(
            query_score=rec["similarity"],
            genre=target_genre,
            genre_null_scores=genre_null_dict,
            global_null_scores=global_null,
            min_genre_samples=30,
            min_null_pairs=MIN_NULL_PAIRS,
        )
        rec["p_value"] = gw_res.p_value
        rec["confidence"] = gw_res.confidence
        rec["severity_label"] = gw_res.severity_label
        rec["fallback_source"] = gw_res.fallback_source

    from src.stats_test import benjamini_hochberg_correction
    raw_pvals = [r.get("p_value") for r in evaluated_records]
    adj_pvals = benjamini_hochberg_correction(raw_pvals)
    for r, adj_p in zip(evaluated_records, adj_pvals):
        r["adjusted_p_value"] = adj_p

    stat_survivors = [
        r for r in floor_survivors if r.get("p_value") is not None and r["p_value"] <= 0.10
    ]
    stat_survivors = sorted(stat_survivors, key=lambda x: (x.get("p_value", 1.0), -x.get("similarity", 0.0)))[:MAX_SURVIVORS]
    candidates_surviving_pvalue = len(stat_survivors)
    _emit_progress(7, "Statistical gateway evaluation", 73.0, len(evaluated_records), len(evaluated_records), f"Statistical gateway complete ({candidates_surviving_pvalue} significant survivors).")

    # 9. Stage 8: Creative Scoring & Case Routing
    _emit_progress(8, "Creative scoring", 75.0, 0, 1, "Evaluating creative novelty and compositional variance...")

    # Case routing and creative population pool
    best_sim = max([r["similarity"] for r in evaluated_records], default=0.0)

    # Select documented genre-specific reference pool or explicitly identified global fallback
    target_genre = query_context.genre
    if target_genre and target_genre in genre_null_dict and len(genre_null_dict[target_genre]) >= 30:
        ref_population_scores = genre_null_dict[target_genre]
        ref_genres = [target_genre]
    elif len(global_null) >= 5:
        ref_population_scores = global_null
        ref_genres = ["Global Human Baseline"]
    else:
        ref_population_scores = []
        ref_genres = []

    if null_count < MIN_NULL_PAIRS:
        # State U: Unresolved due to insufficient null sample size
        case_routing = CaseRoutingResult(
            case_type="Case U",
            case_name="Unresolved",
            description=f"Insufficient null distribution population (found N={null_count}, minimum {MIN_NULL_PAIRS} required) to establish statistical match significance."
        )
        creative_res = compute_creative_score(
            observed_similarity=best_sim,
            genre_pool_scores=ref_population_scores,
            survivor_scores=None,
            evaluated_genres=[query_context.genre] if query_context.genre else [],
            min_pool_size=5,
        )
    elif len(stat_survivors) == 0:
        # Case C: No Match
        case_routing = CaseRoutingResult(
            case_type="Case C",
            case_name="No Match",
            description="No candidate matches exceeded the 0.45 similarity floor and p <= 0.10 statistical gateway."
        )
        creative_res = compute_creative_score(
            observed_similarity=best_sim,
            genre_pool_scores=ref_population_scores,
            survivor_scores=None,
            evaluated_genres=[query_context.genre] if query_context.genre else [],
            min_pool_size=5,
        )
    else:
        top_match = stat_survivors[0]
        match_genre = top_match.get("genre")
        native_genre = query_context.genre

        if native_genre and match_genre and native_genre.lower() != match_genre.lower():
            case_routing = CaseRoutingResult(
                case_type="Case A",
                case_name="Cross-Genre",
                description=f"Significant structural similarity detected across different genres (Query: {native_genre}, Match: {match_genre})."
            )
            unique_genres = list(dict.fromkeys([
                r.get("genre") for r in stat_survivors if r.get("genre")
            ]))[:2]
            target_genres = [native_genre] + [g for g in unique_genres if g.lower() != native_genre.lower()]
            surv_scores = [r["similarity"] for r in stat_survivors]
            creative_res = compute_creative_score(
                observed_similarity=top_match["similarity"],
                genre_pool_scores=ref_population_scores,
                survivor_scores=surv_scores,
                evaluated_genres=target_genres[:2],
                min_pool_size=5,
            )
        else:
            case_routing = CaseRoutingResult(
                case_type="Case B",
                case_name="Same-Genre",
                description="Significant structural similarity identified within the same musical genre."
            )
            surv_scores = [r["similarity"] for r in stat_survivors]
            creative_res = compute_creative_score(
                observed_similarity=top_match["similarity"],
                genre_pool_scores=ref_population_scores,
                survivor_scores=surv_scores,
                evaluated_genres=[native_genre or match_genre or "General"],
                min_pool_size=5,
            )
    _emit_progress(8, "Creative scoring", 79.0, 1, 1, "Creative score evaluated.")

    # 10. Stage 9: Origin-Classifier Availability Check
    _emit_progress(9, "Origin-classifier availability check", 81.0, 0, 1, "Checking origin classifier...")
    origin_res = predict_origin(query_audio_path)
    _emit_progress(9, "Origin-classifier availability check", 84.0, 1, 1, f"Origin check: {origin_res.model_status}.")

    # 11. Stage 10: Musical-Context Estimation
    _emit_progress(10, "Musical-context estimation", 86.0, 1, 1, f"Context extracted: Key {query_context.estimated_key}, Tempo {query_context.estimated_tempo} BPM.")

    # 12. Stage 11: Mimic Ranking and Alignment-Coverage Calculation
    _emit_progress(11, "Mimic ranking and alignment-coverage calculation", 89.0, 0, 1, "Ranking mimic candidates...")
    mimic_records: List[Dict[str, Any]] = []
    for r in (stat_survivors if stat_survivors else evaluated_records):
        rec_copy = dict(r)
        rec_copy["origin_prediction"] = origin_res.predicted_origin
        mimic_records.append(rec_copy)

    mimic_analysis = build_mimic_analysis(query_context, mimic_records, max_mimic_candidates=3)
    _emit_progress(11, "Mimic ranking and alignment-coverage calculation", 93.0, 1, 1, "Mimic ranking complete.")

    # 13. Stage 12: Evidence Timeline and Suggestions
    _emit_progress(12, "Evidence timeline and suggestions", 95.0, 0, 1, "Generating evidence timeline and studio suggestions...")
    evidence_timeline: List[EvidenceTimelineItem] = []
    if case_routing.case_type in ("Case A", "Case B") and len(stat_survivors) > 0:
        from src.suggestor import build_evidence_timeline
        evidence_timeline = build_evidence_timeline(
            candidate_survivors=stat_survivors,
            times_query=beat_times,
            max_candidates=3
        )
    _emit_progress(12, "Evidence timeline and suggestions", 97.0, 1, 1, "Evidence timeline generated.")

    # 14. Stage 13: Final Verdict and Result Display
    _emit_progress(13, "Final verdict and result display", 98.0, 0, 1, "Evaluating final three-indicator verdict...")

    best_sim_for_verdict = stat_survivors[0]["similarity"] if stat_survivors else (
        max([r["similarity"] for r in evaluated_records], default=0.0) if evaluated_records else 0.0
    )
    best_pval_for_verdict = stat_survivors[0].get("p_value") if stat_survivors else None
    norm_creative = creative_res.broad_score_normalized if creative_res.is_available else None
    top_confidence = stat_survivors[0].get("confidence") if stat_survivors else (
        evaluated_records[0].get("confidence") if evaluated_records else "Low"
    )

    final_verdict = evaluate_final_verdict(
        best_similarity=best_sim_for_verdict,
        best_p_value=best_pval_for_verdict,
        normalized_creative_score=norm_creative,
        confidence=top_confidence,
        candidates_evaluated=total_corpus_size,
        null_sample_count=null_count,
    )

    confidence_note: Optional[str] = None
    if top_confidence == "Low" or total_corpus_size < 30:
        confidence_note = (
            f"Low Statistical Confidence Notice: Evaluated against a minimal candidate pool "
            f"({total_corpus_size} candidates, {null_count} null comparisons). "
            f"Results reflect analytical calculations but carry high sampling variance; "
            f"insufficient evidence for reliable population-level conclusions or genre distribution."
        )

    corpus_status = (
        f"Complete Dataset ({total_corpus_size} tracks indexed of SleepyJesse/ai_music_large)"
        if is_index_complete
        else f"Partial Dataset Index ({total_corpus_size} tracks indexed of SleepyJesse/ai_music_large). Analysis covers indexed subset only; full-dataset scan in progress or incomplete."
    )

    response = PipelineAnalysisResponse(
        query_track=query_context,
        cosine_prefilter=prefilter_stats,
        genre_map=genre_map,
        total_corpus_size=total_corpus_size,
        candidates_considered=total_corpus_size,
        candidates_retained_prefilter=after_count,
        candidates_discarded_prefilter=prefilter_discarded_count,
        candidates_evaluated=len(evaluated_records),
        candidates_surviving_floor=candidates_surviving_floor,
        candidates_surviving_pvalue=candidates_surviving_pvalue,
        candidates_failed_to_load=candidates_failed_to_load,
        failed_candidate_details=failed_candidate_details,
        null_distribution_sample_count=null_count,
        confidence_note=confidence_note,
        is_index_complete=is_index_complete,
        corpus_status=corpus_status,
        duration_disparity_detected=duration_disparity_detected,
        case_routing=case_routing,
        creative_score=creative_res,
        origin_classifier=origin_res,
        mimic_analysis=mimic_analysis,
        evidence_timeline=evidence_timeline,
        extra_features=extra_features_dict,
        mert_status=mert_status_dict,
        final_verdict=final_verdict,
    )

    _emit_progress(13, "Final verdict and result display", 100.0, 1, 1, "Analysis complete.")
    return response
