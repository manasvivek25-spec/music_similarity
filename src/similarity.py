"""
Music similarity computation module.

Provides sequence alignment and similarity scoring between chroma representations,
accounting for key transpositions via circular pitch shifting.
Exposes Stage 1 coarse cosine prefiltering, Stage 2 global DTW sweep with +-1
adjacent-shift boundary verification, and similarity tier categorization.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw

from src.config import COSINE_PREFILTER_THRESHOLD, SIMILARITY_FLOOR
from src.schemas import CosinePrefilterStats


@dataclass
class DTWResult:
    """
    Structured outcome of key-invariant DTW alignment.

    Attributes
    ----------
    similarity : float
        Normalized similarity score in [0, 1], calculated as 1 / (1 + normalized_dist).
    normalized_distance : float
        Alignment cost normalized by path length (min_distance / path_length).
    raw_distance : float
        Accumulated minimum DTW distance across all alignment steps.
    best_shift : int
        Optimal circular pitch shift (0 to 11 semitones) applied to chroma_b.
    alignment_path : List[Tuple[int, int]]
        Sequence of (frame_idx_a, frame_idx_b) index pairs along the warping path.
    local_costs : np.ndarray
        Array of Euclidean distances at each step along alignment_path.
    path_length : int
        Total steps along the optimal alignment path.
    adjacent_shift_costs : Dict[int, float]
        Costs recorded for adjacent semitone shifts (best_shift +- 1 mod 12).
    boundary_verified : bool
        True if best_shift has cost <= costs of both adjacent shifts.
    """
    similarity: float
    normalized_distance: float
    raw_distance: float
    best_shift: int
    alignment_path: List[Tuple[int, int]]
    local_costs: np.ndarray
    path_length: int = 0
    adjacent_shift_costs: Dict[int, float] = field(default_factory=dict)
    boundary_verified: bool = True


def get_similarity_tier(similarity: float) -> str:
    """
    Classify a normalized similarity score into predefined scientific tiers.

    Numeric boundaries:
    - [0.85, 1.00]: High Structural Similarity
    - [0.65, 0.85): Moderate Similarity
    - [0.45, 0.65): Low Similarity
    - [0.00, 0.45): Negligible

    Parameters
    ----------
    similarity : float
        Normalized similarity score in [0.0, 1.0].

    Returns
    -------
    str
        Tier description.
    """
    if similarity >= 0.85:
        return "High Structural Similarity"
    elif similarity >= 0.65:
        return "Moderate Similarity"
    elif similarity >= 0.45:
        return "Low Similarity"
    else:
        return "Negligible"


def compute_mean_chroma_vector(chroma: np.ndarray) -> np.ndarray:
    """
    Compute the mean 12-bin chroma vector across time frames.

    Handles empty or non-2D matrices safely, and accepts precomputed 1D 12-bin vectors.
    """
    chroma = np.asarray(chroma, dtype=float)
    if chroma.ndim == 1 and len(chroma) == 12:
        return np.nan_to_num(chroma, nan=0.0, posinf=0.0, neginf=0.0)

    if chroma.ndim != 2:
        return np.zeros(12, dtype=float)

    if chroma.shape[0] != 12 and chroma.shape[1] == 12:
        chroma = chroma.T

    if chroma.shape[0] != 12 or chroma.shape[1] == 0:
        return np.zeros(12, dtype=float)

    vec = np.nanmean(chroma, axis=1)
    # Replace non-finite entries with 0.0
    vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
    return vec


def cosine_similarity_vectors(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """
    Compute cosine similarity between two 1D vectors with zero-division safety.

    Returns 0.0 if either vector has zero or non-finite norm.
    """
    if vec_a.ndim != 1 or vec_b.ndim != 1 or len(vec_a) != len(vec_b):
        return 0.0

    norm_a = float(np.linalg.norm(vec_a))
    norm_b = float(np.linalg.norm(vec_b))

    if norm_a <= 1e-12 or norm_b <= 1e-12 or not (np.isfinite(norm_a) and np.isfinite(norm_b)):
        return 0.0

    dot_prod = float(np.dot(vec_a, vec_b))
    cosine_val = dot_prod / (norm_a * norm_b)
    # Clamp to [-1.0, 1.0] to prevent slight numerical drift
    return float(np.clip(cosine_val, 0.0, 1.0))


def coarse_cosine_prefilter(
    query_chroma: np.ndarray,
    candidates: Dict[str, np.ndarray],
    threshold: float = COSINE_PREFILTER_THRESHOLD
) -> Tuple[Dict[str, np.ndarray], CosinePrefilterStats]:
    """
    Stage 1: Coarse cosine prefilter.

    Computes the mean 12-bin chroma vector for the query and each candidate,
    calculates cosine similarity, and eliminates candidates with similarity < threshold.

    Parameters
    ----------
    query_chroma : np.ndarray
        Query chroma matrix of shape (12, n_frames).
    candidates : Dict[str, np.ndarray]
        Mapping of candidate track_id -> candidate chroma matrix.
    threshold : float, optional
        Minimum cosine similarity to survive, default 0.30.

    Returns
    -------
    surviving_candidates : Dict[str, np.ndarray]
        Filtered dictionary containing only surviving candidate entries.
    stats : CosinePrefilterStats
        Actual filtering statistics (before count, after count, percentage removed).
    """
    before_count = len(candidates)
    if before_count == 0:
        return {}, CosinePrefilterStats(before_count=0, after_count=0, percentage_removed=0.0)

    query_vec = compute_mean_chroma_vector(query_chroma)
    surviving: Dict[str, np.ndarray] = {}

    for cand_id, cand_chroma in candidates.items():
        cand_vec = compute_mean_chroma_vector(cand_chroma)
        cos_sim = cosine_similarity_vectors(query_vec, cand_vec)
        if cos_sim >= threshold:
            surviving[cand_id] = cand_chroma

    after_count = len(surviving)
    if before_count == 0 or after_count >= before_count:
        percentage_removed = 0.0
    else:
        percentage_removed = float(((before_count - after_count) / before_count) * 100.0)

    if abs(percentage_removed) < 1e-6:
        percentage_removed = 0.0

    stats = CosinePrefilterStats(
        before_count=before_count,
        after_count=after_count,
        percentage_removed=round(percentage_removed, 2),
    )
    return surviving, stats


def dtw_align_detailed(chroma_a: np.ndarray, chroma_b: np.ndarray) -> DTWResult:
    """
    Perform key-invariant Dynamic Time Warping and return full alignment details.

    Evaluates all 12 circular pitch shifts of `chroma_b` along the pitch axis,
    identifies the shift yielding the minimum DTW alignment cost, performs +-1
    adjacent-shift boundary verification, and computes per-step local distances
    along the optimal warping path.

    Normalization formula:
        normalized_dist = min_distance / max(len(best_path), 1)
        similarity = 1.0 / (1.0 + normalized_dist)

    Parameters
    ----------
    chroma_a : np.ndarray
        Chroma matrix for track A, shape (12, n_frames_a).
    chroma_b : np.ndarray
        Chroma matrix for track B, shape (12, n_frames_b).

    Returns
    -------
    DTWResult
        Structured outcome containing similarity score, normalized distance,
        best shift, alignment path, local costs, path length, and boundary check.
    """
    if chroma_a.ndim != 2 or chroma_b.ndim != 2:
        raise ValueError("Chroma matrices must be 2-dimensional.")

    # Ensure shape is (12, n_frames)
    if chroma_a.shape[0] != 12 and chroma_a.shape[1] == 12:
        chroma_a = chroma_a.T
    if chroma_b.shape[0] != 12 and chroma_b.shape[1] == 12:
        chroma_b = chroma_b.T

    if chroma_a.shape[0] != 12 or chroma_b.shape[0] != 12:
        raise ValueError("Chroma matrices must have 12 pitch classes along the pitch axis.")

    # Transpose to (n_frames, 12) for fastdtw sequence vector comparisons
    seq_a = chroma_a.T

    min_distance = float("inf")
    best_path: List[Tuple[int, int]] = []
    best_shift = 0
    best_seq_b = chroma_b.T
    all_shift_costs: Dict[int, float] = {}

    # Evaluate all 12 circular shifts along the pitch axis (axis 0 of original chroma)
    for shift in range(12):
        shifted_b = np.roll(chroma_b, shift, axis=0)
        seq_b = shifted_b.T

        dist, path = fastdtw(seq_a, seq_b, dist=euclidean)
        all_shift_costs[shift] = float(dist)

        if dist < min_distance:
            min_distance = dist
            best_path = path
            best_shift = shift
            best_seq_b = seq_b

    # Path length normalization
    path_len = max(len(best_path), 1)
    normalized_dist = min_distance / path_len
    similarity = float(1.0 / (1.0 + normalized_dist))
    # Guarantee finite output in [0, 1]
    similarity = float(np.clip(similarity, 0.0, 1.0))

    # Compute per-step local cost vector along the best alignment path
    local_costs = np.array(
        [euclidean(seq_a[i], best_seq_b[j]) for i, j in best_path],
        dtype=float,
    )

    # Perform +-1 adjacent-shift boundary verification around the best shift
    prev_shift = (best_shift - 1) % 12
    next_shift = (best_shift + 1) % 12
    adjacent_costs = {
        prev_shift: all_shift_costs.get(prev_shift, float("inf")),
        next_shift: all_shift_costs.get(next_shift, float("inf")),
    }
    boundary_verified = bool(
        min_distance <= adjacent_costs[prev_shift] and min_distance <= adjacent_costs[next_shift]
    )

    return DTWResult(
        similarity=similarity,
        normalized_distance=float(normalized_dist),
        raw_distance=float(min_distance),
        best_shift=int(best_shift),
        alignment_path=best_path,
        local_costs=local_costs,
        path_length=path_len,
        adjacent_shift_costs=adjacent_costs,
        boundary_verified=boundary_verified,
    )


def dtw_similarity(chroma_a: np.ndarray, chroma_b: np.ndarray) -> float:
    """
    Compute key-invariant DTW similarity between two chroma matrices.

    Parameters
    ----------
    chroma_a : np.ndarray
        Chroma matrix for first audio sequence (12, n_frames_a).
    chroma_b : np.ndarray
        Chroma matrix for second audio sequence (12, n_frames_b).

    Returns
    -------
    float
        Normalized similarity score in [0.0, 1.0].
    """
    return dtw_align_detailed(chroma_a, chroma_b).similarity
