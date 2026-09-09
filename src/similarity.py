"""
Music similarity computation module.

Provides sequence alignment and similarity scoring between chroma representations,
accounting for key transpositions via circular pitch shifting.
Exposes both overall similarity scores (for hypothesis testing and ranking) and
fine-grained DTW alignment details (paths and local step costs for segment localization).
"""

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw


@dataclass
class DTWResult:
    """
    Structured outcome of key-invariant DTW alignment.

    Attributes
    ----------
    similarity : float
        Normalized similarity score in [0, 1].
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
    """
    similarity: float
    normalized_distance: float
    raw_distance: float
    best_shift: int
    alignment_path: List[Tuple[int, int]]
    local_costs: np.ndarray


def dtw_align_detailed(chroma_a: np.ndarray, chroma_b: np.ndarray) -> DTWResult:
    """
    Perform key-invariant Dynamic Time Warping and return full alignment details.

    Evaluates all 12 circular pitch shifts of `chroma_b` along the pitch axis,
    identifies the shift yielding the minimum DTW alignment cost, and computes
    per-step local distances along the optimal warping path.

    Parameters
    ----------
    chroma_a : np.ndarray
        Chroma matrix for track A, shape (12, n_frames_a).
    chroma_b : np.ndarray
        Chroma matrix for track B, shape (12, n_frames_b).

    Returns
    -------
    DTWResult
        Dataclass containing similarity score, normalized distance, raw distance,
        best semitone shift, alignment path, and local step costs.
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

    # Evaluate all 12 circular shifts along the pitch axis (axis 0 of original chroma)
    for shift in range(12):
        shifted_b = np.roll(chroma_b, shift, axis=0)
        seq_b = shifted_b.T

        dist, path = fastdtw(seq_a, seq_b, dist=euclidean)
        if dist < min_distance:
            min_distance = dist
            best_path = path
            best_shift = shift
            best_seq_b = seq_b

    # Path length normalization
    path_len = max(len(best_path), 1)
    normalized_dist = min_distance / path_len
    similarity = float(1.0 / (1.0 + normalized_dist))

    # Compute per-step local cost vector along the best alignment path
    local_costs = np.array(
        [euclidean(seq_a[i], best_seq_b[j]) for i, j in best_path],
        dtype=float,
    )

    return DTWResult(
        similarity=similarity,
        normalized_distance=float(normalized_dist),
        raw_distance=float(min_distance),
        best_shift=int(best_shift),
        alignment_path=best_path,
        local_costs=local_costs,
    )


def dtw_similarity(chroma_a: np.ndarray, chroma_b: np.ndarray) -> float:
    """
    Compute key-invariant DTW similarity between two chroma matrices.

    The function evaluates all 12 circular pitch shifts of `chroma_b` along
    the pitch axis (semitone transpositions), runs Dynamic Time Warping (DTW)
    for each shift, finds the minimum alignment cost, and converts the
    normalized alignment distance into a similarity score in [0, 1].

    Normalization formula:
        normalized_dist = min_distance / path_length
        similarity = 1.0 / (1.0 + normalized_dist)

    Identical chroma matrices produce an alignment distance of 0, yielding
    a similarity score of 1.0. Higher alignment distances result in lower
    similarity scores approaching 0.

    Parameters
    ----------
    chroma_a : np.ndarray
        Chroma matrix for the first audio sequence, shape (12, n_frames_a).
    chroma_b : np.ndarray
        Chroma matrix for the second audio sequence, shape (12, n_frames_b).

    Returns
    -------
    float
        Normalized similarity score in the interval [0, 1].
    """
    return dtw_align_detailed(chroma_a, chroma_b).similarity
