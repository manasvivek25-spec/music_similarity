"""
Subsequence Dynamic Time Warping (SS-DTW) Module.

Provides an independent research experiment for detecting local melodic/harmonic
matches within longer audio tracks without global endpoint constraints.

Algorithm (Open-End Subsequence Alignment):
- Free start: The alignment can begin at any candidate frame j in [0, M-1] with zero cost.
- Cumulative cost matrix D[i, j] accumulates Euclidean step distances for query frame i in [0, N-1].
- The optimal matched subsequence ends at j_opt = argmin_j D[N-1, j].
- Warping path is traced back from (N-1, j_opt) until i reaches 0 at j_start.
- Tested across all 12 pitch shifts (semitone cyclic shifts) for key invariance.

Note:
This module is strictly an experimental research comparison and does NOT replace
or modify the locked global DTW alignment in src/similarity.py.
"""

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class SubsequenceDTWResult:
    """Outcome of Subsequence DTW alignment."""
    similarity: float
    normalized_distance: float
    raw_distance: float
    best_shift: int
    query_start_frame: int
    query_end_frame: int
    candidate_start_frame: int
    candidate_end_frame: int
    alignment_path: List[Tuple[int, int]]
    path_length: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def subsequence_dtw_single_shift(
    query_chroma: np.ndarray,
    candidate_chroma: np.ndarray,
) -> Tuple[float, float, int, int, List[Tuple[int, int]]]:
    """
    Execute open-end subsequence DTW for a single fixed pitch shift orientation.

    Parameters
    ----------
    query_chroma : np.ndarray
        Shape (12, N) query chroma matrix.
    candidate_chroma : np.ndarray
        Shape (12, M) candidate chroma matrix (M >= N typically).

    Returns
    -------
    Tuple
        (normalized_dist, raw_dist, start_cand_idx, end_cand_idx, warping_path)
    """
    N = query_chroma.shape[1]
    M = candidate_chroma.shape[1]

    if N == 0 or M == 0:
        return float("inf"), float("inf"), 0, 0, []

    # Pairwise Euclidean distances: C[i, j] = ||q[:, i] - c[:, j]||
    diff = query_chroma[:, :, np.newaxis] - candidate_chroma[:, np.newaxis, :]
    C = np.sqrt(np.sum(diff ** 2, axis=0))  # Shape (N, M)

    # Initialize accumulated cost matrix D
    D = np.full((N, M), np.inf, dtype=np.float64)

    # Free start condition: first query frame can align with any candidate frame
    D[0, :] = C[0, :]

    # DP recurrence: Step pattern (1,0), (0,1), (1,1)
    for i in range(1, N):
        for j in range(M):
            cost_prev = D[i - 1, j]  # vertical
            if j > 0:
                cost_prev = min(cost_prev, D[i, j - 1], D[i - 1, j - 1])  # horizontal, diagonal
            D[i, j] = C[i, j] + cost_prev

    # Find optimal endpoint at the end of the query (i = N - 1)
    j_opt = int(np.argmin(D[N - 1, :]))
    raw_dist = float(D[N - 1, j_opt])

    # Traceback optimal warping path from (N-1, j_opt) back to i=0
    path = [(N - 1, j_opt)]
    curr_i, curr_j = N - 1, j_opt

    while curr_i > 0:
        candidates = []
        # Diagonal (i-1, j-1)
        if curr_j > 0:
            candidates.append((D[curr_i - 1, curr_j - 1], curr_i - 1, curr_j - 1))
        # Vertical (i-1, j)
        candidates.append((D[curr_i - 1, curr_j], curr_i - 1, curr_j))
        # Horizontal (i, j-1)
        if curr_j > 0:
            candidates.append((D[curr_i, curr_j - 1], curr_i, curr_j - 1))

        # Sort by lowest cumulative cost
        candidates.sort(key=lambda x: x[0])
        _, next_i, next_j = candidates[0]
        path.append((next_i, next_j))
        curr_i, curr_j = next_i, next_j

    path.reverse()
    path_len = len(path)
    norm_dist = raw_dist / max(path_len, 1)
    cand_start = path[0][1]
    cand_end = path[-1][1]

    return norm_dist, raw_dist, cand_start, cand_end, path


def subsequence_dtw_align(
    query_chroma: np.ndarray,
    candidate_chroma: np.ndarray,
) -> SubsequenceDTWResult:
    """
    Subsequence DTW alignment evaluated over all 12 pitch shifts for key invariance.

    Parameters
    ----------
    query_chroma : np.ndarray
        Shape (12, N) query chroma matrix.
    candidate_chroma : np.ndarray
        Shape (12, M) candidate chroma matrix.

    Returns
    -------
    SubsequenceDTWResult
        Best semitone shift and optimal localized match segment.
    """
    best_shift = 0
    min_norm_dist = float("inf")
    best_raw = float("inf")
    best_start = 0
    best_end = 0
    best_path: List[Tuple[int, int]] = []

    for shift in range(12):
        shifted_cand = np.roll(candidate_chroma, shift, axis=0)
        norm_d, raw_d, c_start, c_end, path = subsequence_dtw_single_shift(
            query_chroma, shifted_cand
        )
        if norm_d < min_norm_dist:
            min_norm_dist = norm_d
            best_raw = raw_d
            best_shift = shift
            best_start = c_start
            best_end = c_end
            best_path = path

    sim = 1.0 / (1.0 + min_norm_dist) if np.isfinite(min_norm_dist) else 0.0

    return SubsequenceDTWResult(
        similarity=round(float(sim), 4),
        normalized_distance=round(float(min_norm_dist), 4),
        raw_distance=round(float(best_raw), 4),
        best_shift=best_shift,
        query_start_frame=0,
        query_end_frame=query_chroma.shape[1] - 1 if query_chroma.shape[1] > 0 else 0,
        candidate_start_frame=best_start,
        candidate_end_frame=best_end,
        alignment_path=best_path,
        path_length=len(best_path),
    )

