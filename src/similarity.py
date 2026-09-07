"""
Music similarity computation module.

Provides sequence alignment and similarity scoring between chroma representations,
accounting for key transpositions via circular pitch shifting.
"""

import numpy as np
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw


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
    best_path_len = 1

    # Evaluate all 12 circular shifts along the pitch axis (axis 0 of original chroma)
    for shift in range(12):
        shifted_b = np.roll(chroma_b, shift, axis=0)
        seq_b = shifted_b.T

        dist, path = fastdtw(seq_a, seq_b, dist=euclidean)
        if dist < min_distance:
            min_distance = dist
            best_path_len = len(path)

    # Normalize by alignment path length to be invariant to track duration
    normalized_dist = min_distance / max(best_path_len, 1)

    # Convert distance to similarity score in [0, 1]
    similarity = 1.0 / (1.0 + normalized_dist)
    return float(similarity)
