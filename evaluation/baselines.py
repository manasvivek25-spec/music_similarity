"""
Comparative Baselines for Offline Evaluation.
Implements the required baselines:
- Majority baseline
- Metadata-only baseline
- Mean-chroma cosine only (no DTW, no pitch-shift search)
- DTW without 12-shift search (shift = 0 only)
- Fixed similarity cutoff (fixed threshold without statistical calibration)
- F4 Classifiers: Logistic Regression (LR), LinearSVC, and Random Forest (RF)
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

from src.similarity import DTWResult


def baseline_mean_chroma_cosine(chroma_a: np.ndarray, chroma_b: np.ndarray) -> float:
    """
    Baseline 1: Mean-chroma cosine similarity only.
    Averages chroma across all time frames into a 12-dim vector and computes cosine similarity.
    No time-warping alignment (no DTW) and no 12-semitone pitch shift search.
    """
    if chroma_a.shape[0] != 12 or chroma_b.shape[0] != 12:
        return 0.0

    mean_a = np.mean(chroma_a, axis=1)
    mean_b = np.mean(chroma_b, axis=1)

    norm_a = np.linalg.norm(mean_a)
    norm_b = np.linalg.norm(mean_b)

    if norm_a < 1e-9 or norm_b < 1e-9:
        return 0.0

    cos_sim = float(np.dot(mean_a, mean_b) / (norm_a * norm_b))
    return max(0.0, min(1.0, (cos_sim + 1.0) / 2.0))  # Map [-1, 1] to [0, 1]


def baseline_dtw_no_shift(chroma_a: np.ndarray, chroma_b: np.ndarray) -> DTWResult:
    """
    Baseline 2: FastDTW without the 12-semitone circular pitch-shift search (shift = 0 only).
    Evaluates alignment strictly at nominal pitch.
    """
    seq_a = chroma_a.T
    seq_b = chroma_b.T

    distance, path = fastdtw(seq_a, seq_b, dist=euclidean)
    norm_dist = distance / max(len(path), 1)
    similarity = float(1.0 / (1.0 + norm_dist))

    local_costs = [float(euclidean(seq_a[i], seq_b[j])) for i, j in path]

    return DTWResult(
        similarity=round(similarity, 4),
        normalized_distance=round(norm_dist, 4),
        raw_distance=round(distance, 4),
        best_shift=0,
        alignment_path=path,
        local_costs=np.array(local_costs, dtype=np.float32),
        path_length=len(path),
    )


def baseline_fixed_similarity_decision(
    similarity: float,
    threshold: float = 0.65,
) -> Dict[str, Any]:
    """
    Baseline 3: Fixed similarity cutoff without empirical null distribution calibration.
    Declares significance purely if raw similarity exceeds static threshold.
    """
    is_sig = bool(similarity >= threshold)
    return {
        "similarity": similarity,
        "threshold": threshold,
        "is_significant": is_sig,
        "rule": f"Similarity >= {threshold}",
    }


def baseline_majority_classifier(y_train: np.ndarray) -> int:
    """
    Baseline 4: Majority class predictor.
    """
    vals, counts = np.unique(y_train, return_counts=True)
    return int(vals[np.argmax(counts)])


def baseline_metadata_match(
    query_genre: Optional[str],
    candidate_genre: Optional[str],
    query_artist: Optional[str] = None,
    candidate_artist: Optional[str] = None,
) -> float:
    """
    Baseline 5: Metadata-only matching score based on genre and artist equality.
    """
    score = 0.0
    if query_genre and candidate_genre and query_genre.lower() == candidate_genre.lower():
        score += 0.5
    if query_artist and candidate_artist and query_artist.lower() == candidate_artist.lower():
        score += 0.5
    return score


def train_f4_baseline_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Baseline 6: Train standard comparison models for Feature 4 Origin Classification:
    - Logistic Regression (LR)
    - Linear Support Vector Classifier (LinearSVC)
    - Random Forest (RF)
    """
    models = {
        "lr": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=seed)),
        ]),
        "linearsvc": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LinearSVC(dual=False, random_state=seed)),
        ]),
        "rf": Pipeline([
            ("clf", RandomForestClassifier(n_estimators=100, random_state=seed)),
        ]),
    }

    fitted = {}
    for name, pipe in models.items():
        pipe.fit(X_train, y_train)
        fitted[name] = pipe

    return fitted
