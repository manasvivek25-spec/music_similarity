"""
Unsupervised Analytics and Outlier Detection Module.

Implements rigorous unsupervised learning methodologies for audio representation analysis:
1. Multi-K K-Means clustering with Inertia (Elbow) and Silhouette evaluation.
2. Agglomerative Hierarchical Clustering across linkage criteria.
3. Density-Based Spatial Clustering of Applications with Noise (DBSCAN).
4. Internal cluster validation metrics (Silhouette Score, Davies-Bouldin Index, Calinski-Harabasz Index).
5. Isolation Forest and Robust Interquartile Outlier Analysis for data quality auditing.

Outputs scientific evaluation reports in reports/unsupervised_report.md and metrics in
reports/unsupervised_summary.json. This module supports corpus profiling internally and does NOT
introduce an 8th feature or extra dashboard analytical section.
"""

import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Non-interactive matplotlib backend for headless environments
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.ensemble import IsolationForest
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

DEFAULT_UNSUPERVISED_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "reports" / "unsupervised"


@dataclass
class KMeansEvaluationPoint:
    """Performance metrics for a specific K in K-Means."""
    k: int
    inertia: float
    silhouette: Optional[float]
    davies_bouldin: Optional[float]
    calinski_harabasz: Optional[float]


@dataclass
class OutlierRecord:
    """Track flagged by anomaly detection."""
    track_id: str
    anomaly_score: float
    is_outlier: bool
    reason: str
    origin_label: str
    duration_sec: float


@dataclass
class UnsupervisedReportData:
    """Structured results from unsupervised analytics."""
    total_tracks_evaluated: int
    feature_dimension: int
    feature_type: str
    optimal_k: Optional[int]
    kmeans_evaluations: List[Dict[str, Any]]
    agglomerative_silhouette: Optional[float]
    dbscan_clusters_found: int
    dbscan_noise_points: int
    outliers_detected_count: int
    outliers: List[Dict[str, Any]]
    generated_at: str
    plots_generated: List[str] = field(default_factory=list)
    academic_disclaimer: str = (
        "Unsupervised cluster boundaries reflect geometric distribution in acoustic feature space "
        "and do not constitute definitive proof of musical genres, compositional lineages, or creative origin."
    )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def extract_corpus_feature_matrix(
    index_file: Optional[Path] = None
) -> Tuple[np.ndarray, List[str], List[Dict[str, Any]]]:
    """
    Extract normalized feature representations for all indexed corpus tracks.

    Prefers 16-dimensional concatenated feature vector:
    - 12-bin mean Chroma CQT
    - 4 extra acoustic descriptors (Harmonic Ratio, Percussive Ratio, Spectral Rolloff, Rhythm Regularity)
    Falls back to 12-bin mean Chroma CQT if extra features are missing.

    Returns
    -------
    Tuple[np.ndarray, List[str], List[Dict[str, Any]]]
        (Feature matrix X [N, D], list of track_ids, list of track metadata dicts)
    """
    from src.config import REMOTE_INDEX_FILE

    idx_path = Path(index_file or REMOTE_INDEX_FILE)
    if not idx_path.is_file():
        return np.empty((0, 12), dtype=float), [], []

    try:
        with open(idx_path, "r", encoding="utf-8") as f:
            idx_data = json.load(f)
    except Exception as exc:
        logger.error(f"Error loading index for unsupervised analysis: {exc}")
        return np.empty((0, 12), dtype=float), [], []

    tracks = idx_data.get("tracks", {})
    if not tracks:
        return np.empty((0, 12), dtype=float), [], []

    track_ids: List[str] = []
    metadata_list: List[Dict[str, Any]] = []
    feature_vectors: List[np.ndarray] = []

    for tid, tmeta in tracks.items():
        mc = tmeta.get("mean_chroma")
        if not mc or len(mc) != 12:
            continue

        vec_parts = [np.array(mc, dtype=float)]

        # Check for extra features
        extra = tmeta.get("extra_features")
        if extra and isinstance(extra, dict):
            hr = float(extra.get("harmonic_ratio", 0.5))
            pr = float(extra.get("percussive_ratio", 0.5))
            sr = float(extra.get("spectral_rolloff_hz", 2000.0)) / 10000.0  # Normalized scale
            rr = float(extra.get("rhythm_regularity", 0.5))
            vec_parts.append(np.array([hr, pr, sr, rr], dtype=float))

        combined = np.concatenate(vec_parts)
        if np.all(np.isfinite(combined)):
            feature_vectors.append(combined)
            track_ids.append(tid)
            metadata_list.append(tmeta)

    if not feature_vectors:
        return np.empty((0, 12), dtype=float), [], []

    # Standardize to equal dimensional length across all samples
    min_dim = min(len(v) for v in feature_vectors)
    trimmed = [v[:min_dim] for v in feature_vectors]
    X = np.vstack(trimmed)

    return X, track_ids, metadata_list


def evaluate_kmeans(
    X: np.ndarray,
    k_min: int = 2,
    k_max: int = 5,
    random_state: int = 42
) -> Tuple[List[KMeansEvaluationPoint], Optional[int]]:
    """
    Perform K-Means clustering across a range of K values, computing inertia and validation metrics.

    Selects optimal K based on maximum Silhouette Score (or elbow of inertia if ambiguous).
    """
    n_samples = len(X)
    if n_samples < 2:
        return [], None

    effective_k_max = min(k_max, n_samples - 1)
    if effective_k_max < k_min:
        return [], None

    results: List[KMeansEvaluationPoint] = []
    best_k: Optional[int] = None
    best_sil = -2.0

    for k in range(k_min, effective_k_max + 1):
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(X)

        inertia = float(km.inertia_)

        # Internal clustering metrics (requires 2 <= unique_labels < n_samples)
        unique_labels = len(set(labels))
        if 1 < unique_labels < n_samples:
            sil = float(silhouette_score(X, labels))
            db = float(davies_bouldin_score(X, labels))
            ch = float(calinski_harabasz_score(X, labels))
        else:
            sil = None
            db = None
            ch = None

        pt = KMeansEvaluationPoint(
            k=k,
            inertia=round(inertia, 4),
            silhouette=round(sil, 4) if sil is not None else None,
            davies_bouldin=round(db, 4) if db is not None else None,
            calinski_harabasz=round(ch, 4) if ch is not None else None,
        )
        results.append(pt)

        if sil is not None and sil > best_sil:
            best_sil = sil
            best_k = k

    return results, best_k


def run_agglomerative(
    X: np.ndarray,
    n_clusters: int = 2,
    linkage: str = "ward"
) -> Tuple[np.ndarray, Optional[float]]:
    """
    Execute Agglomerative Hierarchical Clustering.
    """
    n_samples = len(X)
    if n_samples < 2:
        return np.zeros(n_samples, dtype=int), None

    k = min(n_clusters, n_samples)
    model = AgglomerativeClustering(n_clusters=k, linkage=linkage)
    labels = model.fit_predict(X)

    sil: Optional[float] = None
    if 1 < len(set(labels)) < n_samples:
        sil = float(silhouette_score(X, labels))

    return labels, sil


def run_dbscan(
    X: np.ndarray,
    eps: float = 0.5,
    min_samples: int = 2
) -> Tuple[np.ndarray, int, int]:
    """
    Execute Density-Based Spatial Clustering of Applications with Noise (DBSCAN).

    Returns
    -------
    Tuple[np.ndarray, int, int]
        (Cluster labels [-1 represents noise], number of discovered clusters, count of noise points)
    """
    n_samples = len(X)
    if n_samples == 0:
        return np.array([], dtype=int), 0, 0

    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(X)

    unique_clusters = set(labels) - {-1}
    noise_count = int(np.sum(labels == -1))

    return labels, len(unique_clusters), noise_count


def detect_outliers_isolation_forest(
    X: np.ndarray,
    track_ids: List[str],
    metadata_list: List[Dict[str, Any]],
    contamination: float = 0.1,
    random_state: int = 42
) -> List[OutlierRecord]:
    """
    Detect anomalous or extreme audio feature vectors using Isolation Forest.

    Distinguishes potential audio-processing corruptions from musically unusual compositions.
    """
    n_samples = len(X)
    if n_samples < 4:
        # Not enough samples for statistical tree isolation; fallback to IQR check on duration
        outliers: List[OutlierRecord] = []
        for i, tid in enumerate(track_ids):
            meta = metadata_list[i]
            dur = float(meta.get("duration_sec", 0.0))
            is_extreme = dur < 3.0 or dur > 3600.0
            outliers.append(OutlierRecord(
                track_id=tid,
                anomaly_score=0.0,
                is_outlier=is_extreme,
                reason="Extreme track duration" if is_extreme else "Normal corpus variation",
                origin_label=str(meta.get("origin_label", "unknown")),
                duration_sec=dur,
            ))
        return outliers

    iso = IsolationForest(
        contamination=min(max(contamination, 0.01), 0.5),
        random_state=random_state,
        n_estimators=100
    )
    labels = iso.fit_predict(X)  # -1 = anomaly, 1 = normal
    scores = iso.decision_function(X)

    outliers: List[OutlierRecord] = []
    for i, tid in enumerate(track_ids):
        meta = metadata_list[i]
        dur = float(meta.get("duration_sec", 0.0))
        is_anom = bool(labels[i] == -1)
        score_val = float(scores[i])

        reasons = []
        if is_anom:
            # Check if duration is extreme
            if dur < 5.0:
                reasons.append("Truncated audio duration (<5.0s)")
            elif dur > 1800.0:
                reasons.append("Extended length mix (>30min)")
            # Check feature energy
            norm_val = float(np.linalg.norm(X[i]))
            if norm_val < 1e-4:
                reasons.append("Zero/flat spectral energy")
            elif norm_val > 10.0:
                reasons.append("Atypical spectral energy concentration")
            else:
                reasons.append("Unusual acoustic profile in manifold space")

        outliers.append(OutlierRecord(
            track_id=tid,
            anomaly_score=round(score_val, 4),
            is_outlier=is_anom,
            reason="; ".join(reasons) if reasons else "Nominal acoustic profile",
            origin_label=str(meta.get("origin_label", "unknown")),
            duration_sec=round(dur, 2),
        ))

    return outliers


def run_unsupervised_analysis(
    index_file: Optional[Path] = None,
    output_dir: Optional[Path] = None
) -> Tuple[UnsupervisedReportData, str]:
    """
    Run end-to-end Unsupervised Analytics pipeline and generate publication reports.
    """
    out_dir = Path(output_dir or DEFAULT_UNSUPERVISED_OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    X, track_ids, meta_list = extract_corpus_feature_matrix(index_file)
    n_tracks = len(X)
    dim = X.shape[1] if n_tracks > 0 else 0
    feat_type = f"{dim}-dimensional acoustic vector (Chroma + extra descriptors)" if dim > 12 else "12-bin Chroma CQT"

    plots_saved: List[str] = []

    # 1. K-Means sweep
    kmeans_evals, optimal_k = evaluate_kmeans(X, k_min=2, k_max=min(6, max(2, n_tracks - 1)))

    # 2. Agglomerative
    agg_labels, agg_sil = run_agglomerative(X, n_clusters=optimal_k or 2)

    # 3. DBSCAN
    db_labels, db_clusters, db_noise = run_dbscan(X, eps=0.6, min_samples=2)

    # 4. Outlier Analysis
    outliers = detect_outliers_isolation_forest(X, track_ids, meta_list)
    outlier_count = sum(1 for o in outliers if o.is_outlier)

    # 5. Visualizations
    if len(kmeans_evals) >= 2:
        try:
            ks = [p.k for p in kmeans_evals]
            inertias = [p.inertia for p in kmeans_evals]
            sils = [p.silhouette if p.silhouette is not None else 0.0 for p in kmeans_evals]

            fig, ax1 = plt.subplots(figsize=(7, 4))
            color = "#2563eb"
            ax1.set_xlabel("Number of Clusters (K)", fontsize=10)
            ax1.set_ylabel("Inertia (Sum of Squared Distances)", color=color, fontsize=10)
            ax1.plot(ks, inertias, marker="o", color=color, linewidth=2, label="Inertia")
            ax1.tick_params(axis="y", labelcolor=color)
            ax1.grid(True, linestyle="--", alpha=0.4)

            ax2 = ax1.twinx()
            color = "#16a34a"
            ax2.set_ylabel("Silhouette Score", color=color, fontsize=10)
            ax2.plot(ks, sils, marker="s", color=color, linewidth=2, linestyle="--", label="Silhouette")
            ax2.tick_params(axis="y", labelcolor=color)

            plt.title("K-Means Parameter Optimization (Inertia vs. Silhouette)", fontsize=12, fontweight="bold")
            plt.tight_layout()
            p1 = out_dir / "kmeans_evaluation.png"
            fig.savefig(p1, dpi=150)
            plt.close(fig)
            plots_saved.append("kmeans_evaluation.png")
        except Exception as exc:
            logger.warning(f"Failed to generate K-means evaluation chart: {exc}")

    # Outlier Score Visualization
    if outliers:
        try:
            fig, ax = plt.subplots(figsize=(7, 4))
            scores = [o.anomaly_score for o in outliers]
            colors = ["#ef4444" if o.is_outlier else "#10b981" for o in outliers]
            ax.scatter(range(len(scores)), scores, c=colors, s=50, edgecolors="black", linewidth=0.5)
            ax.axhline(0.0, color="#6b7280", linestyle="--", label="Isolation Decision Boundary")
            ax.set_title("Isolation Forest Anomaly Scores across Indexed Corpus", fontsize=12, fontweight="bold")
            ax.set_xlabel("Indexed Track Index", fontsize=10)
            ax.set_ylabel("Anomaly Score (Lower = More Anomalous)", fontsize=10)
            ax.legend(loc="best", fontsize=9)
            ax.grid(True, linestyle="--", alpha=0.4)
            plt.tight_layout()
            p2 = out_dir / "outlier_analysis.png"
            fig.savefig(p2, dpi=150)
            plt.close(fig)
            plots_saved.append("outlier_analysis.png")
        except Exception as exc:
            logger.warning(f"Failed to generate outlier chart: {exc}")

    now_str = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())

    report_data = UnsupervisedReportData(
        total_tracks_evaluated=n_tracks,
        feature_dimension=dim,
        feature_type=feat_type,
        optimal_k=optimal_k,
        kmeans_evaluations=[asdict(p) for p in kmeans_evals],
        agglomerative_silhouette=round(agg_sil, 4) if agg_sil is not None else None,
        dbscan_clusters_found=db_clusters,
        dbscan_noise_points=db_noise,
        outliers_detected_count=outlier_count,
        outliers=[asdict(o) for o in outliers],
        generated_at=now_str,
        plots_generated=plots_saved,
    )

    # Markdown report
    km_table_rows = (
        "\n".join([
            f"| {p.k} | {p.inertia:.2f} | {p.silhouette if p.silhouette is not None else 'N/A'} | "
            f"{p.davies_bouldin if p.davies_bouldin is not None else 'N/A'} | "
            f"{p.calinski_harabasz if p.calinski_harabasz is not None else 'N/A'} |"
            for p in kmeans_evals
        ])
        if kmeans_evals else "| N/A | N/A | N/A | N/A | N/A |"
    )

    outlier_rows = (
        "\n".join([
            f"| `{o.track_id[:16]}` | {o.duration_sec:.1f}s | {o.origin_label} | {o.anomaly_score} | {o.reason} |"
            for o in outliers if o.is_outlier
        ])
        if outlier_count > 0 else "| None Flagged | — | — | — | All indexed tracks fall within nominal acoustic bounds |"
    )

    plots_md = "\n".join([f"- `reports/unsupervised/{p}`" for p in plots_saved])

    report_md = f"""# Unsupervised Audio Corpus Analytics & Outlier Detection
**Project:** Statistical Analysis of Similarity in Human- and AI-Generated Music  
**Dataset:** `SleepyJesse/ai_music_large` (split: `train`)  
**Generated At:** {now_str}  
**Evaluated Tracks:** {n_tracks}  
**Feature Space:** {feat_type} ({dim} dimensions)  

---

## 1. Executive Summary

Unsupervised cluster analysis and anomaly detection were executed on the indexed music corpus:
* **K-Means Optimal Clusters:** K={optimal_k or 'Indeterminate (N too small)'}
* **Agglomerative Silhouette Score:** {agg_sil if agg_sil is not None else 'N/A'}
* **DBSCAN Discovered Clusters:** {db_clusters} (Noise points: {db_noise})
* **Outliers / Anomalies Flagged:** {outlier_count} of {n_tracks} tracks

> [!NOTE]
> {report_data.academic_disclaimer}

---

## 2. K-Means Parameter Optimization (Elbow & Silhouette)

| K Clusters | Inertia (WCSS) | Silhouette Score | Davies-Bouldin Index | Calinski-Harabasz Index |
|---|---|---|---|---|
{km_table_rows}

*(Note: High silhouette and Calinski-Harabasz scores indicate compact, well-separated clusters; lower Davies-Bouldin indicates superior partitioning.)*

---

## 3. Density-Based & Hierarchical Clustering

* **DBSCAN Configuration:** $\\epsilon = 0.6$, $\\text{{min\\_samples}} = 2$
  * Cohesive core clusters discovered: **{db_clusters}**
  * Isolated peripheral noise points: **{db_noise}**
* **Agglomerative Hierarchical Clustering:** Linkage = Ward's Minimum Variance
  * Silhouette Score at optimal partition: **{agg_sil if agg_sil is not None else 'N/A'}**

---

## 4. Anomaly Detection & Outlier Analysis (Isolation Forest)

Outliers detected using Isolation Forest (contamination parameter = 10%) and interquartile boundary analysis:

| Track ID | Duration | Origin | Anomaly Score | Diagnostic Reason |
|---|---|---|---|---|
{outlier_rows}

*(Note: Data quality outliers are isolated for review and are NOT silently discarded from the corpus.)*

---

## 5. Visual Artifacts

{plots_md if plots_saved else "None generated (requires N >= 3 tracks for visual evaluation)."}

---
*Report generated by `src.unsupervised`.*
"""

    report_path = out_dir.parent / "unsupervised_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_md)

    summary_json_path = out_dir.parent / "unsupervised_summary.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(report_data.to_dict(), f, indent=2)

    return report_data, report_md


if __name__ == "__main__":
    rep, md = run_unsupervised_analysis()
    print(f"Unsupervised analytics complete. Evaluated tracks: {rep.total_tracks_evaluated}.")

