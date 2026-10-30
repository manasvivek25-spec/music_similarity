"""
Exploratory Data Analysis (EDA) Module.

Generates reproducible statistical summaries, distribution analyses,
and visualizations for the indexed audio corpus.
Saves publication-quality figures and an EDA markdown report in reports/eda/.
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

logger = logging.getLogger(__name__)

DEFAULT_EDA_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "reports" / "eda"


@dataclass
class EDASummary:
    """Structured metrics for dataset exploratory data analysis."""
    total_indexed_tracks: int
    human_tracks: int
    ai_tracks: int
    unknown_origin_tracks: int
    verified_genre_tracks: int
    estimated_genre_tracks: int
    unknown_genre_tracks: int
    duration_mean_sec: float
    duration_median_sec: float
    duration_min_sec: float
    duration_max_sec: float
    failed_or_unresolved_count: int
    missing_artist_count: int
    unique_artists_count: int
    duplicate_records_detected: int
    generated_at: str = ""
    plots_generated: List[str] = field(default_factory=list)
    limitations: str = (
        "PCA and UMAP projections are descriptive tools for visual data exploration only, "
        "not empirical proof of musical similarity, melodic copying, or plagiarism."
    )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def run_exploratory_data_analysis(
    index_file: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> Tuple[EDASummary, str]:
    """
    Execute end-to-end Exploratory Data Analysis on the current indexed corpus.

    Parameters
    ----------
    index_file : Optional[Path]
        Path to remote_index.json (defaults to standard location).
    output_dir : Optional[Path]
        Directory where charts and markdown report are saved.

    Returns
    -------
    Tuple[EDASummary, str]
        (Summary object, full markdown report string)
    """
    from src.config import FEATURES_DIR, REMOTE_INDEX_FILE

    idx_path = Path(index_file or REMOTE_INDEX_FILE)
    out_dir = Path(output_dir or DEFAULT_EDA_OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    tracks: Dict[str, Any] = {}
    unresolved: Dict[str, Any] = {}
    if idx_path.is_file():
        try:
            with open(idx_path, "r", encoding="utf-8") as f:
                idx_data = json.load(f)
            tracks = idx_data.get("tracks", {})
            unresolved = idx_data.get("unresolved_records", {})
        except Exception as exc:
            logger.error(f"Error reading index file for EDA: {exc}")

    total = len(tracks)

    # 1. Label distribution
    human_count = sum(1 for t in tracks.values() if t.get("origin_label") == "human")
    ai_count = sum(1 for t in tracks.values() if t.get("origin_label") == "ai_generated")
    unk_count = total - (human_count + ai_count)

    # 2. Duration distribution
    durations = [float(t.get("duration_sec", 0.0)) for t in tracks.values() if t.get("duration_sec") is not None]
    if durations:
        dur_mean = float(np.mean(durations))
        dur_median = float(np.median(durations))
        dur_min = float(np.min(durations))
        dur_max = float(np.max(durations))
    else:
        dur_mean = dur_median = dur_min = dur_max = 0.0

    # 3. Genre coverage (genuine vs estimated vs unknown)
    v_genres = sum(1 for t in tracks.values() if t.get("genre_source") == "metadata" and t.get("genre"))
    e_genres = sum(1 for t in tracks.values() if t.get("genre_source") == "estimated" and t.get("estimated_genre"))
    u_genres = total - (v_genres + e_genres)

    genuine_genres: Dict[str, int] = {}
    for t in tracks.values():
        if t.get("genre_source") == "metadata" and t.get("genre"):
            g = str(t["genre"]).title()
            genuine_genres[g] = genuine_genres.get(g, 0) + 1

    # 4. Artist coverage
    artists: Dict[str, int] = {}
    missing_artists = 0
    for t in tracks.values():
        art = t.get("raw_metadata", {}).get("artist") or t.get("artist")
        if art and str(art).strip().lower() not in ("unknown", "none", ""):
            a_clean = str(art).strip()
            artists[a_clean] = artists.get(a_clean, 0) + 1
        else:
            missing_artists += 1

    # 5. Duplicate analysis
    content_hashes: Dict[str, List[str]] = {}
    for tid, t in tracks.items():
        chash = t.get("content_hash") or tid
        content_hashes.setdefault(chash, []).append(tid)
    duplicates_detected = sum(len(ids) - 1 for ids in content_hashes.values() if len(ids) > 1)

    # 6. Feature vectors (mean chroma)
    chroma_vectors = []
    chroma_labels = []
    for tid, t in tracks.items():
        mc = t.get("mean_chroma")
        if mc and len(mc) == 12:
            chroma_vectors.append(mc)
            chroma_labels.append(t.get("origin_label", "unknown"))

    # Generate Visualizations
    plots_saved: List[str] = []

    # Chart 1: Origin Distribution
    try:
        fig, ax = plt.subplots(figsize=(6, 4))
        labels = ["Human", "AI-Generated", "Unknown"]
        counts = [human_count, ai_count, unk_count]
        colors = ["#10b981", "#ef4444", "#6b7280"]
        bars = ax.bar(labels, counts, color=colors, edgecolor="black", width=0.5)
        ax.set_title("Dataset Origin Label Distribution", fontsize=12, fontweight="bold")
        ax.set_ylabel("Track Count", fontsize=10)
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height}", xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=10)
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()
        plot_p = out_dir / "origin_distribution.png"
        fig.savefig(plot_p, dpi=150)
        plt.close(fig)
        plots_saved.append("origin_distribution.png")
    except Exception as exc:
        logger.warning(f"Failed to generate origin plot: {exc}")

    # Chart 2: Track Duration Distribution
    try:
        if len(durations) > 0:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.hist(durations, bins=min(15, max(3, len(durations))), color="#3b82f6", edgecolor="black", alpha=0.8)
            ax.set_title(f"Track Duration Distribution (N={len(durations)})", fontsize=12, fontweight="bold")
            ax.set_xlabel("Duration (seconds)", fontsize=10)
            ax.set_ylabel("Frequency", fontsize=10)
            ax.grid(axis="y", linestyle="--", alpha=0.5)
            plt.tight_layout()
            plot_p = out_dir / "duration_distribution.png"
            fig.savefig(plot_p, dpi=150)
            plt.close(fig)
            plots_saved.append("duration_distribution.png")
    except Exception as exc:
        logger.warning(f"Failed to generate duration plot: {exc}")

    # Chart 3: Mean Chroma Profile Across Pitch Classes
    try:
        if len(chroma_vectors) > 0:
            mean_c = np.mean(chroma_vectors, axis=0)
            pitch_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(pitch_names, mean_c, marker="o", color="#d97706", linewidth=2)
            ax.fill_between(pitch_names, mean_c, color="#fef3c7", alpha=0.5)
            ax.set_title(f"Average Harmonic Chroma Profile (N={len(chroma_vectors)} tracks)", fontsize=12, fontweight="bold")
            ax.set_xlabel("Pitch Class", fontsize=10)
            ax.set_ylabel("Mean Normalized Energy", fontsize=10)
            ax.set_ylim(0.0, 1.0)
            ax.grid(True, linestyle="--", alpha=0.5)
            plt.tight_layout()
            plot_p = out_dir / "chroma_feature_distributions.png"
            fig.savefig(plot_p, dpi=150)
            plt.close(fig)
            plots_saved.append("chroma_feature_distributions.png")
    except Exception as exc:
        logger.warning(f"Failed to generate chroma plot: {exc}")

    # Chart 4: PCA Projection of Chroma Space (Descriptive exploration only)
    try:
        if len(chroma_vectors) >= 2:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2, random_state=42)
            X_pca = pca.fit_transform(np.array(chroma_vectors))
            var_exp = pca.explained_variance_ratio_

            fig, ax = plt.subplots(figsize=(7, 5))
            color_map = {"human": "#10b981", "ai_generated": "#ef4444", "unknown": "#6b7280"}
            for lbl in set(chroma_labels):
                mask = [i for i, l in enumerate(chroma_labels) if l == lbl]
                ax.scatter(
                    X_pca[mask, 0], X_pca[mask, 1],
                    label=f"{lbl.title()} (N={len(mask)})",
                    color=color_map.get(lbl, "#6b7280"),
                    alpha=0.8, edgecolors="none", s=50
                )
            ax.set_title("PCA Projection of 12-Dimensional Mean Chroma Space", fontsize=12, fontweight="bold")
            ax.set_xlabel(f"PC 1 ({var_exp[0]*100:.1f}% variance)", fontsize=10)
            ax.set_ylabel(f"PC 2 ({var_exp[1]*100:.1f}% variance)", fontsize=10)
            ax.legend(loc="best", fontsize=9)
            ax.grid(True, linestyle="--", alpha=0.4)
            plt.figtext(0.5, 0.01,
                        "Note: PCA is for visual exploration only. Proximity does not prove melodic copying.",
                        wrap=True, horizontalalignment="center", fontsize=8, color="#555555")
            plt.tight_layout(rect=[0, 0.03, 1, 0.97])
            plot_p = out_dir / "pca_projection.png"
            fig.savefig(plot_p, dpi=150)
            plt.close(fig)
            plots_saved.append("pca_projection.png")
    except Exception as exc:
        logger.warning(f"Failed to generate PCA plot: {exc}")

    # Optional UMAP projection
    try:
        import umap
        if len(chroma_vectors) >= 5:
            reducer = umap.UMAP(n_neighbors=min(5, len(chroma_vectors)-1), random_state=42)
            X_umap = reducer.fit_transform(np.array(chroma_vectors))
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.scatter(X_umap[:, 0], X_umap[:, 1], c="#8b5cf6", alpha=0.8, s=40)
            ax.set_title("UMAP Manifold Projection of Harmonic Feature Space", fontsize=12, fontweight="bold")
            ax.grid(True, linestyle="--", alpha=0.4)
            plt.tight_layout()
            plot_p = out_dir / "umap_projection.png"
            fig.savefig(plot_p, dpi=150)
            plt.close(fig)
            plots_saved.append("umap_projection.png")
    except ImportError:
        logger.debug("umap-learn not installed; skipping optional UMAP projection.")
    except Exception as exc:
        logger.debug(f"UMAP projection skipped: {exc}")

    now_str = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())

    summary = EDASummary(
        total_indexed_tracks=total,
        human_tracks=human_count,
        ai_tracks=ai_count,
        unknown_origin_tracks=unk_count,
        verified_genre_tracks=v_genres,
        estimated_genre_tracks=e_genres,
        unknown_genre_tracks=u_genres,
        duration_mean_sec=round(dur_mean, 2),
        duration_median_sec=round(dur_median, 2),
        duration_min_sec=round(dur_min, 2),
        duration_max_sec=round(dur_max, 2),
        failed_or_unresolved_count=len(unresolved),
        missing_artist_count=missing_artists,
        unique_artists_count=len(artists),
        duplicate_records_detected=duplicates_detected,
        generated_at=now_str,
        plots_generated=plots_saved,
    )

    # Markdown Report Generation
    genre_rows = (
        "\n".join([f"| {g} | {c} | {c/max(total, 1)*100:.1f}% |" for g, c in sorted(genuine_genres.items(), key=lambda x: -x[1])])
        if genuine_genres else "| None Verified | 0 | 0.0% |"
    )

    plots_md = "\n".join([f"- `reports/eda/{p}`" for p in plots_saved])

    report_md = f"""# Exploratory Data Analysis (EDA) Report
**Project:** Statistical Analysis of Similarity in Human- and AI-Generated Music  
**Dataset:** `SleepyJesse/ai_music_large` (split: `train`)  
**Generated At:** {now_str}  
**Total Indexed Tracks:** {total}  

---

## 1. Summary Statistics Table

| Metric | Value | Notes / Description |
|---|---|---|
| **Total Indexed Tracks** | {total} | Successfully precomputed 12-dim mean chroma and beat matrices |
| **Human Acoustic Baseline** | {human_count} | Verified via metadata origin flag |
| **AI-Generated Tracks** | {ai_count} | Verified synthetic / generative origin |
| **Unknown Origin Tracks** | {unk_count} | Ambiguous or missing origin labels (excluded from baseline) |
| **Mean Track Duration** | {dur_mean:.1f}s | Sample average |
| **Median Track Duration** | {dur_median:.1f}s | 50th percentile |
| **Duration Range** | {dur_min:.1f}s – {dur_max:.1f}s | Shortest to longest track in indexed corpus |
| **Verified Metadata Genres** | {v_genres} | Genuine genre tags from dataset |
| **Model Estimated Genres** | {e_genres} | Provisional acoustic predictions |
| **Unknown Genre Tracks** | {u_genres} | Unlabeled (no arbitrary ambient fallback) |
| **Unique Artists** | {len(artists)} | Documented artist identities |
| **Unresolved / Failed Records** | {len(unresolved)} | Records requiring streaming retry |
| **Duplicate Records Detected** | {duplicates_detected} | Identical audio content hashes skipped |

---

## 2. Origin and Demographic Distribution

* **Human Tracks:** {human_count} ({human_count / max(total, 1) * 100:.1f}%)
* **AI-Generated Tracks:** {ai_count} ({ai_count / max(total, 1) * 100:.1f}%)
* **Unknown Tracks:** {unk_count} ({unk_count / max(total, 1) * 100:.1f}%)

> [!NOTE]
> Unknown origin tracks are strictly isolated and prevented from contaminating either the Human reference baseline or the AI evaluation pool.

---

## 3. Genuine Genre Breakdown

| Genre | Count | Percentage |
|---|---|---|
{genre_rows}

*(Note: Tracks without verified metadata genres are reported as Unknown rather than artificially labeled.)*

---

## 4. Dimensionality Reduction & Feature Space

* **Feature Representation:** 12-bin beat-synchronized Chroma CQT vectors.
* **PCA Projection:** Computed 2 principal components capturing global harmonic variance.
* **Interpretation Disclaimer:**  
  {summary.limitations}

---

## 5. Generated Visual Artifacts

The following figures were generated and saved in `{out_dir}`:
{plots_md}

---
*Report automatically generated by `src.eda`.*
"""

    report_path = out_dir / "eda_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_md)

    summary_json_path = out_dir / "eda_summary.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary.to_dict(), f, indent=2)

    return summary, report_md


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Exploratory Data Analysis on Indexed Music Corpus.")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_EDA_OUTPUT_DIR), help="Output directory for EDA reports and plots.")
    args = parser.parse_args()

    sum_res, r_md = run_exploratory_data_analysis(output_dir=Path(args.output_dir))
    print(f"EDA Complete. Total tracks: {sum_res.total_indexed_tracks}. Report written to {args.output_dir}/eda_report.md")

