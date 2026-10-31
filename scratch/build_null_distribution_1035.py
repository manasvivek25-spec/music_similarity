"""
Compute exhaustive all-pairs null distribution across all 46 human reference tracks (1,035 unique pairs).
Parallelized across CPU cores using ProcessPoolExecutor.
Persists to features/null_distribution.json.
"""

import json
import logging
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(REPO_ROOT))

from src.config import FEATURES_DIR, NULL_DIST_CACHE_FILE
from src.similarity import dtw_similarity

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("build_null_1035")


def compute_pair_dtw(pair_data):
    i_id, j_id, c_i, c_j = pair_data
    try:
        score = dtw_similarity(c_i, c_j)
        if np.isfinite(score):
            return round(float(score), 5)
    except Exception:
        pass
    return None


def main():
    logger.info("Loading remote_index.json...")
    idx_p = FEATURES_DIR / "remote_index.json"
    with open(idx_p, "r", encoding="utf-8") as f:
        tracks = json.load(f).get("tracks", {})

    human_chroma = {}
    human_meta = {}
    for tid, t in tracks.items():
        lbl = str(t.get("origin_label", "")).lower()
        if lbl in ("human", "0", "false"):
            beat_path = FEATURES_DIR / f"{tid}_beat.npy"
            if beat_path.is_file():
                mat = np.load(str(beat_path))
                # Cap to 300 frames for FastDTW efficiency
                mat_capped = mat[:, :300] if mat.shape[1] > 300 else mat
                human_chroma[tid] = mat_capped
                human_meta[tid] = t

    track_ids = sorted(list(human_chroma.keys()))
    n = len(track_ids)
    total_pairs = (n * (n - 1)) // 2
    logger.info(f"Loaded {n} human tracks. Total exhaustive pairs to evaluate: {total_pairs}")

    tasks = []
    for i in range(n):
        for j in range(i + 1, n):
            tasks.append((track_ids[i], track_ids[j], human_chroma[track_ids[i]], human_chroma[track_ids[j]]))

    t0 = time.time()
    logger.info(f"Launching parallel DTW across 8 worker processes...")
    with ProcessPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(compute_pair_dtw, tasks, chunksize=15))

    null_scores = [r for r in results if r is not None]
    elapsed = time.time() - t0
    logger.info(f"Completed {len(null_scores)} exhaustive null scores in {elapsed:.1f}s!")
    logger.info(f"Summary: min={min(null_scores):.4f}, median={np.median(null_scores):.4f}, max={max(null_scores):.4f}, 95th={np.percentile(null_scores, 95):.4f}")

    # Build genre groups if available
    genre_groups = {}
    for tid, meta in human_meta.items():
        g = meta.get("genre")
        if g:
            genre_groups.setdefault(g, []).append(tid)

    genre_null = {}
    for g, g_tids in genre_groups.items():
        if len(g_tids) >= 2:
            g_tasks = []
            for i in range(len(g_tids)):
                for j in range(i + 1, len(g_tids)):
                    g_tasks.append((g_tids[i], g_tids[j], human_chroma[g_tids[i]], human_chroma[g_tids[j]]))
            with ProcessPoolExecutor(max_workers=8) as executor:
                g_res = list(executor.map(compute_pair_dtw, g_tasks))
            valid_g = [r for r in g_res if r is not None]
            if valid_g:
                genre_null[g] = valid_g

    payload = {
        "version": "2.0",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_track_ids": track_ids,
        "sample_pairs_count": len(null_scores),
        "total_human_tracks": n,
        "is_exhaustive": True,
        "null_scores": null_scores,
        "genre_null_scores": genre_null,
        "pairwise_dependency_note": (
            f"Exhaustive pairwise null observations are formed from all unique pairs within the {n}-track "
            f"reference pool ({total_pairs} pairs). While k/N provides the authoritative empirical p-value, "
            "individual pairwise observations share underlying tracks and are not strictly independent."
        )
    }

    with open(NULL_DIST_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    logger.info(f"Persisted expanded exhaustive null distribution cache ({len(null_scores)} pairs) to {NULL_DIST_CACHE_FILE}")


if __name__ == "__main__":
    main()

