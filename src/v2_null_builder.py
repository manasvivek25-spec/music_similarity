"""
P4 Null Distribution & Statistical Baseline Builder for v2 Reference Population.

Consumes: data/v2/v2_null_population.json (verified Human tracks, strictly disjoint from
          reference_index, f4_train/val/test, and held_out).
Produces: data/v2/v2_null_distribution.json with strict cache invalidation key
          (reference IDs SHA-256, feature_version, algorithm_version, sampling_config,
           preprocessing_settings).

Leaves legacy features/null_distribution.json 100% untouched.
"""

import concurrent.futures
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from src.config import (
    ALGORITHM_VERSION,
    FEATURE_VERSION,
    INDEX_MAX_SECONDS,
    NULL_DIST_SAMPLE_PAIRS,
    REPO_ROOT,
    SAMPLE_RATE,
    V2_DATA_DIR,
    V2_NULL_CACHE_FILE,
)
from src.similarity import dtw_similarity
from src.stats_test import compute_null_cache_key
from src.v2_cleaning_audit import V2_NULL_POP_FILE
from src.v2_indexer import atomic_write_json


def _compute_pairs_parallel(
    chroma_dict: Dict[str, np.ndarray],
    max_pairs: int,
    seed: int = 42,
    max_workers: int = 8,
) -> List[float]:
    track_ids = sorted(chroma_dict.keys())
    n = len(track_ids)
    if n < 2 or max_pairs <= 0:
        return []

    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    if len(pairs) > max_pairs:
        rng = np.random.default_rng(seed)
        sampled = rng.choice(len(pairs), size=max_pairs, replace=False)
        pairs = [pairs[int(idx)] for idx in sampled]

    def eval_pair(ij: Tuple[int, int]) -> float:
        i, j = ij
        c_i = chroma_dict[track_ids[i]]
        c_j = chroma_dict[track_ids[j]]
        c_i = c_i[:, :300] if (c_i.ndim == 2 and c_i.shape[1] > 300) else c_i
        c_j = c_j[:, :300] if (c_j.ndim == 2 and c_j.shape[1] > 300) else c_j
        return float(dtw_similarity(c_i, c_j))

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        scores = list(ex.map(eval_pair, pairs))

    return [round(float(s), 5) for s in scores if np.isfinite(s)]


def build_v2_null_distribution(
    max_pairs: int = NULL_DIST_SAMPLE_PAIRS,
    seed: int = 42,
    max_workers: int = 8,
) -> Dict[str, Any]:
    t0 = time.time()
    with open(V2_NULL_POP_FILE, "r", encoding="utf-8") as f:
        pop_data = json.load(f)

    tracks = pop_data["tracks"]
    chroma_dict: Dict[str, np.ndarray] = {}
    genre_groups: Dict[str, Dict[str, np.ndarray]] = defaultdict(dict)

    for tid, rec in sorted(tracks.items()):
        if rec.get("origin_label") != "human":
            continue
        bp = REPO_ROOT / rec["beat_cache"]
        if bp.is_file():
            mat = np.load(str(bp)).astype(np.float32)
            if mat.ndim == 2 and mat.shape[0] == 12 and mat.shape[1] >= 2:
                chroma_dict[tid] = mat
                g = rec.get("genre")
                if g and str(g).strip().lower() not in ("", "unknown", "none"):
                    genre_groups[str(g).strip()][tid] = mat

    eval_ids = sorted(chroma_dict.keys())
    print(f"[P4] Building v2 null distribution from {len(eval_ids)} disjoint verified human tracks (max_pairs={max_pairs})...", flush=True)

    global_null = _compute_pairs_parallel(chroma_dict, max_pairs=max_pairs, seed=seed, max_workers=max_workers)

    genre_null: Dict[str, List[float]] = {}
    for g, g_chroma in sorted(genre_groups.items()):
        if len(g_chroma) >= 2:
            g_scores = _compute_pairs_parallel(g_chroma, max_pairs=min(max_pairs, 500), seed=seed, max_workers=max_workers)
            if g_scores:
                genre_null[g] = g_scores

    cache_key = compute_null_cache_key(
        reference_ids=eval_ids,
        max_pairs=max_pairs,
        seed=seed,
        feature_version=FEATURE_VERSION,
        algorithm_version=ALGORITHM_VERSION,
        sample_rate=SAMPLE_RATE,
        index_max_seconds=INDEX_MAX_SECONDS,
    )

    payload = {
        "version": "2.0",
        "dataset_id": "v2_balanced_fma_sleepyjesse_real",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cache_key": cache_key,
        "source_track_ids": eval_ids,
        "sample_pairs_count": len(global_null),
        "null_scores": global_null,
        "genre_null_scores": genre_null,
        "summary_stats": {
            "global_n": len(global_null),
            "global_mean": round(float(np.mean(global_null)), 5) if global_null else None,
            "global_std": round(float(np.std(global_null)), 5) if global_null else None,
            "global_median": round(float(np.median(global_null)), 5) if global_null else None,
            "global_p95": round(float(np.percentile(global_null, 95)), 5) if global_null else None,
            "global_p99": round(float(np.percentile(global_null, 99)), 5) if global_null else None,
            "genres_with_ge30_pairs": sorted([g for g, sc in genre_null.items() if len(sc) >= 30]),
            "genre_pair_counts": {g: len(sc) for g, sc in genre_null.items()},
        },
        "pairwise_dependency_note": (
            "Pairwise null observations are formed from pairs within the verified human reference pool "
            "and share underlying tracks. While p = k/N provides the empirical exceedance rate, "
            "pairs sharing tracks are not strictly independent observations."
        ),
    }

    atomic_write_json(V2_NULL_CACHE_FILE, payload)
    dt = round(time.time() - t0, 2)
    print(
        f"[P4] Saved {V2_NULL_CACHE_FILE} in {dt}s: N={len(global_null)}, "
        f"mean={payload['summary_stats']['global_mean']}, std={payload['summary_stats']['global_std']}, "
        f"genres_>=30={len(payload['summary_stats']['genres_with_ge30_pairs'])}",
        flush=True,
    )
    return payload


if __name__ == "__main__":
    build_v2_null_distribution()
