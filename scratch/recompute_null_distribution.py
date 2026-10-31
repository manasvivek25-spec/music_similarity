import json
import time
import itertools
from pathlib import Path
import numpy as np
from multiprocessing import Pool, cpu_count
from src.config import FEATURES_DIR, NULL_DIST_CACHE_FILE, REMOTE_INDEX_FILE
from src.similarity import dtw_similarity

def eval_pair(pair_data):
    id_a, id_b, f_a, f_b = pair_data
    try:
        c_a = np.load(f_a)
        c_b = np.load(f_b)
        sub_a = c_a[:, :300] if c_a.shape[1] > 300 else c_a
        sub_b = c_b[:, :300] if c_b.shape[1] > 300 else c_b
        score = dtw_similarity(sub_a, sub_b)
        return float(score)
    except Exception as e:
        return None

def main():
    print("=== Recomputing Canonical 1,035 Null Distribution Pairs ===")
    with open(REMOTE_INDEX_FILE, "r", encoding="utf-8") as f:
        idx = json.load(f)
    
    tracks = idx.get("tracks", {})
    human_tracks = [
        tid for tid, m in tracks.items()
        if str(m.get("origin_label", "")).lower() in ("human", "0", "false")
    ]
    human_tracks.sort()
    
    valid_human = []
    for tid in human_tracks:
        beat_path = FEATURES_DIR / f"{tid}_beat.npy"
        if beat_path.is_file():
            valid_human.append((tid, str(beat_path)))
    
    N_tracks = len(valid_human)
    print(f"Discovered {N_tracks} valid human tracks with beat features on disk.")
    expected_pairs = N_tracks * (N_tracks - 1) // 2
    print(f"Total combinations to evaluate: {expected_pairs} pairs")
    
    pair_inputs = []
    for (id_a, f_a), (id_b, f_b) in itertools.combinations(valid_human, 2):
        pair_inputs.append((id_a, id_b, f_a, f_b))
        
    start_t = time.time()
    num_workers = min(8, cpu_count())
    print(f"Starting parallel evaluation with {num_workers} worker processes...")
    with Pool(num_workers) as pool:
        scores = pool.map(eval_pair, pair_inputs)
        
    valid_scores = [round(float(s), 5) for s in scores if s is not None]
    elapsed = time.time() - start_t
    print(f"Evaluated {len(valid_scores)} pairs in {elapsed:.2f}s")
    
    valid_scores_arr = np.array(valid_scores)
    print(f"Min: {valid_scores_arr.min():.4f}, Max: {valid_scores_arr.max():.4f}, "
          f"Median: {np.median(valid_scores_arr):.4f}, 95th: {np.percentile(valid_scores_arr, 95):.4f}")
    
    genre_groups = {}
    for tid, _ in valid_human:
        meta = tracks.get(tid, {})
        g = meta.get("genre")
        if g:
            genre_groups.setdefault(g, []).append(tid)
            
    genre_null = {}
    track_to_path = dict(valid_human)
    for g, g_tids in genre_groups.items():
        if len(g_tids) >= 2:
            g_pairs = []
            for t_a, t_b in itertools.combinations(g_tids, 2):
                g_pairs.append((t_a, t_b, track_to_path[t_a], track_to_path[t_b]))
            with Pool(num_workers) as pool:
                g_scores = pool.map(eval_pair, g_pairs)
            valid_g = [round(float(s), 5) for s in g_scores if s is not None]
            if valid_g:
                genre_null[g] = valid_g
                print(f"Genre '{g}': {len(valid_g)} pairs")
                
    payload = {
        "version": "1.0",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_track_ids": [t for t, _ in valid_human],
        "sample_pairs_count": len(valid_scores),
        "null_scores": valid_scores,
        "genre_null_scores": genre_null,
        "pairwise_dependency_note": (
            "Pairwise null observations are formed from pairs within the reference pool "
            "and share underlying tracks. While k/N provides the empirical p-value, "
            "observations are not strictly independent."
        )
    }
    
    with open(NULL_DIST_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        
    print(f"Persisted {len(valid_scores)} null scores to {NULL_DIST_CACHE_FILE}")

if __name__ == "__main__":
    main()

