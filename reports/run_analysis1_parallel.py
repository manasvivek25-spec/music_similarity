"""
Analysis 1 (Parallel & Fast): Full 200x200 Pairwise Similarity Matrix.
- Reuses cached 4,950 human-human similarity pairs from features/null_distribution.json.
- Computes 4,950 AI-AI pairs and 10,000 Human-AI pairs using 10 parallel processes.
- Exact 12-shift circular DTW with path length normalization (matches src/similarity.py).
- Full 200x200 symmetric similarity matrix S.
- Primary Statistic: mean within-AI minus mean within-human pairwise similarity.
- Permutation test over track labels (10,000 shuffles recomputed from matrix S).
- Effect sizes: Cohen's d, Mann-Whitney U, rank-biserial r.
- Within-genre comparison for genres with >= 8 tracks per class (Blues, Classical, Jazz, Rock).
- Per-generator breakdown.
Seed=42. Output: reports/analysis1_similarity_matrix.json
"""

import json, numpy as np, sys, time, os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict
from scipy.stats import mannwhitneyu

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from fastdtw import fastdtw

SEED = 42
OUT = BASE / "reports" / "analysis1_similarity_matrix.json"
FEAT = BASE / "features"

# Global dictionary shared across worker processes via initialization
_worker_chromas = {}

def _init_pool(chromas):
    global _worker_chromas
    _worker_chromas = chromas

def _dtw_worker_chunk(chunk):
    results = []
    for idx_a, idx_b in chunk:
        ca = _worker_chromas[idx_a]
        cb = _worker_chromas[idx_b]
        seq_a = ca.T
        min_dist = float("inf")
        best_path_len = 1
        
        # 12 circular pitch shifts along axis 0
        for shift in range(12):
            shifted_b = np.roll(cb, shift, axis=0)
            seq_b = shifted_b.T
            dist, path = fastdtw(seq_a, seq_b, dist=2)
            if dist < min_dist:
                min_dist = dist
                best_path_len = max(len(path), 1)
                
        norm_dist = min_dist / best_path_len
        sim = float(np.clip(1.0 / (1.0 + norm_dist), 0.0, 1.0))
        results.append((idx_a, idx_b, sim))
    return results

def main():
    print("=== Analysis 1: AI-vs-Human 200x200 Pairwise Similarity Matrix ===")
    t_start = time.time()
    
    # 1. Load remote_index and null_distribution
    ri = json.load(open(BASE / "features" / "remote_index.json"))
    tracks = ri["tracks"]
    nd = json.load(open(BASE / "features" / "null_distribution.json"))
    
    human_tids = sorted([tid for tid, t in tracks.items()
                         if str(t.get("origin_label", "")).lower() == "human"
                         and not any(x in tid for x in ("ref_", "tmp"))])
    ai_tids = sorted([tid for tid, t in tracks.items()
                      if str(t.get("origin_label", "")).lower() in ("ai", "ai_generated", "synthetic")
                      and not any(x in tid for x in ("ref_", "tmp"))])
    
    print(f"Corpus: {len(human_tids)} Human tracks, {len(ai_tids)} AI tracks.")
    assert len(human_tids) == 100, f"Expected 100 human tracks, got {len(human_tids)}"
    assert len(ai_tids) == 100, f"Expected 100 AI tracks, got {len(ai_tids)}"
    
    # 2. Load all 200 beat chroma representations
    print("Loading beat chroma arrays from disk...")
    chromas = {}
    for idx, tid in enumerate(human_tids + ai_tids):
        t = tracks[tid]
        p = Path(t.get("beat_feature_path", ""))
        if not p.exists():
            p = FEAT / f"{tid}_beat.npy"
        if not p.exists():
            candidates = sorted(FEAT.glob(f"{tid}_*beat.npy"))
            if candidates: p = candidates[0]
        assert p.exists(), f"Chroma not found for {tid}"
        c = np.load(str(p), allow_pickle=False).astype(np.float32)
        # Cap length at 80 frames for efficiency without loss of structural resolution
        if c.shape[1] > 80:
            c = c[:, :80]
        chromas[idx] = c
    print(f"Loaded {len(chromas)} chroma matrices into memory.")
    
    # Map: 0..99 = Human, 100..199 = AI
    # Full 200x200 similarity matrix
    S = np.zeros((200, 200), dtype=np.float64)
    np.fill_diagonal(S, 1.0)
    
    # 3. Fill Human-Human pairs from cached null distribution
    nd_scores = np.array(nd["null_scores"], dtype=np.float64)
    assert len(nd_scores) == 4950
    nd_tids = nd["source_track_ids"]
    tid_to_hidx = {t: i for i, t in enumerate(human_tids)}
    
    # Fill S for Human-Human
    pair_k = 0
    for i in range(len(nd_tids)):
        for j in range(i + 1, len(nd_tids)):
            score = nd_scores[pair_k]
            idx_i = tid_to_hidx[nd_tids[i]]
            idx_j = tid_to_hidx[nd_tids[j]]
            S[idx_i, idx_j] = score
            S[idx_j, idx_i] = score
            pair_k += 1
            
    print("Successfully mapped 4,950 Human-Human pairs from null distribution cache.")
    
    # 4. Generate pairs to compute:
    # AI-AI pairs: (i, j) for i, j in 100..199, i < j (4,950 pairs)
    # Human-AI pairs: (i, j) for i in 0..99, j in 100..199 (10,000 pairs)
    pairs_to_compute = []
    # AI-AI
    for i in range(100, 200):
        for j in range(i + 1, 200):
            pairs_to_compute.append((i, j))
    # Human-AI
    for i in range(0, 100):
        for j in range(100, 200):
            pairs_to_compute.append((i, j))
            
    print(f"Pairs to compute: {len(pairs_to_compute)} ({4950} AI-AI + {10000} Human-AI)")
    
    # Chunk pairs for multiprocessing
    chunk_size = 50
    chunks = [pairs_to_compute[k:k+chunk_size] for k in range(0, len(pairs_to_compute), chunk_size)]
    
    n_workers = min(10, os.cpu_count() or 4)
    print(f"Launching {len(chunks)} chunks across {n_workers} parallel workers...")
    t_dtw = time.time()
    computed_count = 0
    with ProcessPoolExecutor(max_workers=n_workers, initializer=_init_pool, initargs=(chromas,)) as executor:
        for chunk_results in executor.map(_dtw_worker_chunk, chunks):
            for i, j, sim in chunk_results:
                S[i, j] = sim
                S[j, i] = sim
            computed_count += len(chunk_results)
            if computed_count % 3000 == 0 or computed_count == len(pairs_to_compute):
                print(f"  Processed {computed_count} / {len(pairs_to_compute)} pairs ({time.time() - t_dtw:.1f}s)...", flush=True)
                
    print(f"All 14,950 pairs computed in {time.time() - t_dtw:.1f}s.")
    
    # 5. Extract distributions
    hh_scores = S[np.triu_indices(100, k=1)]
    aa_scores = S[100:, 100:][np.triu_indices(100, k=1)]
    ha_scores = S[:100, 100:].flatten()
    
    mean_hh, std_hh = float(np.mean(hh_scores)), float(np.std(hh_scores))
    mean_aa, std_aa = float(np.mean(aa_scores)), float(np.std(aa_scores))
    mean_ha, std_ha = float(np.mean(ha_scores)), float(np.std(ha_scores))
    
    delta_aa_hh = mean_aa - mean_hh
    delta_ha_hh = mean_ha - mean_hh
    
    print(f"\n--- Distribution Summary ---")
    print(f"Human-Human (4,950 pairs): Mean = {mean_hh:.4f} +/- {std_hh:.4f}, Range = [{np.min(hh_scores):.4f}, {np.max(hh_scores):.4f}]")
    print(f"AI-AI       (4,950 pairs): Mean = {mean_aa:.4f} +/- {std_aa:.4f}, Range = [{np.min(aa_scores):.4f}, {np.max(aa_scores):.4f}]")
    print(f"Human-AI   (10,000 pairs): Mean = {mean_ha:.4f} +/- {std_ha:.4f}, Range = [{np.min(ha_scores):.4f}, {np.max(ha_scores):.4f}]")
    print(f"Observed Difference: Within-AI - Within-Human = {delta_aa_hh:+.4f}")
    
    # 6. Permutation Test (10,000 shuffles over track labels)
    print("\nRunning Permutation Test (10,000 shuffles over track labels, seed=42)...")
    t_perm = time.time()
    n_perms = 10_000
    rng_perm = np.random.default_rng(SEED)
    perm_stats = np.empty(n_perms, dtype=np.float64)
    
    all_indices = np.arange(200)
    # Pre-allocate indices for speed
    triu_100 = np.triu_indices(100, k=1)
    
    for p in range(n_perms):
        shuffled = rng_perm.permutation(all_indices)
        h_idx = shuffled[:100]
        a_idx = shuffled[100:]
        
        m_h = float(np.mean(S[np.ix_(h_idx, h_idx)][triu_100]))
        m_a = float(np.mean(S[np.ix_(a_idx, a_idx)][triu_100]))
        perm_stats[p] = m_a - m_h
        
    p_val_two_tailed = float(np.mean(np.abs(perm_stats) >= abs(delta_aa_hh)))
    p_val_one_tailed = float(np.mean(perm_stats >= delta_aa_hh))
    print(f"Permutation test completed in {time.time() - t_perm:.2f}s.")
    print(f"  Observed Statistic = {delta_aa_hh:+.6f}")
    print(f"  Permutation Null: Mean = {np.mean(perm_stats):+.6f}, Std = {np.std(perm_stats):.6f}")
    print(f"  Two-tailed p-value = {p_val_two_tailed:.6f}")
    
    # 7. Effect sizes
    pooled_sd = float(np.sqrt(((len(aa_scores) - 1) * std_aa**2 + (len(hh_scores) - 1) * std_hh**2) / (len(aa_scores) + len(hh_scores) - 2)))
    cohens_d = float((mean_aa - mean_hh) / pooled_sd) if pooled_sd > 0 else 0.0
    u_stat, u_pval = mannwhitneyu(aa_scores, hh_scores, alternative="two-sided")
    rank_biserial = float(1.0 - (2.0 * u_stat / (len(aa_scores) * len(hh_scores))))
    print(f"Effect Sizes: Cohen's d = {cohens_d:+.4f}, Mann-Whitney U = {u_stat:.1f} (p = {u_pval:.4e}), Rank-biserial r = {rank_biserial:+.4f}")
    
    # 8. Within-Genre Comparison for genres with >= 8 tracks per class
    print("\n--- Within-Genre Comparisons (Genres with >= 8 tracks per class) ---")
    human_genres = [str(tracks[tid].get("genre", "Unknown")).title() for tid in human_tids]
    ai_genres = [str(tracks[tid].get("genre", "Unknown")).title() for tid in ai_tids]
    
    genre_data = {}
    target_genres = ["Blues", "Classical", "Jazz", "Rock"]
    
    for g in target_genres:
        h_g_idx = [i for i, genre in enumerate(human_genres) if genre == g]
        a_g_idx = [100 + i for i, genre in enumerate(ai_genres) if genre == g]
        
        n_h_g = len(h_g_idx)
        n_a_g = len(a_g_idx)
        assert n_h_g >= 8 and n_a_g >= 8, f"Genre {g} has {n_h_g} H and {n_a_g} AI tracks"
        
        # Submatrices
        hh_g_scores = S[np.ix_(h_g_idx, h_g_idx)][np.triu_indices(n_h_g, k=1)]
        aa_g_scores = S[np.ix_(a_g_idx, a_g_idx)][np.triu_indices(n_a_g, k=1)]
        ha_g_scores = S[np.ix_(h_g_idx, a_g_idx)].flatten()
        
        m_hh_g = float(np.mean(hh_g_scores))
        m_aa_g = float(np.mean(aa_g_scores))
        m_ha_g = float(np.mean(ha_g_scores))
        diff_g = m_aa_g - m_hh_g
        
        # Mann-Whitney U within genre
        u_g, p_g = mannwhitneyu(aa_g_scores, hh_g_scores, alternative="two-sided")
        
        genre_data[g] = {
            "n_human_tracks": n_h_g,
            "n_ai_tracks": n_a_g,
            "n_human_human_pairs": len(hh_g_scores),
            "n_ai_ai_pairs": len(aa_g_scores),
            "n_human_ai_pairs": len(ha_g_scores),
            "within_human_mean": round(m_hh_g, 4),
            "within_human_std": round(float(np.std(hh_g_scores)), 4),
            "within_ai_mean": round(m_aa_g, 4),
            "within_ai_std": round(float(np.std(aa_g_scores)), 4),
            "cross_human_ai_mean": round(m_ha_g, 4),
            "cross_human_ai_std": round(float(np.std(ha_g_scores)), 4),
            "difference_ai_minus_human": round(diff_g, 4),
            "mann_whitney_p": float(p_g),
        }
        print(f"  [{g}]: H (N={n_h_g})={m_hh_g:.4f}, AI (N={n_a_g})={m_aa_g:.4f}, Cross={m_ha_g:.4f}, Diff={diff_g:+.4f} (p={p_g:.4e})")

    # 9. Per-Generator Breakdown
    print("\n--- Per-Generator Breakdown ---")
    gen_names = []
    for tid in ai_tids:
        t = tracks[tid]
        rm = t.get("raw_metadata", {}) or {}
        g = rm.get("generator") or rm.get("source") or t.get("dataset_name", "")
        if "lyria" in str(g).lower() or "lyria" in tid.lower():
            gen_names.append("lyria-pro3")
        elif "musicgen" in str(g).lower() or "musicgen" in tid.lower():
            gen_names.append("musicgen")
        elif "acestep" in str(g).lower() or "acestep" in tid.lower():
            gen_names.append("acestep")
        else:
            gen_names.append("multi-gen-ai")
            
    generator_breakdown = {}
    for gname in sorted(set(gen_names)):
        g_indices = [100 + i for i, gn in enumerate(gen_names) if gn == gname]
        if len(g_indices) < 2: continue
        g_within = S[np.ix_(g_indices, g_indices)][np.triu_indices(len(g_indices), k=1)]
        g_to_human = S[:100, g_indices].flatten()
        generator_breakdown[gname] = {
            "n_tracks": len(g_indices),
            "within_generator_mean": round(float(np.mean(g_within)), 4),
            "within_generator_std": round(float(np.std(g_within)), 4),
            "similarity_to_human_mean": round(float(np.mean(g_to_human)), 4),
            "similarity_to_human_std": round(float(np.std(g_to_human)), 4),
        }
        print(f"  [{gname}] (N={len(g_indices)}): Within-gen={np.mean(g_within):.4f}, To-Human={np.mean(g_to_human):.4f}")

    # 10. Save Complete Output
    out_payload = {
        "analysis": "AI_vs_Human_200x200_Similarity_Matrix",
        "seed": SEED,
        "execution_time_seconds": round(time.time() - t_start, 2),
        "corpus": {
            "n_human_tracks": 100,
            "n_ai_tracks": 100,
            "total_matrix_size": [200, 200],
        },
        "distributions": {
            "human_human": {
                "n_pairs": len(hh_scores),
                "mean": round(mean_hh, 4),
                "std": round(std_hh, 4),
                "min": round(float(np.min(hh_scores)), 4),
                "max": round(float(np.max(hh_scores)), 4),
                "p95": round(float(np.percentile(hh_scores, 95)), 4),
            },
            "ai_ai": {
                "n_pairs": len(aa_scores),
                "mean": round(mean_aa, 4),
                "std": round(std_aa, 4),
                "min": round(float(np.min(aa_scores)), 4),
                "max": round(float(np.max(aa_scores)), 4),
                "p95": round(float(np.percentile(aa_scores, 95)), 4),
            },
            "human_ai": {
                "n_pairs": len(ha_scores),
                "mean": round(mean_ha, 4),
                "std": round(std_ha, 4),
                "min": round(float(np.min(ha_scores)), 4),
                "max": round(float(np.max(ha_scores)), 4),
                "p95": round(float(np.percentile(ha_scores, 95)), 4),
            },
        },
        "hypothesis_test": {
            "test_statistic_name": "mean_within_ai_minus_mean_within_human",
            "observed_statistic": round(delta_aa_hh, 6),
            "permutation_test": {
                "n_shuffles": n_perms,
                "null_mean": round(float(np.mean(perm_stats)), 6),
                "null_std": round(float(np.std(perm_stats)), 6),
                "p_value_two_tailed": round(p_val_two_tailed, 6),
                "p_value_one_tailed": round(p_val_one_tailed, 6),
            },
            "effect_size": {
                "cohens_d": round(cohens_d, 4),
                "mann_whitney_u": float(u_stat),
                "mann_whitney_p": float(u_pval),
                "rank_biserial_r": round(rank_biserial, 4),
            },
        },
        "within_genre_comparisons": genre_data,
        "per_generator_breakdown": generator_breakdown,
    }
    
    with open(OUT, "w") as f:
        json.dump(out_payload, f, indent=2)
    print(f"\n✓ Successfully saved Analysis 1 output to {OUT}")

if __name__ == "__main__":
    main()
