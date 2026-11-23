"""
Analysis 2: F2 Calibration — Leave-One-Out over 100 human tracks.
Null rebuilt without the query track (N-1 tracks → C(N-1,2) pairs via sub-matrix).
Reports type-I error rate at alpha=0.10/0.05/0.01, p-value KS vs Uniform.
Seed=42. Output: reports/analysis2_f2_calibration.json
"""
import json, numpy as np, sys, time
from pathlib import Path
from scipy.stats import kstest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SEED = 42
BASE = Path(__file__).resolve().parent.parent
OUT  = BASE / "reports" / "analysis2_f2_calibration.json"
FEAT = BASE / "features"

print("Loading data...")
nd = json.load(open(BASE / "features" / "null_distribution.json"))
nd_tids  = nd["source_track_ids"]   # 100 human track IDs
nd_scores = np.array(nd["null_scores"], dtype=np.float64)  # 4950 scores

n_tracks = len(nd_tids)
n_pairs  = len(nd_scores)
print(f"Null distribution: {n_tracks} tracks, {n_pairs} pairs")

# ── Reconstruct pair-score matrix from null scores ───────────────────────────
# Pairs are in upper-triangle order: (0,1),(0,2),...,(0,99),(1,2),...
# Score index for pair (i,j) with i<j:
#   idx = i*(n-1) - i*(i-1)//2 + (j-i-1)
def pair_to_idx(i, j, n=100):
    if i > j: i, j = j, i
    return i * (n - 1) - i * (i - 1) // 2 + (j - i - 1)

n = n_tracks
# Verify total
expected = n * (n - 1) // 2
print(f"Expected pairs: {expected}, actual: {n_pairs}")

# Build score matrix S[i,j] from null_scores
print("Building score matrix...")
S = np.zeros((n, n), dtype=np.float64)
pair_idx = 0
for i in range(n):
    for j in range(i + 1, n):
        if pair_idx < n_pairs:
            S[i, j] = nd_scores[pair_idx]
            S[j, i] = nd_scores[pair_idx]
        pair_idx += 1

# ── LOO loop: for each human track, remove it and compute p-value ─────────────
# The "query" is track q: its null p-value is computed from the
# distribution of similarity scores between q and the other 99 tracks.
# Wait — that's not exactly right. The proper LOO-F2 test is:
# When track q is the query, the null distribution should be built from
# the pairwise similarities of the remaining 99 tracks (C(99,2)=4851 pairs).
# The query similarity score is the similarity between q and each of the 99
# other tracks (or its mean, or we evaluate each separately).
#
# For calibration: we evaluate whether the p-values of human tracks
# compared against a LOO-rebuilt null are uniform under H0.
# Specifically: p(q) = fraction of LOO-null scores >= sim(q, random_other)
# 
# Approach: for each query track q, compute p-value of sim(q, q') against
# the LOO-null built from remaining 99 tracks (C(99,2)=4851 pairs).
# We pick q' as each of the other 99 tracks (or a random partner).
# To get a single p-value per track: use the mean sim(q, others) as the
# query score. Under H0, these p-values should be ~Uniform(0,1).

print("Running LOO calibration...")
t0 = time.time()
p_values_loo = []  # one per human track
all_query_scores = []
all_loo_null_means = []

for q in range(n):
    # LOO null: all pairs among tracks {0..n-1} \ {q}
    remaining = [i for i in range(n) if i != q]
    # Extract sub-matrix for remaining tracks
    sub = S[np.ix_(remaining, remaining)]
    # Get upper-triangle values (C(99,2) pairs)
    loo_null = sub[np.triu_indices(len(remaining), k=1)]
    
    # Query similarity: sim(q, each remaining track)
    q_sims = S[q, remaining]
    # Use mean similarity as query score (conservative)
    query_score = float(np.mean(q_sims))
    
    # Empirical p-value: fraction of null ≥ query_score
    k = int(np.sum(loo_null >= query_score))
    N_null = len(loo_null)
    p_val = float(k / N_null)
    p_values_loo.append(p_val)
    all_query_scores.append(round(query_score, 4))
    all_loo_null_means.append(round(float(np.mean(loo_null)), 4))

elapsed = time.time() - t0
print(f"LOO done in {elapsed:.1f}s")

p_arr = np.array(p_values_loo)
print(f"p-values: mean={np.mean(p_arr):.4f}, std={np.std(p_arr):.4f}, min={np.min(p_arr):.4f}, max={np.max(p_arr):.4f}")

# ── Type-I error rates ────────────────────────────────────────────────────────
alphas = [0.10, 0.05, 0.01]
type1_rates = {}
for a in alphas:
    rate = float(np.mean(p_arr <= a))
    type1_rates[str(a)] = round(rate, 4)
    print(f"  Type-I @ α={a}: {rate:.4f} (expected {a:.2f})")

# ── KS test against Uniform(0,1) ─────────────────────────────────────────────
ks_stat, ks_p = kstest(p_arr, 'uniform')
print(f"KS vs Uniform: D={ks_stat:.4f}, p={ks_p:.4f}")

# ── Additional: p-value histogram bins ───────────────────────────────────────
hist_counts, hist_edges = np.histogram(p_arr, bins=10, range=(0,1))
hist_data = {
    "bin_edges": [round(e,2) for e in hist_edges.tolist()],
    "counts": hist_counts.tolist(),
    "expected_uniform": int(n / 10),
}

result = {
    "analysis": "F2_LOO_Calibration",
    "seed": SEED,
    "n_human_tracks": n,
    "n_loo_null_pairs": int(n*(n-1)//2 - (n-1)),  # C(99,2)=4851
    "p_values_loo": [round(p,4) for p in p_values_loo],
    "query_scores_mean_sim": all_query_scores,
    "p_value_summary": {
        "mean": round(float(np.mean(p_arr)),4),
        "std":  round(float(np.std(p_arr)),4),
        "min":  round(float(np.min(p_arr)),4),
        "max":  round(float(np.max(p_arr)),4),
        "median": round(float(np.median(p_arr)),4),
    },
    "type1_error_rates": type1_rates,
    "ks_test_vs_uniform": {
        "ks_statistic": round(float(ks_stat),4),
        "p_value": round(float(ks_p),4),
        "conclusion": "Uniform" if ks_p > 0.05 else "Non-uniform",
    },
    "histogram": hist_data,
}

OUT.parent.mkdir(parents=True, exist_ok=True)
json.dump(result, open(OUT,"w"), indent=2)
print(f"\n✓ Saved to {OUT}")
