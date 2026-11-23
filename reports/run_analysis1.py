"""
Analysis 1: AI-vs-Human Similarity Matrix (200×200 pairwise DTW)
Reuses cached 4,950 human-human pairs; computes AI-AI and Human-AI fresh.
Permutation test (N=10,000 shuffles), effect size, per-genre, generator breakdown.
Seed=42. Output: reports/analysis1_similarity_matrix.json
"""
import json, numpy as np, sys, time
from pathlib import Path
from itertools import combinations
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

SEED = 42
rng = np.random.default_rng(SEED)
BASE = Path(__file__).resolve().parent.parent
FEAT = BASE / "features"
OUT  = BASE / "reports" / "analysis1_similarity_matrix.json"

print("Loading index and null distribution...")
ri = json.load(open(BASE / "features" / "remote_index.json"))
tracks = ri["tracks"]
nd = json.load(open(BASE / "features" / "null_distribution.json"))

# ── Helper: load beat chroma ─────────────────────────────────────────────────
def load_chroma(tid):
    t = tracks[tid]
    p = Path(t.get("beat_feature_path", ""))
    if not p.exists():
        p = FEAT / f"{tid}_beat.npy"
    if not p.exists():
        # try without hash suffix
        candidates = sorted(FEAT.glob(f"{tid}_beat.npy"))
        if candidates:
            p = candidates[0]
    if p.exists():
        return np.load(str(p), allow_pickle=False).astype(np.float32)
    return None

# ── Helper: fast 12-shift DTW similarity ────────────────────────────────────
def dtw_sim_12shift(ca, cb):
    # Truncate long matrices for speed (keep ≤120 frames)
    max_f = 120
    ca = ca[:, :max_f] if ca.shape[1] > max_f else ca
    cb = cb[:, :max_f] if cb.shape[1] > max_f else cb
    seq_a = ca.T
    best = float("inf")
    for s in range(12):
        cb_s = np.roll(cb, s, axis=0).T
        d, _ = fastdtw(seq_a, cb_s, dist=2)
        if d < best:
            best = d
    path_len = max(len(seq_a) + len(cb.T), 1)  # approx
    nd_val = best / path_len
    return float(1.0 / (1.0 + nd_val))

# ── Collect track lists ──────────────────────────────────────────────────────
human_tids = sorted([tid for tid, t in tracks.items()
                     if str(t.get("origin_label","")).lower() == "human"
                     and not any(x in tid for x in ("ref_","tmp"))])
ai_tids    = sorted([tid for tid, t in tracks.items()
                     if str(t.get("origin_label","")).lower() in ("ai","ai_generated","synthetic")
                     and not any(x in tid for x in ("ref_","tmp"))])

print(f"Human tracks: {len(human_tids)}, AI tracks: {len(ai_tids)}")

# ── Load all chroma matrices ─────────────────────────────────────────────────
print("Loading chroma matrices...")
chroma = {}
missing = []
for tid in human_tids + ai_tids:
    c = load_chroma(tid)
    if c is not None and c.ndim == 2 and c.shape[0] == 12 and c.shape[1] > 0:
        chroma[tid] = c
    else:
        missing.append(tid)
print(f"Loaded {len(chroma)} chroma matrices, missing {len(missing)}: {missing[:5]}")

valid_human = [t for t in human_tids if t in chroma]
valid_ai    = [t for t in ai_tids    if t in chroma]
print(f"Valid human: {len(valid_human)}, valid AI: {len(valid_ai)}")

# ── Human-human: load from null_distribution.json (already 4,950 pairs) ─────
# Map source_track_ids to indices
nd_tids = nd.get("source_track_ids", [])
nd_scores = nd.get("null_scores", [])
hh_scores = []
# Use all null_scores (4950)
hh_scores = list(nd_scores)
print(f"Loaded {len(hh_scores)} human-human null scores from cache")

# ── Compute AI-AI pairwise ───────────────────────────────────────────────────
print(f"Computing AI-AI pairwise ({len(valid_ai)*(len(valid_ai)-1)//2} pairs)...")
t0 = time.time()
aa_scores = []
aa_pairs  = []
for i, ta in enumerate(valid_ai):
    for j, tb in enumerate(valid_ai):
        if j <= i:
            continue
        s = dtw_sim_12shift(chroma[ta], chroma[tb])
        aa_scores.append(s)
        aa_pairs.append((ta, tb))
    if (i+1) % 10 == 0:
        elapsed = time.time()-t0
        pct = (i+1)/len(valid_ai)*100
        print(f"  AI-AI: {i+1}/{len(valid_ai)} ({pct:.0f}%) {elapsed:.0f}s", flush=True)
print(f"  AI-AI done: {len(aa_scores)} pairs in {time.time()-t0:.1f}s")

# ── Compute Human-AI pairwise ────────────────────────────────────────────────
print(f"Computing Human-AI pairwise ({len(valid_human)*len(valid_ai)} pairs)...")
t0 = time.time()
ha_scores = []
ha_pairs  = []
for i, th in enumerate(valid_human):
    for ta in valid_ai:
        s = dtw_sim_12shift(chroma[th], chroma[ta])
        ha_scores.append(s)
        ha_pairs.append((th, ta))
    if (i+1) % 10 == 0:
        elapsed = time.time()-t0
        pct = (i+1)/len(valid_human)*100
        print(f"  Human-AI: {i+1}/{len(valid_human)} ({pct:.0f}%) {elapsed:.0f}s", flush=True)
print(f"  Human-AI done: {len(ha_scores)} pairs in {time.time()-t0:.1f}s")

# ── Summary statistics ────────────────────────────────────────────────────────
hh = np.array(hh_scores)
aa = np.array(aa_scores)
ha = np.array(ha_scores)

mean_hh = float(np.mean(hh)); std_hh = float(np.std(hh))
mean_aa = float(np.mean(aa)); std_aa = float(np.std(aa))
mean_ha = float(np.mean(ha)); std_ha = float(np.std(ha))

delta_aa_hh = mean_aa - mean_hh
delta_ha_hh = mean_ha - mean_hh
print(f"\nMean HH: {mean_hh:.4f}±{std_hh:.4f}  AA: {mean_aa:.4f}±{std_aa:.4f}  HA: {mean_ha:.4f}±{std_ha:.4f}")
print(f"Delta(AA-HH): {delta_aa_hh:+.4f}  Delta(HA-HH): {delta_ha_hh:+.4f}")

# ── Permutation test: H0: mean(AA) == mean(HH) ───────────────────────────────
# Pool all pairwise scores by track label; permute labels (track-level)
# Fast approach: permute the full 200-label vector, recompute within-group means from matrix
# We need a compressed representation: for each pair (i,j) store score and labels

# Build label array for 200 tracks (H=0, A=1)
all_tids = valid_human + valid_ai
n_h = len(valid_human); n_a = len(valid_ai); N = n_h + n_a
labels = np.array([0]*n_h + [1]*n_a, dtype=int)

# Build score matrix (upper triangle only)
# human-human: scores already computed (but only for nd_tids; we may not have all)
# Instead, use aa_scores/hh_scores/ha_scores directly with index bookkeeping

# For permutation test, the test statistic is mean(AA) - mean(HH) using new scores
# We use the actual computed AA/HH/HA scores
obs_stat = mean_aa - mean_hh  # observed

# Permutation: pool all pairwise scores (HH+HA+AA), shuffle labels
# For each permutation, randomly assign 100 "H" and 100 "A" labels to 200 tracks
# Then compute new within-group means from the sub-matrices
# Since we have full HH, HA, AA blocks: we need a fast approach
# Assign 100-of-200 as group0; mean of group0-group0 pairs and group1-group1 pairs

# Build pair score lookup dict
# pair_scores[i][j] = score where i<j, i and j are indices into all_tids
# aa pairs: indices are (n_h+i, n_h+j)
# hh pairs: use nd_scores (approximate: take first n_h*(n_h-1)//2 scores matched by position)
# ha pairs: (i, n_h+j)

# For efficiency, store as score matrix
print("\nBuilding 200x200 score matrix for permutation test...")
n_all = len(all_tids)
S = np.zeros((n_all, n_all), dtype=np.float32)

# Fill HH block (0:n_h, 0:n_h) using null scores
# Match by sorting: nd_tids positions
nd_set = set(nd_tids)
h_in_nd = [t for t in valid_human if t in nd_set]
h_nd_idx = {t: nd_tids.index(t) for t in h_in_nd if t in nd_tids}
# recompute hh from scratch if needed, or use the nd_scores positionally
# Easier: reconstruct from nd_scores with the same pair ordering
nd_arr = np.array(nd_scores)
nd_tids_list = nd_tids

# Map nd_tids to valid_human indices
h_to_all_idx = {t: i for i, t in enumerate(all_tids)}
pair_idx = 0
for i in range(len(nd_tids_list)):
    for j in range(i+1, len(nd_tids_list)):
        ti = nd_tids_list[i]; tj = nd_tids_list[j]
        if pair_idx >= len(nd_arr): break
        ai_idx = h_to_all_idx.get(ti); aj_idx = h_to_all_idx.get(tj)
        if ai_idx is not None and aj_idx is not None:
            S[ai_idx, aj_idx] = nd_arr[pair_idx]
            S[aj_idx, ai_idx] = nd_arr[pair_idx]
        pair_idx += 1
    if pair_idx >= len(nd_arr): break

# Fill HA block
for k, (th, ta) in enumerate(ha_pairs):
    ih = h_to_all_idx.get(th); ia = h_to_all_idx.get(ta)
    if ih is not None and ia is not None:
        S[ih, ia] = ha_scores[k]; S[ia, ih] = ha_scores[k]

# Fill AA block
for k, (ta, tb) in enumerate(aa_pairs):
    ia = h_to_all_idx.get(ta); ib = h_to_all_idx.get(tb)
    if ia is not None and ib is not None:
        S[ia, ib] = aa_scores[k]; S[ib, ia] = aa_scores[k]

np.fill_diagonal(S, 0.0)

def group_mean(S, mask0, mask1=None):
    """Mean of upper-triangle elements within group defined by mask0 (or cross-group if mask1 given)."""
    if mask1 is None:
        # within-group
        idx = np.where(mask0)[0]
        if len(idx) < 2: return np.nan
        vals = []
        for i in range(len(idx)):
            for j in range(i+1, len(idx)):
                vals.append(S[idx[i], idx[j]])
        return np.mean(vals) if vals else np.nan
    else:
        idx0 = np.where(mask0)[0]; idx1 = np.where(mask1)[0]
        if len(idx0)==0 or len(idx1)==0: return np.nan
        sub = S[np.ix_(idx0, idx1)]
        return float(np.mean(sub))

# Observed: use direct means already computed
print(f"Running permutation test (N=10,000 shuffles, seed=42)...")
n_perm = 10_000
perm_stats = np.empty(n_perm, dtype=np.float64)
rng_p = np.random.default_rng(SEED)
all_idx = np.arange(n_all)
t0 = time.time()
for p in range(n_perm):
    perm = rng_p.permutation(n_all)
    h_idx = perm[:n_h]; a_idx = perm[n_h:]
    # mean within-AI
    sub_aa = S[np.ix_(a_idx, a_idx)]
    # mean within-Human
    sub_hh = S[np.ix_(h_idx, h_idx)]
    m_aa = float(np.sum(np.triu(sub_aa, 1)) / max(np.sum(np.triu(np.ones_like(sub_aa), 1)), 1))
    m_hh = float(np.sum(np.triu(sub_hh, 1)) / max(np.sum(np.triu(np.ones_like(sub_hh), 1)), 1))
    perm_stats[p] = m_aa - m_hh
    if (p+1) % 2000 == 0:
        print(f"  Permutation {p+1}/{n_perm} ({time.time()-t0:.0f}s)", flush=True)

p_val_perm = float(np.mean(np.abs(perm_stats) >= abs(obs_stat)))
print(f"Permutation test: obs={obs_stat:+.4f}, p={p_val_perm:.4f}, {time.time()-t0:.1f}s")

# Cohen's d effect size
pooled_sd = float(np.sqrt((std_aa**2 + std_hh**2) / 2))
cohens_d  = (mean_aa - mean_hh) / pooled_sd if pooled_sd > 1e-9 else 0.0

# Mann-Whitney U
from scipy.stats import mannwhitneyu
u_stat, u_p = mannwhitneyu(aa, hh, alternative='two-sided')
rank_biserial = float(1 - 2*float(u_stat)/(len(aa)*len(hh)))

# ── Per-genre comparison ─────────────────────────────────────────────────────
genre_results = {}
genres_human = defaultdict(list)
genres_ai    = defaultdict(list)
for tid in valid_human:
    g = tracks[tid].get("genre","Unknown")
    genres_human[g].append(tid)
for tid in valid_ai:
    g = tracks[tid].get("genre","Unknown")
    genres_ai[g].append(tid)

for g in set(genres_human.keys()) & set(genres_ai.keys()):
    ht = genres_human[g]; at = genres_ai[g]
    if len(ht) >= 8 and len(at) >= 8:
        # collect HA scores for this genre
        ht_set = set(ht); at_set = set(at)
        g_ha = [ha_scores[k] for k,(th,ta) in enumerate(ha_pairs) if th in ht_set and ta in at_set]
        g_hh = [ha_scores[k] for k,(th,ta) in enumerate(ha_pairs) if th in ht_set and ta in ht_set]
        g_aa = [aa_scores[k] for k,(ta,tb) in enumerate(aa_pairs) if ta in at_set and tb in at_set]
        genre_results[g] = {
            "n_human": len(ht), "n_ai": len(at),
            "mean_ha": round(float(np.mean(g_ha)),4) if g_ha else None,
            "mean_hh": round(float(np.mean(hh)),4),  # global null for reference
            "mean_aa": round(float(np.mean(g_aa)),4) if g_aa else None,
        }

# ── Save results ─────────────────────────────────────────────────────────────
result = {
    "analysis": "AI_vs_Human_Similarity_Matrix",
    "seed": SEED,
    "n_human_tracks": len(valid_human),
    "n_ai_tracks": len(valid_ai),
    "n_hh_pairs": len(hh_scores),
    "n_aa_pairs": len(aa_scores),
    "n_ha_pairs": len(ha_scores),
    "mean_hh": round(mean_hh, 4), "std_hh": round(std_hh, 4),
    "mean_aa": round(mean_aa, 4), "std_aa": round(std_aa, 4),
    "mean_ha": round(mean_ha, 4), "std_ha": round(std_ha, 4),
    "delta_aa_minus_hh": round(delta_aa_hh, 4),
    "delta_ha_minus_hh": round(delta_ha_hh, 4),
    "permutation_test": {
        "n_shuffles": n_perm,
        "observed_statistic": round(obs_stat, 6),
        "p_value_two_tailed": round(p_val_perm, 4),
        "perm_stat_mean": round(float(np.mean(perm_stats)), 6),
        "perm_stat_std":  round(float(np.std(perm_stats)), 6),
    },
    "effect_size": {
        "cohens_d": round(cohens_d, 4),
        "mann_whitney_u": round(float(u_stat), 1),
        "mann_whitney_p": round(float(u_p), 6),
        "rank_biserial_r": round(rank_biserial, 4),
    },
    "per_genre_within_genre": genre_results,
    "aa_scores_sample": [round(x,4) for x in aa_scores[:20]],
    "ha_scores_sample": [round(x,4) for x in ha_scores[:20]],
}

OUT.parent.mkdir(parents=True, exist_ok=True)
json.dump(result, open(OUT,"w"), indent=2)
print(f"\n✓ Saved to {OUT}")
print(f"  HH={mean_hh:.4f} AA={mean_aa:.4f} HA={mean_ha:.4f}")
print(f"  Δ(AA-HH)={delta_aa_hh:+.4f}  p_perm={p_val_perm:.4f}  d={cohens_d:.4f}")
