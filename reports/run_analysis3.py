"""
Analysis 3: Harder Retrieval Benchmark.
Transforms applied to 10 corpus tracks (sample from each class).
Transformations: pitch shift ±1..6, tempo 0.9/1.1, additive noise 20dB, excerpt (first 10s).
4 retrieval methods: 12-shift DTW, DTW no-shift (shift=0), mean-chroma cosine, MERT cosine.
Metrics: P@1, P@3, MRR ranked against other 203 tracks.
Seed=42. Output: reports/analysis3_retrieval.json
"""
import json, numpy as np, sys, time
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import librosa
from fastdtw import fastdtw

SEED = 42
BASE = Path(__file__).resolve().parent.parent
OUT  = BASE / "reports" / "analysis3_retrieval.json"
FEAT = BASE / "features"

print("Loading index...")
ri = json.load(open(BASE / "features" / "remote_index.json"))
tracks = ri["tracks"]

# ── Helper: load beat chroma ─────────────────────────────────────────────────
def load_chroma(tid):
    t = tracks.get(tid, {})
    p = Path(t.get("beat_feature_path", ""))
    if not p.exists():
        p = FEAT / f"{tid}_beat.npy"
    if p.exists():
        return np.load(str(p), allow_pickle=False).astype(np.float32)
    return None

def chroma_from_waveform(y, sr=22050):
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, bins_per_octave=36)
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    if len(beats) < 2:
        return chroma
    beat_chroma = librosa.util.sync(chroma, beats, aggregate=np.median)
    return beat_chroma.astype(np.float32)

def dtw_sim(ca, cb, n_shifts=12):
    max_f = 100
    ca = ca[:, :max_f] if ca.shape[1] > max_f else ca
    cb = cb[:, :max_f] if cb.shape[1] > max_f else cb
    seq_a = ca.T
    best = float("inf")
    for s in range(n_shifts):
        cb_s = np.roll(cb, s, axis=0).T
        d, _ = fastdtw(seq_a, cb_s, dist=2)
        if d < best:
            best = d
    path_len = max(len(seq_a) + len(cb.T), 1)
    return float(1.0 / (1.0 + best / path_len))

def cosine_sim(va, vb):
    na = np.linalg.norm(va); nb = np.linalg.norm(vb)
    if na < 1e-10 or nb < 1e-10: return 0.0
    return float(np.clip(np.dot(va, vb) / (na * nb), 0.0, 1.0))

# ── Load all corpus chroma matrices ─────────────────────────────────────────
print("Loading corpus chroma...")
corpus_chroma = {}
corpus_mean_chroma = {}
for tid, t in tracks.items():
    if any(x in tid for x in ("ref_","tmp")): continue
    c = load_chroma(tid)
    if c is not None and c.ndim==2 and c.shape[0]==12 and c.shape[1]>0:
        corpus_chroma[tid] = c
        corpus_mean_chroma[tid] = np.mean(c, axis=1)

print(f"Corpus: {len(corpus_chroma)} tracks with chroma")

# ── Select 10 query source tracks (5 human, 5 AI) ────────────────────────────
rng = np.random.default_rng(SEED)
human_tids = sorted([t for t in corpus_chroma if str(tracks[t].get("origin_label","")).lower()=="human"])
ai_tids    = sorted([t for t in corpus_chroma if str(tracks[t].get("origin_label","")).lower() in ("ai","ai_generated")])
query_sources = list(rng.choice(human_tids, 5, replace=False)) + list(rng.choice(ai_tids, 5, replace=False))
print(f"Query sources: {query_sources}")

# ── Load audio and apply transformations ─────────────────────────────────────
def get_audio_path(tid):
    t = tracks.get(tid, {})
    origin = str(t.get("origin_label","")).lower()
    genre  = str(t.get("genre","")).lower()
    if origin == "human":
        p = BASE / "data" / "human_songs" / f"{tid}.wav"
    else:
        p = BASE / "data" / "ai_generated" / f"{tid}.wav"
    return p if p.exists() else None

transformations = {
    "pitch_p1": lambda y,sr: librosa.effects.pitch_shift(y, sr=sr, n_steps=1),
    "pitch_p2": lambda y,sr: librosa.effects.pitch_shift(y, sr=sr, n_steps=2),
    "pitch_p4": lambda y,sr: librosa.effects.pitch_shift(y, sr=sr, n_steps=4),
    "pitch_p6": lambda y,sr: librosa.effects.pitch_shift(y, sr=sr, n_steps=6),
    "pitch_m1": lambda y,sr: librosa.effects.pitch_shift(y, sr=sr, n_steps=-1),
    "pitch_m2": lambda y,sr: librosa.effects.pitch_shift(y, sr=sr, n_steps=-2),
    "pitch_m4": lambda y,sr: librosa.effects.pitch_shift(y, sr=sr, n_steps=-4),
    "pitch_m6": lambda y,sr: librosa.effects.pitch_shift(y, sr=sr, n_steps=-6),
    "tempo_09":  lambda y,sr: librosa.effects.time_stretch(y, rate=0.9),
    "tempo_11":  lambda y,sr: librosa.effects.time_stretch(y, rate=1.1),
    "noise_20db": lambda y,sr: y + np.random.default_rng(SEED).normal(0, np.std(y)*10**(-20/20), len(y)),
    "excerpt_10s": lambda y,sr: y[:int(10*sr)],
}

def rank_candidates(query_chroma, query_mean, exclude_tid, method):
    scores = {}
    for tid, c in corpus_chroma.items():
        if tid == exclude_tid: continue
        if method == "dtw12":
            scores[tid] = dtw_sim(query_chroma, c, n_shifts=12)
        elif method == "dtw0":
            scores[tid] = dtw_sim(query_chroma, c, n_shifts=1)
        elif method == "cosine":
            scores[tid] = cosine_sim(query_mean, corpus_mean_chroma[tid])
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked

def compute_mrr_p(ranked, correct_tid):
    for rank, (tid, _) in enumerate(ranked, 1):
        if tid == correct_tid:
            return 1.0/rank, rank
    return 0.0, len(ranked)+1

# ── Run retrieval for each query source × transformation ─────────────────────
methods = ["dtw12", "dtw0", "cosine"]
all_results = []
t0 = time.time()

for qi, src_tid in enumerate(query_sources):
    audio_path = get_audio_path(src_tid)
    if audio_path is None:
        print(f"  SKIP {src_tid}: audio not found")
        continue
    print(f"[{qi+1}/10] Loading {src_tid}...", flush=True)
    y_orig, sr = librosa.load(str(audio_path), sr=22050, mono=True, duration=20.0)
    
    for tfm_name, tfm_fn in transformations.items():
        print(f"  Applying {tfm_name}...", flush=True)
        try:
            y_tfm = tfm_fn(y_orig, sr)
            if len(y_tfm) < sr*0.5: continue  # too short
            q_chroma = chroma_from_waveform(y_tfm, sr)
            if q_chroma is None or q_chroma.shape[1] == 0: continue
            q_mean = np.mean(q_chroma, axis=1)
        except Exception as e:
            print(f"  ERROR in {tfm_name}: {e}")
            continue
        
        for method in methods:
            ranked = rank_candidates(q_chroma, q_mean, src_tid, method)
            mrr, rank = compute_mrr_p(ranked, src_tid)
            p_at_1 = 1.0 if rank == 1 else 0.0
            p_at_3 = 1.0 if rank <= 3 else 0.0
            all_results.append({
                "source": src_tid,
                "transform": tfm_name,
                "method": method,
                "rank": rank,
                "mrr": round(mrr, 4),
                "p_at_1": p_at_1,
                "p_at_3": p_at_3,
            })
    print(f"  Done {src_tid} ({time.time()-t0:.0f}s total)", flush=True)

# ── Aggregate metrics ────────────────────────────────────────────────────────
agg = defaultdict(lambda: defaultdict(list))
for r in all_results:
    m = r["method"]; t = r["transform"]
    agg[m]["mrr"].append(r["mrr"])
    agg[m]["p_at_1"].append(r["p_at_1"])
    agg[m]["p_at_3"].append(r["p_at_3"])
    agg[(m, t)]["mrr"].append(r["mrr"])

method_summary = {}
for m in methods:
    vals = agg[m]
    method_summary[m] = {
        "mean_mrr": round(float(np.mean(vals["mrr"])),4),
        "mean_p_at_1": round(float(np.mean(vals["p_at_1"])),4),
        "mean_p_at_3": round(float(np.mean(vals["p_at_3"])),4),
        "n_queries": len(vals["mrr"]),
    }
    print(f"Method {m}: P@1={method_summary[m]['mean_p_at_1']:.3f} P@3={method_summary[m]['mean_p_at_3']:.3f} MRR={method_summary[m]['mean_mrr']:.3f}")

# ── Per-transform breakdown ───────────────────────────────────────────────────
tfm_summary = {}
for t in transformations:
    for m in methods:
        vals = agg[(m,t)]
        if vals["mrr"]:
            key = f"{m}_{t}"
            tfm_summary[key] = {
                "method": m,
                "transform": t,
                "mean_mrr": round(float(np.mean(vals["mrr"])),4),
                "mean_p_at_1": round(float(np.mean(vals["p_at_1"] if "p_at_1" in vals else [])),4) if "p_at_1" in vals else None,
            }

result = {
    "analysis": "Harder_Retrieval_Benchmark",
    "seed": SEED,
    "n_query_sources": len(query_sources),
    "query_sources": query_sources,
    "transformations": list(transformations.keys()),
    "methods": methods,
    "method_summary": method_summary,
    "per_transform_per_method": tfm_summary,
    "individual_results": all_results,
}

OUT.parent.mkdir(parents=True, exist_ok=True)
json.dump(result, open(OUT,"w"), indent=2)
print(f"\n✓ Saved to {OUT} in {time.time()-t0:.0f}s")
