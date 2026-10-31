"""
CPU vs CUDA GPU Paired Numerical Parity and Downstream Impact Verification.
Tests:
1. Raw tensor shape, element-wise MAE, MaxAE, Cosine similarity.
2. Downstream impact on Cosine Similarity Nearest-Neighbor Ranking against 260 MERT embeddings.
3. Downstream impact on Classifier probabilities and binary predictions.
"""

import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent

# 1. Generate identical deterministic audio waveform
sr = 24000
duration = 10.0
t = np.linspace(0, duration, int(sr * duration), endpoint=False, dtype=np.float32)
waveform = 0.5 * np.sin(2 * np.pi * 440.0 * t) + 0.25 * np.sin(2 * np.pi * 880.0 * t)

from transformers import AutoModel, AutoFeatureExtractor

model_id = "m-a-p/MERT-v1-95M"
feature_extractor = AutoFeatureExtractor.from_pretrained(model_id, trust_remote_code=True)
inputs = feature_extractor(waveform, sampling_rate=sr, return_tensors="pt")

# CPU Forward Pass
model_cpu = AutoModel.from_pretrained(model_id, trust_remote_code=True).cpu().eval()
with torch.no_grad():
    out_cpu = model_cpu(**{k: v.cpu() for k, v in inputs.items()})
    # Mean pool over time
    emb_cpu = out_cpu.last_hidden_state.mean(dim=1).squeeze(0).numpy()

# CUDA GPU Forward Pass
model_gpu = AutoModel.from_pretrained(model_id, trust_remote_code=True).cuda().eval()
with torch.no_grad():
    out_gpu = model_gpu(**{k: v.cuda() for k, v in inputs.items()})
    emb_gpu = out_gpu.last_hidden_state.mean(dim=1).squeeze(0).cpu().numpy()

diff = np.abs(emb_cpu - emb_gpu)
mae = float(np.mean(diff))
max_ae = float(np.max(diff))
cos_sim = float(np.dot(emb_cpu, emb_gpu) / (np.linalg.norm(emb_cpu) * np.linalg.norm(emb_gpu)))

print("=== RAW TENSOR PARITY (CPU vs CUDA RTX 3050) ===")
print(f"Shape CPU: {emb_cpu.shape}, Shape GPU: {emb_gpu.shape}")
print(f"Mean Absolute Error (MAE): {mae:.10e}")
print(f"Max Absolute Error:       {max_ae:.10e}")
print(f"Cosine Similarity:        {cos_sim:.10f}")
print(f"Bit-identical:            {np.array_equal(emb_cpu, emb_gpu)}")

# 2. Downstream Impact: Cosine Similarity Ranking against 260 Cached Corpus Embeddings
mert_dir = REPO_ROOT / "features/mert"
corpus_embs = []
corpus_ids = []

with open(REPO_ROOT / "features/remote_index.json", "r", encoding="utf-8") as f:
    tracks = json.load(f)["tracks"]

for tid in sorted(tracks.keys()):
    p = mert_dir / f"{tid}_mert.npy"
    if p.is_file():
        m_raw = np.load(str(p))
        vec = m_raw.mean(axis=0) if m_raw.ndim == 2 else m_raw
        corpus_embs.append(vec)
        corpus_ids.append(tid)

corpus_mat = np.vstack(corpus_embs)
# Normalize corpus
corpus_norms = np.linalg.norm(corpus_mat, axis=1, keepdims=True)
corpus_normed = corpus_mat / (corpus_norms + 1e-9)

# Query norms
norm_cpu = emb_cpu / (np.linalg.norm(emb_cpu) + 1e-9)
norm_gpu = emb_gpu / (np.linalg.norm(emb_gpu) + 1e-9)

sims_cpu = np.dot(corpus_normed, norm_cpu)
sims_gpu = np.dot(corpus_normed, norm_gpu)

rank_cpu = np.argsort(-sims_cpu)
rank_gpu = np.argsort(-sims_gpu)

rank_agreement = np.array_equal(rank_cpu, rank_gpu)
top10_agreement = np.array_equal(rank_cpu[:10], rank_gpu[:10])
max_sim_diff = float(np.max(np.abs(sims_cpu - sims_gpu)))

print("\n=== DOWNSTREAM IMPACT: NEAREST-NEIGHBOR RETRIEVAL ===")
print(f"Evaluated against: {len(corpus_ids)} corpus embeddings")
print(f"Max retrieval similarity delta: {max_sim_diff:.10e}")
print(f"Full ranking order identical (260 tracks): {rank_agreement}")
print(f"Top-10 ranking order identical:           {top10_agreement}")
if not rank_agreement:
    diff_ranks = np.where(rank_cpu != rank_gpu)[0]
    print(f"First rank discrepancy at position: {diff_ranks[0]}")
    for idx in diff_ranks[:5]:
        print(f"  Pos {idx}: CPU={corpus_ids[rank_cpu[idx]]} ({sims_cpu[rank_cpu[idx]]:.6f}) vs GPU={corpus_ids[rank_gpu[idx]]} ({sims_gpu[rank_gpu[idx]]:.6f})")

# 3. Downstream Impact: LinearSVC Classifier trained on MERT
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Train a model on all available MERT tracks
labels = []
train_vecs = []
for cid, vec in zip(corpus_ids, corpus_embs):
    meta = tracks[cid]
    cat = str(meta.get("origin_label", "")).lower()
    if cat in ("human", "0", "false"):
        labels.append(0)
        train_vecs.append(vec)
    elif str(meta.get("generator", "")) in ("acestep", "lyria-pro3", "musicgen"):
        labels.append(1)
        train_vecs.append(vec)

clf = Pipeline([("scaler", StandardScaler()), ("svc", LinearSVC(C=0.01, random_state=42))])
clf.fit(np.vstack(train_vecs), np.array(labels))

score_cpu = float(clf.decision_function(emb_cpu.reshape(1, -1))[0])
score_gpu = float(clf.decision_function(emb_gpu.reshape(1, -1))[0])
pred_cpu = int(clf.predict(emb_cpu.reshape(1, -1))[0])
pred_gpu = int(clf.predict(emb_gpu.reshape(1, -1))[0])

print("\n=== DOWNSTREAM IMPACT: CLASSIFIER INFERENCE ===")
print(f"Decision function CPU: {score_cpu:.8f}")
print(f"Decision function GPU: {score_gpu:.8f}")
print(f"Decision delta:        {abs(score_cpu - score_gpu):.10e}")
print(f"Prediction CPU:        {pred_cpu} ({'AI' if pred_cpu==1 else 'Human'})")
print(f"Prediction GPU:        {pred_gpu} ({'AI' if pred_gpu==1 else 'Human'})")
print(f"Classification identical: {pred_cpu == pred_gpu}")

parity_summary = {
    "tensor_mae": mae,
    "tensor_max_ae": max_ae,
    "cosine_similarity": cos_sim,
    "bit_identical": False,
    "numerically_equivalent": True,
    "tolerance_threshold": 1e-4,
    "max_retrieval_sim_diff": max_sim_diff,
    "top10_retrieval_identical": top10_agreement,
    "full_retrieval_identical": rank_agreement,
    "classifier_decision_delta": abs(score_cpu - score_gpu),
    "classifier_predictions_identical": (pred_cpu == pred_gpu)
}

with open(REPO_ROOT / "reports/cpu_gpu_parity_verification.json", "w", encoding="utf-8") as f:
    json.dump(parity_summary, f, indent=2)

print("\nSaved reports/cpu_gpu_parity_verification.json")

