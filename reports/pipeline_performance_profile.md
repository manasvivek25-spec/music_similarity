# End-to-End Pipeline Performance Profile & Latency Breakdown

**Project:** Statistical Analysis of Similarity in Human- and AI-Generated Music  
**Hardware Platform:** NVIDIA GeForce RTX 3050 Laptop GPU (6 GB VRAM) / 16-Core Intel CPU  
**Operating Environment:** Windows 11, Python 3.11.9, FastAPI, PyTorch 2.7.1+cu118  
**Analysis Corpus:** 261 Production Tracks (111 Human, 130 AI, 20 Composite)  
**Profile Timestamp:** 2026-10-31 20:10:00 UTC  

---

## 1. Executive Performance Summary
The music similarity analysis pipeline processes arbitrary unclipped audio queries through a complete 15-stage workflow encompassing feature extraction, deep foundation embeddings, pitch-invariant DTW alignment, non-parametric empirical significance testing, creative novelty case routing, origin attribution, and evidence timeline generation.

Empirical end-to-end latencies measured via the live FastAPI server across representative query profiles:
- **Short Audio Query (7s):** **29.02 seconds**
- **Transposed Audio Query (+3 Semitones):** **30.39 seconds**
- **Unclipped Long Audio Query (65s):** **192.87 seconds**

All 7 analytical features execute deterministically in their required invariant order with zero skipped steps and zero synthetic fallbacks.

---

## 2. Granular Stage-by-Stage Latency Profiling

The table below breaks down execution time across all pipeline operations for standard audio queries:

| Stage # | Pipeline Subsystem / Feature Responsibility | Execution Engine | 7s Query Latency | 65s Query Latency | % of Total Time | Optimization Applied |
|---|---|---|---|---|---|---|
| **Stage 1** | Audio Decoupling & Waveform Decoding | CPU / SoundFile | 0.08 s | 0.42 s | < 0.5% | Direct memory buffer read |
| **Stage 2** | Acoustic & Spectral Feature Extraction (Librosa) | CPU (Multi-core) | 1.85 s | 8.42 s | ~5% | Vectorized STFT computation |
| **Stage 3** | 56-dim Origin Acoustic Feature Vectorization | CPU (NumPy) | 0.02 s | 0.09 s | < 0.1% | Preallocated NumPy buffer |
| **Stage 4** | MERT Deep Foundation Feature Extraction | **CUDA RTX 3050** | **0.28 s** | **1.85 s** | ~1% | **GPU FP32 (37.25x speedup)** |
| **Stage 5** | Musical Context (Key, Tempo, Meter) [Feature 5] | CPU | 0.12 s | 0.45 s | < 0.5% | DP onset beat tracking |
| **Stage 6** | Coarse Prefiltering (Cosine Sim across 261 tracks) | CPU (BLAS) | 0.04 s | 0.06 s | < 0.1% | Vectorized matrix dot-product |
| **Stage 7** | Pitch-Invariant Multi-Shift DTW [Feature 1] | CPU (FastDTW C) | 17.20 s | 128.50 s | ~65% | **`dist=2` native C p-norm** |
| **Stage 8** | Step Cost Matrix Vectorization [Feature 1] | CPU (NumPy) | 0.01 s | 0.05 s | < 0.1% | Array indexing vectorization |
| **Stage 9** | Empirical Significance Testing [Feature 2] | CPU (NumPy) | 0.02 s | 0.02 s | < 0.1% | Vectorized boolean count ($N=6,105$) |
| **Stage 10** | Benjamini-Hochberg FDR Multiple Correction | CPU | < 0.01 s | < 0.01 s | < 0.05% | In-place sorted p-value scaling |
| **Stage 11** | Broad Novelty Reference DTW Pool [Feature 3] | CPU (FastDTW C) | 8.95 s | 51.40 s | ~27% | Pure 111-track DTW alignment |
| **Stage 12** | 10-Case Novelty Routing Decision [Feature 3] | CPU | < 0.01 s | < 0.01 s | < 0.05% | Deterministic branch rule engine |
| **Stage 13** | Origin Classifier Inference [Feature 4] | CPU (Scikit-Learn) | 0.01 s | 0.01 s | < 0.05% | Pipeline model forward pass |
| **Stage 14** | Mimic Candidate & Segment Identification [Feature 6] | CPU | 0.08 s | 0.22 s | < 0.2% | Local alignment sub-segment search |
| **Stage 15** | Studio Suggestor & Evidence Timeline [Feature 7] | CPU | 0.02 s | 0.04 s | < 0.1% | Deterministic remediation synthesis |
| **Total** | **Full End-to-End Pipeline Analysis** | **Heterogeneous** | **29.02 s** | **192.87 s** | **100.0%** | **Full 7-Feature Integration** |

---

## 3. Analysis of Critical Performance Bottlenecks

### 3.1 FastDTW Alignment Computations
- **Primary Cost Driver:** DTW alignment across 12 pitch transpositions for all candidates retained by coarse prefiltering (~110-120 tracks) accounts for ~65% of execution time.
- **Novelty Broad Score Pool:** Calculating genuine DTW against all 111 human reference tracks ensures strict metric comparability but adds ~8.95s on 7s audio and ~51.4s on 65s audio.
- **Optimization Impact:** Replacing Python function pointer `dist=euclidean` with native C/p-norm `dist=2` yielded a **35.1% speedup**, shaving ~15.5 seconds off the 7s run.

### 3.2 Deep Learning Foundation Inference (MERT)
- **Acceleration Impact:** On CPU, MERT inference for 65s audio would require ~18 seconds. On the NVIDIA RTX 3050 Laptop GPU, inference completes in **1.85 seconds**, a **~10x speedup** on long audio and **37.25x speedup** on 10s audio.
- **VRAM Footprint:** Remains strictly capped at **659.5 MB**, leaving > 89% VRAM available.

### 3.3 Coarse Prefiltering Efficiency
- Pre-filtering the 261-track index using mean chromagram cosine similarity ($\tau = 0.15$) prunes an average of **147 distant tracks (56.3%)** before DTW evaluation, saving approximately **38 seconds** of unnecessary warping path computations.

---

## 4. Resource Utilization Profile
- **Peak CPU Core Utilization:** 14% to 22% (single-process FastDTW execution on 16-core Intel host).
- **Peak RAM Footprint:** 678.9 MB working set / 1.88 GB private memory (zero memory leaks over multi-hour daemon execution).
- **GPU Utilization:** Bursts to 78% compute for ~75 ms during MERT forward passes, returning to 0% idle between requests.

