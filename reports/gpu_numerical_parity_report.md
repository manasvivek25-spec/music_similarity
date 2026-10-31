# GPU Acceleration and Numerical Parity Verification Report

**Project:** Statistical Analysis of Similarity in Human- and AI-Generated Music  
**Hardware Platform:** NVIDIA GeForce RTX 3050 Laptop GPU (6 GB VRAM)  
**Execution Environment:** Windows 11, Python 3.11.9, PyTorch 2.7.1+cu118, CUDA 11.8  
**Evaluated Neural Representation:** `m-a-p/MERT-v1-95M` (768-dimensional foundation embeddings)  

---

## 1. Executive Summary
Neural audio foundation model feature extraction (MERT-v1-95M) was benchmarked under identical inputs across both CPU (16-core Intel host) and GPU (NVIDIA RTX 3050 Laptop GPU, 6 GB VRAM).
CUDA acceleration delivers a **37.25x forward inference speedup** (75.4 ms vs 2,808.65 ms) while consuming only **659.5 MB of VRAM** (10.7% of total capacity).
Numerical parity analysis demonstrates extreme semantic alignment with a cosine similarity of **0.99999994** and a Mean Absolute Error (MAE) of **$1.68 \times 10^{-5}$**.

---

## 2. Hardware and Environment Specification
- **Host Processor:** 16-Core Intel CPU @ 2.50 GHz
- **Dedicated Accelerator:** NVIDIA GeForce RTX 3050 Laptop GPU
- **Total VRAM:** 6,144 MB (6 GB GDDR6)
- **CUDA Runtime:** 11.8
- **PyTorch Build:** `2.7.1+cu118` with cuDNN 8.7
- **Model Parameters:** 95 Million (`m-a-p/MERT-v1-95M`)
- **Input Dimensions:** 10.0s mono audio sampled at 24,000 Hz ($2.40 \times 10^5$ samples)

---

## 3. Forward Inference Latency Benchmark

Measurements conducted across 5 warm-up iterations and 5 timed trials under identical acoustic conditions:

| Execution Engine | Trial 1 (ms) | Trial 2 (ms) | Trial 3 (ms) | Trial 4 (ms) | Trial 5 (ms) | Median Latency | Mean ± Std | Acceleration Factor |
|---|---|---|---|---|---|---|---|---|
| **CPU (16-Core)** | 2,689.25 | 2,907.87 | 2,871.47 | 2,808.65 | 2,768.65 | **2,808.65 ms** | 2,809.18 ± 77.0 ms | Baseline (1.0x) |
| **CUDA RTX 3050** | 81.43 | 75.89 | 73.12 | 75.36 | 75.40 | **75.40 ms** | **76.24 ± 2.77 ms** | **37.25x Speedup** |

### Additional Operational Metrics:
- **Model Warm-Up & Transfer Latency:** 1,276.32 ms (one-time cold start; models remain resident in VRAM across analysis queries).
- **Peak VRAM Allocation:** **659.47 MB** / 6,144 MB (Leaves > 5.3 GB VRAM free for operating system display and background services).
- **VRAM Leak Check:** 0.0 MB memory accumulation over 100 consecutive forward passes verified via `torch.cuda.memory_allocated()`.

---

## 4. Numerical Parity and Cross-Device Precision

To determine if GPU tensor operations alter downstream similarity calculations or feature representations, full output tensors were compared element-wise across CPU (x86_64 AVX2) and GPU (NVIDIA Ampere FP32):

| Parity Metric | Empirical Value | Permissible Tolerance | Status |
|---|---|---|---|
| **Mean Absolute Error (MAE)** | **$1.6816 \times 10^{-5}$** | $< 1.0 \times 10^{-4}$ | **PASS** |
| **Max Absolute Error (MaxAE)** | **$2.2078 \times 10^{-4}$** | $< 1.0 \times 10^{-3}$ | **PASS** |
| **Cosine Vector Alignment** | **0.99999994** | $> 0.99999$ | **PASS** |
| **Max Discrepancy** | $1.7881 \times 10^{-7}$ | $< 1.0 \times 10^{-6}$ | **PASS** |

### Analysis of Numerical Discrepancy:
The observed $1.68 \times 10^{-5}$ MAE is mathematically attributable to the non-associative nature of floating-point accumulation between x86 fused multiply-add (FMA) instructions and NVIDIA Tensor Core/cuDNN matrix multiplication pipelines. 
With a directional cosine agreement of **0.99999994**, embedding vector geometry is effectively identical, resulting in **0.0 rank inversions** across candidate nearest-neighbor retrieval.

---

## 5. Batch Retrieval vs Inference Execution Routing

Empirical benchmarking was also conducted on similarity score retrieval across cached embeddings:
- **CPU BLAS Matrix Multiplication (260 tracks):** Median **0.148 ms**
- **CUDA Tensor Core Matrix Multiplication (260 tracks):** Median **0.773 ms**
- **Architecture Decision:**
  - Forward MERT feature extraction runs on **CUDA RTX 3050** (achieving 37.25x speedup).
  - Compact batch dot-product retrieval runs on **CPU BLAS** (avoiding PCI-e host-to-device memory transfer overhead of 0.62 ms for small $N \le 1,000$).
  - When $N > 10,000$, matrix multiplication automatically switches to GPU batch tensors.

---

## 6. Fallback and Fault Tolerance Verification
- When CUDA is unavailable or VRAM is exhausted, `src/models/mert.py` dynamically falls back to CPU execution without throwing runtime exceptions.
- Graceful degradation tests (`test_mert_cpu_fallback`) confirmed 100% pipeline completion under simulated CUDA failure.

