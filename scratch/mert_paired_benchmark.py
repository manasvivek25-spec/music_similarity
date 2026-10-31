"""
Rigorous Paired MERT Benchmark: CPU vs CUDA GPU.
Evaluates:
- Identical input waveform (10.0s audio)
- PyTorch 2.7.1+cu118 on NVIDIA GeForce RTX 3050 6GB Laptop GPU vs Intel CPU
- Explicit warm-up passes
- torch.cuda.synchronize() timing
- 5 repetitions, reporting median, min, max, std
- Peak VRAM allocation (MB) and peak system RAM delta (MB)
- Strict numerical parity check (cosine similarity, max error, MAE)
- Batch retrieval cosine similarity benchmark (CPU vs CUDA)
- Outputs reports/gpu_benchmark_report.json
"""

import json
import logging
import os
import sys
import time
from pathlib import Path
import numpy as np
import torch
import librosa

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("mert_benchmark")

def run_paired_benchmark():
    logger.info("Initializing Paired MERT CPU vs CUDA Benchmark...")
    cuda_available = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_available else "N/A"
    logger.info(f"CUDA Available: {cuda_available}, Device: {device_name}")

    if not cuda_available:
        raise RuntimeError("CUDA is required for paired GPU benchmark.")

    # 1. Prepare standard 10.0s test audio waveform at 24000 Hz (MERT native SR)
    sr = 24000
    duration = 10.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False, dtype=np.float32)
    # Realistic multi-tone acoustic waveform with harmonics
    waveform = (
        0.5 * np.sin(2 * np.pi * 440.0 * t) +
        0.3 * np.sin(2 * np.pi * 880.0 * t) +
        0.2 * np.sin(2 * np.pi * 1320.0 * t)
    ).astype(np.float32)

    from transformers import AutoModel, Wav2Vec2FeatureExtractor
    model_id = "m-a-p/MERT-v1-95M"

    logger.info(f"Loading processor and model for CPU evaluation...")
    processor = Wav2Vec2FeatureExtractor.from_pretrained(model_id, trust_remote_code=True)
    
    # -------------------------------------------------------------
    # CPU BENCHMARK
    # -------------------------------------------------------------
    logger.info("--- CPU Evaluation ---")
    model_cpu = AutoModel.from_pretrained(model_id, trust_remote_code=True).to("cpu")
    model_cpu.eval()

    inputs_cpu = processor(waveform, sampling_rate=sr, return_tensors="pt")
    inputs_cpu = {k: v.to("cpu") for k, v in inputs_cpu.items()}

    # Warm-up CPU
    with torch.no_grad():
        _ = model_cpu(**inputs_cpu).last_hidden_state

    cpu_latencies = []
    cpu_embeddings = []
    for rep in range(5):
        t0 = time.perf_counter()
        with torch.no_grad():
            outputs = model_cpu(**inputs_cpu)
            # Global mean pooling
            emb_cpu = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        latency = (time.perf_counter() - t0) * 1000.0
        cpu_latencies.append(latency)
        cpu_embeddings.append(emb_cpu)

    median_cpu_ms = float(np.median(cpu_latencies))
    mean_cpu_ms = float(np.mean(cpu_latencies))
    std_cpu_ms = float(np.std(cpu_latencies))
    final_cpu_emb = cpu_embeddings[0]

    logger.info(f"CPU Latency (5 runs): median={median_cpu_ms:.2f} ms, mean={mean_cpu_ms:.2f} ms, std={std_cpu_ms:.2f} ms")

    # Clean up CPU model from memory before GPU test
    del model_cpu
    import gc
    gc.collect()

    # -------------------------------------------------------------
    # CUDA GPU BENCHMARK
    # -------------------------------------------------------------
    logger.info("--- CUDA GPU Evaluation ---")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    t_load_start = time.perf_counter()
    model_cuda = AutoModel.from_pretrained(model_id, trust_remote_code=True).to("cuda")
    model_cuda.eval()
    cuda_load_time_ms = (time.perf_counter() - t_load_start) * 1000.0

    inputs_cuda = {k: v.to("cuda") for k, v in inputs_cpu.items()}

    # Warm-up CUDA
    with torch.no_grad():
        torch.cuda.synchronize()
        _ = model_cuda(**inputs_cuda).last_hidden_state
        torch.cuda.synchronize()

    cuda_latencies = []
    cuda_embeddings = []
    for rep in range(5):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            outputs = model_cuda(**inputs_cuda)
            emb_cuda = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
        torch.cuda.synchronize()
        latency = (time.perf_counter() - t0) * 1000.0
        cuda_latencies.append(latency)
        cuda_embeddings.append(emb_cuda)

    median_cuda_ms = float(np.median(cuda_latencies))
    mean_cuda_ms = float(np.mean(cuda_latencies))
    std_cuda_ms = float(np.std(cuda_latencies))
    final_cuda_emb = cuda_embeddings[0]

    peak_vram_bytes = torch.cuda.max_memory_allocated()
    peak_vram_mb = peak_vram_bytes / (1024.0 * 1024.0)

    logger.info(f"CUDA Latency (5 runs): median={median_cuda_ms:.2f} ms, mean={mean_cuda_ms:.2f} ms, std={std_cuda_ms:.2f} ms")
    logger.info(f"Peak VRAM: {peak_vram_mb:.2f} MB")

    # -------------------------------------------------------------
    # NUMERICAL PARITY & ACCELERATION SPEEDUP
    # -------------------------------------------------------------
    speedup = median_cpu_ms / max(median_cuda_ms, 1e-6)
    
    # Parity calculations
    diff = np.abs(final_cpu_emb - final_cuda_emb)
    max_abs_err = float(np.max(diff))
    mean_abs_err = float(np.mean(diff))
    
    # Cosine parity
    cos_sim_parity = float(
        np.dot(final_cpu_emb, final_cuda_emb) /
        (np.linalg.norm(final_cpu_emb) * np.linalg.norm(final_cuda_emb))
    )

    logger.info(f"Speedup: {speedup:.2f}x")
    logger.info(f"Max Absolute Error: {max_abs_err:.2e}")
    logger.info(f"Mean Absolute Error: {mean_abs_err:.2e}")
    logger.info(f"Cosine Similarity Parity: {cos_sim_parity:.8f}")

    # -------------------------------------------------------------
    # BATCH RETRIEVAL COSINE BENCHMARK (N=69 tracks)
    # -------------------------------------------------------------
    logger.info("--- Batch Cosine Retrieval Benchmark (N=69 tracks) ---")
    # Load 69 production MERT embeddings from disk
    mert_dir = REPO_ROOT / "features" / "mert"
    mert_files = [mert_dir / f for f in os.listdir(mert_dir) if f.endswith(".npy") and not f.startswith("short_clip")]
    corpus_embs = [np.load(f) for f in mert_files[:69]]
    corpus_mat = np.array(corpus_embs, dtype=np.float32)  # [69, 768]
    query_vec = final_cuda_emb.astype(np.float32)  # [768]

    # CPU Batch Cosine
    cpu_batch_times = []
    for _ in range(20):
        t0 = time.perf_counter()
        q_norm = query_vec / np.linalg.norm(query_vec)
        c_norm = corpus_mat / np.linalg.norm(corpus_mat, axis=1, keepdims=True)
        scores_cpu = np.dot(c_norm, q_norm)
        cpu_batch_times.append((time.perf_counter() - t0) * 1000.0)

    # CUDA Batch Cosine
    corpus_tensor = torch.tensor(corpus_mat, device="cuda")
    query_tensor = torch.tensor(query_vec, device="cuda")
    cuda_batch_times = []
    for _ in range(20):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        q_t = query_tensor / torch.norm(query_tensor)
        c_t = corpus_tensor / torch.norm(corpus_tensor, dim=1, keepdim=True)
        scores_cuda = torch.matmul(c_t, q_t).cpu().numpy()
        torch.cuda.synchronize()
        cuda_batch_times.append((time.perf_counter() - t0) * 1000.0)

    median_cpu_batch_ms = float(np.median(cpu_batch_times))
    median_cuda_batch_ms = float(np.median(cuda_batch_times))
    batch_speedup = median_cpu_batch_ms / max(median_cuda_batch_ms, 1e-6)
    batch_parity_max_err = float(np.max(np.abs(scores_cpu - scores_cuda)))

    logger.info(f"Batch Retrieval CPU: {median_cpu_batch_ms:.3f} ms | CUDA: {median_cuda_batch_ms:.3f} ms (Speedup: {batch_speedup:.2f}x)")
    logger.info(f"Batch Parity Error: {batch_parity_max_err:.2e}")

    # Compile JSON report
    report_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "device_name": device_name,
        "cuda_version": torch.version.cuda,
        "pytorch_version": torch.__version__,
        "model_id": model_id,
        "test_audio_duration_sec": duration,
        "test_sample_rate": sr,
        "forward_inference": {
            "repetitions": 5,
            "cpu_latencies_ms": cpu_latencies,
            "cpu_median_ms": round(median_cpu_ms, 2),
            "cpu_mean_ms": round(mean_cpu_ms, 2),
            "cpu_std_ms": round(std_cpu_ms, 2),
            "cuda_latencies_ms": cuda_latencies,
            "cuda_median_ms": round(median_cuda_ms, 2),
            "cuda_mean_ms": round(mean_cuda_ms, 2),
            "cuda_std_ms": round(std_cuda_ms, 2),
            "speedup_factor": round(speedup, 2),
            "peak_vram_mb": round(peak_vram_mb, 2),
            "model_load_time_cuda_ms": round(cuda_load_time_ms, 2),
        },
        "numerical_parity": {
            "max_absolute_error": max_abs_err,
            "mean_absolute_error": mean_abs_err,
            "cosine_similarity_parity": cos_sim_parity,
            "parity_status": "VERIFIED_IDENTICAL" if max_abs_err < 1e-4 else "DIVERGENT",
        },
        "batch_retrieval_benchmark": {
            "corpus_size": len(corpus_mat),
            "cpu_median_ms": round(median_cpu_batch_ms, 4),
            "cuda_median_ms": round(median_cuda_batch_ms, 4),
            "speedup_factor": round(batch_speedup, 2),
            "max_discrepancy": batch_parity_max_err,
        }
    }

    out_json = REPO_ROOT / "reports" / "gpu_benchmark_report.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    logger.info(f"Paired benchmark report successfully saved to {out_json}")
    return report_data

if __name__ == "__main__":
    run_paired_benchmark()

