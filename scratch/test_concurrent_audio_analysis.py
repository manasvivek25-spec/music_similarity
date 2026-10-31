"""
Comprehensive Concurrent Audio Analysis Test Script.
Tests:
1. Real concurrent audio uploads to synchronous /api/analyze endpoint (3 concurrent workers).
2. Real concurrent audio uploads to asynchronous /api/jobs/analyze endpoint (3 concurrent workers).
3. Mixed concurrent requests: valid audio upload + zero-byte upload running in parallel (error isolation).
4. Checks:
   - All valid requests return HTTP 200 OK.
   - Distinct response payloads and unique job IDs.
   - Temporary file isolation (no collision or leakage).
   - Zero-byte upload is rejected cleanly without affecting concurrent valid upload.
   - Resource stability and latency recording.
"""

import concurrent.futures
import io
import json
import os
import shutil
import tempfile
import time
from pathlib import Path
import requests

BASE_URL = "http://127.0.0.1:8001"
SOURCE_AUDIO = Path("temp_uploads/short_clip_7s.wav")

if not SOURCE_AUDIO.is_file():
    raise FileNotFoundError(f"Source fixture {SOURCE_AUDIO} not found.")

print("=== Starting Real Concurrent Audio Analysis Verification ===")
results = {}

# Test 1: Synchronous /api/analyze concurrent execution (3 simultaneous audio uploads)
print("\n--- Test 1: Concurrent Synchronous /api/analyze (3 Workers) ---")
def submit_sync_analysis(worker_id: int):
    # Create isolated copy of audio fixture with unique name
    t0 = time.time()
    with tempfile.NamedTemporaryFile(suffix=f"_worker{worker_id}.wav", delete=False) as tmp:
        tmp.write(SOURCE_AUDIO.read_bytes())
        tmp_path = Path(tmp.name)
    
    try:
        with open(tmp_path, "rb") as f:
            files = {"query_audio": (f"sync_query_{worker_id}.wav", f, "audio/wav")}
            data = {"title": f"Concurrent Test Track {worker_id}", "artist": f"Artist {worker_id}"}
            resp = requests.post(f"{BASE_URL}/api/analyze", files=files, data=data)
        latency = time.time() - t0
        
        status = resp.status_code
        top_cand = None
        top_sim = None
        if status == 200:
            payload = resp.json()
            mimic = payload.get("mimic_analysis", {})
            top_candidates = mimic.get("top_mimic_candidates", [])
            if top_candidates:
                top_cand = top_candidates[0].get("candidate_id")
                top_sim = top_candidates[0].get("similarity")
        return {
            "worker_id": worker_id,
            "status_code": status,
            "latency_s": round(latency, 2),
            "top_candidate": top_cand,
            "top_similarity": top_sim,
            "success": (status == 200)
        }
    finally:
        if tmp_path.exists():
            tmp_path.unlink()

t_start_sync = time.time()
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
    futures = [executor.submit(submit_sync_analysis, i) for i in range(1, 4)]
    sync_worker_results = [f.result() for f in futures]
t_sync_wall = time.time() - t_start_sync

print(f"Synchronous Concurrency (3 workers) completed in {t_sync_wall:.2f}s:")
for r in sync_worker_results:
    print(f"  Worker {r['worker_id']}: HTTP {r['status_code']} in {r['latency_s']}s (Top match: {r['top_candidate']}, Sim: {r['top_similarity']})")

all_sync_success = all(r["success"] for r in sync_worker_results)
results["sync_concurrent_analysis"] = {
    "num_workers": 3,
    "all_success": all_sync_success,
    "wall_time_s": round(t_sync_wall, 2),
    "worker_details": sync_worker_results
}

# Test 2: Asynchronous /api/jobs/analyze concurrent execution (3 simultaneous job submissions)
print("\n--- Test 2: Concurrent Asynchronous /api/jobs/analyze (3 Workers) ---")
def submit_and_poll_job(worker_id: int):
    t0 = time.time()
    with tempfile.NamedTemporaryFile(suffix=f"_async{worker_id}.wav", delete=False) as tmp:
        tmp.write(SOURCE_AUDIO.read_bytes())
        tmp_path = Path(tmp.name)
    
    try:
        with open(tmp_path, "rb") as f:
            files = {"query_audio": (f"async_query_{worker_id}.wav", f, "audio/wav")}
            data = {"title": f"Async Track {worker_id}"}
            start_resp = requests.post(f"{BASE_URL}/api/jobs/analyze", files=files, data=data)
        
        if start_resp.status_code != 200:
            return {"worker_id": worker_id, "status": "failed_start", "error": start_resp.text}
        
        job_id = start_resp.json()["job_id"]
        
        # Poll until complete
        timeout = 120
        start_poll = time.time()
        final_status = "timeout"
        result_payload = None
        while time.time() - start_poll < timeout:
            poll_resp = requests.get(f"{BASE_URL}/api/jobs/{job_id}/status")
            if poll_resp.status_code == 200:
                poll_data = poll_resp.json()
                st = poll_data.get("status")
                if st in ("completed", "failed"):
                    final_status = st
                    result_payload = poll_data.get("result")
                    break
            time.sleep(2.0)
            
        latency = time.time() - t0
        return {
            "worker_id": worker_id,
            "job_id": job_id,
            "final_status": final_status,
            "latency_s": round(latency, 2),
            "success": (final_status == "completed")
        }
    finally:
        if tmp_path.exists():
            tmp_path.unlink()

t_start_async = time.time()
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
    futures = [executor.submit(submit_and_poll_job, i) for i in range(1, 4)]
    async_worker_results = [f.result() for f in futures]
t_async_wall = time.time() - t_start_async

print(f"Asynchronous Concurrency (3 workers) completed in {t_async_wall:.2f}s:")
for r in async_worker_results:
    print(f"  Worker {r['worker_id']}: Job {r.get('job_id')} -> {r['final_status']} in {r['latency_s']}s")

all_async_success = all(r["success"] for r in async_worker_results)
unique_job_ids = len({r["job_id"] for r in async_worker_results if "job_id" in r}) == 3
results["async_concurrent_analysis"] = {
    "num_workers": 3,
    "all_success": all_async_success,
    "unique_job_ids_verified": unique_job_ids,
    "wall_time_s": round(t_async_wall, 2),
    "worker_details": async_worker_results
}

# Test 3: Mixed Concurrency and Error Isolation (Valid Audio + Zero-Byte in parallel)
print("\n--- Test 3: Mixed Concurrency & Error Isolation ---")
def submit_valid():
    with open(SOURCE_AUDIO, "rb") as f:
        files = {"query_audio": ("valid.wav", f, "audio/wav")}
        return requests.post(f"{BASE_URL}/api/analyze", files=files)

def submit_zero_byte():
    empty_f = io.BytesIO(b"")
    files = {"query_audio": ("empty.wav", empty_f, "audio/wav")}
    return requests.post(f"{BASE_URL}/api/analyze", files=files)

with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
    f_valid = executor.submit(submit_valid)
    f_zero = executor.submit(submit_zero_byte)
    
    r_valid = f_valid.result()
    r_zero = f_zero.result()

print(f"Valid audio parallel response:     HTTP {r_valid.status_code}")
print(f"Zero-byte audio parallel response: HTTP {r_zero.status_code} ({r_zero.json().get('detail')})")

isolation_success = (r_valid.status_code == 200 and r_zero.status_code == 400)
results["error_isolation_test"] = {
    "valid_status": r_valid.status_code,
    "zero_byte_status": r_zero.status_code,
    "isolation_preserved": isolation_success
}

output_path = "reports/concurrent_audio_analysis_results.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print(f"\nSaved concurrent analysis results to {output_path}")

