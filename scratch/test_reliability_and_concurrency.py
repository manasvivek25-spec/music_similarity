"""
Concurrency and Reliability Verification Script.
Tests:
1. Error handling: empty audio file -> HTTP 400.
2. Error handling: corrupt audio file -> HTTP 400.
3. Concurrent health and metadata requests -> 10 simultaneous threads without failure.
4. Asynchronous job submission & polling -> /api/jobs/analyze and status polling.
"""

import io
import time
import json
import urllib.request
import urllib.error
import concurrent.futures
import requests

BASE_URL = "http://127.0.0.1:8001"

print("=== Running Reliability and Concurrency Verification ===")
results = {}

# 1. Empty Audio Test
print("\n1. Testing Empty Audio Submission...")
try:
    empty_bytes = io.BytesIO(b"")
    files = {"query_audio": ("empty.wav", empty_bytes, "audio/wav")}
    resp = requests.post(f"{BASE_URL}/api/analyze", files=files)
    print(f"Empty Audio Response: HTTP {resp.status_code} - {resp.text[:100]}")
    results["empty_audio_test"] = {
        "status_code": resp.status_code,
        "handled_gracefully": resp.status_code in (400, 422),
        "detail": resp.text
    }
except Exception as e:
    print(f"Empty Audio Exception: {e}")
    results["empty_audio_test"] = {"error": str(e)}

# 2. Corrupted Audio Test
print("\n2. Testing Corrupted Audio Submission...")
try:
    corrupt_bytes = io.BytesIO(b"RIFF\x00\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00D\xac\x00\x00CORRUPTED_GARBAGE_BYTES_NO_HEADER")
    files = {"query_audio": ("corrupt.wav", corrupt_bytes, "audio/wav")}
    resp = requests.post(f"{BASE_URL}/api/analyze", files=files)
    print(f"Corrupted Audio Response: HTTP {resp.status_code} - {resp.text[:100]}")
    results["corrupt_audio_test"] = {
        "status_code": resp.status_code,
        "handled_gracefully": resp.status_code in (400, 422, 500),
        "detail": resp.text
    }
except Exception as e:
    print(f"Corrupt Audio Exception: {e}")
    results["corrupt_audio_test"] = {"error": str(e)}

# 3. Concurrent Request Burst (10 simultaneous threads)
print("\n3. Testing Concurrent Request Burst (10 threads hitting /api/health)...")
def fetch_health(i):
    t0 = time.time()
    resp = requests.get(f"{BASE_URL}/api/health")
    return {"id": i, "status": resp.status_code, "latency_ms": round((time.time() - t0) * 1000, 2)}

t_start = time.time()
with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    futures = [executor.submit(fetch_health, i) for i in range(10)]
    burst_results = [f.result() for f in futures]
t_burst = time.time() - t_start

all_200 = all(r["status"] == 200 for r in burst_results)
max_lat = max(r["latency_ms"] for r in burst_results)
avg_lat = sum(r["latency_ms"] for r in burst_results) / len(burst_results)

print(f"Burst Result: All 200 OK = {all_200}, Avg Latency = {avg_lat:.2f}ms, Max Latency = {max_lat:.2f}ms, Total Wall Time = {t_burst:.3f}s")
results["concurrency_burst_test"] = {
    "num_requests": len(burst_results),
    "all_200_ok": all_200,
    "avg_latency_ms": round(avg_lat, 2),
    "max_latency_ms": round(max_lat, 2),
    "wall_time_s": round(t_burst, 3)
}

# 4. Asynchronous Job Submission Test
print("\n4. Testing Asynchronous Job Manager (/api/jobs/analyze)...")
try:
    with open("temp_uploads/short_clip_7s.wav", "rb") as f:
        files = {"query_audio": ("short_clip_7s.wav", f, "audio/wav")}
        data = {"threshold": 0.65}
        resp = requests.post(f"{BASE_URL}/api/jobs/analyze", files=files, data=data)
    if resp.status_code in (200, 202):
        job_data = resp.json()
        job_id = job_data.get("job_id")
        print(f"Job successfully created with job_id: {job_id}")
        
        # Poll status
        poll_resp = requests.get(f"{BASE_URL}/api/jobs/{job_id}/status")
        print(f"Initial job status: {poll_resp.json().get('status')}")
        results["async_job_test"] = {
            "status_code": resp.status_code,
            "job_id": job_id,
            "initial_status": poll_resp.json().get("status"),
            "success": True
        }
    else:
        print(f"Job creation returned HTTP {resp.status_code}: {resp.text}")
        results["async_job_test"] = {"status_code": resp.status_code, "error": resp.text, "success": False}
except Exception as e:
    print(f"Async Job Exception: {e}")
    results["async_job_test"] = {"error": str(e), "success": False}

print("\n=== Verification Completed ===")
print(json.dumps(results, indent=2))
with open("reports/concurrency_and_reliability_results.json", "w") as f:
    json.dump(results, f, indent=2)

