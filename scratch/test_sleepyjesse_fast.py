"""
Fast diagnostic on SleepyJesse/ai_music_large remote capabilities with immediate flushing.
"""
import requests
import time

print("--- Check 1: Hugging Face Serverless Dataset API ---", flush=True)
url = "https://datasets-server.huggingface.co/rows?dataset=SleepyJesse%2Fai_music_large&config=default&split=train&offset=0&limit=1"
try:
    t0 = time.time()
    r = requests.get(url, timeout=8)
    print(f"Status: {r.status_code} in {time.time()-t0:.2f}s", flush=True)
    if r.status_code == 200:
        data = r.json()
        print("Rows response keys:", list(data.keys()), flush=True)
        print("First row keys:", list(data["rows"][0]["row"].keys()), flush=True)
    else:
        print(f"Serverless API Error: {r.status_code} - {r.text[:300]}", flush=True)
except Exception as e:
    print(f"Check 1 Exception: {e}", flush=True)

print("\n--- Check 2: HTTP Range Request on Shard 0 ---", flush=True)
url_parquet = "https://huggingface.co/datasets/SleepyJesse/ai_music_large/resolve/main/data/train-00000-of-00147.parquet"
try:
    t0 = time.time()
    # Check HEAD first
    r_head = requests.head(url_parquet, timeout=8, allow_redirects=True)
    print(f"HEAD Status: {r_head.status_code} in {time.time()-t0:.2f}s", flush=True)
    print(f"Final URL: {r_head.url[:80]}...", flush=True)
    print(f"Content-Length: {r_head.headers.get('Content-Length')} bytes ({int(r_head.headers.get('Content-Length', 0))/(1024*1024):.1f} MB)", flush=True)
    print(f"Accept-Ranges: {r_head.headers.get('Accept-Ranges')}", flush=True)

    # Now Range GET 0-1023
    t0 = time.time()
    r_range = requests.get(url_parquet, headers={"Range": "bytes=0-1023"}, timeout=8, allow_redirects=True)
    print(f"GET Range (bytes=0-1023) Status: {r_range.status_code} in {time.time()-t0:.2f}s", flush=True)
    print(f"Content-Range: {r_range.headers.get('Content-Range')}", flush=True)
    print(f"Bytes received: {len(r_range.content)}", flush=True)
    print(f"Magic header bytes: {r_range.content[:4]} (Expected b'PAR1' for Parquet)", flush=True)
except Exception as e:
    print(f"Check 2 Exception: {e}", flush=True)

print("\n--- Check 3: Parquet Footer Range Request (Metadata) ---", flush=True)
try:
    total_len = int(r_head.headers.get('Content-Length', 0))
    if total_len > 10000:
        # Parquet footer is at the end of the file. Read last 16384 bytes
        footer_start = total_len - 16384
        headers = {"Range": f"bytes={footer_start}-{total_len-1}"}
        t0 = time.time()
        r_footer = requests.get(url_parquet, headers=headers, timeout=8, allow_redirects=True)
        print(f"GET Footer Status: {r_footer.status_code} in {time.time()-t0:.2f}s", flush=True)
        print(f"Footer magic end: {r_footer.content[-4:]} (Expected b'PAR1')", flush=True)
        # Read footer size
        footer_size = int.from_bytes(r_footer.content[-8:-4], 'little')
        print(f"Parquet FileMetaData length: {footer_size} bytes", flush=True)
except Exception as e:
    print(f"Check 3 Exception: {e}", flush=True)

