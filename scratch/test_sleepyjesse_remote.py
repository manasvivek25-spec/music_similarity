"""
Diagnostic Script: Remote access verification for SleepyJesse/ai_music_large.
Tests 3 non-destructive remote access methods without downloading full dataset:
1. Hugging Face Serverless Datasets API (/info, /splits, /rows)
2. HTTP Range Requests on Parquet shards
3. Hugging Face datasets streaming=True generator
"""

import json
import os
import sys
import time
import requests

def test_hf_datasets_server_api():
    print("=== Method 1: Hugging Face Datasets Server API ===")
    url_info = "https://datasets-server.huggingface.co/info?dataset=SleepyJesse%2Fai_music_large"
    try:
        t0 = time.time()
        r = requests.get(url_info, timeout=10)
        dt = time.time() - t0
        print(f"API /info status: {r.status_code} in {dt:.2f}s")
        if r.status_code == 200:
            data = r.json()
            configs = list(data.get("dataset_info", {}).keys())
            print(f"Configs found: {configs}")
            print(f"Features: {list(data.get('dataset_info', {}).get('default', {}).get('features', {}).keys())}")
        else:
            print(f"Response: {r.text[:200]}")
    except Exception as e:
        print(f"Error querying /info: {e}")

    url_rows = "https://datasets-server.huggingface.co/rows?dataset=SleepyJesse%2Fai_music_large&config=default&split=train&offset=0&limit=2"
    try:
        t0 = time.time()
        r = requests.get(url_rows, timeout=10)
        dt = time.time() - t0
        print(f"API /rows status: {r.status_code} in {dt:.2f}s")
        if r.status_code == 200:
            rows_data = r.json()
            num_rows = len(rows_data.get("rows", []))
            print(f"Retrieved {num_rows} rows via serverless API.")
            if num_rows > 0:
                row0 = rows_data["rows"][0]["row"]
                print(f"Sample row keys: {list(row0.keys())}")
                # check audio field
                if "audio" in row0:
                    audio_meta = row0["audio"]
                    print(f"Audio metadata type: {type(audio_meta)}, keys/preview: {str(audio_meta)[:150]}")
        else:
            print(f"Response: {r.text[:200]}")
    except Exception as e:
        print(f"Error querying /rows: {e}")


def test_http_range_request():
    print("\n=== Method 2: HTTP Range Requests on Parquet Shard ===")
    # Check shard 0
    url = "https://huggingface.co/datasets/SleepyJesse/ai_music_large/resolve/main/data/train-00000-of-00147.parquet"
    try:
        # First send HEAD or GET range 0-1024
        headers = {"Range": "bytes=0-1023"}
        t0 = time.time()
        r = requests.get(url, headers=headers, timeout=10, allow_redirects=True)
        dt = time.time() - t0
        print(f"Range request (0-1023 bytes): status {r.status_code} in {dt:.2f}s")
        print(f"Content-Range: {r.headers.get('Content-Range')}")
        print(f"Accept-Ranges: {r.headers.get('Accept-Ranges')}")
        print(f"Content-Length: {r.headers.get('Content-Length')}")
        print(f"Is Partial Content (206)? {r.status_code == 206}")
    except Exception as e:
        print(f"Error testing range request: {e}")


def test_hf_streaming_iterator():
    print("\n=== Method 3: Hugging Face datasets streaming=True ===")
    try:
        from datasets import load_dataset
        t0 = time.time()
        print("Initializing load_dataset with streaming=True...")
        ds = load_dataset("SleepyJesse/ai_music_large", split="train", streaming=True)
        init_time = time.time() - t0
        print(f"Initialized streaming dataset in {init_time:.2f}s. Fetching 1 sample...")
        
        t0 = time.time()
        iterator = iter(ds)
        first_sample = next(iterator)
        fetch_time = time.time() - t0
        print(f"Successfully fetched first sample in {fetch_time:.2f}s!")
        print(f"Sample keys: {list(first_sample.keys())}")
        if "audio" in first_sample:
            audio = first_sample["audio"]
            print(f"Audio dict keys: {list(audio.keys()) if isinstance(audio, dict) else type(audio)}")
            if isinstance(audio, dict):
                print(f"Audio sampling_rate: {audio.get('sampling_rate')}, array shape: {getattr(audio.get('array'), 'shape', None)}")
    except Exception as e:
        print(f"Error in streaming test: {e}")

if __name__ == "__main__":
    test_hf_datasets_server_api()
    test_http_range_request()
    test_hf_streaming_iterator()

