import sys
import json
import requests
from huggingface_hub import HfApi

api = HfApi()

def probe_haim():
    print("\n--- 1. Probing mippia/HAIM ---", flush=True)
    try:
        root_items = list(api.list_repo_tree("mippia/HAIM", repo_type="dataset", recursive=False))
        print("Root items:", [item.path for item in root_items], flush=True)
        
        # Check A_full_generation
        a_items = list(api.list_repo_tree("mippia/HAIM", repo_type="dataset", path_in_repo="A_full_generation", recursive=False))
        print("A_full_generation items:", [item.path for item in a_items], flush=True)

        # Check A1_real
        a1_items = list(api.list_repo_tree("mippia/HAIM", repo_type="dataset", path_in_repo="A_full_generation/A1_real", recursive=False))
        print("A1_real items:", [item.path for item in a1_items], flush=True)

        for a1_sub in a1_items:
            sub_files = list(api.list_repo_tree("mippia/HAIM", repo_type="dataset", path_in_repo=a1_sub.path, recursive=False))
            mp3_count = sum(1 for f in sub_files if f.path.endswith('.mp3'))
            print(f"  {a1_sub.path}: {len(sub_files)} items ({mp3_count} MP3s)", flush=True)

        # Check A2_fake
        a2_items = list(api.list_repo_tree("mippia/HAIM", repo_type="dataset", path_in_repo="A_full_generation/A2_fake", recursive=False))
        print("A2_fake generators:", [item.path for item in a2_items], flush=True)
        for gen in a2_items:
            gen_sub = list(api.list_repo_tree("mippia/HAIM", repo_type="dataset", path_in_repo=gen.path, recursive=False))
            print(f"  Generator {gen.path}: subfolders = {[s.path for s in gen_sub]}", flush=True)
            for sub in gen_sub:
                files = list(api.list_repo_tree("mippia/HAIM", repo_type="dataset", path_in_repo=sub.path, recursive=False))
                mp3_count = sum(1 for f in files if f.path.endswith('.mp3'))
                print(f"    {sub.path}: {len(files)} items ({mp3_count} MP3s)", flush=True)

        # Check B_hybrid
        b_items = list(api.list_repo_tree("mippia/HAIM", repo_type="dataset", path_in_repo="B_hybrid", recursive=False))
        print("B_hybrid items:", [item.path for item in b_items], flush=True)
        for b_sub in b_items:
            b_folders = list(api.list_repo_tree("mippia/HAIM", repo_type="dataset", path_in_repo=b_sub.path, recursive=False))
            for bf in b_folders:
                files = list(api.list_repo_tree("mippia/HAIM", repo_type="dataset", path_in_repo=bf.path, recursive=False))
                mp3_count = sum(1 for f in files if f.path.endswith('.mp3'))
                print(f"  {bf.path}: {len(files)} items ({mp3_count} MP3s)", flush=True)

        # Check C_mixing
        c_items = list(api.list_repo_tree("mippia/HAIM", repo_type="dataset", path_in_repo="C_mixing", recursive=False))
        print("C_mixing items:", [item.path for item in c_items], flush=True)
        for c_sub in c_items:
            files = list(api.list_repo_tree("mippia/HAIM", repo_type="dataset", path_in_repo=c_sub.path, recursive=False))
            mp3_count = sum(1 for f in files if f.path.endswith('.mp3'))
            print(f"  {c_sub.path}: {len(files)} items ({mp3_count} MP3s)", flush=True)

    except Exception as e:
        print("Error probing HAIM:", e, flush=True)

def probe_sonics():
    print("\n--- 2. Probing awsaf49/sonics ---", flush=True)
    try:
        items = list(api.list_repo_tree("awsaf49/sonics", repo_type="dataset", recursive=False))
        print("Root items in awsaf49/sonics:", [item.path for item in items], flush=True)
        for item in items:
            print(f"  {item.path} (type={getattr(item, 'type', 'unknown')}, size={getattr(item, 'size', 'unknown')})", flush=True)
    except Exception as e:
        print("Error probing SONICS:", e, flush=True)

def probe_sleepyjesse():
    print("\n--- 3. Probing SleepyJesse/ai_music_large ---", flush=True)
    try:
        items = list(api.list_repo_tree("SleepyJesse/ai_music_large", repo_type="dataset", recursive=False))
        data_files = [item for item in items if "data" in item.path or item.path.endswith(".parquet")]
        print(f"Total root items: {len(items)}, data/parquet items: {len(data_files)}", flush=True)
        
        # Test rows endpoint
        url = "https://datasets-server.huggingface.co/rows?dataset=SleepyJesse%2Fai_music_large&config=default&split=train&offset=0&limit=2"
        resp = requests.get(url, timeout=10)
        print(f"Rows API status: {resp.status_code}", flush=True)
        if resp.status_code == 200:
            d = resp.json()
            rows = d.get("rows", [])
            print(f"Rows returned: {len(rows)}", flush=True)
            if rows:
                print("First row keys:", list(rows[0].get("row", {}).keys()), flush=True)
                audio_meta = rows[0].get("row", {}).get("audio", {})
                print("Audio meta:", audio_meta, flush=True)
    except Exception as e:
        print("Error probing SleepyJesse:", e, flush=True)

def probe_jamendo():
    print("\n--- 4. Probing MTG-Jamendo ---", flush=True)
    # Check MTG-Jamendo raw audio endpoints:
    # MTG provides raw audio download through jamendo or archive.org
    test_url = "https://prod-1.storage.jamendo.com/?trackid=6001&format=mp31&from=app-976"
    try:
        r = requests.head(test_url, timeout=10, allow_redirects=True)
        print(f"Jamendo direct audio HEAD status for track 6001: {r.status_code}", flush=True)
        print("Content-Type:", r.headers.get("Content-Type"), "Content-Length:", r.headers.get("Content-Length"), flush=True)
    except Exception as e:
        print("Jamendo URL check error:", e, flush=True)

if __name__ == "__main__":
    probe_haim()
    probe_sonics()
    probe_sleepyjesse()
    probe_jamendo()

