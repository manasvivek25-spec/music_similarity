"""
P1 Parallel Resumable Shard Downloader for v2 Balanced Real Dataset.
Downloads selected verified shards from:
- benjamin-paine/free-music-archive-small (Human, CC, with genuine genres & artists)
- SleepyJesse/ai_music_large (Human shard 0 already cached; AI MusicSet shards 76-80, SunoCaps shard 75, Udio shard 111)
"""
import concurrent.futures
import json
import os
import time
from pathlib import Path
from huggingface_hub import hf_hub_download

SHARD_MANIFEST = [
    # Human with genuine genres & artists (FMA-small, ~533 tracks/shard)
    {"repo": "benjamin-paine/free-music-archive-small", "file": "data/train-00000-of-00015.parquet", "role": "human_fma_small", "expected_mb": 458},
    {"repo": "benjamin-paine/free-music-archive-small", "file": "data/train-00001-of-00015.parquet", "role": "human_fma_small", "expected_mb": 472},
    # Human FMA-full (already cached in P1 probe, 137 tracks)
    {"repo": "SleepyJesse/ai_music_large", "file": "data/train-00000-of-00147.parquet", "role": "human_fma_full", "expected_mb": 645},
    # AI: MusicSet (~136 tracks/shard, ~59 MB/shard)
    {"repo": "SleepyJesse/ai_music_large", "file": "data/train-00076-of-00147.parquet", "role": "ai_musicset", "expected_mb": 60},
    {"repo": "SleepyJesse/ai_music_large", "file": "data/train-00077-of-00147.parquet", "role": "ai_musicset", "expected_mb": 60},
    {"repo": "SleepyJesse/ai_music_large", "file": "data/train-00078-of-00147.parquet", "role": "ai_musicset", "expected_mb": 60},
    {"repo": "SleepyJesse/ai_music_large", "file": "data/train-00079-of-00147.parquet", "role": "ai_musicset", "expected_mb": 58},
    {"repo": "SleepyJesse/ai_music_large", "file": "data/train-00080-of-00147.parquet", "role": "ai_musicset", "expected_mb": 60},
    # AI: SunoCaps (~136 tracks, 157 MB)
    {"repo": "SleepyJesse/ai_music_large", "file": "data/train-00075-of-00147.parquet", "role": "ai_sunocaps", "expected_mb": 157},
    # AI: Udio (~136 tracks, 538 MB, CC0)
    {"repo": "SleepyJesse/ai_music_large", "file": "data/train-00111-of-00147.parquet", "role": "ai_udio", "expected_mb": 539},
]

OUT_PATH = Path("data/v2/downloaded_shards.json")


def download_one(spec):
    t0 = time.time()
    try:
        local_path = hf_hub_download(spec["repo"], spec["file"], repo_type="dataset")
        dt = round(time.time() - t0, 2)
        sz = round(os.path.getsize(local_path) / 1e6, 2)
        print(f"[DONE] {spec['role']} {spec['file']} -> {sz} MB in {dt}s", flush=True)
        return {**spec, "local_path": str(local_path), "size_mb": sz, "elapsed_sec": dt, "status": "ok"}
    except Exception as e:
        dt = round(time.time() - t0, 2)
        print(f"[FAIL] {spec['role']} {spec['file']} ({dt}s): {e}", flush=True)
        return {**spec, "local_path": None, "error": str(e), "elapsed_sec": dt, "status": "failed"}


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    t_all = time.time()
    print(f"Starting parallel shard acquisition ({len(SHARD_MANIFEST)} shards, max_workers=4)...", flush=True)
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
        futs = [ex.submit(download_one, s) for s in SHARD_MANIFEST]
        for fut in concurrent.futures.as_completed(futs):
            results.append(fut.result())
            # Save incremental progress atomically
            tmp = OUT_PATH.with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump({"updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "shards": results}, f, indent=2)
            tmp.replace(OUT_PATH)
    ok_n = sum(1 for r in results if r["status"] == "ok")
    print(f"\nCompleted shard acquisition: {ok_n}/{len(SHARD_MANIFEST)} succeeded in {time.time()-t_all:.1f}s.", flush=True)


if __name__ == "__main__":
    main()
