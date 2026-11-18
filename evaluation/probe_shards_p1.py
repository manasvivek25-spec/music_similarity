"""Inspect SleepyJesse shard 76 (MusicSet) & shard 75 (SunoCaps) and FMA-small shard 0."""
import io
import time
import soundfile as sf
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download, HfFileSystem


def inspect_shard(repo_id, filename, n_sample=6):
    t0 = time.time()
    path = hf_hub_download(repo_id, filename, repo_type="dataset")
    dt = time.time() - t0
    pf = pq.ParquetFile(path)
    sz_mb = round(pf.metadata.serialized_size / 1e6, 2)
    import os
    disk_mb = round(os.path.getsize(path) / 1e6, 1)
    print(f"\n=== {repo_id} / {filename} ({disk_mb} MB in {dt:.1f}s, {disk_mb/max(dt,0.1):.1f} MB/s) rows={pf.metadata.num_rows} ===")
    rows = next(pf.iter_batches(batch_size=n_sample)).to_pylist()
    for i, r in enumerate(rows[:n_sample]):
        aud = r["audio"]
        b = aud["bytes"] if isinstance(aud, dict) else aud
        try:
            info = sf.info(io.BytesIO(b))
            info_s = f"fmt={info.format}/{info.subtype} sr={info.samplerate} ch={info.channels} dur={info.duration:.2f}s bytes={len(b)}"
        except Exception as e:
            info_s = f"sf_err={e} bytes={len(b)}"
        meta = {k: v for k, v in r.items() if k != "audio"}
        print(f"  [{i}] {info_s} | meta={meta}")


if __name__ == "__main__":
    inspect_shard("SleepyJesse/ai_music_large", "data/train-00076-of-00147.parquet", n_sample=6)
    inspect_shard("SleepyJesse/ai_music_large", "data/train-00075-of-00147.parquet", n_sample=6)
