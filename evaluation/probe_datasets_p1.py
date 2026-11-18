"""P1 Dataset Schema, Polarity, and Throughput Probe."""
import io
import time
import json
from collections import Counter
from pathlib import Path
import numpy as np
import soundfile as sf
import librosa
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download, HfFileSystem

from src.config import SAMPLE_RATE
from src.extract import extract_chroma, extract_chroma_beatsync
from src.origin_classifier import extract_origin_features_from_audio
from src.mert_embedding import extract_mert_embedding
from src.similarity import dtw_align_detailed


def probe_sleepyjesse():
    print("=== Probing SleepyJesse/ai_music_large ===")
    fs = HfFileSystem()
    # Read only non-audio columns from 5 shards spread across 0..146 to inspect polarity & distribution
    shards = [0, 35, 73, 110, 146]
    source_ai_counts = Counter()
    for s in shards:
        fname = f"datasets/SleepyJesse/ai_music_large/data/train-{s:05d}-of-00147.parquet"
        try:
            t0 = time.time()
            with fs.open(fname, "rb") as f:
                pf = pq.ParquetFile(f)
                cols = [c for c in pf.schema.names if c != "audio"]
                tbl = pf.read(columns=cols)
                d = tbl.to_pydict()
                n = len(d["ai_generated"])
                c_local = Counter(zip(d["source"], d["ai_generated"]))
                source_ai_counts.update(c_local)
                print(f"  Shard {s:03d}: n={n} ({time.time()-t0:.2f}s) counts={dict(c_local)}")
        except Exception as e:
            print(f"  Shard {s:03d} ERR: {e}")
    print("  Total sampled (source, ai_generated):", dict(source_ai_counts))

    # Download 1 shard (or read first 12 rows including audio bytes) to benchmark real download+decode+features
    print("\n--- Benchmarking audio decode + features + MERT(GPU) on 10 rows of shard 0 & shard 100 ---")
    t_dl0 = time.time()
    p0 = hf_hub_download("SleepyJesse/ai_music_large", "data/train-00000-of-00147.parquet", repo_type="dataset")
    t_dl = time.time() - t_dl0
    pf = pq.ParquetFile(p0)
    print(f"  Downloaded shard 0 ({Path(p0).stat().st_size/1e6:.1f} MB) in {t_dl:.2f}s; rows={pf.metadata.num_rows}")
    batch = next(pf.iter_batches(batch_size=12))
    rows = batch.to_pylist()
    tmp_dir = Path("temp_uploads/p1_bench")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    t_feat_start = time.time()
    chromas = []
    sr_orig_list = []
    dur_orig_list = []
    for i, r in enumerate(rows[:10]):
        aud = r["audio"]
        raw_bytes = aud["bytes"] if isinstance(aud, dict) else aud
        data, sr_orig = sf.read(io.BytesIO(raw_bytes), always_2d=False)
        if data.ndim > 1:
            data = np.mean(data, axis=1)
        dur_orig = len(data) / float(sr_orig)
        sr_orig_list.append(sr_orig)
        dur_orig_list.append(round(dur_orig, 2))
        # Resample + center 30s window
        y = librosa.resample(data.astype(np.float32), orig_sr=sr_orig, target_sr=SAMPLE_RATE) if sr_orig != SAMPLE_RATE else data.astype(np.float32)
        target_len = int(30.0 * SAMPLE_RATE)
        if len(y) > target_len:
            start = (len(y) - target_len) // 2
            y = y[start:start + target_len]
        # RMS loudness normalize
        rms = float(np.sqrt(np.mean(y ** 2) + 1e-12))
        y = np.clip(y * (0.1 / rms), -0.98, 0.98).astype(np.float32)
        wav_p = tmp_dir / f"bench_{i}.wav"
        sf.write(str(wav_p), y, SAMPLE_RATE)
        c_beat, b_times, dur = extract_chroma_beatsync(wav_p)
        f56 = extract_origin_features_from_audio(wav_p)
        chromas.append(c_beat)
    t_feat = time.time() - t_feat_start
    print(f"  10 tracks decode+chroma+56dim: {t_feat:.2f}s ({10.0/t_feat*60:.1f} tracks/min)")
    print(f"  Orig SRs: {sr_orig_list}, Orig Durs: {dur_orig_list}")

    # Benchmark MERT on GPU
    t_mert0 = time.time()
    for i in range(10):
        emb, meta = extract_mert_embedding(tmp_dir / f"bench_{i}.wav")
    t_mert = time.time() - t_mert0
    print(f"  10 tracks MERT ({meta.get('device')}): {t_mert:.2f}s ({10.0/t_mert*60:.1f} tracks/min), shape={emb.shape if emb is not None else None}")

    # Benchmark DTW pairs/sec
    t_dtw0 = time.time()
    n_pairs = 0
    for i in range(len(chromas)):
        for j in range(i + 1, len(chromas)):
            dtw_align_detailed(chromas[i], chromas[j])
            n_pairs += 1
    t_dtw = time.time() - t_dtw0
    print(f"  {n_pairs} DTW pairs (30s beat-chroma): {t_dtw:.2f}s ({n_pairs/t_dtw:.1f} pairs/sec)")


def probe_fma_small():
    print("\n=== Probing benjamin-paine/free-music-archive-small ===")
    fs = HfFileSystem()
    fname = "datasets/benjamin-paine/free-music-archive-small/data/train-00000-of-00015.parquet"
    with fs.open(fname, "rb") as f:
        pf = pq.ParquetFile(f)
        cols = [c for c in pf.schema.names if c != "audio"]
        tbl = pf.read(columns=cols)
        d = tbl.to_pydict()
        print("  Shard 0 rows:", len(d["title"]), "cols:", cols)
        print("  Sample titles/artists/genres:", [(d["title"][i], d["artist"][i], d["genres"][i]) for i in range(5)])


def probe_sonics_csv():
    print("\n=== Probing awsaf49/sonics fake_songs.csv ===")
    p = hf_hub_download("awsaf49/sonics", "fake_songs.csv", repo_type="dataset")
    import pandas as pd
    df = pd.read_csv(p)
    print("  Shape:", df.shape, "Cols:", list(df.columns))
    if "algorithm" in df.columns:
        print("  Algorithms:", df["algorithm"].value_counts().to_dict())
    print(df.head(3).to_dict(orient="records"))


if __name__ == "__main__":
    probe_sleepyjesse()
    probe_fma_small()
    probe_sonics_csv()
