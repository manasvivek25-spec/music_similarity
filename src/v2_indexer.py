"""
Resumable v2 Dataset Indexer & Normalized Feature Extractor (P1-P3).

Processes verified real human and AI audio shards through an identical
confound-controlled normalization pipeline:
  - Resample to 22,050 Hz mono
  - Exact center INDEX_MAX_SECONDS (10.0 s) window
  - 4th-order zero-phase Butterworth low-pass filter at 7,500 Hz
    (eliminates 16 kHz vs 44.1 kHz sample-rate / Nyquist confounds)
  - RMS loudness normalization (target RMS = 0.10, peak clip [-0.98, 0.98])

Extracts and caches compact .npy features under data/v2/features/:
  - Beat-synchronized 12-bin Chroma CQT + beat_times + mean_chroma
  - 56-dim acoustic features (Feature 4)
  - 768-dim MERT-v1-95M embeddings (batched GPU/CPU, graceful fallback)
  - Feature 5 musical context (estimated key, estimated tempo, duration)

Tracks real indexing state in data/v2/index_state.json with states:
  NOT_STARTED, INDEXING, PAUSED, COMPLETED, FAILED
"""

import concurrent.futures
import hashlib
import io
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import librosa
import numpy as np
import pyarrow.parquet as pq
import scipy.signal as signal
import soundfile as sf
import torch

from src.config import (
    ALGORITHM_VERSION,
    FEATURE_VERSION,
    INDEX_MAX_SECONDS,
    REPO_ROOT,
    SAMPLE_RATE,
    V2_DATA_DIR,
    V2_INDEX_FILE,
    V2_INDEX_STATE_FILE,
)
from src.feature5 import estimate_musical_key
from src.similarity import compute_mean_chroma_vector

V2_FEATURES_DIR = V2_DATA_DIR / "features"
V2_AUDIO_NORM_DIR = V2_DATA_DIR / "audio" / "norm10s"
V2_AUDIO_LONG_DIR = V2_DATA_DIR / "audio" / "long25s"
V2_RAW_CATALOG_FILE = V2_DATA_DIR / "raw_indexed_catalog.json"
SHARDS_MANIFEST_FILE = V2_DATA_DIR / "downloaded_shards.json"

LOWPASS_HZ = 7500.0
TARGET_RMS = 0.10
MIN_ORIG_DURATION_SEC = 9.5

# Precompute 4th-order Butterworth low-pass SOS at 7,500 Hz for 22,050 Hz SR
_LP_SOS = signal.butter(4, LOWPASS_HZ / (SAMPLE_RATE / 2.0), btype="low", output="sos")


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)


def normalize_waveform(
    data: np.ndarray,
    orig_sr: int,
    target_sec: float = INDEX_MAX_SECONDS,
    target_sr: int = SAMPLE_RATE,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Identical confound-controlled normalization applied to EVERY track.
    Returns (normalized_mono_float32_array, post_norm_diagnostics).
    """
    if data.ndim > 1:
        mono = np.mean(data, axis=1).astype(np.float32)
    else:
        mono = data.astype(np.float32)

    # Remove NaN/Inf safely
    if not np.all(np.isfinite(mono)):
        mono = np.nan_to_num(mono, nan=0.0, posinf=0.0, neginf=0.0)

    # Resample to 22,050 Hz
    if orig_sr != target_sr:
        y = librosa.resample(mono, orig_sr=orig_sr, target_sr=target_sr, res_type="soxr_hq").astype(np.float32)
    else:
        y = mono

    # Extract exact center target_sec window
    target_len = int(round(target_sec * target_sr))
    if len(y) > target_len:
        start = (len(y) - target_len) // 2
        y = y[start:start + target_len]
    elif len(y) < target_len:
        pad_total = target_len - len(y)
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        y = np.pad(y, (pad_left, pad_right), mode="constant")

    # Remove DC offset
    y = (y - float(np.mean(y))).astype(np.float32)

    # Identical 7,500 Hz zero-phase lowpass filter (removes sample-rate Nyquist confounds)
    y = signal.sosfiltfilt(_LP_SOS, y).astype(np.float32)

    # Identical RMS loudness normalization
    rms = float(np.sqrt(np.mean(y ** 2) + 1e-12))
    if rms < 1e-5:
        raise ValueError("Audio window is effectively silent (rms < 1e-5)")
    y = np.clip(y * (TARGET_RMS / rms), -0.98, 0.98).astype(np.float32)

    # Post-normalization diagnostics for confound probe audit
    post_rms = float(np.sqrt(np.mean(y ** 2) + 1e-12))
    post_peak = float(np.max(np.abs(y)))
    post_dc = float(np.mean(y))
    silence_ratio_norm = float(np.mean(np.abs(y) < 0.005))

    diag = {
        "norm_duration_sec": round(len(y) / float(target_sr), 4),
        "norm_sr": target_sr,
        "norm_channels": 1,
        "norm_rms": round(post_rms, 6),
        "norm_peak": round(post_peak, 6),
        "norm_dc_offset": round(post_dc, 6),
        "silence_ratio_norm": round(silence_ratio_norm, 6),
    }
    return y, diag


def extract_features_from_waveform(y: np.ndarray, sr: int = SAMPLE_RATE) -> Dict[str, Any]:
    """
    Extract Chroma CQT (raw + beat-sync), beat_times, mean_chroma,
    56-dim acoustic features (identical formula to src/origin_classifier.py),
    and Feature 5 musical context (key, tempo).
    """
    # 1. Chroma CQT & Beat Sync (exact Feature 1 spec)
    chroma_raw = librosa.feature.chroma_cqt(y=y, sr=sr).astype(np.float32)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo_val: Optional[float] = None
    beat_frames = np.array([], dtype=int)
    if np.var(onset_env) >= 1e-4:
        try:
            tempo_raw, beat_frames = librosa.beat.beat_track(y=y, sr=sr, onset_envelope=onset_env)
            tv = float(tempo_raw[0] if isinstance(tempo_raw, (np.ndarray, list)) else tempo_raw)
            if np.isfinite(tv) and tv >= 30.0 and len(beat_frames) >= 2:
                tempo_val = round(tv, 1)
        except Exception:
            beat_frames = np.array([], dtype=int)

    if len(beat_frames) >= 2:
        chroma_beat = librosa.util.sync(chroma_raw, beat_frames, aggregate=np.median).astype(np.float32)
        beat_times = librosa.frames_to_time(beat_frames, sr=sr).astype(np.float32)
        mlen = min(chroma_beat.shape[1], len(beat_times))
        chroma_beat = chroma_beat[:, :mlen]
        beat_times = beat_times[:mlen]
    else:
        chroma_beat = chroma_raw
        beat_times = librosa.frames_to_time(np.arange(chroma_raw.shape[1]), sr=sr).astype(np.float32)

    mean_chroma = compute_mean_chroma_vector(chroma_beat)
    est_key = estimate_musical_key(chroma_raw)

    # 2. 56-dim acoustic features (identical to src/origin_classifier.extract_origin_features_from_audio)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)

    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    zcr = librosa.feature.zero_crossing_rate(y=y)[0]

    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_stft_mean = np.mean(chroma_stft, axis=1)
    chroma_stft_std = np.std(chroma_stft, axis=1)

    f56 = np.concatenate([
        mfcc_mean, mfcc_std,
        [np.mean(centroid), np.std(centroid)],
        [np.mean(rolloff), np.std(rolloff)],
        [np.mean(zcr), np.std(zcr)],
        chroma_stft_mean, chroma_stft_std,
    ]).astype(np.float32)
    f56 = np.nan_to_num(f56, nan=0.0, posinf=0.0, neginf=0.0)

    return {
        "chroma_beat": chroma_beat,
        "beat_times": beat_times,
        "mean_chroma": mean_chroma,
        "f56": f56,
        "estimated_key": est_key,
        "estimated_tempo": tempo_val,
    }


def verify_label_and_provenance(role: str, raw_meta: Dict[str, Any]) -> Tuple[str, str, str]:
    """
    Verify label polarity against dataset card and metadata consistency (Section 6.1).
    Returns (origin_label, generator, provenance).
    Ambiguous or conflicting labels become 'unknown'.
    """
    if role == "human_fma_small":
        return "human", "human_fma_small", "verified_fma_small_cc"

    src = str(raw_meta.get("source", "")).strip().lower()
    ai_flag = raw_meta.get("ai_generated")

    if "free-music-archive" in src:
        if ai_flag is False or str(ai_flag).lower() == "false":
            return "human", "human_fma_full", "verified_sleepyjesse_fma_full"
        return "unknown", "conflict", "polarity_conflict_fma_flagged_ai"

    if "musicset" in src:
        if ai_flag is True or str(ai_flag).lower() == "true":
            return "ai", "MusicSet", "verified_sleepyjesse_musicset"
        return "unknown", "conflict", "polarity_conflict_musicset_flagged_human"

    if "sunocaps" in src:
        if ai_flag is True or str(ai_flag).lower() == "true":
            return "ai", "SunoCaps", "verified_sleepyjesse_sunocaps"
        return "unknown", "conflict", "polarity_conflict_sunocaps_flagged_human"

    if "udio" in src:
        if ai_flag is True or str(ai_flag).lower() == "true":
            return "ai", "Udio", "verified_sleepyjesse_udio_cc0"
        return "unknown", "conflict", "polarity_conflict_udio_flagged_human"

    return "unknown", "unknown", f"unverified_source:{src[:40]}"


def extract_mert_batch_gpu(
    track_wav_pairs: List[Tuple[str, Path]],
    features_dir: Path,
    batch_size: int = 8,
) -> Dict[str, Any]:
    """
    Batched MERT-v1-95M embedding extraction on CUDA GPU (fallback CPU).
    Resumable: skips tracks that already have <tid>_mert.npy on disk.
    Never lets an optional MERT failure break the Chroma+DTW pipeline.
    """
    todo = []
    reused = 0
    for tid, wav_p in track_wav_pairs:
        out_p = features_dir / f"{tid}_mert.npy"
        if out_p.is_file():
            reused += 1
        elif wav_p.is_file():
            todo.append((tid, wav_p, out_p))

    stats = {"reused": reused, "extracted": 0, "failed": 0, "device": "none"}
    if not todo:
        stats["device"] = "cached"
        return stats

    try:
        from transformers import AutoModel, Wav2Vec2FeatureExtractor
        device = "cuda" if torch.cuda.is_available() else "cpu"
        stats["device"] = device
        model_id = "m-a-p/MERT-v1-95M"
        print(f"[MERT] Loading {model_id} on {device} for {len(todo)} tracks (reused={reused})...", flush=True)
        processor = Wav2Vec2FeatureExtractor.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_id, trust_remote_code=True).to(device)
        model.eval()
        mert_sr = int(getattr(processor, "sampling_rate", 24000))
        target_mert_len = int(round(INDEX_MAX_SECONDS * mert_sr))

        t0 = time.time()
        for start_idx in range(0, len(todo), batch_size):
            batch = todo[start_idx:start_idx + batch_size]
            wavs_24k = []
            valid_items = []
            for tid, wav_p, out_p in batch:
                try:
                    y, sr = sf.read(str(wav_p), dtype="float32")
                    if y.ndim > 1:
                        y = np.mean(y, axis=1)
                    if sr != mert_sr:
                        y = librosa.resample(y, orig_sr=sr, target_sr=mert_sr, res_type="soxr_hq").astype(np.float32)
                    if len(y) > target_mert_len:
                        y = y[:target_mert_len]
                    elif len(y) < target_mert_len:
                        y = np.pad(y, (0, target_mert_len - len(y)))
                    wavs_24k.append(y)
                    valid_items.append((tid, out_p))
                except Exception:
                    stats["failed"] += 1

            if not wavs_24k:
                continue

            try:
                inputs = processor(wavs_24k, sampling_rate=mert_sr, return_tensors="pt", padding=True)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                with torch.no_grad():
                    if device == "cuda":
                        with torch.amp.autocast("cuda", dtype=torch.float16):
                            outputs = model(**inputs)
                    else:
                        outputs = model(**inputs)
                    hidden = outputs.last_hidden_state.float().mean(dim=1).cpu().numpy()  # (B, 768)
                for (tid, out_p), vec in zip(valid_items, hidden):
                    np.save(str(out_p), vec.astype(np.float32))
                    stats["extracted"] += 1
            except Exception as exc:
                # Graceful per-item fallback if batch OOMs
                print(f"[MERT] Batch fallback at {start_idx}: {exc}", flush=True)
                for (tid, out_p), y_item in zip(valid_items, wavs_24k):
                    try:
                        inp = processor(y_item, sampling_rate=mert_sr, return_tensors="pt")
                        inp = {k: v.to(device) for k, v in inp.items()}
                        with torch.no_grad():
                            out = model(**inp).last_hidden_state.float().mean(dim=1).squeeze(0).cpu().numpy()
                        np.save(str(out_p), out.astype(np.float32))
                        stats["extracted"] += 1
                    except Exception:
                        stats["failed"] += 1

            done_so_far = start_idx + len(batch)
            if done_so_far % 160 == 0 or done_so_far >= len(todo):
                rate = stats["extracted"] / max(time.time() - t0, 0.1) * 60.0
                print(f"[MERT] {done_so_far}/{len(todo)} processed ({rate:.1f} tracks/min)", flush=True)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as e:
        print(f"[MERT] Optional MERT extraction degraded cleanly: {e}", flush=True)
        stats["error"] = str(e)

    return stats


def run_resumable_v2_indexing(
    max_workers: int = 8,
    resume: bool = True,
    max_tracks_per_shard: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Execute resumable v2 indexing across all downloaded shards.
    Writes:
      - data/v2/index_state.json (NOT_STARTED -> INDEXING -> COMPLETED/FAILED)
      - data/v2/raw_indexed_catalog.json (all processed records with raw & normalized metadata)
      - data/v2/features/*.npy (chroma_beat, beat_times, f56, mert_768)
      - data/v2/audio/norm10s/*.wav (10s normalized audio for evaluation/playback)
    """
    V2_DATA_DIR.mkdir(parents=True, exist_ok=True)
    V2_FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    V2_AUDIO_NORM_DIR.mkdir(parents=True, exist_ok=True)
    V2_AUDIO_LONG_DIR.mkdir(parents=True, exist_ok=True)

    if not SHARDS_MANIFEST_FILE.is_file():
        raise FileNotFoundError(f"Shards manifest not found: {SHARDS_MANIFEST_FILE}")

    with open(SHARDS_MANIFEST_FILE, "r", encoding="utf-8") as f:
        shard_specs = [s for s in json.load(f)["shards"] if s.get("status") == "ok" and s.get("local_path")]

    # Load existing catalog if resuming
    catalog: Dict[str, Any] = {"tracks": {}, "failures": [], "duplicates": [], "skipped": []}
    if resume and V2_RAW_CATALOG_FILE.is_file():
        try:
            with open(V2_RAW_CATALOG_FILE, "r", encoding="utf-8") as f:
                catalog = json.load(f)
        except Exception:
            pass

    existing_tracks: Dict[str, Dict[str, Any]] = catalog.get("tracks", {})
    seen_raw_hashes = {t.get("raw_content_sha256") for t in existing_tracks.values() if t.get("raw_content_sha256")}
    seen_norm_hashes = {t.get("norm_sha256") for t in existing_tracks.values() if t.get("norm_sha256")}

    # Discover total expected rows across shards
    total_expected = 0
    shard_row_counts = []
    for s in shard_specs:
        pf = pq.ParquetFile(s["local_path"])
        nr = pf.metadata.num_rows
        if max_tracks_per_shard is not None:
            nr = min(nr, max_tracks_per_shard)
        total_expected += nr
        shard_row_counts.append((s, nr))

    t_start = time.time()
    state = {
        "status": "INDEXING",
        "is_complete": False,
        "dataset_version": "v2",
        "feature_version": FEATURE_VERSION,
        "algorithm_version": ALGORITHM_VERSION,
        "index_max_seconds": INDEX_MAX_SECONDS,
        "sample_rate": SAMPLE_RATE,
        "lowpass_hz": LOWPASS_HZ,
        "total_expected": total_expected,
        "discovered": 0,
        "indexed": len(existing_tracks),
        "reused": len(existing_tracks),
        "failed": len(catalog.get("failures", [])),
        "skipped": len(catalog.get("skipped", [])),
        "duplicates": len(catalog.get("duplicates", [])),
        "progress_pct": round(len(existing_tracks) / max(total_expected, 1) * 100.0, 2),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    atomic_write_json(V2_INDEX_STATE_FILE, state)
    print(f"[P3] Starting resumable v2 indexing: {total_expected} expected tracks across {len(shard_specs)} shards (reused={len(existing_tracks)}).", flush=True)

    def process_single_row(task: Tuple[str, str, int, Dict[str, Any], List[str]]) -> Dict[str, Any]:
        role, shard_name, row_idx, row, genre_names = task
        tid = f"v2_{role}_{shard_name.replace('.parquet', '').split('-')[1]}_{row_idx:04d}"

        # Fast resume check
        if tid in existing_tracks:
            return {"status": "reused", "track_id": tid}

        aud = row.get("audio")
        raw_bytes = aud.get("bytes") if isinstance(aud, dict) else aud
        if not raw_bytes or len(raw_bytes) < 512:
            return {"status": "failed", "track_id": tid, "reason": "empty_or_tiny_audio_bytes", "role": role}

        raw_meta = {k: (str(v) if not isinstance(v, (int, float, bool, str, list, type(None))) else v)
                    for k, v in row.items() if k != "audio"}
        origin_label, generator, provenance = verify_label_and_provenance(role, raw_meta)

        # Decode audio
        try:
            info = sf.info(io.BytesIO(raw_bytes))
            orig_fmt = f"{info.format}/{info.subtype}"
            orig_sr = int(info.samplerate)
            orig_ch = int(info.channels)
            orig_dur = float(info.duration)
            data, sr_loaded = sf.read(io.BytesIO(raw_bytes), always_2d=False, dtype="float32")
        except Exception as e:
            return {"status": "failed", "track_id": tid, "reason": f"decode_error:{str(e)[:80]}", "role": role}

        if data is None or len(data) == 0:
            return {"status": "failed", "track_id": tid, "reason": "decoded_empty_waveform", "role": role}

        if orig_dur < MIN_ORIG_DURATION_SEC:
            return {"status": "skipped", "track_id": tid, "reason": f"too_short:{orig_dur:.2f}s<{MIN_ORIG_DURATION_SEC}s", "role": role}

        mono_raw = np.mean(data, axis=1).astype(np.float32) if data.ndim > 1 else data.astype(np.float32)
        silence_ratio_raw = float(np.mean(np.abs(mono_raw) < 0.005))
        raw_content_sha256 = hashlib.sha256(mono_raw.tobytes()).hexdigest()

        # Identical normalization
        try:
            y_norm, norm_diag = normalize_waveform(data, sr_loaded, target_sec=INDEX_MAX_SECONDS, target_sr=SAMPLE_RATE)
        except Exception as e:
            return {"status": "skipped", "track_id": tid, "reason": f"norm_reject:{str(e)[:80]}", "role": role}

        norm_sha256 = hashlib.sha256(y_norm.tobytes()).hexdigest()

        # Extract Chroma, Beat Sync, 56-dim, F5 Context
        try:
            feats = extract_features_from_waveform(y_norm, sr=SAMPLE_RATE)
        except Exception as e:
            return {"status": "failed", "track_id": tid, "reason": f"feature_extract_error:{str(e)[:80]}", "role": role}

        # Save compact .npy features & 10s normalized WAV
        beat_p = V2_FEATURES_DIR / f"{tid}_beat.npy"
        times_p = V2_FEATURES_DIR / f"{tid}_times.npy"
        f56_p = V2_FEATURES_DIR / f"{tid}_f56.npy"
        wav_p = V2_AUDIO_NORM_DIR / f"{tid}.wav"

        np.save(str(beat_p), feats["chroma_beat"])
        np.save(str(times_p), feats["beat_times"])
        np.save(str(f56_p), feats["f56"])
        sf.write(str(wav_p), y_norm, SAMPLE_RATE, subtype="PCM_16")

        # Also save a 25s normalized clip (for long tracks >= 22s, capped to first 60 per source) for spliced real evaluation
        long_wav_rel = None
        if orig_dur >= 22.0 and row_idx < 60:
            try:
                y_long, _ = normalize_waveform(data, sr_loaded, target_sec=22.0, target_sr=SAMPLE_RATE)
                long_p = V2_AUDIO_LONG_DIR / f"{tid}_22s.wav"
                sf.write(str(long_p), y_long, SAMPLE_RATE, subtype="PCM_16")
                long_wav_rel = str(long_p.relative_to(REPO_ROOT))
            except Exception:
                pass

        # Genre & Artist handling (NEVER invent genres!)
        genre_str = None
        genre_source = "unknown"
        artist_str = None
        title_str = None
        album_str = None
        if role == "human_fma_small":
            title_str = str(raw_meta.get("title") or "").strip() or None
            artist_str = str(raw_meta.get("artist") or "").strip() or None
            album_str = str(raw_meta.get("album_title") or "").strip() or None
            g_ids = raw_meta.get("genres") or []
            if isinstance(g_ids, list) and len(g_ids) > 0 and genre_names:
                g0 = g_ids[0]
                if isinstance(g0, int) and 0 <= g0 < len(genre_names):
                    genre_str = genre_names[g0]
                    genre_source = "verified_fma_metadata"
                elif isinstance(g0, str) and g0.strip():
                    genre_str = g0.strip()
                    genre_source = "verified_fma_metadata"
        else:
            title_str = f"{generator} Track {row_idx:04d}"

        record = {
            "track_id": tid,
            "origin_label": origin_label,
            "generator": generator,
            "label_provenance": provenance,
            "source_role": role,
            "source_shard": shard_name,
            "title": title_str,
            "artist": artist_str,
            "album_title": album_str,
            "genre": genre_str,
            "genre_source": genre_source,
            "estimated_key": feats["estimated_key"],
            "estimated_tempo": feats["estimated_tempo"],
            "duration_sec": INDEX_MAX_SECONDS,
            "mean_chroma": [round(float(x), 5) for x in feats["mean_chroma"].tolist()],
            "raw_content_sha256": raw_content_sha256,
            "norm_sha256": norm_sha256,
            "beat_cache": str(beat_p.relative_to(REPO_ROOT)),
            "times_cache": str(times_p.relative_to(REPO_ROOT)),
            "f56_cache": str(f56_p.relative_to(REPO_ROOT)),
            "mert_cache": str((V2_FEATURES_DIR / f"{tid}_mert.npy").relative_to(REPO_ROOT)),
            "norm_wav_path": str(wav_p.relative_to(REPO_ROOT)),
            "long_wav_path": long_wav_rel,
            "pre_norm_metadata": {
                "orig_sr": orig_sr,
                "orig_channels": orig_ch,
                "orig_duration_sec": round(orig_dur, 3),
                "orig_format": orig_fmt,
                "orig_bytes": len(raw_bytes),
                "silence_ratio_raw": round(silence_ratio_raw, 6),
            },
            "post_norm_metadata": norm_diag,
            "raw_metadata": raw_meta,
            "feature_version": FEATURE_VERSION,
        }
        return {"status": "ok", "track_id": tid, "record": record}

    discovered = 0
    for s, n_rows in shard_row_counts:
        role = s["role"]
        shard_name = Path(s["file"]).name
        pf = pq.ParquetFile(s["local_path"])
        genre_names: List[str] = []
        meta = pf.schema_arrow.metadata
        if role == "human_fma_small" and meta and b"huggingface" in meta:
            try:
                hf_m = json.loads(meta[b"huggingface"])
                genre_names = hf_m["info"]["features"]["genres"]["feature"]["names"]
            except Exception:
                pass

        print(f"[P3] Processing shard {role} / {shard_name} ({n_rows} rows)...", flush=True)
        row_idx = 0
        for batch in pf.iter_batches(batch_size=64):
            rows = batch.to_pylist()
            tasks = []
            for r in rows:
                if row_idx >= n_rows:
                    break
                tasks.append((role, shard_name, row_idx, r, genre_names))
                row_idx += 1
                discovered += 1

            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
                outcomes = list(ex.map(process_single_row, tasks))

            for out in outcomes:
                st = out["status"]
                if st == "reused":
                    continue
                elif st == "ok":
                    rec = out["record"]
                    rh = rec["raw_content_sha256"]
                    nh = rec["norm_sha256"]
                    if rh in seen_raw_hashes or nh in seen_norm_hashes:
                        catalog["duplicates"].append({
                            "track_id": rec["track_id"],
                            "reason": "exact_content_hash_duplicate",
                            "raw_content_sha256": rh,
                        })
                        # Remove duplicate files to save disk
                        for pkey in ("beat_cache", "times_cache", "f56_cache", "norm_wav_path"):
                            if rec.get(pkey):
                                fp = REPO_ROOT / rec[pkey]
                                if fp.is_file():
                                    fp.unlink(missing_ok=True)
                    else:
                        seen_raw_hashes.add(rh)
                        seen_norm_hashes.add(nh)
                        existing_tracks[rec["track_id"]] = rec
                elif st == "skipped":
                    catalog["skipped"].append(out)
                else:
                    catalog["failures"].append(out)

            # Incremental atomic checkpoint after every batch
            catalog["tracks"] = existing_tracks
            state.update({
                "discovered": discovered,
                "indexed": len(existing_tracks),
                "failed": len(catalog["failures"]),
                "skipped": len(catalog["skipped"]),
                "duplicates": len(catalog["duplicates"]),
                "progress_pct": round(discovered / max(total_expected, 1) * 100.0, 2),
                "elapsed_sec": round(time.time() - t_start, 1),
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            })
            atomic_write_json(V2_INDEX_STATE_FILE, state)

            if row_idx >= n_rows:
                break

        # Persist catalog after each shard for fast resume
        atomic_write_json(V2_RAW_CATALOG_FILE, catalog)
        print(
            f"  -> Shard done: indexed={len(existing_tracks)} dupes={len(catalog['duplicates'])} "
            f"skipped={len(catalog['skipped'])} failed={len(catalog['failures'])} ({time.time()-t_start:.1f}s)",
            flush=True,
        )

    # Step 2: Batched MERT-v1-95M GPU Extraction across all indexed tracks
    wav_pairs = [(tid, REPO_ROOT / rec["norm_wav_path"]) for tid, rec in sorted(existing_tracks.items())]
    mert_stats = extract_mert_batch_gpu(wav_pairs, V2_FEATURES_DIR, batch_size=8)

    # Mark MERT readiness per track
    for tid, rec in existing_tracks.items():
        mp = REPO_ROOT / rec["mert_cache"]
        rec["has_mert"] = mp.is_file()

    catalog["tracks"] = existing_tracks
    catalog["mert_stats"] = mert_stats
    atomic_write_json(V2_RAW_CATALOG_FILE, catalog)

    state.update({
        "status": "COMPLETED",
        "is_complete": True,
        "discovered": discovered,
        "indexed": len(existing_tracks),
        "failed": len(catalog["failures"]),
        "skipped": len(catalog["skipped"]),
        "duplicates": len(catalog["duplicates"]),
        "mert_stats": mert_stats,
        "progress_pct": 100.0,
        "elapsed_sec": round(time.time() - t_start, 1),
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })
    atomic_write_json(V2_INDEX_STATE_FILE, state)
    print(f"\n[P3] COMPLETED v2 indexing in {time.time()-t_start:.1f}s! Total clean unique indexed: {len(existing_tracks)}", flush=True)
    return state


if __name__ == "__main__":
    run_resumable_v2_indexing(max_workers=8, resume=True)
