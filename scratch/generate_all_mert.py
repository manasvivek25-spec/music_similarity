import io
import json
import logging
import os
import sys
import time
from pathlib import Path

# Ensure repo root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import librosa
import numpy as np
import requests
import torch

from src.config import FEATURES_DIR
from src.mert_embedding import (
    MERT_CACHE_DIR,
    MERT_EMBEDDING_DIM,
    MERT_MODEL_ID,
    MERT_SAMPLE_RATE,
    MERT_VERSION,
    detect_mert_environment,
    extract_mert_embedding,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("mert_ingestion")


def generate_missing_mert_embeddings():
    env = detect_mert_environment()
    logger.info(f"MERT Environment: device={env.device}, name={env.device_name}, cuda={env.cuda_available}")

    with open("features/remote_index.json", "r", encoding="utf-8") as f:
        idx = json.load(f)

    tracks = idx.get("tracks", {})
    logger.info(f"Loaded {len(tracks)} tracks from remote_index.json")

    manifest = {
        "version": MERT_VERSION,
        "model_id": MERT_MODEL_ID,
        "sample_rate": MERT_SAMPLE_RATE,
        "embedding_dim": MERT_EMBEDDING_DIM,
        "device": env.device,
        "device_name": env.device_name,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "tracks": {},
    }

    # First, audit existing valid embeddings
    cached_count = 0
    missing_tracks = []

    for tid, info in tracks.items():
        cache_file = MERT_CACHE_DIR / f"{tid}_mert.npy"
        if cache_file.is_file():
            try:
                emb = np.load(str(cache_file))
                if emb.shape == (MERT_EMBEDDING_DIM,) and not np.isnan(emb).any():
                    manifest["tracks"][tid] = {
                        "status": "cached",
                        "shape": list(emb.shape),
                        "norm": round(float(np.linalg.norm(emb)), 4),
                        "file": cache_file.name,
                    }
                    cached_count += 1
                    continue
            except Exception as e:
                logger.warning(f"Invalid cache for {tid}: {e}")

        missing_tracks.append((tid, info))

    logger.info(f"Initial status: {cached_count} tracks already valid, {len(missing_tracks)} missing")

    # Process missing tracks
    processed_count = 0
    failed_records = []

    for idx_num, (tid, info) in enumerate(missing_tracks, 1):
        s_url = info.get("source_url")
        logger.info(f"[{idx_num}/{len(missing_tracks)}] Processing {tid}...")

        if not s_url:
            raw_m = info.get("raw_metadata", {})
            src = info.get("source") or raw_m.get("source")
            logger.warning(f"No direct source_url for {tid} (source: {src}) - recording external dependency blocker")
            manifest["tracks"][tid] = {
                "status": "ineligible_no_stream_url",
                "source": src,
                "notes": "Parquet archive lacks direct public HTTP audio stream endpoint; requires external archive extraction.",
            }
            failed_records.append({"track_id": tid, "reason": "No direct stream URL"})
            continue

        try:
            t0 = time.time()
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
            resp = requests.get(s_url, headers=headers, timeout=45)
            if resp.status_code != 200:
                raise RuntimeError(f"HTTP {resp.status_code}: Failed to download audio")

            fetch_time = time.time() - t0

            # Decode full audio into 24kHz mono in memory
            y, sr = librosa.load(io.BytesIO(resp.content), sr=MERT_SAMPLE_RATE, mono=True)
            dur = len(y) / sr
            logger.info(f"  Downloaded {len(resp.content)} bytes in {fetch_time:.2f}s. Audio duration: {dur:.2f}s")

            # Extract MERT embedding on GPU
            t_inf = time.time()
            emb = extract_mert_embedding(y, track_id=tid, sr=MERT_SAMPLE_RATE, force_recompute=True)
            inf_time = time.time() - t_inf

            logger.info(f"  Extracted MERT embedding in {inf_time:.2f}s on {env.device}. Norm: {np.linalg.norm(emb):.4f}")

            manifest["tracks"][tid] = {
                "status": "valid",
                "audio_duration_sec": round(dur, 2),
                "shape": list(emb.shape),
                "norm": round(float(np.linalg.norm(emb)), 4),
                "extraction_time_sec": round(inf_time, 2),
                "device": env.device,
                "file": f"{tid}_mert.npy",
            }
            processed_count += 1
            cached_count += 1

        except Exception as exc:
            logger.error(f"  Failed to process {tid}: {exc}")
            manifest["tracks"][tid] = {
                "status": "extraction_error",
                "error": str(exc),
            }
            failed_records.append({"track_id": tid, "reason": str(exc)})

    # Final summary in manifest
    manifest["total_canonical_tracks"] = len(tracks)
    manifest["valid_production_embeddings"] = cached_count
    manifest["missing_or_blocked_tracks"] = len(failed_records)
    manifest["coverage_percentage"] = round((cached_count / len(tracks)) * 100.0, 2)
    manifest["failed_records"] = failed_records

    manifest_path = FEATURES_DIR / "mert_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"=== MERT INGESTION COMPLETE ===")
    logger.info(f"Total canonical tracks: {len(tracks)}")
    logger.info(f"Valid production embeddings: {cached_count} ({manifest['coverage_percentage']}%)")
    logger.info(f"Failed / Blocked: {len(failed_records)}")
    logger.info(f"Manifest written to {manifest_path}")


if __name__ == "__main__":
    generate_missing_mert_embeddings()
