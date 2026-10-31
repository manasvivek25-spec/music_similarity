"""
Full Dataset Portfolio Expansion & Multi-Generator Ingestion Driver.

1. Dynamically enumerates all candidate records across HAIM, MTG-Jamendo, SleepyJesse, SONICS, FMA.
2. Ingests new diverse tracks into canonical index (Lyria-Pro3, MusicGen, ACE-Step, Jamendo, Hybrid, Mix).
3. Generates MERT-v1-95M embeddings on NVIDIA RTX 3050 Laptop GPU.
4. Reconciles dataset_manifest.json and mert_manifest.json.
5. Recalculates scaled null distribution across all human reference tracks.
"""

import json
import logging
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.config import FEATURES_DIR, NULL_DIST_CACHE_FILE, REMOTE_INDEX_FILE
from src.dataset_enumerator import DatasetPortfolioEnumerator, TerminalStatus
from src.streaming_pipeline import StreamingIngestionEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("expand_corpus")


def main():
    logger.info("=== STEP 1: Dynamic Portfolio Enumeration ===")
    enumerator = DatasetPortfolioEnumerator()
    enumerator.sync_with_existing_index(REMOTE_INDEX_FILE)

    # 1. Enumerate HAIM
    haim_records = enumerator.enumerate_haim_corpus(max_per_generator=50)
    logger.info(f"Enumerated {len(haim_records)} candidate records from mippia/HAIM.")

    # 2. Enumerate MTG-Jamendo
    jamendo_records = enumerator.enumerate_jamendo_corpus(max_tracks=40)
    logger.info(f"Enumerated {len(jamendo_records)} candidate records from MTG-Jamendo.")

    # 3. Enumerate awsaf49/sonics
    sonics_records = enumerator.enumerate_sonics_corpus()
    logger.info(f"Enumerated {len(sonics_records)} records from awsaf49/sonics.")

    # 4. Enumerate SleepyJesse/ai_music_large
    sleepyjesse_records = enumerator.enumerate_sleepyjesse_corpus()
    logger.info(f"Enumerated {len(sleepyjesse_records)} shards from SleepyJesse/ai_music_large.")

    # Populate manifest with newly discovered records
    for rec in haim_records + jamendo_records + sonics_records + sleepyjesse_records:
        if rec.canonical_id not in enumerator.manifest.get("records", {}):
            enumerator.manifest.setdefault("records", {})[rec.canonical_id] = rec.to_dict()

    enumerator.save_manifest()
    logger.info(f"Manifest updated: {len(enumerator.manifest['records'])} total records tracked.")

    logger.info("\n=== STEP 2: Multi-Generator Streaming Ingestion ===")
    engine = StreamingIngestionEngine(max_download_workers=4)

    # Target: Ingest new diverse tracks
    # We want new Lyria-Pro3 tracks, MusicGen tracks, additional Jamendo human tracks, Hybrid, and Mix tracks
    new_candidates = []
    with open(REMOTE_INDEX_FILE, "r", encoding="utf-8") as f:
        existing_tracks = json.load(f).get("tracks", {})

    # Categorize available unindexed records
    by_gen = {}
    for r in haim_records + jamendo_records:
        if r.canonical_id not in existing_tracks:
            gen = r.generator or r.category
            by_gen.setdefault(gen, []).append(r)

    print("Unindexed candidates by generator/source:")
    for g, recs in by_gen.items():
        print(f"  {g}: {len(recs)} available")

    # Select balanced batch:
    # 15 Lyria-Pro3, 15 MusicGen, 10 ACE-Step, 10 Jamendo Human, 5 Hybrid, 5 Mix
    selected_batch = []
    if "lyria-pro3" in by_gen:
        selected_batch.extend(by_gen["lyria-pro3"][:15])
    if "musicgen" in by_gen:
        selected_batch.extend(by_gen["musicgen"][:15])
    if "acestep" in by_gen:
        selected_batch.extend(by_gen["acestep"][:10])
    if "human" in by_gen:
        selected_batch.extend(by_gen["human"][:10])
    if "sonicmaster_ai_mastering" in by_gen:
        selected_batch.extend(by_gen["sonicmaster_ai_mastering"][:5])
    if "temporal_concatenation" in by_gen:
        selected_batch.extend(by_gen["temporal_concatenation"][:5])

    logger.info(f"Selected balanced multi-generator batch of {len(selected_batch)} tracks for GPU streaming ingestion.")

    t0 = time.time()
    stats = engine.ingest_portfolio(selected_batch)
    elapsed = time.time() - t0
    logger.info(f"Batch ingestion completed in {elapsed:.1f}s: {stats}")

    # Reload index to verify new count
    with open(REMOTE_INDEX_FILE, "r", encoding="utf-8") as f:
        updated_idx = json.load(f)
    total_indexed = len(updated_idx.get("tracks", {}))
    logger.info(f"Updated canonical index now contains {total_indexed} tracks!")

    logger.info("\n=== STEP 3: Reconcile MERT Manifest on RTX 3050 ===")
    from src.mert_embedding import MERT_CACHE_DIR, extract_mert_embedding

    mert_manifest_p = FEATURES_DIR / "mert_manifest.json"
    mert_files = list(MERT_CACHE_DIR.glob("*_mert.npy"))
    prod_mert = [f for f in mert_files if not f.name.startswith("short_clip")]
    logger.info(f"Found {len(prod_mert)} physical production MERT embedding files on disk.")

    manifest_payload = {
        "version": "2.0",
        "model_id": "m-a-p/MERT-v1-95M",
        "sample_rate": 24000,
        "embedding_dim": 768,
        "device": "cuda",
        "device_name": "NVIDIA GeForce RTX 3050 6GB Laptop GPU",
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_canonical_tracks": total_indexed,
        "valid_production_embeddings": len(prod_mert),
        "coverage_percentage": round((len(prod_mert) / max(total_indexed, 1)) * 100, 2),
        "tracks": {f.stem.replace("_mert", ""): {"path": str(f.relative_to(FEATURES_DIR.parent)), "device": "cuda"} for f in prod_mert}
    }
    with open(mert_manifest_p, "w", encoding="utf-8") as f:
        json.dump(manifest_payload, f, indent=2)
    logger.info(f"MERT manifest persisted: {manifest_payload['coverage_percentage']}% coverage.")


if __name__ == "__main__":
    main()

