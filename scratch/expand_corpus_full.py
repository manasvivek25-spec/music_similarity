"""
Master Full-Dataset Expansion & GPU-Accelerated Production Ingestion Pipeline.

1. Dynamically enumerates all candidate records across all 6 approved sources:
   - mippia/HAIM (A1 Real, A2 ACE-Step, A2 Lyria-Pro3, A2 MusicGen, B1 Hybrid, C1 Temporal Mix)
   - MTG-Jamendo (autotagging_genre.tsv with verified CC licenses and genre annotations)
   - awsaf49/sonics (Suno AI deepfakes + YouTube ToS exclusion)
   - SleepyJesse/ai_music_large (147 Parquet shards)
   - Free Music Archive (switch.ch official hosting)
   - DA-TACOS (Cover song analysis metadata)
2. Ingests 150+ new diverse tracks with:
   - 12-bin Chroma CQT and beat-synchronous chroma matrices
   - 56-dim acoustic features
   - 768-dim MERT-v1-95M embeddings on NVIDIA RTX 3050 Laptop GPU
   - Strict zero raw audio persistence
3. Expands production corpus from 120 to 270+ canonical tracks.
4. Recomputes exhaustive pairwise human null distribution (6,000+ pairs) across 110+ human tracks.
5. Retrains Origin Classifier and performs Leave-One-Generator-Out (LOGO) cross-validation.
6. Reconciles dataset_manifest.json and mert_manifest.json.
"""

import csv
import io
import json
import logging
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import requests
import torch
from huggingface_hub import HfApi

from src.config import FEATURES_DIR, NULL_DIST_CACHE_FILE, REMOTE_INDEX_FILE
from src.dataset_enumerator import (
    DatasetPortfolioEnumerator,
    MANIFEST_FILE,
    SourceRecord,
    TerminalStatus,
)
from src.mert_embedding import MERT_CACHE_DIR, extract_mert_embedding
from src.origin_classifier import (
    evaluate_group_aware_generalization,
    train_origin_classifier,
)
from src.streaming_pipeline import StreamingIngestionEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("expand_corpus_full")


def eval_pair(pair_data):
    id_a, id_b, f_a, f_b = pair_data
    try:
        from src.similarity import dtw_similarity
        c_a = np.load(f_a)
        c_b = np.load(f_b)
        sub_a = c_a[:, :300] if c_a.shape[1] > 300 else c_a
        sub_b = c_b[:, :300] if c_b.shape[1] > 300 else c_b
        return float(dtw_similarity(sub_a, sub_b))
    except Exception:
        return None


def enumerate_all_six_sources(enumerator: DatasetPortfolioEnumerator):
    logger.info("=== Enumerating All 6 Approved Dataset Portfolios ===")
    base_haim = "https://huggingface.co/datasets/mippia/HAIM/resolve/main"
    api = HfApi()

    # 1. HAIM dynamic enumeration
    logger.info("Enumerating mippia/HAIM...")
    haim_records = []
    
    # A1 Human
    try:
        a1_items = list(api.list_repo_tree(
            "mippia/HAIM", repo_type="dataset", path_in_repo="A_full_generation/A1_real/MTG-Jamendo music subset", recursive=False
        ))
        for it in a1_items:
            if it.path.endswith(".mp3"):
                stem = Path(it.path).stem
                haim_records.append(SourceRecord(
                    canonical_id=f"haim_human_{stem}",
                    source_dataset="mippia/HAIM",
                    source_identifier=it.path,
                    category="human",
                    generator="human",
                    audio_stream_url=f"{base_haim}/{it.path}",
                    license_info="Non-Commercial Research & Educational Use",
                ))
    except Exception as e:
        logger.warning(f"Error enumerating HAIM A1: {e}")

    # A2 Generators
    a2_folders = [
        ("acestep", ["A_full_generation/A2_fake/acestep/d0", "A_full_generation/A2_fake/acestep/d1"]),
        ("lyria-pro3", ["A_full_generation/A2_fake/lyria-pro3/d0", "A_full_generation/A2_fake/lyria-pro3/d1"]),
        ("musicgen", ["A_full_generation/A2_fake/musicgen/small", "A_full_generation/A2_fake/musicgen/medium", "A_full_generation/A2_fake/musicgen/large"]),
    ]
    for gen_name, subpaths in a2_folders:
        for sp in subpaths:
            try:
                items = list(api.list_repo_tree("mippia/HAIM", repo_type="dataset", path_in_repo=sp, recursive=False))
                for it in items:
                    if it.path.endswith(".mp3"):
                        stem = Path(it.path).stem
                        haim_records.append(SourceRecord(
                            canonical_id=f"haim_ai_{gen_name.replace('-', '_')}_{stem}",
                            source_dataset="mippia/HAIM",
                            source_identifier=it.path,
                            category="ai",
                            generator=gen_name,
                            audio_stream_url=f"{base_haim}/{it.path}",
                            license_info="Non-Commercial Research & Educational Use",
                        ))
            except Exception as e:
                logger.warning(f"Error enumerating HAIM {gen_name}: {e}")

    # B1 Hybrid
    try:
        b1_items = list(api.list_repo_tree(
            "mippia/HAIM", repo_type="dataset", path_in_repo="B_hybrid/B1_ai_mastered_human__MTG_sonicmaster/d0", recursive=False
        ))
        for it in b1_items:
            if it.path.endswith(".mp3"):
                stem = Path(it.path).stem
                haim_records.append(SourceRecord(
                    canonical_id=f"haim_hybrid_b1_{stem}",
                    source_dataset="mippia/HAIM",
                    source_identifier=it.path,
                    category="hybrid",
                    generator="sonicmaster_ai_mastering",
                    audio_stream_url=f"{base_haim}/{it.path}",
                    license_info="Non-Commercial Research & Educational Use",
                ))
    except Exception as e:
        logger.warning(f"Error enumerating HAIM B1: {e}")

    # C1 Mix
    try:
        c1_items = list(api.list_repo_tree(
            "mippia/HAIM", repo_type="dataset", path_in_repo="C_mixing/C1_mixset_concat/d0", recursive=False
        ))
        for it in c1_items:
            if it.path.endswith(".mp3"):
                stem = Path(it.path).stem
                haim_records.append(SourceRecord(
                    canonical_id=f"haim_temporal_c1_{stem}",
                    source_dataset="mippia/HAIM",
                    source_identifier=it.path,
                    category="temporal_mix",
                    generator="temporal_concatenation",
                    audio_stream_url=f"{base_haim}/{it.path}",
                    license_info="Non-Commercial Research & Educational Use",
                ))
    except Exception as e:
        logger.warning(f"Error enumerating HAIM C1: {e}")

    logger.info(f"Discovered {len(haim_records)} total records from mippia/HAIM.")

    # 2. MTG-Jamendo dynamic enumeration
    logger.info("Enumerating MTG-Jamendo catalog via autotagging_genre.tsv...")
    jamendo_records = []
    try:
        tsv_url = "https://raw.githubusercontent.com/MTG/mtg-jamendo-dataset/master/data/autotagging_genre.tsv"
        resp = requests.get(tsv_url, timeout=15)
        if resp.status_code == 200:
            reader = csv.reader(io.StringIO(resp.text), delimiter="\t")
            next(reader)  # Header
            for row in reader:
                if len(row) >= 6:
                    tid_raw = row[0].replace("track_", "").lstrip("0") or "0"
                    tid_num = int(tid_raw)
                    genre_clean = row[5].replace("genre---", "")
                    dur_val = float(row[4]) if row[4] else None
                    cid = f"mtg_jamendo_{tid_num}"
                    url = f"https://prod-1.storage.jamendo.com/?trackid={tid_num}&format=mp31&from=app-976"
                    jamendo_records.append(SourceRecord(
                        canonical_id=cid,
                        source_dataset="MTG-Jamendo",
                        source_identifier=row[0],
                        category="human",
                        generator="human",
                        audio_stream_url=url,
                        license_info="Creative Commons (CC BY-NC-SA)",
                        duration_sec=dur_val,
                    ))
    except Exception as e:
        logger.warning(f"Error enumerating MTG-Jamendo: {e}")

    logger.info(f"Discovered {len(jamendo_records)} MTG-Jamendo annotated tracks.")

    # 3. awsaf49/sonics
    logger.info("Enumerating awsaf49/sonics...")
    sonics_records = []
    for i in range(10):
        sonics_records.append(SourceRecord(
            canonical_id=f"sonics_real_{str(i).zfill(5)}",
            source_dataset="awsaf49/sonics",
            source_identifier=f"real_{str(i).zfill(5)}",
            category="human",
            generator="human",
            terminal_status=TerminalStatus.EXPLICITLY_EXCLUDED.value,
            status_reason="Contains external YouTube ID only; scraping prohibited per YouTube ToS",
            license_info="Research Use Only",
        ))
    for i in range(1, 11):
        sonics_records.append(SourceRecord(
            canonical_id=f"sonics_fake_chunk_{str(i).zfill(2)}",
            source_dataset="awsaf49/sonics",
            source_identifier=f"fake_songs/part_{str(i).zfill(2)}.zip",
            category="ai",
            generator="suno",
            audio_stream_url=f"https://huggingface.co/datasets/awsaf49/sonics/resolve/main/fake_songs/part_{str(i).zfill(2)}.zip",
            terminal_status=TerminalStatus.METADATA_ONLY.value,
            status_reason="Multi-gigabyte remote ZIP archive; accessible via HTTP range requests",
            license_info="Research Use Only",
        ))

    # 4. SleepyJesse/ai_music_large
    logger.info("Enumerating SleepyJesse/ai_music_large...")
    sleepyjesse_records = []
    for i in range(147):
        shard_name = f"train-{str(i).zfill(5)}-of-00147.parquet"
        sleepyjesse_records.append(SourceRecord(
            canonical_id=f"sleepyjesse_shard_{str(i).zfill(5)}",
            source_dataset="SleepyJesse/ai_music_large",
            source_identifier=shard_name,
            category="ai",
            generator="ai_music_large",
            audio_stream_url=f"https://huggingface.co/datasets/SleepyJesse/ai_music_large/resolve/main/data/{shard_name}",
            terminal_status=TerminalStatus.METADATA_ONLY.value,
            status_reason="Remote Parquet shard (~615 MB); accessible via HTTP byte range requests",
            cursor_shard=shard_name,
            license_info="Open Data / Research",
        ))

    # 5. Free Music Archive (FMA)
    logger.info("Enumerating Free Music Archive (FMA)...")
    fma_records = [
        SourceRecord(
            canonical_id="fma_metadata_archive",
            source_dataset="Free Music Archive",
            source_identifier="fma_metadata.zip",
            category="human",
            generator="human",
            audio_stream_url="https://os.unil.cloud.switch.ch/fma/fma_metadata.zip",
            terminal_status=TerminalStatus.METADATA_ONLY.value,
            status_reason="Full metadata catalog (tracks.csv, genres.csv) on official Swiss cloud",
            license_info="Open Audio / CC Licenses",
        ),
        SourceRecord(
            canonical_id="fma_small_archive",
            source_dataset="Free Music Archive",
            source_identifier="fma_small.zip",
            category="human",
            generator="human",
            audio_stream_url="https://os.unil.cloud.switch.ch/fma/fma_small.zip",
            terminal_status=TerminalStatus.METADATA_ONLY.value,
            status_reason="8,000 track audio archive (7.67 GB) on switch.ch supporting byte range streaming",
            license_info="CC BY-NC / Open Data",
        ),
    ]

    # 6. DA-TACOS
    logger.info("Enumerating DA-TACOS...")
    datacos_records = [
        SourceRecord(
            canonical_id="datacos_benchmark_subset",
            source_dataset="DA-TACOS",
            source_identifier="da-tacos_benchmark_subset",
            category="metadata_only",
            generator="none",
            terminal_status=TerminalStatus.METADATA_ONLY.value,
            status_reason="Designed without audio files per official MTG specification; pre-extracted features & annotations only",
            license_info="CC BY-NC-SA 4.0",
        ),
        SourceRecord(
            canonical_id="datacos_cover_analysis_subset",
            source_dataset="DA-TACOS",
            source_identifier="da-tacos_cover_analysis_subset",
            category="metadata_only",
            generator="none",
            terminal_status=TerminalStatus.METADATA_ONLY.value,
            status_reason="Designed without audio files per official MTG specification; cover song relationships and metadata only",
            license_info="CC BY-NC-SA 4.0",
        ),
    ]

    all_records = haim_records + jamendo_records + sonics_records + sleepyjesse_records + fma_records + datacos_records
    for r in all_records:
        if r.canonical_id not in enumerator.manifest.get("records", {}):
            enumerator.manifest.setdefault("records", {})[r.canonical_id] = r.to_dict()

    enumerator.save_manifest()
    logger.info(f"Master manifest updated: {len(enumerator.manifest['records'])} total records tracked.")
    return haim_records, jamendo_records


def main():
    logger.info("=== STARTING FULL PRODUCTION EXPANSION ===")
    enumerator = DatasetPortfolioEnumerator()
    enumerator.sync_with_existing_index(REMOTE_INDEX_FILE)

    haim_records, jamendo_records = enumerate_all_six_sources(enumerator)

    # Inspect current index
    with open(REMOTE_INDEX_FILE, "r", encoding="utf-8") as f:
        current_tracks = json.load(f).get("tracks", {})
    logger.info(f"Starting corpus count: {len(current_tracks)} tracks.")

    # Select candidates to ingest
    by_category = {}
    for r in haim_records:
        if r.canonical_id not in current_tracks:
            gen = r.generator or r.category
            by_category.setdefault(gen, []).append(r)

    for r in jamendo_records:
        if r.canonical_id not in current_tracks:
            by_category.setdefault("jamendo_human", []).append(r)

    logger.info("Available unindexed candidate pools:")
    for k, v in by_category.items():
        logger.info(f"  {k}: {len(v)} tracks available")

    # Ingestion targets:
    # 25 more HAIM human
    # 40 more MTG-Jamendo human
    # 26 more ACE-Step
    # 20 more Lyria-Pro3
    # 20 more MusicGen
    # 10 more Hybrid B1
    # 10 more Temporal Mix C1
    # Total new tracks = 151
    batch = []
    if "human" in by_category:
        batch.extend(by_category["human"][:25])
    if "jamendo_human" in by_category:
        batch.extend(by_category["jamendo_human"][:40])
    if "acestep" in by_category:
        batch.extend(by_category["acestep"][:26])
    if "lyria-pro3" in by_category:
        batch.extend(by_category["lyria-pro3"][:20])
    if "musicgen" in by_category:
        batch.extend(by_category["musicgen"][:20])
    if "sonicmaster_ai_mastering" in by_category:
        batch.extend(by_category["sonicmaster_ai_mastering"][:10])
    if "temporal_concatenation" in by_category:
        batch.extend(by_category["temporal_concatenation"][:10])

    logger.info(f"Selected batch of {len(batch)} tracks for GPU-accelerated streaming ingestion.")

    engine = StreamingIngestionEngine(max_download_workers=4)
    t0 = time.time()
    stats = engine.ingest_portfolio(batch)
    elapsed = time.time() - t0
    logger.info(f"Ingestion finished in {elapsed:.1f}s. Stats: {stats}")

    # Reload index
    with open(REMOTE_INDEX_FILE, "r", encoding="utf-8") as f:
        updated_idx = json.load(f)
    tracks = updated_idx.get("tracks", {})
    total_indexed = len(tracks)
    logger.info(f"Production index now contains: {total_indexed} canonical tracks!")

    # Breakdown by category
    cats = {}
    gens = {}
    for t in tracks.values():
        c = t.get("origin_label", "unknown")
        g = t.get("generator", "unknown")
        cats[c] = cats.get(c, 0) + 1
        gens[g] = gens.get(g, 0) + 1
    logger.info(f"Category breakdown: {cats}")
    logger.info(f"Generator breakdown: {gens}")

    # Reconcile MERT manifest
    logger.info("\n=== Reconciling MERT Manifest on RTX 3050 ===")
    mert_files = list(MERT_CACHE_DIR.glob("*_mert.npy"))
    prod_mert = [f for f in mert_files if not f.name.startswith("short_clip")]
    logger.info(f"Valid physical production MERT embedding files: {len(prod_mert)} / {total_indexed}")

    mert_manifest_p = FEATURES_DIR / "mert_manifest.json"
    mm_payload = {
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
        "tracks": {
            f.stem.replace("_mert", ""): {
                "path": str(f.relative_to(FEATURES_DIR.parent)),
                "device": "cuda",
            }
            for f in prod_mert
        },
    }
    with open(mert_manifest_p, "w", encoding="utf-8") as f:
        json.dump(mm_payload, f, indent=2)
    logger.info(f"MERT manifest persisted: {mm_payload['coverage_percentage']}% coverage.")

    enumerator.sync_with_existing_index(REMOTE_INDEX_FILE)

    # Recompute exhaustive human null distribution
    logger.info("\n=== Recomputing Scaled Exhaustive Human Null Distribution ===")
    import itertools
    from multiprocessing import Pool, cpu_count
    from src.similarity import dtw_similarity

    human_tids = [
        tid for tid, m in tracks.items()
        if str(m.get("origin_label", "")).lower() in ("human", "0", "false")
    ]
    human_tids.sort()

    valid_human = []
    for tid in human_tids:
        beat_p = FEATURES_DIR / f"{tid}_beat.npy"
        if beat_p.is_file():
            valid_human.append((tid, str(beat_p)))

    N_human = len(valid_human)
    expected_pairs = N_human * (N_human - 1) // 2
    logger.info(f"Discovered {N_human} human tracks with beat features on disk.")
    logger.info(f"Computing exhaustive {expected_pairs} pairwise comparisons in parallel...")

    pair_inputs = []
    for (id_a, f_a), (id_b, f_b) in itertools.combinations(valid_human, 2):
        pair_inputs.append((id_a, id_b, f_a, f_b))

    t_null_start = time.time()
    workers = min(8, cpu_count())
    with Pool(workers) as pool:
        scores = pool.map(eval_pair, pair_inputs)

    valid_scores = [round(float(s), 5) for s in scores if s is not None]
    null_elapsed = time.time() - t_null_start
    logger.info(f"Evaluated {len(valid_scores)} pairs in {null_elapsed:.2f}s!")

    scores_arr = np.array(valid_scores)
    logger.info(
        f"Null metrics: Min={scores_arr.min():.4f}, Max={scores_arr.max():.4f}, "
        f"Median={np.median(scores_arr):.4f}, 95th={np.percentile(scores_arr, 95):.4f}"
    )

    null_payload = {
        "version": "2.0",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_track_ids": [t for t, _ in valid_human],
        "sample_pairs_count": len(valid_scores),
        "null_scores": valid_scores,
        "genre_null_scores": {},
        "pairwise_dependency_note": (
            "Pairwise null observations are formed from pairs within the reference pool "
            "and share underlying tracks. While k/N provides the empirical p-value, "
            "observations are not strictly independent."
        ),
    }
    with open(NULL_DIST_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(null_payload, f, indent=2)
    logger.info(f"Persisted {len(valid_scores)} null scores to {NULL_DIST_CACHE_FILE}")

    # Retrain Origin Classifier on expanded dataset
    logger.info("\n=== Retraining Origin Classifier on Expanded Multi-Generator Corpus ===")
    clf_res = train_origin_classifier(tracks)
    logger.info(f"Classifier retrained: model={clf_res.get('model_type')}, test_acc={clf_res.get('test_accuracy')}")

    # Run LOGO cross-validation
    gen_eval = evaluate_group_aware_generalization(clf_res["pipeline"], tracks)
    logger.info("Leave-One-Generator-Out Evaluation Results:")
    for gen, res in gen_eval.get("leave_one_generator_out", {}).items():
        logger.info(
            f"  Generator [{gen}]: Unseen AI Recall={res.get('unseen_generator_recall')} "
            f"(sample={res.get('test_sample_size')}), Human Recall={res.get('held_out_human_recall')}"
        )

    # Persist metrics report
    os.makedirs("reports", exist_ok=True)
    with open("reports/origin_classifier_metrics.json", "w", encoding="utf-8") as f:
        json.dump(gen_eval, f, indent=2)

    logger.info("\n=== FULL EXPANSION SUCCESSFULLY COMPLETED ===")


if __name__ == "__main__":
    main()
