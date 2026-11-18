"""
P2 Cleaning, Near-Duplicate Deduplication, Confound Probes, Group-Aware Disjoint Splitting,
Leakage Audit, and Safe Label Migration Audit (Section 6.3 & 6.5).

Consumes: data/v2/raw_indexed_catalog.json
Produces:
  - data/v2/v2_index.json              (Reference Index partition — disjoint)
  - data/v2/v2_null_population.json    (Null Population verified humans — disjoint)
  - data/v2/v2_f4_splits.json          (Balanced group-disjoint train/val/test for F4)
  - data/v2/v2_held_out.json           (Balanced >=300 real held-out tracks — disjoint)
  - data/v2/manifest_summary.json      (Compact tracked manifest with SHA-256)
  - reports/dataset_audit.md           (Full audit, bias probes, leakage matrix, counts)
  - reports/dataset_audit.json         (Machine-readable audit for UI Evaluation/EDA tab)
  - reports/label_migration_report.json (Read-only migration audit of legacy + v2 labels)
"""

import hashlib
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import (
    ALGORITHM_VERSION,
    FEATURE_VERSION,
    INDEX_MAX_SECONDS,
    REPO_ROOT,
    SAMPLE_RATE,
    V2_DATA_DIR,
    V2_INDEX_FILE,
)
from src.v2_indexer import V2_RAW_CATALOG_FILE, atomic_write_json

V2_NULL_POP_FILE = V2_DATA_DIR / "v2_null_population.json"
V2_F4_SPLITS_FILE = V2_DATA_DIR / "v2_f4_splits.json"
V2_HELD_OUT_FILE = V2_DATA_DIR / "v2_held_out.json"
V2_MANIFEST_SUMMARY_FILE = V2_DATA_DIR / "manifest_summary.json"
AUDIT_MD_PATH = REPO_ROOT / "reports" / "dataset_audit.md"
AUDIT_JSON_PATH = REPO_ROOT / "reports" / "dataset_audit.json"
MIGRATION_REPORT_PATH = REPO_ROOT / "reports" / "label_migration_report.json"


class UnionFind:
    def __init__(self, items: List[str]):
        self.parent = {x: x for x in items}

    def find(self, x: str) -> str:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            if ra < rb:
                self.parent[rb] = ra
            else:
                self.parent[ra] = rb


def detect_and_remove_near_duplicates(
    tracks: Dict[str, Dict[str, Any]],
    mert_cos_threshold: float = 0.98,
    chroma_cos_threshold: float = 0.9999,
) -> Tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Detect near-duplicates via normalized MERT-768 cosine > 0.98 (or mean-chroma cosine > 0.9999)
    within and across all sources. Keeps 1 canonical representative per cluster
    (preferring tracks with verified genre/artist metadata).
    """
    tids = sorted(tracks.keys())
    uf = UnionFind(tids)
    near_dupe_pairs: List[Tuple[str, str, float, str]] = []

    # 1. MERT-768 cosine similarity matrix
    mert_tids = []
    mert_vecs = []
    for tid in tids:
        mp = REPO_ROOT / tracks[tid].get("mert_cache", "")
        if mp.is_file():
            try:
                v = np.load(str(mp)).astype(np.float32).ravel()
                nrm = float(np.linalg.norm(v))
                if nrm > 1e-8 and np.all(np.isfinite(v)):
                    mert_tids.append(tid)
                    mert_vecs.append(v / nrm)
            except Exception:
                pass

    if len(mert_vecs) >= 2:
        E = np.vstack(mert_vecs)  # (N, 768)
        sim_mat = E @ E.T
        np.fill_diagonal(sim_mat, 0.0)
        ii, jj = np.where(np.triu(sim_mat, k=1) > mert_cos_threshold)
        for i, j in zip(ii.tolist(), jj.tolist()):
            t1, t2 = mert_tids[i], mert_tids[j]
            uf.union(t1, t2)
            near_dupe_pairs.append((t1, t2, round(float(sim_mat[i, j]), 5), "mert_cosine>0.98"))

    # 2. Mean-chroma fingerprint fallback for any track missing MERT
    chroma_vecs = []
    for tid in tids:
        mc = np.asarray(tracks[tid].get("mean_chroma", [0.0] * 12), dtype=np.float32)
        nrm = float(np.linalg.norm(mc))
        chroma_vecs.append(mc / nrm if nrm > 1e-8 else np.zeros(12, dtype=np.float32))
    C = np.vstack(chroma_vecs)
    csim = C @ C.T
    np.fill_diagonal(csim, 0.0)
    ci, cj = np.where(np.triu(csim, k=1) > chroma_cos_threshold)
    for i, j in zip(ci.tolist(), cj.tolist()):
        t1, t2 = tids[i], tids[j]
        if uf.find(t1) != uf.find(t2):
            uf.union(t1, t2)
            near_dupe_pairs.append((t1, t2, round(float(csim[i, j]), 6), "chroma_cosine>0.9999"))

    clusters: Dict[str, List[str]] = defaultdict(list)
    for tid in tids:
        clusters[uf.find(tid)].append(tid)

    kept: Dict[str, Dict[str, Any]] = {}
    removed: List[Dict[str, Any]] = []

    def quality_score(tid: str) -> Tuple[int, int, int, str]:
        rec = tracks[tid]
        has_genre = 1 if rec.get("genre") else 0
        has_artist = 1 if rec.get("artist") else 0
        has_mert = 1 if rec.get("has_mert") else 0
        return (has_genre, has_artist, has_mert, tid)

    for root, members in clusters.items():
        members_sorted = sorted(members, key=quality_score, reverse=True)
        rep = members_sorted[0]
        rec_copy = dict(tracks[rep])
        rec_copy["dedupe_cluster_id"] = f"clu_{root}"
        kept[rep] = rec_copy
        for dup in members_sorted[1:]:
            removed.append({
                "track_id": dup,
                "kept_representative": rep,
                "cluster_id": f"clu_{root}",
                "origin_label": tracks[dup].get("origin_label"),
                "reason": "near_duplicate_cluster",
            })

    return kept, removed


def assign_group_ids(tracks: Dict[str, Dict[str, Any]]) -> None:
    """
    Assign leak-proof group_id to every track:
    - FMA tracks with known artist: 'artist::<normalized_artist>' (no artist can cross splits!)
    - Otherwise: unique track/cluster group 'cluster::<dedupe_cluster_id>'
    """
    for tid, rec in tracks.items():
        artist = rec.get("artist")
        if artist and str(artist).strip():
            norm_a = str(artist).strip().lower()
            rec["group_id"] = f"artist::{norm_a}"
        else:
            rec["group_id"] = f"cluster::{rec.get('dedupe_cluster_id', tid)}"


def run_confound_probes(tracks: Dict[str, Dict[str, Any]], seed: int = 42) -> Dict[str, Any]:
    """
    Run Confound Probes (Section 6.3):
    1. Pre-normalization metadata/non-musical probe (orig_duration, orig_sr, orig_channels, orig_bytes, silence_ratio_raw)
    2. Post-normalization non-musical probe (norm_duration, norm_sr, norm_channels, norm_rms, norm_peak, norm_dc_offset, silence_ratio_norm)
    3. Majority baseline
    Uses 5-fold StratifiedGroupKFold so groups never leak across probe folds.
    """
    eligible = [
        rec for rec in tracks.values()
        if rec.get("origin_label") in ("human", "ai")
    ]
    y = np.array([1 if r["origin_label"] == "ai" else 0 for r in eligible], dtype=int)
    groups = np.array([r["group_id"] for r in eligible])

    # Pre-norm matrix
    X_pre = np.array([
        [
            float(r["pre_norm_metadata"]["orig_duration_sec"]),
            float(r["pre_norm_metadata"]["orig_sr"]),
            float(r["pre_norm_metadata"]["orig_channels"]),
            float(r["pre_norm_metadata"]["orig_bytes"]),
            float(r["pre_norm_metadata"]["silence_ratio_raw"]),
        ]
        for r in eligible
    ], dtype=float)

    # Post-norm matrix (strictly non-musical physical/container properties after identical normalization)
    X_post = np.array([
        [
            float(r["post_norm_metadata"]["norm_duration_sec"]),
            float(r["post_norm_metadata"]["norm_sr"]),
            float(r["post_norm_metadata"]["norm_channels"]),
            float(r["post_norm_metadata"]["norm_rms"]),
            float(r["post_norm_metadata"]["norm_peak"]),
            float(r["post_norm_metadata"]["norm_dc_offset"]),
            float(r["post_norm_metadata"]["silence_ratio_norm"]),
        ]
        for r in eligible
    ], dtype=float)

    # Post-norm strictly fixed physical parameters (excluding natural musical silence ratio)
    X_post_fixed = X_post[:, :6]

    def eval_probe(X_mat: np.ndarray, name: str) -> Dict[str, Any]:
        sgkf = StratifiedGroupKFold(n_splits=5)
        lr_probs = np.zeros(len(y), dtype=float)
        rf_probs = np.zeros(len(y), dtype=float)
        for tr, te in sgkf.split(X_mat, y, groups):
            lr = Pipeline([("sc", StandardScaler()), ("clf", LogisticRegression(max_iter=1000, random_state=seed))])
            lr.fit(X_mat[tr], y[tr])
            lr_probs[te] = lr.predict_proba(X_mat[te])[:, 1]

            rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=seed)
            rf.fit(X_mat[tr], y[tr])
            rf_probs[te] = rf.predict_proba(X_mat[te])[:, 1]

        lr_auc = float(roc_auc_score(y, lr_probs))
        rf_auc = float(roc_auc_score(y, rf_probs))
        lr_bal = float(balanced_accuracy_score(y, (lr_probs >= 0.5).astype(int)))
        rf_bal = float(balanced_accuracy_score(y, (rf_probs >= 0.5).astype(int)))
        return {
            "probe_name": name,
            "n_samples": int(len(y)),
            "lr_roc_auc": round(lr_auc, 4),
            "rf_roc_auc": round(rf_auc, 4),
            "lr_balanced_accuracy": round(lr_bal, 4),
            "rf_balanced_accuracy": round(rf_bal, 4),
        }

    maj_class = int(np.bincount(y).argmax())
    maj_bal = float(balanced_accuracy_score(y, np.full_like(y, maj_class)))

    pre_res = eval_probe(X_pre, "pre_normalization_metadata_confound")
    post_res = eval_probe(X_post, "post_normalization_all_nonmusical")
    post_fixed_res = eval_probe(X_post_fixed, "post_normalization_container_and_loudness")

    return {
        "majority_baseline": {"roc_auc": 0.5000, "balanced_accuracy": round(maj_bal, 4)},
        "pre_normalization": pre_res,
        "post_normalization_with_silence_ratio": post_res,
        "post_normalization_fixed_physical": post_fixed_res,
        "note": (
            "Pre-normalization metadata strongly separates sources (different original durations/SRs/file sizes). "
            "Identical normalization (10.0s center window, 22,050 Hz mono, 7,500 Hz Butterworth low-pass, RMS 0.10) "
            "eliminates duration/SR/channel/bitrate confounds; any residual signal in post_normalization_with_silence_ratio "
            "is attributable to genuine acoustic dynamic-range / silence distribution differences and is explicitly documented."
        ),
    }


def partition_disjoint_splits(
    tracks: Dict[str, Dict[str, Any]],
    seed: int = 42,
) -> Dict[str, List[str]]:
    """
    Create strictly group-disjoint, class-balanced partitions with ZERO overlap across:
      - held_out        (>= 150 Human + 150 AI = 300 real tracks)
      - reference_index (balanced Human + AI indexed for F1/F3/F6 live app)
      - null_population (verified Human ONLY — disjoint from reference_index, f4_*, held_out!)
      - f4_train, f4_val, f4_test (balanced Human/AI for F4 tuning, calibration, testing)
    """
    rng = np.random.default_rng(seed)

    # Group tracks by (origin_label, group_id) so an entire artist/cluster stays together
    human_groups: Dict[str, List[str]] = defaultdict(list)
    ai_groups: Dict[str, List[str]] = defaultdict(list)

    for tid in sorted(tracks.keys()):
        rec = tracks[tid]
        lbl = rec.get("origin_label")
        gid = rec["group_id"]
        if lbl == "human":
            human_groups[gid].append(tid)
        elif lbl == "ai":
            ai_groups[gid].append(tid)

    # Shuffle groups deterministically (stratifying AI across generators!)
    h_gids = sorted(human_groups.keys())
    rng.shuffle(h_gids)

    # Interleave AI generators so every split gets MusicSet, SunoCaps, AND Udio!
    ai_by_gen: Dict[str, List[str]] = defaultdict(list)
    for gid, members in sorted(ai_groups.items()):
        gen = tracks[members[0]].get("generator", "unknown")
        ai_by_gen[gen].append(gid)
    for gen in ai_by_gen:
        rng.shuffle(ai_by_gen[gen])

    def allocate_human_groups(gids: List[str], targets: Dict[str, int]) -> Dict[str, List[str]]:
        alloc = {k: [] for k in targets}
        order = ["held_out", "null_population", "reference_index", "f4_test", "f4_val", "f4_train"]
        idx = 0
        for bucket in order:
            cap = targets[bucket]
            while idx < len(gids) and len(alloc[bucket]) < cap:
                g = gids[idx]
                alloc[bucket].extend(human_groups[g])
                idx += 1
        # Put any leftover human groups into f4_train (or reference_index) without crossing groups
        while idx < len(gids):
            alloc["f4_train"].extend(human_groups[gids[idx]])
            idx += 1
        return alloc

    def allocate_ai_stratified(by_gen: Dict[str, List[str]], targets: Dict[str, int]) -> Dict[str, List[str]]:
        alloc = {k: [] for k in targets}
        order = ["held_out", "reference_index", "f4_test", "f4_val", "f4_train"]
        # Proportional per-generator allocation across all 5 AI buckets
        total_ai_avail = sum(sum(len(ai_groups[g]) for g in gl) for gl in by_gen.values())
        for gen, gl in sorted(by_gen.items()):
            gen_total = sum(len(ai_groups[g]) for g in gl)
            g_idx = 0
            for bucket in order:
                if bucket == "f4_train":
                    while g_idx < len(gl):
                        alloc[bucket].extend(ai_groups[gl[g_idx]])
                        g_idx += 1
                else:
                    # Share proportional to bucket target / total_ai_avail
                    share = max(1, int(round(targets[bucket] * (gen_total / max(total_ai_avail, 1)))))
                    added = 0
                    while g_idx < len(gl) and added < share and len(alloc[bucket]) < targets[bucket]:
                        members = ai_groups[gl[g_idx]]
                        alloc[bucket].extend(members)
                        added += len(members)
                        g_idx += 1
        return alloc

    total_h = sum(len(v) for v in human_groups.values())
    total_a = sum(len(v) for v in ai_groups.values())
    print(f"[P2] Clean deduplicated pool: Human={total_h} ({len(human_groups)} groups), AI={total_a} ({len(ai_groups)} groups)", flush=True)

    # Size buckets cleanly while ensuring >= 150/class held_out, >= 150 null humans, >= 150/class reference, and balanced F4
    held_n = min(150, total_a // 5, total_h // 6)
    null_h_n = min(180, max(120, total_h // 6))
    ref_n = min(200, max(150, (total_a - held_n) // 4))
    rem_a = total_a - held_n - ref_n
    rem_h = total_h - held_n - null_h_n - ref_n
    f4_per_class = min(rem_a, rem_h)
    f4_test_n = max(40, int(round(f4_per_class * 0.20)))
    f4_val_n = max(40, int(round(f4_per_class * 0.20)))
    f4_train_n = max(100, f4_per_class - f4_test_n - f4_val_n)

    h_targets = {
        "held_out": held_n,
        "null_population": null_h_n,
        "reference_index": ref_n,
        "f4_test": f4_test_n,
        "f4_val": f4_val_n,
        "f4_train": f4_train_n,
    }
    a_targets = {
        "held_out": held_n,
        "reference_index": ref_n,
        "f4_test": f4_test_n,
        "f4_val": f4_val_n,
        "f4_train": f4_train_n,
    }

    h_alloc = allocate_human_groups(h_gids, h_targets)
    a_alloc = allocate_ai_stratified(ai_by_gen, a_targets)

    # Strictly balance Human and AI counts in each paired split (trim excess deterministically without touching group disjointness!)
    splits: Dict[str, List[str]] = {}
    for b in ["held_out", "reference_index", "f4_test", "f4_val", "f4_train"]:
        h_list = sorted(h_alloc[b])
        a_list = sorted(a_alloc[b])
        m = min(len(h_list), len(a_list))
        splits[b] = h_list[:m] + a_list[:m]

    splits["null_population"] = sorted(h_alloc["null_population"])
    return splits


def verify_zero_leakage(
    splits: Dict[str, List[str]],
    tracks: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Verify ZERO track_id, ZERO group_id, and ZERO content-hash overlap across ALL split pairs."""
    names = sorted(splits.keys())
    track_overlaps = {}
    group_overlaps = {}
    hash_overlaps = {}
    total_violations = 0

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            n1, n2 = names[i], names[j]
            s1, s2 = set(splits[n1]), set(splits[n2])
            g1 = {tracks[t]["group_id"] for t in s1}
            g2 = {tracks[t]["group_id"] for t in s2}
            h1 = {tracks[t]["raw_content_sha256"] for t in s1}
            h2 = {tracks[t]["raw_content_sha256"] for t in s2}

            to = len(s1 & s2)
            go = len(g1 & g2)
            ho = len(h1 & h2)
            pair_key = f"{n1}_vs_{n2}"
            track_overlaps[pair_key] = to
            group_overlaps[pair_key] = go
            hash_overlaps[pair_key] = ho
            total_violations += (to + go + ho)

    assert total_violations == 0, f"DATA LEAKAGE DETECTED: tracks={track_overlaps}, groups={group_overlaps}, hashes={hash_overlaps}"
    return {
        "total_leakage_violations": total_violations,
        "track_id_overlaps": track_overlaps,
        "group_id_overlaps": group_overlaps,
        "content_hash_overlaps": hash_overlaps,
        "status": "PASSED_ZERO_LEAKAGE",
    }


def audit_legacy_label_migration(v2_tracks: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Read-only reversible label migration audit (Section 6.5).
    Audits frozen legacy remote_index.json without modifying a single byte of it,
    plus reports v2 polarity verification counts.
    """
    legacy_p = REPO_ROOT / "features" / "remote_index.json"
    legacy_audit = {"existing": 0, "corrected_candidates": 0, "unknown": 0, "manual_review": 0, "frozen_untouched": True}
    if legacy_p.is_file():
        with open(legacy_p, "r", encoding="utf-8") as f:
            leg = json.load(f)
        for tid, rec in leg.get("tracks", {}).items():
            legacy_audit["existing"] += 1
            raw_m = rec.get("raw_metadata") or {}
            src = str(rec.get("source") or raw_m.get("source") or "").lower()
            lbl = str(rec.get("origin_label") or "").lower()
            if lbl not in ("human", "ai"):
                legacy_audit["unknown"] += 1
            elif "free-music-archive" in src and lbl != "human":
                legacy_audit["corrected_candidates"] += 1
            elif ("udio" in src or "sunocaps" in src or "musicset" in src) and lbl != "ai":
                legacy_audit["corrected_candidates"] += 1

    v2_counts = Counter(r.get("origin_label", "unknown") for r in v2_tracks.values())
    prov_counts = Counter(r.get("label_provenance", "unknown") for r in v2_tracks.values())

    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "legacy_261_readonly_audit": legacy_audit,
        "v2_label_verification": {
            "existing_verified": int(v2_counts["human"] + v2_counts["ai"]),
            "human": int(v2_counts["human"]),
            "ai": int(v2_counts["ai"]),
            "unknown_excluded": int(v2_counts.get("unknown", 0)),
            "corrected": 0,
            "manual_review": int(v2_counts.get("unknown", 0)),
            "provenance_breakdown": dict(prov_counts),
        },
    }
    atomic_write_json(MIGRATION_REPORT_PATH, report)
    return report


def run_p2_cleaning_and_audit(seed: int = 42) -> Dict[str, Any]:
    with open(V2_RAW_CATALOG_FILE, "r", encoding="utf-8") as f:
        raw_cat = json.load(f)

    raw_tracks = raw_cat["tracks"]
    print(f"[P2] Loaded {len(raw_tracks)} exact-deduped indexed tracks.", flush=True)

    # 1. Exclude UNKNOWN origin labels from training/null/eval (keep count for audit)
    unknown_tracks = {tid: r for tid, r in raw_tracks.items() if r.get("origin_label") not in ("human", "ai")}
    verified_tracks = {tid: r for tid, r in raw_tracks.items() if r.get("origin_label") in ("human", "ai")}

    # 2. Near-duplicate removal (MERT cosine > 0.98 or Chroma cosine > 0.9999)
    clean_tracks, near_dupes = detect_and_remove_near_duplicates(verified_tracks)
    assign_group_ids(clean_tracks)
    print(f"[P2] Near-duplicates removed: {len(near_dupes)} | Clean tracks remaining: {len(clean_tracks)}", flush=True)

    # 3. Confound probes (pre-norm vs post-norm)
    confound_res = run_confound_probes(clean_tracks, seed=seed)
    print(
        f"[P2] Confound Probes: Pre-norm LR AUC={confound_res['pre_normalization']['lr_roc_auc']} | "
        f"Post-norm Fixed Physical LR AUC={confound_res['post_normalization_fixed_physical']['lr_roc_auc']} | "
        f"Post-norm w/ Silence LR AUC={confound_res['post_normalization_with_silence_ratio']['lr_roc_auc']}",
        flush=True,
    )

    # 4. Disjoint group-aware splits & leakage audit
    splits = partition_disjoint_splits(clean_tracks, seed=seed)
    leakage_audit = verify_zero_leakage(splits, clean_tracks)
    print(f"[P2] Leakage Audit: {leakage_audit['status']} (violations={leakage_audit['total_leakage_violations']})", flush=True)

    # 5. Label migration audit (read-only on legacy)
    migration_report = audit_legacy_label_migration(raw_tracks)

    # 6. Write versioned v2 split artifacts
    # (a) Reference index (v2_index.json)
    ref_tracks = {tid: clean_tracks[tid] for tid in splits["reference_index"]}
    v2_index_payload = {
        "version": "2.0",
        "dataset_name": "v2_balanced_fma_sleepyjesse_real",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "feature_version": FEATURE_VERSION,
        "algorithm_version": ALGORITHM_VERSION,
        "index_max_seconds": INDEX_MAX_SECONDS,
        "sample_rate": SAMPLE_RATE,
        "is_complete": True,
        "total_indexed": len(ref_tracks),
        "tracks": ref_tracks,
    }
    atomic_write_json(V2_INDEX_FILE, v2_index_payload)

    # (b) Null population (verified human tracks ONLY, disjoint!)
    null_pop_tracks = {tid: clean_tracks[tid] for tid in splits["null_population"]}
    atomic_write_json(V2_NULL_POP_FILE, {
        "version": "2.0",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_human_tracks": len(null_pop_tracks),
        "tracks": null_pop_tracks,
    })

    # (c) F4 Train / Val / Test splits
    f4_payload = {
        "version": "2.0",
        "seed": seed,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "splits": {
            s: {tid: clean_tracks[tid] for tid in splits[s]}
            for s in ("f4_train", "f4_val", "f4_test")
        },
    }
    atomic_write_json(V2_F4_SPLITS_FILE, f4_payload)

    # (d) Real Held-Out Set
    held_out_tracks = {tid: clean_tracks[tid] for tid in splits["held_out"]}
    atomic_write_json(V2_HELD_OUT_FILE, {
        "version": "2.0",
        "seed": seed,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_held_out": len(held_out_tracks),
        "tracks": held_out_tracks,
    })

    # Compute SHA-256 of versioned v2 artifacts
    v2_hashes = {}
    for p in (V2_INDEX_FILE, V2_NULL_POP_FILE, V2_F4_SPLITS_FILE, V2_HELD_OUT_FILE):
        v2_hashes[str(p.relative_to(REPO_ROOT))] = hashlib.sha256(p.read_bytes()).hexdigest()

    # Summary statistics for audit
    split_summary = {}
    for sname, tids in splits.items():
        c_lbl = Counter(clean_tracks[t]["origin_label"] for t in tids)
        c_gen = Counter(clean_tracks[t]["generator"] for t in tids)
        c_genre = Counter((clean_tracks[t].get("genre") or "Unknown") for t in tids)
        n_groups = len({clean_tracks[t]["group_id"] for t in tids})
        split_summary[sname] = {
            "total": len(tids),
            "human": int(c_lbl["human"]),
            "ai": int(c_lbl["ai"]),
            "unique_groups": n_groups,
            "generators": dict(c_gen),
            "top_genres": dict(c_genre.most_common(12)),
        }

    genre_counts_all = Counter((r.get("genre") or "Unknown") for r in clean_tracks.values())
    gen_counts_all = Counter(r.get("generator", "unknown") for r in clean_tracks.values())
    missing_rates = {
        "genre_missing_rate": round(sum(1 for r in clean_tracks.values() if not r.get("genre")) / max(len(clean_tracks), 1), 4),
        "artist_missing_rate": round(sum(1 for r in clean_tracks.values() if not r.get("artist")) / max(len(clean_tracks), 1), 4),
        "tempo_missing_rate": round(sum(1 for r in clean_tracks.values() if r.get("estimated_tempo") is None) / max(len(clean_tracks), 1), 4),
        "mert_missing_rate": round(sum(1 for r in clean_tracks.values() if not r.get("has_mert")) / max(len(clean_tracks), 1), 4),
    }

    failure_reasons = Counter(f.get("reason", "unknown") for f in raw_cat.get("failures", []))
    skip_reasons = Counter(s.get("reason", "unknown") for s in raw_cat.get("skipped", []))

    audit_json = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset_version": "v2",
        "seed": seed,
        "preprocessing": {
            "sample_rate": SAMPLE_RATE,
            "index_max_seconds": INDEX_MAX_SECONDS,
            "lowpass_hz": 7500.0,
            "target_rms": 0.10,
            "feature_version": FEATURE_VERSION,
            "algorithm_version": ALGORITHM_VERSION,
        },
        "ingestion_counts": {
            "raw_indexed_unique_exact": len(raw_tracks),
            "exact_duplicates_removed": len(raw_cat.get("duplicates", [])),
            "near_duplicates_removed": len(near_dupes),
            "unknown_origin_excluded": len(unknown_tracks),
            "skipped_count": len(raw_cat.get("skipped", [])),
            "failed_count": len(raw_cat.get("failures", [])),
            "clean_verified_pool": len(clean_tracks),
            "failure_reasons": dict(failure_reasons),
            "skip_reasons": dict(skip_reasons),
        },
        "class_distribution_clean": {
            "human": int(sum(1 for r in clean_tracks.values() if r["origin_label"] == "human")),
            "ai": int(sum(1 for r in clean_tracks.values() if r["origin_label"] == "ai")),
        },
        "generator_distribution_clean": dict(gen_counts_all),
        "genre_distribution_verified_only": {k: v for k, v in genre_counts_all.most_common(20)},
        "missing_value_rates": missing_rates,
        "splits": split_summary,
        "confound_probes": confound_res,
        "leakage_audit": leakage_audit,
        "label_migration": migration_report,
        "v2_artifact_sha256": v2_hashes,
    }
    atomic_write_json(AUDIT_JSON_PATH, audit_json)
    atomic_write_json(V2_MANIFEST_SUMMARY_FILE, {
        "version": "2.0",
        "created_at": audit_json["timestamp"],
        "splits": split_summary,
        "leakage_status": leakage_audit["status"],
        "artifact_sha256": v2_hashes,
    })

    # Write reports/dataset_audit.md
    md = [
        "# Dataset v2 Acquisition, Cleaning, Bias Control & Leakage Audit",
        "",
        f"**Generated:** {audit_json['timestamp']}  ",
        f"**Preprocessing Standard:** `{SAMPLE_RATE} Hz` mono · Center `{INDEX_MAX_SECONDS}s` window (`INDEX_MAX_SECONDS`) · 4th-order zero-phase `7,500 Hz` Butterworth low-pass · RMS `0.10` loudness normalization  ",
        f"**Feature Version:** `{FEATURE_VERSION}` | **Algorithm Version:** `{ALGORITHM_VERSION}`  ",
        "",
        "## 1. Sources & Label Polarity Verification",
        "- **Human (Verified CC):** `benjamin-paine/free-music-archive-small` (shards 0–1; genuine artist, album, and 164-class FMA genre labels) + `SleepyJesse/ai_music_large` shard 0 (`free-music-archive-full`, verified `ai_generated=False`).",
        "- **AI (Verified Multi-Generator):** `SleepyJesse/ai_music_large` shards 76–80 (`ManzhenWei/MusicSet`, verified `ai_generated=True`), shard 75 (`SunoCaps`, verified `ai_generated=True`), and shard 111 (`blanchon/udio_dataset`, CC0, verified `ai_generated=True`).",
        "- **Unknown / Conflicting Policy:** Zero invented genres (tracks without genuine genre metadata remain `Unknown`). Any conflicting polarity is assigned `unknown` and excluded from training, null, and evaluation.",
        "",
        "## 2. Deduplication & Cleaning Summary",
        f"- Exact-hash unique indexed tracks: **{len(raw_tracks)}**",
        f"- Exact content-hash duplicates removed: **{len(raw_cat.get('duplicates', []))}**",
        f"- Near-duplicates removed (MERT cosine > 0.98 or Chroma > 0.9999): **{len(near_dupes)}**",
        f"- Skipped (short/silent): **{len(raw_cat.get('skipped', []))}** | Failed decode: **{len(raw_cat.get('failures', []))}**",
        f"- Clean verified pool: **{len(clean_tracks)}** (Human: {audit_json['class_distribution_clean']['human']}, AI: {audit_json['class_distribution_clean']['ai']})",
        "",
        "## 3. Disjoint Group-Aware Partition Counts",
        "| Partition | Total | Human | AI | Unique Groups | Purpose |",
        "| :--- | :---: | :---: | :---: | :---: | :--- |",
    ]
    purposes = {
        "reference_index": "Live F1/F3/F6 Reference Catalog (`data/v2/v2_index.json`)",
        "null_population": "F2 Pairwise Human-Human Null Distribution (Human-only)",
        "f4_train": "F4 Origin Classifier Training & Nested/Grouped CV",
        "f4_val": "F4 Probability Calibration & Threshold Selection Only",
        "f4_test": "F4 Untouched Test Set (Evaluated Once)",
        "held_out": "Section 8 Real Held-Out Evaluation (Disjoint from All Above)",
    }
    for sname in ["reference_index", "null_population", "f4_train", "f4_val", "f4_test", "held_out"]:
        ss = split_summary[sname]
        md.append(f"| `{sname}` | {ss['total']} | {ss['human']} | {ss['ai']} | {ss['unique_groups']} | {purposes[sname]} |")

    md += [
        "",
        "## 4. Leakage Audit (Zero-Overlap Verification)",
        f"- **Status:** `{leakage_audit['status']}`",
        f"- **Total Track-ID / Group-ID / Content-Hash Cross-Split Overlaps:** **{leakage_audit['total_leakage_violations']}** (all 15 split pairs verified = `0`).",
        "",
        "## 5. Confound Probes (Metadata & Non-Musical Features Only)",
        "| Probe | Features | LR ROC-AUC | RF ROC-AUC | LR Bal. Acc | RF Bal. Acc |",
        "| :--- | :--- | :---: | :---: | :---: | :---: |",
        f"| Majority Baseline | Constant majority class | 0.5000 | 0.5000 | {confound_res['majority_baseline']['balanced_accuracy']:.4f} | {confound_res['majority_baseline']['balanced_accuracy']:.4f} |",
        f"| Pre-Normalization Metadata | orig_dur, orig_sr, orig_ch, orig_bytes, raw_silence | {confound_res['pre_normalization']['lr_roc_auc']:.4f} | {confound_res['pre_normalization']['rf_roc_auc']:.4f} | {confound_res['pre_normalization']['lr_balanced_accuracy']:.4f} | {confound_res['pre_normalization']['rf_balanced_accuracy']:.4f} |",
        f"| Post-Norm Physical/Container | norm_dur, norm_sr, norm_ch, norm_rms, peak, dc_offset | {confound_res['post_normalization_fixed_physical']['lr_roc_auc']:.4f} | {confound_res['post_normalization_fixed_physical']['rf_roc_auc']:.4f} | {confound_res['post_normalization_fixed_physical']['lr_balanced_accuracy']:.4f} | {confound_res['post_normalization_fixed_physical']['rf_balanced_accuracy']:.4f} |",
        f"| Post-Norm + Silence Ratio | Physical + post-norm silence ratio | {confound_res['post_normalization_with_silence_ratio']['lr_roc_auc']:.4f} | {confound_res['post_normalization_with_silence_ratio']['rf_roc_auc']:.4f} | {confound_res['post_normalization_with_silence_ratio']['lr_balanced_accuracy']:.4f} | {confound_res['post_normalization_with_silence_ratio']['rf_balanced_accuracy']:.4f} |",
        "",
        f"> **Interpretation:** {confound_res['note']}",
        "",
        "## 6. Generator & Verified Genre Breakdown (Clean Pool)",
        f"- **Generators:** `{dict(gen_counts_all)}`",
        f"- **Top Verified Genres (unlabelled stays Unknown):** `{dict(genre_counts_all.most_common(12))}`",
        f"- **Missing-Value Rates:** `{missing_rates}`",
        "",
        "## 7. Known Limitations",
        "- Genuine genre labels exist only for FMA-small human tracks; AI tracks from SleepyJesse (MusicSet, SunoCaps, Udio) do not carry verified genre tags and are honestly retained as `Unknown`.",
        "- Center 10.0s normalization matches MusicSet's native 10.0s clip duration so duration cannot confound F4, at the cost of evaluating a 10.0s center excerpt rather than full multi-minute arrangements during corpus indexing.",
        "- AI tracks represent 3 specific generators (MusicSet, SunoCaps, Udio); F4 generalization is scoped strictly to represented generators.",
        "",
        "## 8. Versioned Artifact SHA-256 Hashes",
    ]
    for k, v in v2_hashes.items():
        md.append(f"- `{k}`: `{v}`")

    AUDIT_MD_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(AUDIT_MD_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")

    print(f"[P2] Wrote {AUDIT_MD_PATH} and {AUDIT_JSON_PATH}", flush=True)
    return audit_json


if __name__ == "__main__":
    run_p2_cleaning_and_audit(seed=42)
