"""
Truth Panel Ground-Truth Manifest Lookup (Section 9 Item 5).
Matches uploaded audio content hash (raw file SHA-256 or normalized waveform SHA-256)
against the held-out evaluation manifest without altering any predictions.
"""

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import librosa
import numpy as np

from src.config import INDEX_MAX_SECONDS, REPO_ROOT, SAMPLE_RATE

MANIFEST_CANDIDATES = [
    REPO_ROOT / "evaluation" / "evaluation_manifest.csv",
    REPO_ROOT / "evaluation_manifest.csv",
]


def _load_manifest_by_hash() -> Dict[str, Dict[str, Any]]:
    index: Dict[str, Dict[str, Any]] = {}
    for mp in MANIFEST_CANDIDATES:
        if not mp.is_file():
            continue
        try:
            with open(mp, "r", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    h = (row.get("sha256") or "").strip().lower()
                    if h:
                        index[h] = row
        except Exception:
            continue

    # Also index held-out raw & norm hashes from data/v2/v2_held_out.json if present
    v2_ho = REPO_ROOT / "data" / "v2" / "v2_held_out.json"
    if v2_ho.is_file():
        try:
            with open(v2_ho, "r", encoding="utf-8") as f:
                tracks = json.load(f).get("tracks", {})
            for tid, rec in tracks.items():
                row_equiv = {
                    "track_id": tid,
                    "sha256": rec.get("norm_sha256", ""),
                    "label": rec.get("origin_label", "unknown"),
                    "generator": rec.get("generator", "unknown"),
                    "source": rec.get("label_provenance", "v2_held_out"),
                    "group_id": rec.get("group_id", ""),
                    "bpm": rec.get("estimated_tempo"),
                    "key": rec.get("estimated_key"),
                    "related_ids": tid,
                    "transform_params": json.dumps({"type": "original"}),
                }
                for hk in ("raw_content_sha256", "norm_sha256"):
                    hv = (rec.get(hk) or "").strip().lower()
                    if hv and hv not in index:
                        index[hv] = row_equiv
        except Exception:
            pass

    return index


def lookup_upload_ground_truth(
    audio_path: Path,
    predicted_origin: Optional[str] = None,
    detected_best_shift: Optional[int] = None,
    detected_segments: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Match uploaded audio against held-out manifest via content hash and build
    Ground Truth vs Prediction comparison payload for the UI Truth Panel.
    """
    try:
        file_sha = hashlib.sha256(Path(audio_path).read_bytes()).hexdigest().lower()
    except Exception:
        return {"available": False, "message": "No ground truth available for this upload."}

    manifest_index = _load_manifest_by_hash()
    matched = manifest_index.get(file_sha)

    # Secondary check on normalized waveform hash (handles WAV header differences)
    if matched is None:
        try:
            from src.v2_indexer import normalize_waveform
            y, sr = librosa.load(str(audio_path), sr=None, mono=True)
            raw_wav_sha = hashlib.sha256(y.astype(np.float32).tobytes()).hexdigest().lower()
            matched = manifest_index.get(raw_wav_sha)
            if matched is None:
                y_norm, _ = normalize_waveform(y, int(sr), target_sec=INDEX_MAX_SECONDS, target_sr=SAMPLE_RATE)
                norm_sha = hashlib.sha256(y_norm.tobytes()).hexdigest().lower()
                matched = manifest_index.get(norm_sha)
        except Exception:
            matched = None

    if matched is None:
        return {
            "available": False,
            "file_sha256": file_sha,
            "message": "No ground truth available for this upload.",
        }

    true_label = str(matched.get("label", "unknown")).strip().lower()
    true_origin_display = "AI" if true_label == "ai" else ("Human" if true_label == "human" else "Unknown")
    pred_norm = (predicted_origin or "").strip().lower()
    origin_match: Optional[bool] = None
    if true_label in ("human", "ai") and pred_norm in ("human", "ai"):
        origin_match = (true_label == pred_norm)

    params: Dict[str, Any] = {}
    try:
        params = json.loads(matched.get("transform_params") or "{}")
    except Exception:
        params = {}

    ttype = params.get("type", "original")

    # Transposition shift comparison
    shift_comparison = None
    if ttype == "transposition" and "k" in params:
        true_k = int(params["k"])
        shift_match = None
        recovered_k = None
        if detected_best_shift is not None:
            recovered_k = int(detected_best_shift) % 12
            shift_match = (recovered_k == (true_k % 12))
        shift_comparison = {
            "true_semitone_shift": true_k,
            "detected_best_shift": detected_best_shift,
            "recovered_shift_mod12": recovered_k,
            "match": shift_match,
        }

    # Spliced segment comparison
    splice_comparison = None
    if ttype == "spliced" and "start_sec" in params and "end_sec" in params:
        t_start = float(params["start_sec"])
        t_end = float(params["end_sec"])
        pred_iv = None
        iou = None
        if detected_segments:
            seg0 = detected_segments[0]
            p_start = float(seg0.get("start_time_a", 0.0))
            p_end = float(seg0.get("end_time_a", 0.0))
            pred_iv = [p_start, p_end]
            ov = max(0.0, min(t_end, p_end) - max(t_start, p_start))
            un = max(1e-6, max(t_end, p_end) - min(t_start, p_start))
            iou = round(ov / un, 4)
        splice_comparison = {
            "true_interval_sec": [t_start, t_end],
            "detected_interval_sec": pred_iv,
            "iou": iou,
            "match": bool(iou is not None and iou >= 0.30),
        }

    return {
        "available": True,
        "file_sha256": file_sha,
        "manifest_source": matched.get("source"),
        "group_id": matched.get("group_id"),
        "generator": matched.get("generator"),
        "transform_type": ttype,
        "transform_params": params,
        "origin_comparison": {
            "true_origin": true_origin_display,
            "predicted_origin": predicted_origin,
            "match": origin_match,
        },
        "shift_comparison": shift_comparison,
        "splice_comparison": splice_comparison,
        "message": "Matched held-out evaluation manifest entry by cryptographic content hash.",
    }
