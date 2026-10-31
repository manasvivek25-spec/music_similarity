"""
Remote Dataset Streaming and Incremental Indexing Module.

Supports streaming the Hugging Face dataset (default: 'SleepyJesse/ai_music_large')
without downloading the entire raw corpus to disk.
Dynamically inspects and validates the schema, processes audio records in memory,
precomputes and persists compact 12-dimensional mean-chroma vectors in a local index,
and caches detailed beat-synchronous representations only for shortlisted candidates.

Network note: Streaming downloads Parquet column chunks and audio records over HTTP
on demand. While it avoids storing the complete raw corpus on disk, network bandwidth
is consumed for all streamed and decoded records.
"""

import argparse
import io
import json
import logging
import os
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Generator, Iterator, List, Optional, Tuple, Union

# Ensure repo root is on sys.path for both direct execution and module execution
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import soundfile as sf
import librosa

from src.config import (
    COSINE_PREFILTER_THRESHOLD,
    DEFAULT_REMOTE_DATASET,
    DEFAULT_REMOTE_SAMPLE_SIZE,
    DEFAULT_REMOTE_SPLIT,
    FEATURES_DIR,
    REMOTE_INDEX_FILE,
    SAMPLE_RATE,
    SIMILARITY_FLOOR,
)
from src.similarity import (
    DTWResult,
    compute_mean_chroma_vector,
    cosine_similarity_vectors,
    dtw_align_detailed,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Schema Definitions and Detection
# ---------------------------------------------------------------------------

@dataclass
class DatasetSchemaInfo:
    """
    Metadata representation of detected dataset columns and audio encodings.
    """
    audio_col: Optional[str] = None
    id_col: Optional[str] = None
    label_col: Optional[str] = None
    genre_col: Optional[str] = None
    all_columns: List[str] = field(default_factory=list)
    audio_type: str = "unknown"  # 'datasets.Audio', 'dict_bytes', 'dict_array', 'raw_bytes', 'path', 'unknown'
    is_valid: bool = False
    validation_error: Optional[str] = None


@dataclass
class IndexingReport:
    """
    Structured metrics report generated after incremental remote dataset indexing.
    """
    dataset_name: str
    split: str
    is_full_pass: bool
    requested_sample_size: Optional[int]
    processed_records: int
    failed_records: int
    cache_hits: int
    downloaded_bytes: int
    elapsed_sec: float
    index_file: str
    errors: List[str] = field(default_factory=list)
    records_scanned: int = 0
    duplicate_records: int = 0
    unresolved_records_count: int = 0
    is_complete: bool = False
    status: str = "Partial"
    disclaimer: str = (
        "Streaming downloads Parquet chunks and audio records over HTTP incrementally on demand. "
        "While it avoids storing the complete raw corpus on disk, network bandwidth is consumed "
        "for all streamed and decoded records."
    )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def detect_dataset_schema(features_or_sample: Any) -> DatasetSchemaInfo:
    """
    Inspect and validate the dataset schema without hardcoding column names.

    Accepts either a Hugging Face `Features` mapping/object or a dictionary
    representing an individual data record.

    Identifies:
      - audio_col: column containing audio payload (bytes, array, or Audio feature)
      - id_col: column containing unique identifier or title/filename
      - label_col: column indicating origin (AI vs Human, generator, source, model)
      - genre_col: column indicating musical style or genre tag

    Returns
    -------
    DatasetSchemaInfo
        Validation summary containing identified columns and validity flag.
    """
    schema = DatasetSchemaInfo()

    # Extract column names and features dictionary
    if hasattr(features_or_sample, "items"):
        col_names = list(features_or_sample.keys())
    elif hasattr(features_or_sample, "column_names"):
        col_names = list(features_or_sample.column_names)
    elif isinstance(features_or_sample, (list, tuple)):
        col_names = list(features_or_sample)
    else:
        col_names = []

    schema.all_columns = col_names

    if not col_names:
        schema.is_valid = False
        schema.validation_error = "Dataset features or record is empty."
        return schema

    # 1. Detect Audio Column
    audio_candidates = ["audio", "audio_data", "audio_bytes", "waveform", "sound", "music", "bytes", "wav", "mp3"]
    detected_audio_col = None
    detected_audio_type = "unknown"

    # Priority A: Check for Hugging Face Audio feature type
    for col in col_names:
        val = features_or_sample.get(col) if hasattr(features_or_sample, "get") else None
        type_str = str(type(val)).lower()
        repr_str = str(val).lower()
        if "audio" in type_str or "audio(" in repr_str:
            detected_audio_col = col
            detected_audio_type = "datasets.Audio"
            break

    # Priority B: Check for dict structure {'bytes': ...} or {'array': ...} or raw bytes
    if not detected_audio_col and hasattr(features_or_sample, "get"):
        for col in col_names:
            val = features_or_sample.get(col)
            if isinstance(val, dict):
                if "bytes" in val:
                    detected_audio_col = col
                    detected_audio_type = "dict_bytes"
                    break
                elif "array" in val:
                    detected_audio_col = col
                    detected_audio_type = "dict_array"
                    break
            elif isinstance(val, (bytes, bytearray)):
                detected_audio_col = col
                detected_audio_type = "raw_bytes"
                break

    # Priority C: Fall back to candidate column name matching
    if not detected_audio_col:
        for cand in audio_candidates:
            matched = [c for c in col_names if c.lower() == cand]
            if matched:
                detected_audio_col = matched[0]
                detected_audio_type = "named_audio"
                break

    schema.audio_col = detected_audio_col
    schema.audio_type = detected_audio_type

    if not detected_audio_col:
        schema.is_valid = False
        schema.validation_error = (
            f"No recognizable audio column detected among columns: {col_names}. "
            f"Expected one of {audio_candidates} or an Audio feature type."
        )
        return schema

    # 2. Detect Identifier Column
    id_candidates = ["id", "track_id", "song_id", "title", "name", "filename", "path", "file", "index", "idx"]
    for cand in id_candidates:
        matched = [c for c in col_names if c.lower() == cand]
        if matched and matched[0] != detected_audio_col:
            schema.id_col = matched[0]
            break

    # 3. Detect Label / Source Column
    label_candidates = [
        "label", "source", "origin", "origin_label", "model", "generator",
        "is_ai", "ai_or_human", "class", "target", "category", "tag",
    ]
    for cand in label_candidates:
        matched = [c for c in col_names if c.lower() == cand]
        if matched and matched[0] != detected_audio_col and matched[0] != schema.id_col:
            schema.label_col = matched[0]
            break

    # 4. Detect Genre Column
    genre_candidates = ["genre", "style", "musical_genre", "music_genre", "tags", "category"]
    for cand in genre_candidates:
        matched = [c for c in col_names if c.lower() == cand]
        if matched and matched[0] not in (detected_audio_col, schema.id_col, schema.label_col):
            schema.genre_col = matched[0]
            break

    schema.is_valid = True
    return schema


# ---------------------------------------------------------------------------
# In-Memory Audio Decoding and Feature Extraction
# ---------------------------------------------------------------------------

def decode_audio_record(
    audio_data: Any,
    target_sr: int = SAMPLE_RATE
) -> Tuple[np.ndarray, float, int]:
    """
    Decode an audio record entirely in memory without writing any raw file to disk.

    Supports:
      - dict with 'bytes' containing raw encoded audio (e.g., MP3, WAV, FLAC, OGG)
      - dict with 'array' (numpy array or list) and optional 'sampling_rate'
      - raw bytes or bytearray
      - local file path or URI (fallback)

    Parameters
    ----------
    audio_data : Any
        Audio field extracted from the dataset record.
    target_sr : int, optional
        Target sample rate for resampling, default SAMPLE_RATE (22050 Hz).

    Returns
    -------
    waveform : np.ndarray
        1D float32 mono audio waveform.
    duration_sec : float
        Duration of the decoded audio in seconds.
    byte_count : int
        Measurable bytes of audio data received/processed.

    Raises
    ------
    ValueError
        If the audio data format is unsupported or cannot be decoded.
    """
    if audio_data is None:
        raise ValueError("Audio data record is None.")

    waveform: Optional[np.ndarray] = None
    sr_loaded: int = target_sr
    byte_count: int = 0

    # Case 1: Dictionary containing 'bytes' or 'array'
    if isinstance(audio_data, dict):
        if "bytes" in audio_data and audio_data["bytes"] is not None:
            raw_bytes = audio_data["bytes"]
            byte_count = len(raw_bytes)
            bio = io.BytesIO(raw_bytes)
            try:
                data, sr_loaded = sf.read(bio, dtype="float32")
                if data.ndim > 1:
                    data = np.mean(data, axis=1)
                waveform = data
            except Exception:
                # Fallback to librosa with BytesIO
                bio.seek(0)
                try:
                    data, sr_loaded = librosa.load(bio, sr=None, mono=True)
                    waveform = data
                except Exception as exc:
                    raise ValueError(f"Failed to decode audio bytes: {exc}") from exc

        elif "array" in audio_data and audio_data["array"] is not None:
            arr = np.asarray(audio_data["array"], dtype=np.float32)
            byte_count = arr.nbytes
            if arr.ndim > 1:
                arr = np.mean(arr, axis=1)
            waveform = arr
            sr_loaded = int(audio_data.get("sampling_rate", target_sr))

        elif "path" in audio_data and audio_data["path"]:
            path_str = str(audio_data["path"])
            if os.path.isfile(path_str):
                byte_count = os.path.getsize(path_str)
                waveform, sr_loaded = librosa.load(path_str, sr=None, mono=True)
            else:
                raise ValueError(f"Audio dict specifies non-existent path: {path_str}")

    # Case 2: Direct raw bytes or bytearray
    elif isinstance(audio_data, (bytes, bytearray)):
        byte_count = len(audio_data)
        bio = io.BytesIO(audio_data)
        try:
            data, sr_loaded = sf.read(bio, dtype="float32")
            if data.ndim > 1:
                data = np.mean(data, axis=1)
            waveform = data
        except Exception:
            bio.seek(0)
            try:
                data, sr_loaded = librosa.load(bio, sr=None, mono=True)
                waveform = data
            except Exception as exc:
                raise ValueError(f"Failed to decode raw audio bytes: {exc}") from exc

    # Case 3: Direct numpy array
    elif isinstance(audio_data, np.ndarray):
        byte_count = audio_data.nbytes
        waveform = audio_data.astype(np.float32)
        if waveform.ndim > 1:
            waveform = np.mean(waveform, axis=1)
        sr_loaded = target_sr

    # Case 4: File path string
    elif isinstance(audio_data, str) and os.path.isfile(audio_data):
        byte_count = os.path.getsize(audio_data)
        waveform, sr_loaded = librosa.load(audio_data, sr=None, mono=True)

    if waveform is None or len(waveform) == 0:
        raise ValueError("Decoded audio waveform is empty or invalid.")

    # Resample to target_sr if required
    if sr_loaded != target_sr:
        waveform = librosa.resample(waveform, orig_sr=sr_loaded, target_sr=target_sr)

    duration_sec = float(len(waveform) / target_sr)
    return waveform.astype(np.float32), duration_sec, byte_count


def extract_features_from_waveform(
    waveform: np.ndarray,
    sr: int = SAMPLE_RATE
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Extract raw chroma CQT and beat-synchronized chroma directly from an in-memory waveform.

    Parameters
    ----------
    waveform : np.ndarray
        Mono audio signal at `sr` Hz.
    sr : int, optional
        Sample rate (default 22050 Hz).

    Returns
    -------
    chroma_raw : np.ndarray
        Raw chroma CQT matrix (12, n_frames).
    chroma_beat : np.ndarray
        Beat-synchronized chroma matrix (12, n_beats).
    beat_times : np.ndarray
        Timestamps in seconds for each beat frame.
    duration_sec : float
        Audio duration in seconds.
    """
    if waveform is None or len(waveform) == 0:
        raise ValueError("Waveform is empty.")

    duration_sec = float(len(waveform) / sr)

    # 1. Raw Chroma CQT (with STFT fallback if too short)
    try:
        if len(waveform) < sr * 3:
            n_fft = min(1024, len(waveform))
            chroma_raw = librosa.feature.chroma_stft(y=waveform, sr=sr, n_fft=n_fft)
        else:
            chroma_raw = librosa.feature.chroma_cqt(y=waveform, sr=sr)
    except Exception:
        n_fft = min(1024, len(waveform))
        chroma_raw = librosa.feature.chroma_stft(y=waveform, sr=sr, n_fft=n_fft)

    if chroma_raw.shape[0] != 12 and chroma_raw.shape[1] == 12:
        chroma_raw = chroma_raw.T

    # 2. Beat Tracking and Beat-Synchronization
    try:
        tempo, beat_frames = librosa.beat.beat_track(y=waveform, sr=sr)
    except Exception:
        beat_frames = np.array([], dtype=int)

    if len(beat_frames) >= 2:
        chroma_beat = librosa.util.sync(chroma_raw, beat_frames, aggregate=np.median)
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)
        min_len = min(chroma_beat.shape[1], len(beat_times))
        chroma_beat = chroma_beat[:, :min_len]
        beat_times = beat_times[:min_len]
    else:
        chroma_beat = chroma_raw
        n_frames = chroma_raw.shape[1]
        beat_times = librosa.frames_to_time(np.arange(n_frames), sr=sr)

    return (
        chroma_raw.astype(np.float32),
        chroma_beat.astype(np.float32),
        beat_times.astype(np.float32),
        duration_sec,
    )


# ---------------------------------------------------------------------------
# Metadata-Only Inspection
# ---------------------------------------------------------------------------

def inspect_remote_metadata(
    dataset_name: str = DEFAULT_REMOTE_DATASET,
    split: str = DEFAULT_REMOTE_SPLIT,
    limit: int = 10,
    dataset_override: Any = None,
) -> Dict[str, Any]:
    """
    Inspect dataset schema and metadata records without downloading audio columns.

    Parameters
    ----------
    dataset_name : str
        Hugging Face dataset identifier.
    split : str
        Dataset split (e.g. 'train').
    limit : int
        Maximum number of metadata records to inspect.
    dataset_override : Any, optional
        Pre-instantiated iterable dataset for testing.

    Returns
    -------
    Dict[str, Any]
        Dictionary with detected schema information and stripped metadata records.
    """
    import datasets

    start_time = time.time()

    if dataset_override is not None:
        ds = dataset_override
    else:
        logger.info(f"Loading remote dataset '{dataset_name}' (split='{split}', streaming=True)")
        ds = datasets.load_dataset(dataset_name, split=split, streaming=True)

    # Detect schema from dataset.features if available
    schema_info: Optional[DatasetSchemaInfo] = None
    if hasattr(ds, "features") and ds.features:
        schema_info = detect_dataset_schema(ds.features)

    # If only inspecting schema and no records requested (limit <= 0)
    if limit <= 0 and schema_info and schema_info.is_valid:
        return {
            "dataset_name": dataset_name,
            "split": split,
            "schema": asdict(schema_info),
            "limit": 0,
            "records_retrieved": 0,
            "records": [],
            "elapsed_sec": round(time.time() - start_time, 3),
        }

    # If schema is known and has an audio column, select metadata columns only
    if schema_info and schema_info.audio_col and hasattr(ds, "select_columns"):
        meta_cols = [c for c in schema_info.all_columns if c != schema_info.audio_col]
        if meta_cols:
            try:
                ds = ds.select_columns(meta_cols)
            except Exception as exc:
                logger.debug(f"select_columns failed: {exc}")

    # Create iterator
    record_iter = iter(ds)
    sample_record = None

    if schema_info is None or not schema_info.is_valid:
        try:
            sample_record = next(record_iter)
            schema_info = detect_dataset_schema(sample_record)
        except StopIteration:
            return {
                "dataset_name": dataset_name,
                "split": split,
                "schema": asdict(DatasetSchemaInfo(is_valid=False, validation_error="Dataset is empty")),
                "records": [],
                "elapsed_sec": round(time.time() - start_time, 3),
            }

    # Extract metadata records, deliberately excluding audio column
    records: List[Dict[str, Any]] = []
    audio_col = schema_info.audio_col

    def _clean_record(rec: Dict[str, Any], idx: int) -> Dict[str, Any]:
        cleaned = {}
        for k, v in rec.items():
            if k == audio_col:
                continue
            # Keep serializable primitives
            if isinstance(v, (str, int, float, bool)) or v is None:
                cleaned[k] = v
            elif isinstance(v, list) and all(isinstance(x, (str, int, float, bool)) for x in v):
                cleaned[k] = v
            else:
                cleaned[k] = str(v)

        # Normalize standard fields
        prov_id = rec.get(schema_info.id_col) if schema_info.id_col else None
        track_id = generate_stable_track_id(rec.get(schema_info.audio_col), provenance_id=str(prov_id) if prov_id else f"rec_{idx}")

        if "ai_generated" in rec:
            val = rec["ai_generated"]
            if isinstance(val, bool):
                origin_label = "ai_generated" if val else "human"
            elif str(val).lower() in ("false", "0", "no"):
                origin_label = "human"
            else:
                origin_label = "ai_generated"
        else:
            raw = str(rec.get(schema_info.label_col or "label", "unknown")).lower()
            origin_label = "human" if "human" in raw else ("ai_generated" if "ai" in raw else raw)

        raw_genre = rec.get(schema_info.genre_col) if schema_info.genre_col else None
        genre = str(raw_genre).strip() if raw_genre and str(raw_genre).strip() and str(raw_genre).strip().lower() not in ("none", "ambient", "unknown") else "Unknown"

        cleaned["_normalized_track_id"] = track_id
        cleaned["_normalized_origin"] = origin_label
        cleaned["_normalized_genre"] = genre
        return cleaned

    if sample_record is not None:
        records.append(_clean_record(sample_record, 0))

    while len(records) < limit:
        try:
            rec = next(record_iter)
            records.append(_clean_record(rec, len(records)))
        except StopIteration:
            break

    return {
        "dataset_name": dataset_name,
        "split": split,
        "schema": asdict(schema_info),
        "limit": limit,
        "records_retrieved": len(records),
        "records": records,
        "elapsed_sec": round(time.time() - start_time, 3),
    }


# ---------------------------------------------------------------------------
# Incremental Dataset Indexing and Compact Cache Management
# ---------------------------------------------------------------------------

def load_remote_index(index_file: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load existing precomputed remote index from disk.
    Normalizes legacy origin labels and ensures presence of status metadata.
    """
    path = Path(index_file) if index_file is not None else REMOTE_INDEX_FILE
    if not path.is_file():
        return {
            "version": "1.0",
            "dataset_name": DEFAULT_REMOTE_DATASET,
            "split": DEFAULT_REMOTE_SPLIT,
            "created_at": None,
            "updated_at": None,
            "total_indexed": 0,
            "records_scanned": 0,
            "duplicate_records": 0,
            "scan_status": "NOT_STARTED",
            "is_complete": False,
            "unresolved_records": {},
            "tracks": {},
        }
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            data.setdefault("is_complete", False)
            data.setdefault("unresolved_records", {})
            tracks = data.get("tracks", {})
            unresolved = data.get("unresolved_records", {})
            if data.get("is_complete"):
                def_status = "COMPLETED" if len(unresolved) == 0 else "COMPLETED_WITH_EXCLUSIONS"
            else:
                def_status = "PAUSED" if len(tracks) > 0 else "NOT_STARTED"
            data.setdefault("scan_status", def_status)
            data.setdefault("records_scanned", len(tracks) + len(unresolved))
            data.setdefault("duplicate_records", 0)

            # Clean and normalize any legacy origin labels where raw "False" was saved as "ai_generated"
            for tid, tinfo in tracks.items():
                raw_meta = tinfo.get("raw_metadata", {})
                if "ai_generated" in raw_meta and tinfo.get("origin_label") == "ai_generated":
                    normalized = parse_origin_label(raw_meta["ai_generated"])
                    if normalized != "ai_generated":
                        tinfo["origin_label"] = normalized

            data["tracks"] = tracks
            data["total_indexed"] = len(tracks)
            return data
    except Exception as exc:
        logger.warning(f"Failed to read remote index file {path}: {exc}. Starting fresh.")
        return {
            "version": "1.0",
            "dataset_name": DEFAULT_REMOTE_DATASET,
            "split": DEFAULT_REMOTE_SPLIT,
            "created_at": None,
            "updated_at": None,
            "total_indexed": 0,
            "records_scanned": 0,
            "duplicate_records": 0,
            "scan_status": "Not Started",
            "is_complete": False,
            "unresolved_records": {},
            "tracks": {},
        }


def parse_origin_label(raw_val: Any) -> str:
    """
    Authoritative, strict origin-label parser delegating to validation layer.
    """
    from src.validation import validate_origin_label
    return validate_origin_label(raw_val).normalized_label


def generate_stable_track_id(
    audio_val: Any,
    waveform: Optional[np.ndarray] = None,
    existing_ids: Optional[Union[set, Dict[str, Any]]] = None,
    provenance_id: Optional[str] = None,
) -> str:
    """
    Derive a stable, content-derived track identifier following Authoritative Decision 3.

    Format:
        track_<12_hex_characters>

    Derivation:
        - When actual encoded audio bytes are available:
          hashlib.sha256(audio_bytes[:4096]).hexdigest()[:12]
        - If only a decoded waveform is available:
          hashlib.sha256(np.ascontiguousarray(waveform[:4096]).tobytes()).hexdigest()[:12]
        - Where a stable original source identifier is available, retain it separately
          as provenance metadata in the record.

    Collision handling:
        This is a truncated prefix hash, not a mathematically collision-free identifier.
        Collision detection checks against existing_ids and extends the hex prefix if
        an identical prefix is already used by an unrelated record.
    """
    import hashlib

    data_bytes: Optional[bytes] = None

    if isinstance(audio_val, (bytes, bytearray)):
        data_bytes = bytes(audio_val)
    elif isinstance(audio_val, dict):
        if "bytes" in audio_val and audio_val["bytes"]:
            raw = audio_val["bytes"]
            data_bytes = bytes(raw) if isinstance(raw, (bytes, bytearray)) else bytes(str(raw), "utf-8")
        elif "array" in audio_val and audio_val["array"] is not None:
            arr = np.asarray(audio_val["array"], dtype=np.float32)
            data_bytes = np.ascontiguousarray(arr).tobytes()
    elif isinstance(audio_val, np.ndarray):
        data_bytes = np.ascontiguousarray(audio_val.astype(np.float32)).tobytes()

    if data_bytes is None and waveform is not None and len(waveform) > 0:
        data_bytes = np.ascontiguousarray(waveform.astype(np.float32)).tobytes()

    if data_bytes and len(data_bytes) > 0:
        full_hash = hashlib.sha256(data_bytes).hexdigest()
    elif provenance_id:
        full_hash = hashlib.sha256(str(provenance_id).encode("utf-8")).hexdigest()
    else:
        full_hash = hashlib.sha256(b"unknown_audio_content").hexdigest()

    prefix_len = 12
    cand_id = f"track_{full_hash[:prefix_len]}"

    if isinstance(existing_ids, dict):
        # If cand_id exists, check whether it is the same track (re-indexing / cache check)
        # or a true hash collision with an unrelated track
        if cand_id in existing_ids:
            existing_rec = existing_ids[cand_id]
            # Match by provenance ID or content hash -> same track, return identical ID
            if provenance_id is not None:
                existing_src = existing_rec.get("source_id") or existing_rec.get("raw_metadata", {}).get("original_source_id")
                if existing_src == str(provenance_id):
                    return cand_id
            if existing_rec.get("content_hash") == full_hash:
                return cand_id

            # True collision with unrelated track: extend hex prefix
            step = 2
            while cand_id in existing_ids and prefix_len < len(full_hash):
                prefix_len += step
                cand_id = f"track_{full_hash[:prefix_len]}"
    elif isinstance(existing_ids, set):
        # Set without metadata: only extend if cand_id in set
        pass

    return cand_id


def save_remote_index(index_data: Dict[str, Any], index_file: Optional[Path] = None) -> None:
    """
    Save remote index atomically to disk in JSON format.

    Uses a unique temporary file and atomic filesystem replace so an interrupted
    operation does not corrupt the target JSON.
    """
    path = Path(index_file) if index_file is not None else REMOTE_INDEX_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f"{path.stem}_{os.getpid()}_{int(time.time() * 1000)}.tmp")
    try:
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(index_data, f, indent=2)
        os.replace(temp_path, path)
    except Exception as exc:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except Exception:
                pass
        raise exc


def index_remote_dataset(
    dataset_name: str = DEFAULT_REMOTE_DATASET,
    split: str = DEFAULT_REMOTE_SPLIT,
    sample_size: int = DEFAULT_REMOTE_SAMPLE_SIZE,
    is_full: bool = False,
    force_reindex: bool = False,
    index_file: Optional[Path] = None,
    features_cache_dir: Optional[Path] = None,
    cache_detailed: bool = True,
    dataset_override: Any = None,
    progress_callback: Optional[Callable[[int, int, Dict[str, Any]], None]] = None,
    cancel_event: Optional[Any] = None,
) -> IndexingReport:
    """
    Incrementally stream and index audio records from Hugging Face.

    Safeties & Constraints:
      - By default processes only `sample_size` records (default 10) for development.
      - Requires `is_full=True` for a full-corpus pass.
      - Decodes embedded audio strictly in memory without writing raw files to disk.
      - Precomputes and persists compact 12-dimensional mean-chroma vectors in `remote_index.json`.
      - Caches detailed beat-synchronized features under `features/` separately.
      - Accurately tracks downloaded audio bytes, processed records, failed records, and cache hits.

    Parameters
    ----------
    dataset_name : str
        Hugging Face dataset identifier.
    split : str
        Split name ('train', etc.).
    sample_size : int
        Number of records to process when is_full=False (default: 10).
    is_full : bool
        If True, processes all records in the split. If False, processes up to sample_size.
    force_reindex : bool
        If True, recomputes features even if the track is already in the local index.
    index_file : Optional[Path]
        Target JSON index file path (default: FEATURES_DIR / 'remote_index.json').
    features_cache_dir : Optional[Path]
        Directory for storing detailed candidate beat matrices (default: FEATURES_DIR).
    cache_detailed : bool
        Whether to cache beat-synchronous chroma matrices for Stage 2 DTW.
    dataset_override : Any, optional
        Pre-instantiated iterable dataset (used for unit testing with mocked datasets).
    progress_callback : Optional[Callable]
        Callback invoked after each processed item: (current_count, max_count, current_track_info).
    cancel_event : Optional[Any]
        Optional threading.Event to pause or cancel indexing safely.

    Returns
    -------
    IndexingReport
        Comprehensive metrics report detailing execution, transferred bytes, and errors.
    """
    import datasets

    start_time = time.time()
    idx_path = Path(index_file) if index_file is not None else REMOTE_INDEX_FILE
    feat_dir = Path(features_cache_dir) if features_cache_dir is not None else FEATURES_DIR
    feat_dir.mkdir(parents=True, exist_ok=True)

    # Load existing index
    index_data = load_remote_index(idx_path)
    existing_tracks = index_data.get("tracks", {})

    # Load streaming dataset
    if dataset_override is not None:
        ds = dataset_override
    else:
        logger.info(f"Connecting to remote dataset: {dataset_name} (split={split}, streaming=True)")
        ds = datasets.load_dataset(dataset_name, split=split, streaming=True)

    # Detect Schema
    schema: Optional[DatasetSchemaInfo] = None
    if hasattr(ds, "features") and ds.features:
        schema = detect_dataset_schema(ds.features)
        if schema.is_valid and schema.audio_col and hasattr(ds, "cast_column"):
            try:
                ds = ds.cast_column(schema.audio_col, datasets.Audio(decode=False))
            except Exception as exc:
                logger.debug(f"cast_column(decode=False) failed: {exc}")

    # Initialize metrics
    processed_records = 0
    failed_records = 0
    cache_hits = 0
    duplicate_records = 0
    downloaded_bytes = 0
    errors: List[str] = []
    stream_exhausted = False

    limit = None if is_full else max(1, int(sample_size))
    record_iter = iter(ds)

    item_idx = 0
    while True:
        if cancel_event is not None and getattr(cancel_event, "is_set", lambda: False)():
            logger.info("Indexing cancelled by user/task signal.")
            break

        if limit is not None and item_idx >= limit:
            break

        try:
            record = next(record_iter)
        except StopIteration:
            stream_exhausted = True
            break
        except Exception as exc:
            failed_records += 1
            errors.append(f"Failed to stream record #{item_idx}: {exc}")
            item_idx += 1
            continue

        # Validate schema on first record if not already established
        if schema is None or not schema.is_valid:
            schema = detect_dataset_schema(record)
            if not schema.is_valid:
                failed_records += 1
                errors.append(f"Invalid record schema at #{item_idx}: {schema.validation_error}")
                item_idx += 1
                break

        # Extract audio payload first to generate content-derived stable ID
        audio_val = record.get(schema.audio_col)
        if audio_val is None:
            failed_records += 1
            errors.append(f"Record #{item_idx} missing audio field '{schema.audio_col}'")
            item_idx += 1
            continue

        raw_id = record.get(schema.id_col) if schema.id_col else None
        provenance_id = str(raw_id).strip() if raw_id is not None and str(raw_id).strip() else None

        # Content-derived ID: track_<12_hex_characters>
        track_id = generate_stable_track_id(
            audio_val,
            existing_ids=existing_tracks,
            provenance_id=provenance_id,
        )

        raw_origin_val = (
            record.get("ai_generated")
            if "ai_generated" in record
            else record.get(schema.label_col or "label")
        )
        origin_label = parse_origin_label(raw_origin_val)

        # Detailed cache file paths
        beat_cache_file = feat_dir / f"{track_id}_beat.npy"
        times_cache_file = feat_dir / f"{track_id}_times.npy"

        # Safe Incremental Indexing: Check Cache Hit and skip already indexed tracks
        if (
            not force_reindex
            and track_id in existing_tracks
            and beat_cache_file.is_file()
            and times_cache_file.is_file()
        ):
            cache_hits += 1
            item_idx += 1
            if progress_callback:
                prog_info = dict(existing_tracks[track_id])
                prog_info.update({
                    "records_scanned": item_idx,
                    "cache_hits": cache_hits,
                    "processed_records": processed_records,
                    "failed_records": failed_records,
                    "unresolved_count": len(index_data.get("unresolved_records", {})),
                    "downloaded_bytes": downloaded_bytes,
                    "elapsed_sec": round(time.time() - start_time, 2),
                    "current_operation": f"Cache hit: {track_id} (already indexed)",
                })
                try:
                    progress_callback(item_idx, limit or -1, prog_info)
                except TypeError:
                    progress_callback(prog_info)
            continue

        try:
            waveform, duration_sec, byte_count = decode_audio_record(audio_val, target_sr=SAMPLE_RATE)
            downloaded_bytes += byte_count

            # Extract features from in-memory waveform
            chroma_raw, chroma_beat, beat_times, _ = extract_features_from_waveform(
                waveform, sr=SAMPLE_RATE
            )

            # Compute compact 12-dimensional mean chroma vector
            mean_chroma = compute_mean_chroma_vector(chroma_beat)

            # Resolve genre via verified metadata or genuine model estimate; never default to ambient
            raw_genre = record.get(schema.genre_col) if schema.genre_col else None
            from src.genre import resolve_track_genre
            genre_info = resolve_track_genre(raw_genre, waveform=waveform, sr=SAMPLE_RATE)

            # Persist detailed features separately for Stage 2 DTW if requested
            if cache_detailed:
                np.save(str(beat_cache_file), chroma_beat)
                np.save(str(times_cache_file), beat_times)

            # Build compact track descriptor with provenance metadata
            provenance_meta = {
                k: str(v)
                for k, v in record.items()
                if k != schema.audio_col and isinstance(v, (str, int, float, bool))
            }
            if provenance_id:
                provenance_meta["original_source_id"] = provenance_id

            track_meta = {
                "track_id": track_id,
                "source_id": provenance_id,
                "origin_label": origin_label,
                "genre": genre_info["genre"],
                "estimated_genre": genre_info["estimated_genre"],
                "genre_confidence": genre_info["genre_confidence"],
                "genre_source": genre_info["genre_source"],
                "mean_chroma": [round(float(v), 5) for v in mean_chroma],
                "duration_sec": round(duration_sec, 2),
                "beat_cache": str(beat_cache_file.relative_to(idx_path.parent.parent)),
                "times_cache": str(times_cache_file.relative_to(idx_path.parent.parent)),
                "raw_metadata": provenance_meta,
            }

            existing_tracks[track_id] = track_meta
            processed_records += 1

            # Remove from unresolved records if previously recorded
            unresolved = index_data.get("unresolved_records", {})
            if track_id in unresolved:
                unresolved.pop(track_id, None)

            if progress_callback:
                prog_info = dict(track_meta)
                prog_info.update({
                    "records_scanned": item_idx + 1,
                    "processed_records": processed_records,
                    "cache_hits": cache_hits,
                    "failed_records": failed_records,
                    "unresolved_count": len(index_data.get("unresolved_records", {})),
                    "downloaded_bytes": downloaded_bytes,
                    "elapsed_sec": round(time.time() - start_time, 2),
                    "current_operation": f"Indexed {track_id} (Genre: {track_meta.get('genre_source')}, Origin: {origin_label})",
                })
                try:
                    progress_callback(item_idx + 1, limit or -1, prog_info)
                except TypeError:
                    progress_callback(prog_info)

        except Exception as exc:
            failed_records += 1
            err_msg = f"Record #{item_idx} ({track_id}) audio feature extraction failed: {exc}"
            errors.append(err_msg)
            unresolved = index_data.setdefault("unresolved_records", {})
            unresolved[track_id] = {
                "index": item_idx,
                "track_id": track_id,
                "provenance_id": provenance_id,
                "error": str(exc),
                "recoverable": True,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }

        item_idx += 1

        # Periodic atomic save every 10 processed items to persist progress
        if processed_records > 0 and processed_records % 10 == 0:
            index_data["total_indexed"] = len(existing_tracks)
            index_data["total_scanned"] = item_idx
            index_data["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            index_data["tracks"] = existing_tracks
            save_remote_index(index_data, idx_path)

    # Invalidate null distribution cache only if real production index was modified
    if processed_records > 0 and Path(idx_path).resolve() == Path(REMOTE_INDEX_FILE).resolve():
        from src.stats_test import invalidate_null_distribution_cache
        invalidate_null_distribution_cache()

    # Determine complete status
    unresolved_count = len(index_data.get("unresolved_records", {}))
    if is_full and stream_exhausted:
        if unresolved_count == 0:
            scan_status = "COMPLETED"
            is_complete = True
        else:
            scan_status = "COMPLETED_WITH_EXCLUSIONS"
            is_complete = True
    else:
        scan_status = "PAUSED" if len(existing_tracks) > 0 else "NOT_STARTED"
        is_complete = False

    # Update and write persistent index
    index_data["dataset_name"] = dataset_name
    index_data["split"] = split
    index_data["total_indexed"] = len(existing_tracks)
    index_data["total_scanned"] = item_idx
    index_data["records_scanned"] = item_idx
    index_data["duplicate_records"] = duplicate_records
    index_data["scan_status"] = scan_status
    index_data["is_complete"] = is_complete
    index_data["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    if not index_data.get("created_at"):
        index_data["created_at"] = index_data["updated_at"]
    index_data["tracks"] = existing_tracks

    save_remote_index(index_data, idx_path)

    elapsed_sec = round(time.time() - start_time, 2)
    report = IndexingReport(
        dataset_name=dataset_name,
        split=split,
        is_full_pass=is_full,
        requested_sample_size=sample_size if not is_full else None,
        processed_records=processed_records,
        failed_records=failed_records,
        cache_hits=cache_hits,
        downloaded_bytes=downloaded_bytes,
        elapsed_sec=elapsed_sec,
        index_file=str(idx_path),
        errors=errors[:20],  # cap error strings
        records_scanned=item_idx,
        duplicate_records=duplicate_records,
        unresolved_records_count=unresolved_count,
        is_complete=is_complete,
        status=scan_status,
    )

    return report


# ---------------------------------------------------------------------------
# Two-Stage Query Search Against Local Remote Index
# ---------------------------------------------------------------------------

def filter_remote_candidates(
    query_chroma: np.ndarray,
    index_file: Optional[Path] = None,
    threshold: float = COSINE_PREFILTER_THRESHOLD,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Stage 1: Pre-filter candidate tracks from local index using cosine similarity.

    Avoids rescanning or connecting to the remote dataset during query time.
    Uses precomputed 12-dimensional mean chroma vectors stored in `remote_index.json`.

    Parameters
    ----------
    query_chroma : np.ndarray
        Query chroma matrix (12, n_frames).
    index_file : Optional[Path]
        Path to local remote_index.json.
    threshold : float, optional
        Cosine similarity cutoff, default COSINE_PREFILTER_THRESHOLD (0.30).

    Returns
    -------
    shortlisted : List[Dict[str, Any]]
        List of surviving candidate track dictionaries with added 'cosine_similarity'.
    stats : Dict[str, Any]
        Filtering statistics (before count, after count, percentage removed).
    """
    index_data = load_remote_index(index_file)
    tracks = index_data.get("tracks", {})
    before_count = len(tracks)

    if before_count == 0:
        return [], {"before_count": 0, "after_count": 0, "percentage_removed": 0.0}

    query_mean = compute_mean_chroma_vector(query_chroma)
    shortlisted: List[Dict[str, Any]] = []

    for track_id, info in tracks.items():
        cand_mean = np.array(info.get("mean_chroma", []), dtype=float)
        cos_sim = cosine_similarity_vectors(query_mean, cand_mean)
        if cos_sim >= threshold:
            item = dict(info)
            item["cosine_similarity"] = round(float(cos_sim), 4)
            shortlisted.append(item)

    after_count = len(shortlisted)
    percentage_removed = (
        round(((before_count - after_count) / before_count) * 100.0, 2)
        if before_count > 0
        else 0.0
    )

    stats = {
        "before_count": before_count,
        "after_count": after_count,
        "percentage_removed": percentage_removed,
    }

    # Sort surviving candidates descending by cosine similarity
    shortlisted.sort(key=lambda x: x["cosine_similarity"], reverse=True)
    return shortlisted, stats


def align_shortlisted_remote_candidate(
    query_chroma_beat: np.ndarray,
    candidate_info: Dict[str, Any],
    repo_root: Optional[Path] = None,
) -> Optional[Tuple[DTWResult, np.ndarray]]:
    """
    Stage 2: Perform detailed key-invariant DTW alignment on a shortlisted candidate.

    Loads the cached beat-synchronous representation from disk without re-extracting
    or streaming raw audio.

    Parameters
    ----------
    query_chroma_beat : np.ndarray
        Beat-synchronized chroma matrix for query track.
    candidate_info : Dict[str, Any]
        Candidate dictionary containing 'beat_cache' or 'track_id'.
    repo_root : Optional[Path]
        Repository root directory to resolve relative cache paths.

    Returns
    -------
    Tuple[DTWResult, np.ndarray] or None
        Alignment result and candidate beat times array, or None if cache is missing.
    """
    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parent.parent
    beat_rel = candidate_info.get("beat_cache")
    times_rel = candidate_info.get("times_cache")

    if not beat_rel:
        track_id = candidate_info.get("track_id", "")
        beat_path = FEATURES_DIR / f"{track_id}_beat.npy"
        times_path = FEATURES_DIR / f"{track_id}_times.npy"
    else:
        beat_path = root / beat_rel
        times_path = (root / times_rel) if times_rel else (beat_path.with_name(beat_path.name.replace("_beat.npy", "_times.npy")))

    if not beat_path.is_file():
        logger.warning(f"Detailed beat cache missing for candidate {candidate_info.get('track_id')}: {beat_path}")
        return None

    try:
        cand_beat = np.load(str(beat_path))
        cand_times = np.load(str(times_path)) if times_path.is_file() else np.array([])
        dtw_res = dtw_align_detailed(query_chroma_beat, cand_beat)
        return dtw_res, cand_times
    except Exception as exc:
        logger.error(f"DTW alignment failed for remote candidate {candidate_info.get('track_id')}: {exc}")
        return None


# ---------------------------------------------------------------------------
# CLI Command Interface
# ---------------------------------------------------------------------------

def build_cli_parser() -> argparse.ArgumentParser:
    """
    Construct command line arguments parser for remote dataset operations.
    """
    parser = argparse.ArgumentParser(
        description="Stream and index Hugging Face audio datasets incrementally without full download."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=DEFAULT_REMOTE_DATASET,
        help=f"Hugging Face dataset identifier (default: {DEFAULT_REMOTE_DATASET})",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=DEFAULT_REMOTE_SPLIT,
        help=f"Dataset split (default: {DEFAULT_REMOTE_SPLIT})",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=DEFAULT_REMOTE_SAMPLE_SIZE,
        help=f"Number of records to sample for development (default: {DEFAULT_REMOTE_SAMPLE_SIZE})",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Perform an explicit full-corpus pass. If omitted, strictly defaults to --sample.",
    )
    parser.add_argument(
        "--inspect-schema",
        action="store_true",
        help="Inspect and print detected dataset column schema only.",
    )
    parser.add_argument(
        "--inspect-metadata",
        action="store_true",
        help="Inspect and print metadata records without reading audio payloads.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-indexing and overwrite existing local feature cache.",
    )
    parser.add_argument(
        "--retry-unresolved",
        action="store_true",
        help="Retry previously failed or unresolved audio records in the index.",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Display current index statistics and exit without indexing.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results formatted as JSON.",
    )
    return parser


def main() -> None:
    """
    CLI entry point.
    """
    parser = build_cli_parser()
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    if args.stats:
        idx = load_remote_index()
        tracks = idx.get("tracks", {})
        unresolved = idx.get("unresolved_records", {})
        human = sum(1 for t in tracks.values() if t.get("origin_label") == "human")
        ai = sum(1 for t in tracks.values() if t.get("origin_label") == "ai_generated")
        unknown_orig = sum(1 for t in tracks.values() if t.get("origin_label") == "unknown")
        meta_genre = sum(1 for t in tracks.values() if t.get("genre_source") == "metadata")
        est_genre = sum(1 for t in tracks.values() if t.get("genre_source") == "estimated")
        unknown_genre = sum(1 for t in tracks.values() if t.get("genre_source") == "unknown")

        print("\n================== CURRENT INDEX METRICS ==================")
        print(f"  Dataset:             {idx.get('dataset_name')} (split: {idx.get('split')})")
        print(f"  Indexed Tracks:      {len(tracks)}")
        print(f"  Index Complete:      {idx.get('is_complete', False)}")
        print(f"  Unresolved Failures: {len(unresolved)}")
        print(f"  Origin Breakdown:    Human: {human} | AI: {ai} | Unknown: {unknown_orig}")
        print(f"  Genre Breakdown:     Verified: {meta_genre} | Estimated: {est_genre} | Unknown: {unknown_genre}")
        print(f"  Updated At:          {idx.get('updated_at')}")
        print("===========================================================\n")
        return

    if args.inspect_schema or args.inspect_metadata:
        schema_only = args.inspect_schema and not args.inspect_metadata
        limit = 0 if schema_only else args.sample
        print(f"\n--- Inspecting Remote Dataset Metadata [{args.dataset}] ---")
        meta_result = inspect_remote_metadata(
            dataset_name=args.dataset,
            split=args.split,
            limit=limit,
        )
        if args.json:
            print(json.dumps(meta_result, indent=2))
        else:
            schema = meta_result["schema"]
            print(f"Dataset: {meta_result['dataset_name']} (split: {meta_result['split']})")
            print(f"Valid Schema: {schema['is_valid']}")
            print(f"Audio Column: {schema['audio_col']} (Type: {schema['audio_type']})")
            print(f"ID Column: {schema['id_col']}")
            print(f"Label Column: {schema['label_col']}")
            print(f"Genre Column: {schema['genre_col']}")
            print(f"All Columns: {', '.join(schema['all_columns'])}")
            print(f"Sample Records Inspected: {meta_result['records_retrieved']}")
            if meta_result["records"]:
                print(f"\nFirst record preview:")
                first = meta_result["records"][0]
                for k, v in first.items():
                    print(f"  {k}: {v}")
        return

    # Indexing Mode
    is_full_pass = bool(args.full)
    target_count = "FULL CORPUS" if is_full_pass else f"{args.sample} records"
    print(f"\n=================================================================")
    print(f"  Incremental Remote Dataset Indexing: {args.dataset}")
    print(f"  Pass Mode: {target_count} (Split: {args.split})")
    print(f"  Audio decoded in-memory; raw dataset is NOT saved to disk.")
    print(f"=================================================================\n")

    def _progress(cur: int, total: int, info: Dict[str, Any]) -> None:
        total_str = str(total) if total > 0 else "unknown"
        mb = round(info.get("downloaded_bytes", 0) / (1024 * 1024), 2)
        op = info.get("current_operation", "processing")
        print(
            f"  -> [{cur}/{total_str}] {op} | Cache Hits: {info.get('cache_hits', 0)} | "
            f"Failures: {info.get('failed_records', 0)} | {mb} MB | Elapsed: {info.get('elapsed_sec', 0.0)}s"
        )

    report = index_remote_dataset(
        dataset_name=args.dataset,
        split=args.split,
        sample_size=args.sample,
        is_full=is_full_pass,
        force_reindex=args.force,
        progress_callback=_progress,
    )

    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print("\n=================== INDEXING SUMMARY REPORT ===================")
        print(f"  Dataset:            {report.dataset_name} ({report.split})")
        print(f"  Full Pass:          {report.is_full_pass}")
        print(f"  Processed Records:  {report.processed_records}")
        print(f"  Failed Records:     {report.failed_records}")
        print(f"  Cache Hits:         {report.cache_hits}")
        mb_downloaded = round(report.downloaded_bytes / (1024 * 1024), 2)
        print(f"  Downloaded Bytes:   {report.downloaded_bytes} bytes ({mb_downloaded} MB)")
        print(f"  Elapsed Time:       {report.elapsed_sec} sec")
        print(f"  Index File:         {report.index_file}")
        if report.errors:
            print(f"  Errors Encountered: {len(report.errors)}")
            for err in report.errors[:5]:
                print(f"    - {err}")
        print(f"\nNote: {report.disclaimer}")
        print("===============================================================\n")


if __name__ == "__main__":
    main()
