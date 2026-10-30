"""
Data Cleaning, Validation, and Normalization Layer.

Provides centralized validation for audio payloads, origin metadata,
genre labels, artist provenance, and index migration.
Enforces non-destructive normalization and preserves raw metadata.
"""

import hashlib
import json
import logging
import os
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Validation Result Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class AudioValidationResult:
    """Outcome of audio waveform and file structure validation."""
    is_valid: bool
    duration_sec: float = 0.0
    sample_rate: int = 0
    num_channels: int = 1
    num_samples: int = 0
    error_message: Optional[str] = None
    warning_message: Optional[str] = None


@dataclass
class OriginValidationResult:
    """Outcome of origin label resolution and normalization."""
    normalized_label: str  # 'human', 'ai_generated', 'unknown'
    raw_value: Any
    confidence: float = 1.0
    provenance_field: Optional[str] = None
    is_contradictory: bool = False
    notes: Optional[str] = None


@dataclass
class GenreValidationResult:
    """Outcome of genre resolution with strict provenance tracking."""
    normalized_genre: Optional[str]  # None if unavailable
    genre_source: str  # 'metadata', 'estimated', 'unknown'
    raw_value: Any = None
    confidence: Optional[float] = None
    notes: Optional[str] = None


@dataclass
class FeatureValidationResult:
    """Outcome of numerical matrix validation for pipeline safety."""
    is_valid: bool
    expected_dim: int
    actual_shape: Tuple[int, ...]
    has_nans: bool = False
    has_infs: bool = False
    error_message: Optional[str] = None


@dataclass
class IndexMigrationReport:
    """Audit metrics for safe index migration."""
    total_records: int
    existing_human_count: int
    existing_ai_count: int
    existing_unknown_count: int
    corrected_human_count: int
    corrected_ai_count: int
    corrected_unknown_count: int
    contradictory_records: List[str] = field(default_factory=list)
    backup_file_path: Optional[str] = None
    migration_timestamp: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# 1. Origin Label Normalization and Validation
# ---------------------------------------------------------------------------

def validate_origin_label(
    raw_value: Any,
    field_name: Optional[str] = None,
    additional_metadata: Optional[Dict[str, Any]] = None,
) -> OriginValidationResult:
    """
    Validate and normalize origin metadata into ('human', 'ai_generated', 'unknown').

    Strict Rules:
    - Never automatically convert missing or ambiguous metadata into 'ai_generated'.
    - Unknown must remain 'unknown'.
    - Boolean True/False mapped according to confirmed field semantics.
    - Preserves raw original value and flags contradictory metadata.
    """
    meta = additional_metadata or {}
    field_lower = (field_name or "origin").lower()

    if raw_value is None:
        return OriginValidationResult(
            normalized_label="unknown",
            raw_value=raw_value,
            provenance_field=field_name,
            notes="Missing origin metadata (value is None)",
        )

    # Boolean types
    if isinstance(raw_value, (bool, np.bool_)):
        norm = "ai_generated" if raw_value else "human"
        return OriginValidationResult(
            normalized_label=norm,
            raw_value=raw_value,
            provenance_field=field_name,
            notes=f"Boolean resolved with field context '{field_name}': {norm}",
        )

    # String representations
    val_str = str(raw_value).strip().lower()

    if val_str in ("ai", "ai_generated", "synthetic", "model_generated", "suno", "udio", "musiclm"):
        return OriginValidationResult(
            normalized_label="ai_generated",
            raw_value=raw_value,
            provenance_field=field_name,
            notes="Explicit AI origin label",
        )

    if val_str in ("human", "acoustic", "real", "organic", "fma", "fma_full", "jamendo", "gtzan"):
        return OriginValidationResult(
            normalized_label="human",
            raw_value=raw_value,
            provenance_field=field_name,
            notes="Explicit Human acoustic origin label",
        )

    # Boolean strings: "true", "false", "1", "0", "yes", "no"
    if val_str in ("false", "0", "no"):
        return OriginValidationResult(
            normalized_label="human",
            raw_value=raw_value,
            provenance_field=field_name,
            notes="Negated AI flag resolves to human acoustic baseline",
        )

    if val_str in ("true", "1", "yes"):
        # Check for contradictory metadata (e.g. source is Free Music Archive)
        source_url = str(meta.get("source", "")).lower()
        if "free-music-archive" in source_url or "fma" in source_url:
            return OriginValidationResult(
                normalized_label="unknown",
                raw_value=raw_value,
                provenance_field=field_name,
                is_contradictory=True,
                notes="Contradiction: ai_generated=True but source is Free Music Archive",
            )
        return OriginValidationResult(
            normalized_label="ai_generated",
            raw_value=raw_value,
            provenance_field=field_name,
            notes="Confirmed AI generation flag",
        )

    # Missing, null, unknown representations
    if val_str in ("", "none", "null", "nan", "unknown", "unlabeled", "indeterminate", "?"):
        return OriginValidationResult(
            normalized_label="unknown",
            raw_value=raw_value,
            provenance_field=field_name,
            notes="Missing or uninformative origin label",
        )

    # Fallback: ambiguous strings must remain unknown
    return OriginValidationResult(
        normalized_label="unknown",
        raw_value=raw_value,
        provenance_field=field_name,
        notes=f"Unrecognized origin label format: '{val_str}'",
    )


# ---------------------------------------------------------------------------
# 2. Audio Payload and Waveform Validation
# ---------------------------------------------------------------------------

def validate_audio_waveform(
    waveform: np.ndarray,
    sr: int = 22050,
    min_duration_sec: float = 3.0,
    max_duration_sec: float = 7200.0,
) -> AudioValidationResult:
    """
    Validate in-memory audio waveform for pipeline processing.

    Detects:
    - Empty or zero-length audio.
    - Non-finite (NaN, Inf) samples.
    - Extremely short (< 3.0s) or anomalously long (> 2 hours) tracks.
    - Silent / constant signals.
    """
    if waveform is None or len(waveform) == 0:
        return AudioValidationResult(
            is_valid=False,
            error_message="Audio waveform is empty (zero samples).",
        )

    if not np.all(np.isfinite(waveform)):
        return AudioValidationResult(
            is_valid=False,
            error_message="Audio waveform contains non-finite values (NaN or Inf).",
        )

    num_samples = len(waveform)
    duration_sec = float(num_samples / max(sr, 1))

    if duration_sec < min_duration_sec:
        return AudioValidationResult(
            is_valid=False,
            duration_sec=duration_sec,
            sample_rate=sr,
            num_samples=num_samples,
            error_message=(
                f"Audio duration ({duration_sec:.2f}s) is below research minimum threshold "
                f"of {min_duration_sec:.1f} seconds."
            ),
        )

    warning: Optional[str] = None
    if duration_sec > max_duration_sec:
        warning = f"Audio duration ({duration_sec:.1f}s) is anomalously long (> {max_duration_sec:.0f}s)."

    # Check for silent signal (peak amplitude below 1e-5)
    peak_amp = float(np.max(np.abs(waveform)))
    if peak_amp < 1e-5:
        return AudioValidationResult(
            is_valid=False,
            duration_sec=duration_sec,
            sample_rate=sr,
            num_samples=num_samples,
            error_message="Audio waveform is silent (peak amplitude < 1e-5).",
        )

    return AudioValidationResult(
        is_valid=True,
        duration_sec=duration_sec,
        sample_rate=sr,
        num_channels=1 if waveform.ndim == 1 else waveform.shape[0],
        num_samples=num_samples,
        warning_message=warning,
    )


# ---------------------------------------------------------------------------
# 3. Genre and Artist Provenance Validation
# ---------------------------------------------------------------------------

def validate_genre_metadata(
    genre: Optional[str],
    genre_source: Optional[str] = None,
    confidence: Optional[float] = None,
) -> GenreValidationResult:
    """
    Validate genre label without inventing default genres (e.g. ambient).
    Preserves genre provenance: 'metadata', 'estimated', or 'unknown'.
    """
    if genre is None:
        return GenreValidationResult(
            normalized_genre=None,
            genre_source="unknown",
            notes="No genre metadata available",
        )

    cleaned = str(genre).strip()
    if not cleaned or cleaned.lower() in ("unknown", "none", "null", "nan", "unlabeled", "general", "?"):
        return GenreValidationResult(
            normalized_genre=None,
            genre_source="unknown",
            raw_value=genre,
            notes="Uninformative genre label treated as unknown",
        )

    # Standardize capitalization
    norm_genre = cleaned.title()
    source = genre_source if genre_source in ("metadata", "estimated") else "metadata"

    return GenreValidationResult(
        normalized_genre=norm_genre,
        genre_source=source,
        raw_value=genre,
        confidence=confidence,
        notes=f"Genre verified via {source}",
    )


# ---------------------------------------------------------------------------
# 4. Feature Matrix Numerical Validation
# ---------------------------------------------------------------------------

def validate_feature_matrix(
    matrix: np.ndarray,
    expected_dim: int = 12,
    allow_time_major: bool = True,
) -> FeatureValidationResult:
    """
    Verify numerical integrity of extracted chroma or embedding matrices.
    Ensures dimensions match expected pitch classes / channels and values are finite.
    """
    if matrix is None:
        return FeatureValidationResult(
            is_valid=False,
            expected_dim=expected_dim,
            actual_shape=(),
            error_message="Feature matrix is None.",
        )

    shape = matrix.shape
    if len(shape) != 2:
        return FeatureValidationResult(
            is_valid=False,
            expected_dim=expected_dim,
            actual_shape=shape,
            error_message=f"Expected 2D matrix (dim, frames), but got shape {shape}.",
        )

    # Check dimension alignment (expected_dim on axis 0 or axis 1)
    if shape[0] != expected_dim:
        if allow_time_major and shape[1] == expected_dim:
            pass  # Can be transposed safely
        else:
            return FeatureValidationResult(
                is_valid=False,
                expected_dim=expected_dim,
                actual_shape=shape,
                error_message=f"Matrix dimension mismatch: expected {expected_dim}, got {shape}.",
            )

    has_nans = bool(np.isnan(matrix).any())
    has_infs = bool(np.isinf(matrix).any())

    if has_nans or has_infs:
        return FeatureValidationResult(
            is_valid=False,
            expected_dim=expected_dim,
            actual_shape=shape,
            has_nans=has_nans,
            has_infs=has_infs,
            error_message="Matrix contains non-finite values (NaN or Inf).",
        )

    return FeatureValidationResult(
        is_valid=True,
        expected_dim=expected_dim,
        actual_shape=shape,
        has_nans=False,
        has_infs=False,
    )


# ---------------------------------------------------------------------------
# 5. Robust Content Hashing
# ---------------------------------------------------------------------------

def compute_full_audio_content_hash(audio_bytes: bytes) -> str:
    """
    Compute a robust SHA-256 hash of the complete available audio content.
    Avoids truncated prefix collisions where headers might be identical.
    """
    if not audio_bytes:
        return hashlib.sha256(b"empty_audio_content").hexdigest()
    return hashlib.sha256(audio_bytes).hexdigest()


# ---------------------------------------------------------------------------
# 6. Safe and Reversible Index Migration Tool
# ---------------------------------------------------------------------------

def migrate_remote_index(
    index_file_path: Optional[Union[str, Path]] = None,
    report_output_path: Optional[Union[str, Path]] = None,
) -> IndexMigrationReport:
    """
    Inspect the existing remote index for stale or ambiguous origin labels.
    Creates a timestamped backup before migration, normalizes labels using
    verified semantics, and produces a complete audit report.
    """
    from src.config import FEATURES_DIR, REMOTE_INDEX_FILE

    target_path = Path(index_file_path or REMOTE_INDEX_FILE)
    if not target_path.is_file():
        return IndexMigrationReport(
            total_records=0,
            existing_human_count=0,
            existing_ai_count=0,
            existing_unknown_count=0,
            corrected_human_count=0,
            corrected_ai_count=0,
            corrected_unknown_count=0,
            notes="Target index file does not exist.",
        )

    # 1. Create a safe backup before any modification
    backup_path = target_path.with_suffix(".migration_backup.json")
    shutil.copy2(target_path, backup_path)

    with open(target_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    tracks = data.get("tracks", {})
    total = len(tracks)

    orig_human = sum(1 for t in tracks.values() if t.get("origin_label") == "human")
    orig_ai = sum(1 for t in tracks.values() if t.get("origin_label") == "ai_generated")
    orig_unk = total - (orig_human + orig_ai)

    contradictory: List[str] = []

    for track_id, track_info in tracks.items():
        raw_meta = track_info.get("raw_metadata", {})
        raw_val = raw_meta.get("ai_generated") or raw_meta.get("origin_label") or track_info.get("origin_label")

        res = validate_origin_label(
            raw_value=raw_val,
            field_name="ai_generated" if "ai_generated" in raw_meta else "origin_label",
            additional_metadata=raw_meta,
        )

        track_info["origin_label"] = res.normalized_label
        if res.is_contradictory:
            contradictory.append(track_id)

        # Ensure genre preserves provenance
        if track_info.get("genre") is None or str(track_info.get("genre")).strip().lower() in ("unknown", "none"):
            track_info["genre"] = None
            track_info["genre_source"] = "unknown"

    corr_human = sum(1 for t in tracks.values() if t.get("origin_label") == "human")
    corr_ai = sum(1 for t in tracks.values() if t.get("origin_label") == "ai_generated")
    corr_unk = sum(1 for t in tracks.values() if t.get("origin_label") not in ("human", "ai_generated"))

    data["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    data["migration_version"] = "2.0"

    # Atomic write
    tmp_path = target_path.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp_path, target_path)

    report = IndexMigrationReport(
        total_records=total,
        existing_human_count=orig_human,
        existing_ai_count=orig_ai,
        existing_unknown_count=orig_unk,
        corrected_human_count=corr_human,
        corrected_ai_count=corr_ai,
        corrected_unknown_count=corr_unk,
        contradictory_records=contradictory,
        backup_file_path=str(backup_path),
        migration_timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )

    if report_output_path:
        out_p = Path(report_output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2)

    return report
