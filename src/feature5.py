"""
Feature 5: Metadata Validation and Musical Context Module.

Validates and standardizes track metadata (IDs, artist, title, genre, declared origin),
and extracts estimated musical context (tempo in BPM and musical key via Krumhansl-Schmuckler
chroma profile correlation). Distinguishes user-provided metadata from audio estimates.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import librosa

from src.schemas import MetadataContext
from src.utils import get_audio_duration


# Krumhansl-Schmuckler Key Profiles (12 pitch classes: C, C#, D, D#, E, F, F#, G, G#, A, A#, B)
MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def estimate_musical_key(chroma: np.ndarray) -> str:
    """
    Estimate musical key using Krumhansl-Schmuckler key-finding algorithm.

    Correlates the mean chroma vector across all 12 major and 12 minor profile shifts.
    """
    if chroma.ndim != 2 or chroma.shape[0] != 12 or chroma.shape[1] == 0:
        return "Unknown"

    mean_chroma = np.mean(chroma, axis=1)
    norm = np.linalg.norm(mean_chroma)
    if norm < 1e-9 or not np.isfinite(norm):
        return "Unknown"

    mean_chroma = mean_chroma / norm

    best_corr = -1.0
    best_key = "Unknown"

    for shift in range(12):
        pitch_name = PITCH_CLASSES[shift]

        # Shifted profiles
        maj_profile = np.roll(MAJOR_PROFILE, shift)
        min_profile = np.roll(MINOR_PROFILE, shift)

        maj_corr = float(np.corrcoef(mean_chroma, maj_profile)[0, 1])
        min_corr = float(np.corrcoef(mean_chroma, min_profile)[0, 1])

        if np.isfinite(maj_corr) and maj_corr > best_corr:
            best_corr = maj_corr
            best_key = f"{pitch_name} Major"

        if np.isfinite(min_corr) and min_corr > best_corr:
            best_corr = min_corr
            best_key = f"{pitch_name} Minor"

    return best_key


def estimate_tempo(audio_path: Union[str, Path], sr: int = 22050) -> Optional[float]:
    """
    Estimate track tempo in beats per minute (BPM) using librosa beat tracking.
    """
    path = Path(audio_path)
    if not path.is_file():
        return None

    try:
        y, sr_loaded = librosa.load(str(path), sr=sr, mono=True, duration=60.0)
        if len(y) == 0:
            return None
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr_loaded)
        # Handle scalar or array tempo output from librosa versions
        val = float(tempo[0] if isinstance(tempo, (np.ndarray, list)) else tempo)
        return round(val, 1) if np.isfinite(val) else None
    except Exception:
        return None


def extract_metadata_context(
    audio_path: Union[str, Path],
    provided_metadata: Optional[Dict[str, Any]] = None,
    chroma_matrix: Optional[np.ndarray] = None
) -> MetadataContext:
    """
    Validate metadata and compute musical context for pipeline processing.

    Parameters
    ----------
    audio_path : Union[str, Path]
        Path to audio file.
    provided_metadata : Optional[Dict[str, Any]]
        User or catalog provided metadata dictionary.
    chroma_matrix : Optional[np.ndarray]
        Pre-extracted chroma matrix for fast key estimation.

    Returns
    -------
    MetadataContext
        Structured container with validated metadata, duration, estimated key, and BPM.
    """
    path = Path(audio_path)
    meta = provided_metadata or {}

    track_id = str(meta.get("track_id") or path.stem).strip()
    title = str(meta.get("title")).strip() if meta.get("title") else path.stem.replace("_", " ").title()
    artist = str(meta.get("artist")).strip() if meta.get("artist") else None
    genre = str(meta.get("genre")).strip().title() if meta.get("genre") else None

    prov_key = str(meta.get("key")).strip() if meta.get("key") else None
    prov_tempo = float(meta["tempo"]) if meta.get("tempo") and str(meta["tempo"]).replace(".", "", 1).isdigit() else None

    duration_sec = get_audio_duration(path)

    # Estimate key
    est_key: Optional[str] = None
    if chroma_matrix is not None:
        est_key = estimate_musical_key(chroma_matrix)

    # Estimate tempo if not provided
    est_tempo = prov_tempo
    if est_tempo is None:
        est_tempo = estimate_tempo(path)

    return MetadataContext(
        track_id=track_id,
        file_path=str(path),
        title=title,
        artist=artist,
        genre=genre,
        provided_key=prov_key,
        estimated_key=est_key,
        provided_tempo=prov_tempo,
        estimated_tempo=est_tempo,
        duration_sec=round(duration_sec, 2),
    )
