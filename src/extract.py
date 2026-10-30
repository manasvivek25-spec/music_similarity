"""
Audio feature extraction module.

Extracts chroma features from audio files, supports beat-synchronization for
time-alignment in DTW, handles caching under features/, and converts frame
indices to audio timestamps.
"""

import hashlib
from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import numpy as np
import librosa
import soundfile as sf

from src.config import FEATURES_DIR, SAMPLE_RATE


class AudioProcessingError(Exception):
    """Raised when an audio file cannot be loaded, decoded, or processed."""
    pass


def extract_chroma(audio_path: Union[str, Path], sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Load an audio file and extract its 12-bin Chroma Constant-Q (chroma_cqt) feature matrix.

    Matrix convention:
        Shape: (12, n_frames) where axis 0 represents the 12 pitch classes (C, C#, ..., B)
        and axis 1 represents time frames.

    Parameters
    ----------
    audio_path : Union[str, Path]
        Path to the audio file.
    sr : int, optional
        Target sample rate for loading audio as mono, by default 22050 Hz.

    Returns
    -------
    np.ndarray
        Chroma CQT feature matrix of shape (12, n_frames).

    Raises
    ------
    AudioProcessingError
        If the file does not exist, is empty, or cannot be decoded.
    """
    path = Path(audio_path)
    if not path.is_file():
        raise AudioProcessingError(f"Audio file does not exist: {path}")

    try:
        y, sr_loaded = librosa.load(str(path), sr=sr, mono=True)
    except Exception as exc:
        raise AudioProcessingError(f"Failed to decode audio file {path}: {exc}") from exc

    if y is None or len(y) == 0:
        raise AudioProcessingError(f"Audio signal is empty in file: {path}")

    # Compute Chroma CQT
    try:
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr_loaded)
    except Exception as exc:
        # UserWarning / fallback if signal is too short for default n_fft / bins
        try:
            chroma = librosa.feature.chroma_stft(y=y, sr=sr_loaded, n_fft=min(1024, len(y)))
        except Exception as inner_exc:
            raise AudioProcessingError(f"Chroma extraction failed for {path}: {inner_exc}") from exc

    # Ensure shape is (12, n_frames)
    if chroma.shape[0] != 12 and chroma.shape[1] == 12:
        chroma = chroma.T

    if chroma.shape[0] != 12:
        raise AudioProcessingError(f"Unexpected chroma matrix shape {chroma.shape}; expected 12 pitch classes.")

    return chroma.astype(np.float32)


def extract_chroma_beatsync(
    audio_path: Union[str, Path],
    sr: int = SAMPLE_RATE
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Extract beat-synchronized chroma features and corresponding time mapping.

    Loads the audio file as mono at `sr` Hz, computes raw chroma CQT and tempo/beat
    frame intervals, then synchronizes chroma across beat intervals via median aggregation.
    Retains beat timestamps so DTW alignment frame indices can be mapped to seconds.

    Parameters
    ----------
    audio_path : Union[str, Path]
        Path to audio file.
    sr : int, optional
        Sample rate (default 22050 Hz).

    Returns
    -------
    chroma_beat : np.ndarray
        Beat-synchronized chroma matrix of shape (12, n_beats).
    beat_times : np.ndarray
        Array of timestamps in seconds corresponding to each beat frame.
    duration_sec : float
        Total duration of the audio file in seconds.
    """
    path = Path(audio_path)
    if not path.is_file():
        raise AudioProcessingError(f"Audio file does not exist: {path}")

    try:
        y, sr_loaded = librosa.load(str(path), sr=sr, mono=True)
    except Exception as exc:
        raise AudioProcessingError(f"Failed to load audio {path}: {exc}") from exc

    if y is None or len(y) == 0:
        raise AudioProcessingError(f"Audio signal is empty: {path}")

    duration_sec = float(len(y) / sr_loaded)

    # 1. Raw Chroma CQT
    try:
        chroma_raw = librosa.feature.chroma_cqt(y=y, sr=sr_loaded)
    except Exception:
        chroma_raw = librosa.feature.chroma_stft(y=y, sr=sr_loaded, n_fft=min(1024, len(y)))

    if chroma_raw.shape[0] != 12 and chroma_raw.shape[1] == 12:
        chroma_raw = chroma_raw.T

    # 2. Beat Tracking
    try:
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr_loaded)
    except Exception:
        beat_frames = np.array([], dtype=int)

    # If beat tracking detected at least 2 beats, synchronize by beat
    if len(beat_frames) >= 2:
        # Synchronize
        chroma_beat = librosa.util.sync(chroma_raw, beat_frames, aggregate=np.median)
        beat_times = librosa.frames_to_time(beat_frames, sr=sr_loaded)
        # Ensure lengths match
        min_len = min(chroma_beat.shape[1], len(beat_times))
        chroma_beat = chroma_beat[:, :min_len]
        beat_times = beat_times[:min_len]
    else:
        # Fallback to standard frame hop times if audio lacks clear beats
        chroma_beat = chroma_raw
        n_frames = chroma_raw.shape[1]
        beat_times = librosa.frames_to_time(np.arange(n_frames), sr=sr_loaded)

    return chroma_beat.astype(np.float32), beat_times.astype(np.float32), duration_sec


def save_features(features: np.ndarray, output_path: Union[str, Path]) -> None:
    """
    Save a numpy feature array to disk in .npy format.

    Ensures parent directories exist before saving.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(path), features)


def load_features(feature_path: Union[str, Path]) -> np.ndarray:
    """
    Load a cached numpy feature matrix from disk (.npy).
    """
    path = Path(feature_path)
    if not path.is_file():
        raise FileNotFoundError(f"Feature cache file not found: {path}")
    return np.load(str(path))


def get_cache_paths(
    audio_path: Union[str, Path],
    cache_dir: Optional[Union[str, Path]] = None
) -> Dict[str, Path]:
    """
    Generate deterministic cache file paths for raw chroma, beat-sync chroma, and timing info.

    Uses an MD5 hash of the resolved audio path combined with the file stem.
    """
    cdir = Path(cache_dir) if cache_dir is not None else FEATURES_DIR
    path = Path(audio_path).resolve()
    hash_id = hashlib.md5(str(path).encode("utf-8")).hexdigest()[:10]
    stem = path.stem.replace(" ", "_")

    return {
        "raw": cdir / f"{stem}_{hash_id}_raw.npy",
        "beat": cdir / f"{stem}_{hash_id}_beat.npy",
        "times": cdir / f"{stem}_{hash_id}_times.npy",
        "meta": cdir / f"{stem}_{hash_id}_meta.npy",
    }


def get_or_extract_features(
    audio_path: Union[str, Path],
    cache_dir: Optional[Union[str, Path]] = None,
    sr: int = SAMPLE_RATE
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Retrieve cached feature arrays or extract and cache them safely.

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
    cache_paths = get_cache_paths(audio_path, cache_dir)

    # Check if all 4 cache files exist
    if all(p.is_file() for p in cache_paths.values()):
        try:
            chroma_raw = load_features(cache_paths["raw"])
            chroma_beat = load_features(cache_paths["beat"])
            beat_times = load_features(cache_paths["times"])
            meta = load_features(cache_paths["meta"])
            duration_sec = float(meta[0])
            return chroma_raw, chroma_beat, beat_times, duration_sec
        except Exception:
            # If cache is corrupted, recompute
            pass

    # Extract
    chroma_raw = extract_chroma(audio_path, sr=sr)
    chroma_beat, beat_times, duration_sec = extract_chroma_beatsync(audio_path, sr=sr)

    # Cache
    try:
        save_features(chroma_raw, cache_paths["raw"])
        save_features(chroma_beat, cache_paths["beat"])
        save_features(beat_times, cache_paths["times"])
        save_features(np.array([duration_sec], dtype=np.float32), cache_paths["meta"])
    except Exception:
        # Non-fatal if cache saving fails
        pass

    return chroma_raw, chroma_beat, beat_times, duration_sec
