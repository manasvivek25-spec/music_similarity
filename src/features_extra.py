"""
Additional Audio Feature Engineering Module.

Provides independent descriptive representations alongside the core Chroma pipeline:
1. MFCC statistics (13 coefficients: mean and standard deviation).
2. Onset strength features (average novelty energy).
3. Rhythmic descriptors and tempogram mean.
4. Spectral contrast (7 sub-band peak-to-valley dynamic contrasts).
5. Harmonic/Percussive Separation (HPSS) energy distribution.
6. Spectral shape descriptors (flatness, rolloff, bandwidth).

All features are calculated independently and cached in features/extra/.
They are NOT merged into the locked Chroma DTW alignment formula.
"""

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import librosa
import numpy as np

from src.config import FEATURES_DIR, SAMPLE_RATE

logger = logging.getLogger(__name__)

EXTRA_FEATURES_DIR = FEATURES_DIR / "extra"
EXTRA_FEATURES_DIR.mkdir(parents=True, exist_ok=True)
EXTRA_FEATURE_VERSION = "1.0"


@dataclass
class ExtraAudioFeatures:
    """Independent multi-dimensional acoustic and rhythmic descriptors."""
    track_id: str
    mfcc_mean: List[float]  # 13 dimensions: timbral envelope means
    mfcc_std: List[float]   # 13 dimensions: timbral variance
    onset_strength_mean: float  # Scalar: note attack prominence
    tempo_bpm: float  # Estimated beats per minute
    rhythm_regularity: float  # Scalar: tempogram periodicity peak ratio
    spectral_contrast_mean: List[float]  # 7 dimensions: sub-band peak-valley differences
    harmonic_ratio: float  # [0, 1]: harmonic energy fraction
    percussive_ratio: float  # [0, 1]: percussive energy fraction
    spectral_flatness_mean: float  # [0, 1]: noise-like vs tone-like measure
    spectral_rolloff_hz: float  # 85% energy frequency cutoff
    spectral_bandwidth_hz: float  # Frequency spread around centroid
    extraction_version: str = EXTRA_FEATURE_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExtraAudioFeatures":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


def extract_extra_features_from_waveform(
    waveform: np.ndarray,
    sr: int = SAMPLE_RATE,
    track_id: str = "query",
) -> ExtraAudioFeatures:
    """
    Extract comprehensive independent audio descriptors from a mono waveform.

    Parameters
    ----------
    waveform : np.ndarray
        Audio signal normalized to [-1.0, 1.0].
    sr : int
        Sample rate (default: 22050 Hz).
    track_id : str
        Identifier for the audio track.

    Returns
    -------
    ExtraAudioFeatures
        Validated structured audio descriptor.
    """
    if len(waveform) == 0:
        raise ValueError("Cannot extract features from empty waveform.")

    if len(waveform) < 512:
        return ExtraAudioFeatures(
            track_id=track_id,
            mfcc_mean=[0.0] * 13,
            mfcc_std=[0.0] * 13,
            onset_strength_mean=0.0,
            tempo_bpm=120.0,
            rhythm_regularity=1.0,
            spectral_contrast_mean=[0.0] * 7,
            harmonic_ratio=0.0,
            percussive_ratio=0.0,
            spectral_flatness_mean=0.0,
            spectral_rolloff_hz=0.0,
            spectral_bandwidth_hz=0.0,
            extraction_version=EXTRA_FEATURE_VERSION,
        )

    n_fft = min(2048, len(waveform))

    # 1. MFCC Statistics (13 coefficients)
    try:
        mfcc = librosa.feature.mfcc(y=waveform, sr=sr, n_mfcc=13, n_fft=n_fft)
        mfcc_mean = [round(float(x), 4) for x in np.mean(mfcc, axis=1)]
        mfcc_std = [round(float(x), 4) for x in np.std(mfcc, axis=1)]
    except Exception as exc:
        logger.warning(f"MFCC extraction failed for {track_id}: {exc}")
        mfcc_mean = [0.0] * 13
        mfcc_std = [0.0] * 13

    # 2. Onset Strength & Rhythm
    try:
        onset_env = librosa.onset.onset_strength(y=waveform, sr=sr)
        onset_mean = round(float(np.mean(onset_env)), 4)
    except Exception as exc:
        logger.warning(f"Onset extraction failed for {track_id}: {exc}")
        onset_mean = 0.0

    # Tempo & Tempogram Regularity
    try:
        tempo, _ = librosa.beat.beat_track(y=waveform, sr=sr)
        bpm = round(float(np.atleast_1d(tempo)[0]), 1)
        tempogram = librosa.feature.tempogram(y=waveform, sr=sr)
        tg_mean = np.mean(tempogram, axis=1)
        rhythm_reg = round(float(np.max(tg_mean) / (np.mean(tg_mean) + 1e-6)), 4)
    except Exception as exc:
        logger.warning(f"Tempo/rhythm extraction failed for {track_id}: {exc}")
        bpm = 120.0
        rhythm_reg = 1.0

    # 3. Spectral Contrast (7 sub-bands)
    try:
        contrast = librosa.feature.spectral_contrast(y=waveform, sr=sr, n_bands=6)
        contrast_mean = [round(float(x), 4) for x in np.mean(contrast, axis=1)]
    except Exception as exc:
        logger.warning(f"Spectral contrast extraction failed for {track_id}: {exc}")
        contrast_mean = [0.0] * 7

    # 4. Harmonic / Percussive Separation (HPSS)
    try:
        y_harm, y_perc = librosa.effects.hpss(waveform)
        rms_harm = float(np.sqrt(np.mean(y_harm**2)))
        rms_perc = float(np.sqrt(np.mean(y_perc**2)))
        rms_total = rms_harm + rms_perc + 1e-9
        harm_ratio = round(rms_harm / rms_total, 4)
        perc_ratio = round(rms_perc / rms_total, 4)
    except Exception as exc:
        logger.warning(f"HPSS extraction failed for {track_id}: {exc}")
        harm_ratio = 0.5
        perc_ratio = 0.5

    # 5. Additional Spectral Descriptors
    try:
        flatness = float(np.mean(librosa.feature.spectral_flatness(y=waveform, n_fft=n_fft)))
        rolloff = float(np.mean(librosa.feature.spectral_rolloff(y=waveform, sr=sr, roll_percent=0.85, n_fft=n_fft)))
        bandwidth = float(np.mean(librosa.feature.spectral_bandwidth(y=waveform, sr=sr, n_fft=n_fft)))
    except Exception as exc:
        logger.warning(f"Spectral descriptors failed for {track_id}: {exc}")
        flatness = 0.01
        rolloff = 3000.0
        bandwidth = 1500.0

    return ExtraAudioFeatures(
        track_id=track_id,
        mfcc_mean=mfcc_mean,
        mfcc_std=mfcc_std,
        onset_strength_mean=onset_mean,
        tempo_bpm=bpm,
        rhythm_regularity=rhythm_reg,
        spectral_contrast_mean=contrast_mean,
        harmonic_ratio=harm_ratio,
        percussive_ratio=perc_ratio,
        spectral_flatness_mean=round(flatness, 6),
        spectral_rolloff_hz=round(rolloff, 1),
        spectral_bandwidth_hz=round(bandwidth, 1),
        extraction_version=EXTRA_FEATURE_VERSION,
    )


def get_or_extract_extra_features(
    audio_file_or_waveform: Union[str, Path, np.ndarray],
    track_id: str,
    sr: int = SAMPLE_RATE,
    cache_dir: Optional[Path] = None,
    force_recompute: bool = False,
) -> ExtraAudioFeatures:
    """
    Retrieve cached independent audio features or compute and persist them atomically.

    Parameters
    ----------
    audio_file_or_waveform : Union[str, Path, np.ndarray]
        Path to audio file on disk, or pre-decoded numpy array.
    track_id : str
        Unique track identifier used for cache filename.
    sr : int
        Audio sampling rate.
    cache_dir : Optional[Path]
        Directory for cache files (defaults to features/extra/).
    force_recompute : bool
        If True, recomputes features even if cache exists.

    Returns
    -------
    ExtraAudioFeatures
        Structured descriptor instance.
    """
    target_dir = Path(cache_dir or EXTRA_FEATURES_DIR)
    target_dir.mkdir(parents=True, exist_ok=True)
    cache_file = target_dir / f"{track_id}_extra.json"

    if not force_recompute and cache_file.is_file():
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            if data.get("extraction_version") == EXTRA_FEATURE_VERSION:
                return ExtraAudioFeatures.from_dict(data)
        except Exception:
            pass  # Fall through to recompute on corrupted cache

    # Compute features
    if isinstance(audio_file_or_waveform, (str, Path)):
        p = Path(audio_file_or_waveform)
        if not p.is_file():
            raise FileNotFoundError(f"Audio file not found: {p}")
        y, _ = librosa.load(str(p), sr=sr, mono=True)
    elif isinstance(audio_file_or_waveform, np.ndarray):
        y = audio_file_or_waveform
    else:
        raise TypeError(f"Unsupported audio type: {type(audio_file_or_waveform)}")

    features = extract_extra_features_from_waveform(y, sr=sr, track_id=track_id)

    # Persist cache atomically
    try:
        tmp_cache = cache_file.with_suffix(".tmp")
        with open(tmp_cache, "w", encoding="utf-8") as f:
            json.dump(features.to_dict(), f, indent=2)
        os.replace(tmp_cache, cache_file)
    except Exception as exc:
        logger.warning(f"Failed to persist extra features cache for {track_id}: {exc}")

    return features

