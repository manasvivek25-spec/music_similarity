"""
Audio feature extraction module.

Extracts chroma features from audio files and provides utilities for caching
and loading feature representations.
"""

from pathlib import Path
from typing import Union
import numpy as np
import librosa


def extract_chroma(audio_path: Union[str, Path], sr: int = 22050) -> np.ndarray:
    """
    Load an audio file and extract its Chroma Constant-Q (chroma_cqt) feature matrix.

    Parameters
    ----------
    audio_path : Union[str, Path]
        Path to the audio file.
    sr : int, optional
        Target sample rate for loading audio, by default 22050 Hz.

    Returns
    -------
    np.ndarray
        Chroma CQT feature matrix of shape (12, n_frames).
    """
    audio_path_str = str(audio_path)
    y, sr_loaded = librosa.load(audio_path_str, sr=sr)
    chroma: np.ndarray = librosa.feature.chroma_cqt(y=y, sr=sr_loaded)
    return chroma


def save_features(features: np.ndarray, output_path: Union[str, Path]) -> None:
    """
    Save a numpy feature array to disk in .npy format.

    Parameters
    ----------
    features : np.ndarray
        Feature matrix to cache.
    output_path : Union[str, Path]
        Target file path (typically inside /features/).
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(path), features)


def load_features(feature_path: Union[str, Path]) -> np.ndarray:
    """
    Load a cached numpy feature matrix from disk (.npy).

    Parameters
    ----------
    feature_path : Union[str, Path]
        Path to the saved .npy feature file.

    Returns
    -------
    np.ndarray
        The loaded feature matrix.
    """
    return np.load(str(feature_path))
