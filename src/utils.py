"""
Utility functions for audio file processing and pipeline helpers.
"""

from pathlib import Path
from typing import List, Tuple, Union
import soundfile as sf
import librosa


SUPPORTED_AUDIO_EXTENSIONS = (".wav", ".mp3", ".flac", ".ogg", ".m4a")


def list_audio_files(
    directory: Union[str, Path],
    extensions: Tuple[str, ...] = SUPPORTED_AUDIO_EXTENSIONS
) -> List[Path]:
    """
    Recursively find all audio files with supported extensions in a directory.

    Parameters
    ----------
    directory : Union[str, Path]
        Directory path to search.
    extensions : tuple, optional
        Tuple of audio file extensions to match (case-insensitive).

    Returns
    -------
    List[Path]
        Sorted list of Path objects for discovered audio files.
    """
    dir_path = Path(directory)
    if not dir_path.is_dir():
        return []

    found = [
        p for p in dir_path.rglob("*")
        if p.is_file() and p.suffix.lower() in extensions
    ]
    return sorted(found)


def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Ensure a directory exists, creating parent directories if necessary.
    """
    dir_path = Path(path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def get_audio_duration(file_path: Union[str, Path]) -> float:
    """
    Fast duration estimation of an audio file in seconds.

    Attempts soundfile header inspection first, falling back to librosa if necessary.
    """
    path = Path(file_path)
    if not path.is_file():
        return 0.0

    try:
        info = sf.info(str(path))
        return float(info.duration)
    except Exception:
        try:
            return float(librosa.get_duration(path=str(path)))
        except Exception:
            return 0.0
