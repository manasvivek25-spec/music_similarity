"""
Utility functions for audio file processing and pipeline helpers.
"""

from pathlib import Path
from typing import List, Union


def list_audio_files(directory: Union[str, Path], extensions: tuple = (".mp3", ".wav", ".flac", ".ogg")) -> List[Path]:
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
        List of Path objects for discovered audio files.
    """
    dir_path = Path(directory)
    if not dir_path.exists():
        return []

    return [
        p for p in dir_path.rglob("*")
        if p.is_file() and p.suffix.lower() in extensions
    ]


def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Ensure a directory exists, creating parent directories if necessary.

    Parameters
    ----------
    path : Union[str, Path]
        Path to directory.

    Returns
    -------
    Path
        Created or existing Path object.
    """
    dir_path = Path(path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path
