"""
Feature extraction module for acoustic feature vectors.
Provides the 133-dimensional rich acoustic feature extraction for origin classification,
spectral descriptors, and musical timbre representation.
"""

from typing import Union, Optional
from pathlib import Path
import numpy as np

from src.origin_classifier import (
    extract_origin_features_from_audio,
    extract_origin_features_from_audio as extract_rich_acoustic_features,
)

__all__ = [
    "extract_origin_features_from_audio",
    "extract_rich_acoustic_features",
]
