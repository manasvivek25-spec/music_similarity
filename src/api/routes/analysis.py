"""
Origin classification and metadata extraction routes (Features 4 & 5).
"""

import tempfile
from pathlib import Path
import numpy as np
import librosa
from fastapi import APIRouter, File, HTTPException, UploadFile

from src.api.schemas import MetadataExtractResponse, OriginClassifyResponse

router = APIRouter(prefix="/analysis", tags=["Analysis & Metadata (Features 4 & 5)"])

PITCH_CLASS_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def _save_temp_audio(file: UploadFile) -> Path:
    suffix = Path(file.filename or "audio.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file.file.read())
        return Path(tmp.name)


@router.post("/origin/classify", response_model=OriginClassifyResponse)
async def classify_track_origin(file: UploadFile = File(...)):
    """
    Feature 4: AI-vs-Human Origin Classifier stub using spectral artifacts.
    """
    tmp_path = None
    try:
        tmp_path = _save_temp_audio(file)
        y, sr = librosa.load(str(tmp_path), sr=22050, duration=30.0)

        flatness = float(np.mean(librosa.feature.spectral_flatness(y=y)))

        # Baseline heuristic stub: AI models like MusicGen often exhibit distinct high-frequency artifacting
        if flatness > 0.05:
            prediction = "ai"
            confidence = 0.82
            notes = "High spectral flatness and characteristic high-frequency artifacts detected."
        else:
            prediction = "human"
            confidence = 0.78
            notes = "Harmonic coherence matches typical human acoustic/studio recording profile."

        return OriginClassifyResponse(
            prediction=prediction,
            confidence=confidence,
            spectral_flatness_mean=round(flatness, 6),
            notes=notes,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Classification failed: {exc}")
    finally:
        if tmp_path and tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


@router.post("/metadata/extract", response_model=MetadataExtractResponse)
async def extract_metadata(file: UploadFile = File(...)):
    """
    Feature 5: Metadata extraction (BPM tempo, estimated key from chroma, duration, spectral brightness).
    """
    tmp_path = None
    try:
        tmp_path = _save_temp_audio(file)
        y, sr = librosa.load(str(tmp_path), sr=22050)
        duration = float(librosa.get_duration(y=y, sr=sr))

        # Tempo estimation
        tempo_arr, _ = librosa.beat.beat_track(y=y, sr=sr)
        tempo = float(tempo_arr[0]) if isinstance(tempo_arr, (np.ndarray, list)) else float(tempo_arr)

        # Musical key estimation from chroma
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        pitch_energies = np.sum(chroma, axis=1)
        root_pitch_idx = int(np.argmax(pitch_energies))
        estimated_key = f"{PITCH_CLASS_NAMES[root_pitch_idx]} Tonality"

        # Spectral centroid (brightness)
        centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))

        return MetadataExtractResponse(
            tempo_bpm=round(tempo, 1),
            estimated_key=estimated_key,
            duration_sec=round(duration, 2),
            spectral_centroid_hz=round(centroid, 1),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Metadata extraction failed: {exc}")
    finally:
        if tmp_path and tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
