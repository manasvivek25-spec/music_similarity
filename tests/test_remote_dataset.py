"""
Unit and integration tests for remote dataset streaming, schema detection,
in-memory incremental decoding, compact feature caching, and Stage 1/2 search.

All tests operate offline using mocked streaming generators and synthesized
in-memory audio records, ensuring zero external network dependency during tests.
"""

import io
import json
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import soundfile as sf

from src.remote_dataset import (
    DatasetSchemaInfo,
    IndexingReport,
    align_shortlisted_remote_candidate,
    build_cli_parser,
    decode_audio_record,
    detect_dataset_schema,
    extract_features_from_waveform,
    filter_remote_candidates,
    index_remote_dataset,
    inspect_remote_metadata,
    load_remote_index,
    save_remote_index,
)
from src.similarity import compute_mean_chroma_vector, cosine_similarity_vectors


# ---------------------------------------------------------------------------
# Test Helpers
# ---------------------------------------------------------------------------

def create_synthetic_wav_bytes(duration_sec: float = 1.0, sr: int = 22050, freq: float = 440.0) -> bytes:
    """Generate in-memory WAV audio bytes for testing without writing to disk."""
    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)
    # Sine wave + harmonic
    signal = 0.5 * np.sin(2 * np.pi * freq * t) + 0.25 * np.sin(2 * np.pi * freq * 2 * t)
    bio = io.BytesIO()
    sf.write(bio, signal.astype(np.float32), sr, format="WAV")
    return bio.getvalue()


# ---------------------------------------------------------------------------
# 1. CLI Parser Tests
# ---------------------------------------------------------------------------

def test_cli_parser_defaults():
    """Verify CLI defaults to small sample for development and full=False."""
    parser = build_cli_parser()
    args = parser.parse_args([])
    assert args.dataset == "SleepyJesse/ai_music_large"
    assert args.split == "train"
    assert args.sample == 10
    assert args.full is False
    assert args.inspect_schema is False
    assert args.inspect_metadata is False
    assert args.force is False


def test_cli_parser_explicit_options():
    """Verify CLI flags are parsed accurately."""
    parser = build_cli_parser()
    args = parser.parse_args(["--full", "--sample", "50", "--force", "--inspect-schema", "--dataset", "Custom/ds"])
    assert args.full is True
    assert args.sample == 50
    assert args.force is True
    assert args.inspect_schema is True
    assert args.dataset == "Custom/ds"


# ---------------------------------------------------------------------------
# 2. Dynamic Schema Detection and Validation Tests
# ---------------------------------------------------------------------------

def test_detect_schema_standard_keys():
    """Detect standard column names (audio, label, genre, id)."""
    record = {
        "audio": {"bytes": b"mock_wav_bytes"},
        "label": "ai_generated",
        "genre": "electronic",
        "id": "track_101",
    }
    schema = detect_dataset_schema(record)
    assert schema.is_valid is True
    assert schema.audio_col == "audio"
    assert schema.audio_type == "dict_bytes"
    assert schema.id_col == "id"
    assert schema.label_col == "label"
    assert schema.genre_col == "genre"


def test_detect_schema_alternative_keys():
    """Detect non-standard column names (sound, source, style, track_id)."""
    record = {
        "sound": b"raw_sound_bytes",
        "source": "human",
        "style": "classical",
        "track_id": "trk_202",
        "extra_info": "unrelated",
    }
    schema = detect_dataset_schema(record)
    assert schema.is_valid is True
    assert schema.audio_col == "sound"
    assert schema.audio_type == "raw_bytes"
    assert schema.id_col == "track_id"
    assert schema.label_col == "source"
    assert schema.genre_col == "style"


def test_detect_schema_missing_audio_fails():
    """Validation fails gracefully when no audio column exists in record."""
    record = {
        "title": "A Song About Antigravity",
        "rating": 5,
        "genre": "rock",
    }
    schema = detect_dataset_schema(record)
    assert schema.is_valid is False
    assert "No recognizable audio column" in schema.validation_error


def test_detect_schema_empty_features():
    """Empty record or schema returns is_valid=False."""
    schema = detect_dataset_schema({})
    assert schema.is_valid is False
    assert "empty" in schema.validation_error.lower()


# ---------------------------------------------------------------------------
# 3. In-Memory Incremental Audio Decoding Tests
# ---------------------------------------------------------------------------

def test_decode_audio_record_from_bytes_dict():
    """Decode audio directly from dict with 'bytes' without disk writes."""
    wav_bytes = create_synthetic_wav_bytes(duration_sec=1.5, sr=22050, freq=440.0)
    audio_data = {"bytes": wav_bytes}

    waveform, duration, byte_count = decode_audio_record(audio_data, target_sr=22050)
    assert isinstance(waveform, np.ndarray)
    assert waveform.ndim == 1
    assert len(waveform) == pytest.approx(int(1.5 * 22050), abs=100)
    assert duration == pytest.approx(1.5, abs=0.05)
    assert byte_count == len(wav_bytes)


def test_decode_audio_record_from_raw_bytes():
    """Decode raw audio bytes directly."""
    wav_bytes = create_synthetic_wav_bytes(duration_sec=0.8, sr=22050, freq=523.25)
    waveform, duration, byte_count = decode_audio_record(wav_bytes, target_sr=22050)
    assert isinstance(waveform, np.ndarray)
    assert duration == pytest.approx(0.8, abs=0.05)
    assert byte_count == len(wav_bytes)


def test_decode_audio_record_from_array_dict():
    """Decode audio from dict containing pre-decoded array."""
    synth_arr = np.sin(np.linspace(0, 10, 22050)).astype(np.float32)
    audio_data = {"array": synth_arr, "sampling_rate": 22050}

    waveform, duration, byte_count = decode_audio_record(audio_data, target_sr=22050)
    assert len(waveform) == 22050
    assert duration == pytest.approx(1.0, abs=0.01)
    assert byte_count == synth_arr.nbytes


def test_decode_audio_record_resamples_automatically():
    """Automatically resample when source sampling rate differs from target."""
    wav_bytes = create_synthetic_wav_bytes(duration_sec=1.0, sr=44100, freq=440.0)
    waveform, duration, byte_count = decode_audio_record(wav_bytes, target_sr=22050)
    assert len(waveform) == pytest.approx(22050, abs=50)
    assert duration == pytest.approx(1.0, abs=0.05)


def test_decode_audio_record_invalid_raises():
    """Invalid or empty audio record raises ValueError."""
    with pytest.raises(ValueError, match="Audio data record is None"):
        decode_audio_record(None)

    with pytest.raises(ValueError):
        decode_audio_record(b"corrupted_garbage_header")


# ---------------------------------------------------------------------------
# 4. In-Memory Feature Extraction Tests
# ---------------------------------------------------------------------------

def test_extract_features_from_waveform():
    """Extract raw and beat-synced chroma directly from in-memory waveform."""
    t = np.linspace(0, 2.0, int(22050 * 2.0))
    waveform = (0.5 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)

    chroma_raw, chroma_beat, beat_times, duration = extract_features_from_waveform(waveform, sr=22050)
    assert chroma_raw.shape[0] == 12
    assert chroma_beat.shape[0] == 12
    assert chroma_raw.ndim == 2
    assert chroma_beat.ndim == 2
    assert len(beat_times) == chroma_beat.shape[1]
    assert duration == pytest.approx(2.0, abs=0.01)

    mean_chroma = compute_mean_chroma_vector(chroma_beat)
    assert len(mean_chroma) == 12
    assert np.all(np.isfinite(mean_chroma))


# ---------------------------------------------------------------------------
# 5. Metadata-Only Inspection Tests
# ---------------------------------------------------------------------------

def test_inspect_remote_metadata_without_audio():
    """Inspect dataset metadata and verify audio column is stripped from output records."""
    wav_bytes = create_synthetic_wav_bytes(duration_sec=0.5)
    mock_records = [
        {"audio": {"bytes": wav_bytes}, "title": "Song A", "label": "ai_generated", "genre": "ambient"},
        {"audio": {"bytes": wav_bytes}, "title": "Song B", "label": "human", "genre": "jazz"},
    ]

    result = inspect_remote_metadata(limit=2, dataset_override=mock_records)
    assert result["records_retrieved"] == 2
    schema = result["schema"]
    assert schema["is_valid"] is True
    assert schema["audio_col"] == "audio"

    for rec in result["records"]:
        assert "audio" not in rec
        assert "_normalized_origin" in rec
        assert "_normalized_genre" in rec


# ---------------------------------------------------------------------------
# 6. Incremental Indexing and Cache Reuse Tests
# ---------------------------------------------------------------------------

def test_incremental_indexing_with_mocked_stream():
    """
    Simulate incremental streaming indexing:
      - 2 valid records + 1 corrupted record
      - Precomputes 12-dim mean-chroma in JSON
      - Caches detailed beat matrices in features/
      - Tracks downloaded bytes and error count
      - Zero raw audio files saved to disk
    """
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        index_file = tmp_dir / "test_remote_index.json"
        features_dir = tmp_dir / "features"
        features_dir.mkdir(parents=True, exist_ok=True)

        wav1 = create_synthetic_wav_bytes(duration_sec=1.0, freq=261.63)  # Middle C
        wav2 = create_synthetic_wav_bytes(duration_sec=1.0, freq=329.63)  # E4

        mock_stream = [
            {"audio": {"bytes": wav1}, "id": "song_c", "label": "human", "genre": "classical"},
            {"audio": {"bytes": b"BAD_BYTES"}, "id": "corrupt_song", "label": "ai", "genre": "noise"},
            {"audio": {"bytes": wav2}, "id": "song_e", "label": "ai_generated", "genre": "electronic"},
        ]

        report = index_remote_dataset(
            sample_size=3,
            is_full=False,
            force_reindex=False,
            index_file=index_file,
            features_cache_dir=features_dir,
            cache_detailed=True,
            dataset_override=mock_stream,
        )

        # Assert report metrics
        assert report.processed_records == 2
        assert report.failed_records == 1
        assert report.cache_hits == 0
        assert report.downloaded_bytes > 0
        assert len(report.errors) == 1
        assert "audio feature extraction failed" in report.errors[0]
        assert "disclaimer" in report.to_dict()

        # Check persisted JSON index
        assert index_file.is_file()
        index_data = load_remote_index(index_file)
        assert index_data["total_indexed"] == 2

        # Verify tracks were indexed with stable content-hash IDs (Decision 3)
        tracks = index_data["tracks"]
        c_tracks = [t for t in tracks.values() if t.get("source_id") == "song_c"]
        e_tracks = [t for t in tracks.values() if t.get("source_id") == "song_e"]
        assert len(c_tracks) == 1
        assert len(e_tracks) == 1

        c_track = c_tracks[0]
        c_track_id = c_track["track_id"]
        assert c_track_id.startswith("track_")
        assert len(c_track["mean_chroma"]) == 12
        assert c_track["origin_label"] == "human"
        assert c_track["genre"] == "classical"

        # Check detailed feature matrices saved with stable track ID
        beat_c = features_dir / f"{c_track_id}_beat.npy"
        times_c = features_dir / f"{c_track_id}_times.npy"
        assert beat_c.is_file()
        assert times_c.is_file()

        # Verify NO raw audio files exist in temp directory or features directory
        wav_files = list(tmp_dir.glob("**/*.wav")) + list(tmp_dir.glob("**/*.mp3"))
        assert len(wav_files) == 0

        # --- Second Pass: Verify Cache Reuse (Cache Hits) ---
        mock_stream_pass2 = [
            {"audio": {"bytes": wav1}, "id": "song_c", "label": "human", "genre": "classical"},
            {"audio": {"bytes": wav2}, "id": "song_e", "label": "ai_generated", "genre": "electronic"},
        ]

        report_pass2 = index_remote_dataset(
            sample_size=2,
            is_full=False,
            force_reindex=False,
            index_file=index_file,
            features_cache_dir=features_dir,
            dataset_override=mock_stream_pass2,
        )

        assert report_pass2.processed_records == 0
        assert report_pass2.cache_hits == 2
        assert report_pass2.downloaded_bytes == 0

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# 7. Two-Stage Candidate Filtering & Shortlisted Alignment Tests
# ---------------------------------------------------------------------------

def test_filter_remote_candidates_stage1():
    """Test Stage 1 Cosine prefiltering using precomputed compact mean chroma."""
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        index_file = tmp_dir / "remote_index.json"
        # Track 1: strongly dominant at bin 0 (C)
        vec1 = [1.0, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]
        # Track 2: strongly dominant at bin 6 (F#), orthogonal to C
        vec2 = [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 1.0, 0.05, 0.05, 0.05, 0.05, 0.05]

        mock_index_data = {
            "tracks": {
                "remote_track_similar": {
                    "track_id": "remote_track_similar",
                    "origin_label": "human",
                    "genre": "pop",
                    "mean_chroma": vec1,
                },
                "remote_track_distant": {
                    "track_id": "remote_track_distant",
                    "origin_label": "ai_generated",
                    "genre": "jazz",
                    "mean_chroma": vec2,
                },
            }
        }
        save_remote_index(mock_index_data, index_file)

        # Query that is also dominant at bin 0 (C)
        query_chroma = np.zeros((12, 10), dtype=float)
        query_chroma[0, :] = 1.0  # C pitch class

        shortlisted, stats = filter_remote_candidates(
            query_chroma=query_chroma,
            index_file=index_file,
            threshold=0.30,  # Locked Feature 1 threshold
        )

        assert stats["before_count"] == 2
        # Track 1 should survive because cosine similarity is high (~0.99)
        assert any(t["track_id"] == "remote_track_similar" for t in shortlisted)
        # Track 2 has cosine similarity < 0.30 and should be pruned
        assert not any(t["track_id"] == "remote_track_distant" for t in shortlisted)
        assert stats["after_count"] == 1
        assert stats["percentage_removed"] == 50.0

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_align_shortlisted_remote_candidate_stage2():
    """Test Stage 2 DTW alignment on shortlisted candidate using cached beat chroma."""
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        # Create synthetic beat matrices
        t = np.linspace(0, 1.0, 100)
        chroma_a = np.tile(np.array([[1.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]]), (1, 30)).astype(np.float32)
        # Shifted by 2 semitones
        chroma_b = np.roll(chroma_a, 2, axis=0)

        beat_path = tmp_dir / "remote_cand_beat.npy"
        times_path = tmp_dir / "remote_cand_times.npy"
        np.save(str(beat_path), chroma_b)
        np.save(str(times_path), np.linspace(0, 1.0, 30))

        candidate_info = {
            "track_id": "remote_cand",
            "beat_cache": str(beat_path.relative_to(tmp_dir)),
            "times_cache": str(times_path.relative_to(tmp_dir)),
        }

        alignment_result = align_shortlisted_remote_candidate(
            query_chroma_beat=chroma_a,
            candidate_info=candidate_info,
            repo_root=tmp_dir,
        )

        assert alignment_result is not None
        dtw_res, times = alignment_result
        assert dtw_res.similarity > 0.90  # Key-invariance should detect shift 2 or 10
        assert dtw_res.best_shift in (2, 10)
        assert len(times) == 30

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
