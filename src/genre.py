"""
Stage 3: Genre Influence Map & Metadata Processing.

Groups candidates by genre metadata, computes actual percentage contributions,
identifies cross-genre comparisons, and provides standardized metadata catalog loading.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import pandas as pd

from src.schemas import GenreContribution, GenreMapResult


METADATA_COLUMNS = [
    "track_id",
    "file_path",
    "title",
    "artist",
    "genre",
    "source_type",
    "origin_label",
    "key",
    "tempo",
]


def load_metadata_catalog(csv_path: Union[str, Path]) -> Dict[str, Dict[str, Any]]:
    """
    Load track metadata from a documented CSV file into a memory catalog.

    Parameters
    ----------
    csv_path : Union[str, Path]
        Path to metadata CSV. Expected columns:
        track_id, file_path, title, artist, genre, source_type, origin_label, key, tempo.

    Returns
    -------
    Dict[str, Dict[str, Any]]
        Dictionary keyed by track_id or file stem.
    """
    path = Path(csv_path)
    if not path.is_file():
        return {}

    try:
        df = pd.read_csv(str(path))
    except Exception:
        return {}

    catalog: Dict[str, Dict[str, Any]] = {}
    for _, row in df.iterrows():
        track_id = str(row.get("track_id", "")).strip()
        if not track_id:
            # Fallback to stem of file_path
            fp = str(row.get("file_path", "")).strip()
            if fp:
                track_id = Path(fp).stem
            else:
                continue

        catalog[track_id] = {
            "track_id": track_id,
            "file_path": str(row.get("file_path", "")).strip() if pd.notna(row.get("file_path")) else None,
            "title": str(row.get("title", "")).strip() if pd.notna(row.get("title")) else None,
            "artist": str(row.get("artist", "")).strip() if pd.notna(row.get("artist")) else None,
            "genre": str(row.get("genre", "")).strip() if pd.notna(row.get("genre")) else None,
            "source_type": str(row.get("source_type", "")).strip() if pd.notna(row.get("source_type")) else None,
            "origin_label": str(row.get("origin_label", "")).strip() if pd.notna(row.get("origin_label")) else None,
            "key": str(row.get("key", "")).strip() if pd.notna(row.get("key")) else None,
            "tempo": float(row["tempo"]) if pd.notna(row.get("tempo")) and str(row.get("tempo")).replace(".", "", 1).isdigit() else None,
        }

    return catalog


def resolve_track_genre(
    metadata_genre: Optional[str] = None,
    waveform: Optional[Any] = None,
    sr: int = 22050,
    classifier: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Resolve track genre following Authoritative Decision 2:
    - Use verified metadata when available -> genre_source = 'metadata'
    - Otherwise use real model estimate if one is configured -> genre_source = 'estimated'
    - Otherwise keep genre unknown -> genre_source = 'unknown'

    Never use a hardcoded default such as 'ambient' as if it were an actual genre observation.
    """
    clean_meta = None
    if metadata_genre:
        cand = str(metadata_genre).strip()
        if cand and cand.lower() not in ("none", "null", "unknown", "ambient", ""):
            clean_meta = cand

    if clean_meta:
        return {
            "genre": clean_meta,
            "estimated_genre": None,
            "genre_confidence": None,
            "genre_source": "metadata",
        }

    # Attempt classifier estimation if waveform is available
    if waveform is not None and len(waveform) > 0:
        try:
            if classifier is not None:
                clf = classifier
            else:
                from src.genre_classifier import get_genre_classifier
                clf = get_genre_classifier()

            if clf.is_available:
                est, conf = clf.predict(waveform, sr=sr)
                if est:
                    return {
                        "genre": None,
                        "estimated_genre": est,
                        "genre_confidence": conf,
                        "genre_source": "estimated",
                    }
        except Exception:
            pass

    return {
        "genre": None,
        "estimated_genre": None,
        "genre_confidence": None,
        "genre_source": "unknown",
    }


def compute_genre_influence_map(
    query_genre: Optional[str],
    candidate_genres: Dict[str, Optional[str]]
) -> GenreMapResult:
    """
    Group candidates by genre, compute percentage breakdown, and detect cross-genre presence.

    Parameters
    ----------
    query_genre : Optional[str]
        Declared or detected native genre of the query track.
    candidate_genres : Dict[str, Optional[str]]
        Mapping of candidate track_id -> genre string (or None if unclassified).

    Returns
    -------
    GenreMapResult
        Structured representation of genre distribution, cross-genre flag, and matched genres.
    """
    total = len(candidate_genres)
    if total == 0:
        return GenreMapResult(
            distribution=[],
            is_cross_genre=False,
            native_genre=query_genre,
            matched_genres=[],
        )

    # Count occurrences
    counts: Dict[str, int] = {}
    for cand_id, g in candidate_genres.items():
        genre_name = str(g).strip().title() if g and str(g).strip() else "Unknown"
        counts[genre_name] = counts.get(genre_name, 0) + 1

    distribution: List[GenreContribution] = []
    # Sort descending by count
    for genre_name, count in sorted(counts.items(), key=lambda x: x[1], reverse=True):
        pct = float((count / total) * 100.0)
        distribution.append(GenreContribution(
            genre=genre_name,
            count=count,
            percentage=round(pct, 2)
        ))

    norm_query_genre = str(query_genre).strip().title() if query_genre and str(query_genre).strip() else None

    # Check cross-genre presence: any non-unknown genre different from query genre
    is_cross_genre = False
    if norm_query_genre:
        for g_contrib in distribution:
            if g_contrib.genre != "Unknown" and g_contrib.genre.lower() != norm_query_genre.lower():
                is_cross_genre = True
                break
    else:
        # If query genre is unspecified but candidates have multiple genres
        known_genres = [g.genre for g in distribution if g.genre != "Unknown"]
        if len(known_genres) > 1:
            is_cross_genre = True

    matched_genres = [g.genre for g in distribution]

    return GenreMapResult(
        distribution=distribution,
        is_cross_genre=is_cross_genre,
        native_genre=norm_query_genre,
        matched_genres=matched_genres,
    )
