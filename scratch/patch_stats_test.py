with open("src/stats_test.py", "rb") as f:
    content = f.read().decode("utf-8")

# 1. Update imports
old_import = """from src.config import (
    GENRE_MIN_SAMPLE_SIZE,
    MAX_SURVIVORS,
    MIN_NULL_PAIRS,
    NULL_DIST_SAMPLE_PAIRS,
    P_VALUE_BORDERLINE_THRESHOLD,
    P_VALUE_CRITICAL_THRESHOLD,
)"""

new_import = """from src.config import (
    GENRE_MIN_SAMPLE_SIZE,
    MAX_SURVIVORS,
    MIN_NULL_PAIRS,
    NULL_DIST_SAMPLE_PAIRS,
    P_VALUE_BORDERLINE_THRESHOLD,
    P_VALUE_CRITICAL_THRESHOLD,
    NULL_DIST_CACHE_FILE,
    LegacyArtifactError,
    LEGACY_ARTIFACT_HASHES,
    is_fresh_production_mode_active,
    is_legacy_mode_active,
    get_system_mode,
)"""

content = content.replace(old_import.replace("\n", "\r\n"), new_import.replace("\n", "\r\n"))
if new_import not in content:
    content = content.replace(old_import, new_import)

# 2. Add load_null_distribution and update get_or_build_cached_null_distribution
old_get_or_build = """def get_or_build_cached_null_distribution(
    chroma_dict: Dict[str, np.ndarray],
    similarity_func,
    cache_file: Optional[Path] = None,
    max_pairs: int = NULL_DIST_SAMPLE_PAIRS,
    candidate_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
    force_rebuild: bool = False,
) -> Tuple[List[float], Dict[str, List[float]]]:
    \"\"\"
    Load precomputed pairwise null distribution from disk cache, or compute and cache it.

    Avoids repeated pairwise DTW computations across queries. Caches are invalidated
    automatically if source candidate IDs change or force_rebuild=True.

    Parameters
    ----------
    chroma_dict : Dict[str, np.ndarray]
        Mapping of candidate track_id -> beat-synchronized chroma matrix.
    similarity_func : callable
        Function accepting (chroma_a, chroma_b) -> float.
    cache_file : Optional[Path]
        Location to save/load null distribution JSON.
    max_pairs : int
        Maximum pairwise combinations to evaluate.
    candidate_metadata : Optional[Dict[str, Dict[str, Any]]]
        Candidate metadata (for per-genre null distributions).
    force_rebuild : bool
        If True, ignores existing cache and recomputes.

    Returns
    -------
    Tuple[List[float], Dict[str, List[float]]]
        (global_null_scores, genre_null_scores_dict)
    \"\"\"
    import json
    import os
    import time
    from src.config import NULL_DIST_CACHE_FILE

    target_file = cache_file or NULL_DIST_CACHE_FILE

    # Determine baseline tracks: filter strictly to non-fixture human tracks if origin metadata exists
    has_origin = False
    eval_chroma = {}
    for cid, c in chroma_dict.items():
        if is_fixture_identifier(cid):
            continue
        if candidate_metadata:
            meta = candidate_metadata.get(cid, {})
            if meta.get("is_fixture", False):
                continue
            origin = str(meta.get("origin_label", "")).lower()
            if origin in ("human", "0", "false"):
                eval_chroma[cid] = c
        else:
            eval_chroma[cid] = c

    eval_ids = sorted(list(eval_chroma.keys()))

    if not force_rebuild and target_file.is_file():
        try:
            with open(target_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            cached_ids = data.get("source_track_ids", [])
            cached_scores = data.get("null_scores", [])
            # Return cached null distribution if it has cached scores and covers the requested tracks
            if len(cached_scores) > 0:
                if (not eval_ids) or set(eval_ids).issubset(set(cached_ids)) or set(cached_ids) == set(eval_ids):
                    return cached_scores, data.get("genre_null_scores", {})
        except Exception:
            pass  # Fall through to rebuild on parse/read error"""

new_get_or_build = """def load_null_distribution(
    cache_file: Optional[Path] = None,
    allow_legacy: bool = False,
    strict: bool = True,
) -> Dict[str, Any]:
    \"\"\"
    Safely load pairwise null distribution JSON cache from disk.

    In fresh-production mode:
    - Strictly rejects legacy 261-record corpus null distribution (features/null_distribution.json).
    - If strict=True, raises LegacyArtifactError.
    - If strict=False, logs warning and returns empty dict.

    In legacy mode (MUSIC_SIMILARITY_MODE="legacy" or allow_legacy=True):
    - Loads historical 6,105-pair null distribution for benchmark verification.
    \"\"\"
    import json
    import hashlib
    target_file = Path(cache_file) if cache_file is not None else NULL_DIST_CACHE_FILE
    if not target_file.is_file():
        return {}

    is_legacy_dist = False
    if target_file.resolve() == NULL_DIST_CACHE_FILE.resolve():
        is_legacy_dist = True
    else:
        try:
            h = hashlib.sha256(target_file.read_bytes()).hexdigest()
            if h == LEGACY_ARTIFACT_HASHES.get("null_distribution"):
                is_legacy_dist = True
        except Exception:
            pass

    if is_legacy_dist:
        if not allow_legacy and is_fresh_production_mode_active():
            msg = (
                f"Historical null distribution ({target_file.name}) from retired 261-record corpus "
                f"cannot be loaded in fresh-production mode (mode='{get_system_mode()}'). "
                "Explicit mode='legacy' or allow_legacy=True is strictly required."
            )
            if strict:
                raise LegacyArtifactError(msg)
            return {}

    with open(target_file, "r", encoding="utf-8") as f:
        return json.load(f)


def get_or_build_cached_null_distribution(
    chroma_dict: Dict[str, np.ndarray],
    similarity_func,
    cache_file: Optional[Path] = None,
    max_pairs: int = NULL_DIST_SAMPLE_PAIRS,
    candidate_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
    force_rebuild: bool = False,
    allow_legacy: bool = False,
    strict: bool = False,
) -> Tuple[List[float], Dict[str, List[float]]]:
    \"\"\"
    Load precomputed pairwise null distribution from disk cache, or compute and cache it.

    In fresh-production mode:
    - Strictly isolates and rejects legacy 261-record null distribution cache (6105 pairs).
    - If strict=True, raises LegacyArtifactError on legacy cache.
    - If strict=False, computes null distribution freshly from provided candidates.

    In legacy mode (MUSIC_SIMILARITY_MODE="legacy" or allow_legacy=True):
    - Loads historical cached null distribution.
    \"\"\"
    import json
    import os
    import time
    import hashlib

    target_file = Path(cache_file) if cache_file is not None else NULL_DIST_CACHE_FILE

    # Determine baseline tracks: filter strictly to non-fixture human tracks if origin metadata exists
    has_origin = False
    eval_chroma = {}
    for cid, c in chroma_dict.items():
        if is_fixture_identifier(cid):
            continue
        if candidate_metadata:
            meta = candidate_metadata.get(cid, {})
            if meta.get("is_fixture", False):
                continue
            origin = str(meta.get("origin_label", "")).lower()
            if origin in ("human", "0", "false"):
                eval_chroma[cid] = c
        else:
            eval_chroma[cid] = c

    eval_ids = sorted(list(eval_chroma.keys()))

    is_legacy_dist = False
    if target_file.resolve() == NULL_DIST_CACHE_FILE.resolve():
        is_legacy_dist = True
    else:
        try:
            h = hashlib.sha256(target_file.read_bytes()).hexdigest()
            if h == LEGACY_ARTIFACT_HASHES.get("null_distribution"):
                is_legacy_dist = True
        except Exception:
            pass

    if is_legacy_dist and not allow_legacy and is_fresh_production_mode_active():
        if strict:
            raise LegacyArtifactError(
                f"Historical null distribution ({target_file.name}) from retired 261-record corpus "
                f"cannot be loaded in fresh-production mode. Explicit mode='legacy' or allow_legacy=True required."
            )
        can_use_cache = False
    else:
        can_use_cache = not force_rebuild and target_file.is_file()

    if can_use_cache:
        try:
            with open(target_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            cached_ids = data.get("source_track_ids", [])
            cached_scores = data.get("null_scores", [])
            # Return cached null distribution if it has cached scores and covers the requested tracks
            if len(cached_scores) > 0:
                if (not eval_ids) or set(eval_ids).issubset(set(cached_ids)) or set(cached_ids) == set(eval_ids):
                    return cached_scores, data.get("genre_null_scores", {})
        except Exception:
            pass  # Fall through to rebuild on parse/read error"""

old_crlf = old_get_or_build.replace("\r\n", "\n").replace("\n", "\r\n")
old_lf = old_get_or_build.replace("\r\n", "\n")

if old_crlf in content:
    content = content.replace(old_crlf, new_get_or_build.replace("\n", "\r\n"))
    print("Replaced get_or_build CRLF")
elif old_lf in content:
    content = content.replace(old_lf, new_get_or_build)
    print("Replaced get_or_build LF")
else:
    print("old_get_or_build not found!")

with open("src/stats_test.py", "wb") as f:
    f.write(content.encode("utf-8"))
print("Done patching src/stats_test.py")

