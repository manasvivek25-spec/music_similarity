with open("src/remote_dataset.py", "rb") as f:
    content = f.read().decode("utf-8")

# 1. Update imports
old_import = """from src.config import (
    COSINE_PREFILTER_THRESHOLD,
    DEFAULT_REMOTE_DATASET,
    DEFAULT_REMOTE_SAMPLE_SIZE,
    DEFAULT_REMOTE_SPLIT,
    FEATURES_DIR,
    REMOTE_INDEX_FILE,
    SAMPLE_RATE,
    SIMILARITY_FLOOR,
)"""

new_import = """from src.config import (
    COSINE_PREFILTER_THRESHOLD,
    DEFAULT_REMOTE_DATASET,
    DEFAULT_REMOTE_SAMPLE_SIZE,
    DEFAULT_REMOTE_SPLIT,
    FEATURES_DIR,
    REMOTE_INDEX_FILE,
    SAMPLE_RATE,
    SIMILARITY_FLOOR,
    LegacyArtifactError,
    LEGACY_ARTIFACT_HASHES,
    is_fresh_production_mode_active,
    is_legacy_mode_active,
    get_system_mode,
)"""

content = content.replace(old_import.replace("\n", "\r\n"), new_import.replace("\n", "\r\n"))
if new_import not in content:
    content = content.replace(old_import, new_import)

# 2. Update load_remote_index and add validate_feature_cache_path
old_load = """def load_remote_index(index_file: Optional[Path] = None) -> Dict[str, Any]:
    \"\"\"
    Load existing precomputed remote index from disk.
    Normalizes legacy origin labels and ensures presence of status metadata.
    \"\"\"
    path = Path(index_file) if index_file is not None else REMOTE_INDEX_FILE
    if not path.is_file():
        return {
            \"version\": \"1.0\",
            \"dataset_name\": DEFAULT_REMOTE_DATASET,
            \"split\": DEFAULT_REMOTE_SPLIT,
            \"created_at\": None,
            \"updated_at\": None,
            \"total_indexed\": 0,
            \"records_scanned\": 0,
            \"duplicate_records\": 0,
            \"scan_status\": \"NOT_STARTED\",
            \"is_complete\": False,
            \"unresolved_records\": {},
            \"tracks\": {},
        }
    try:
        with open(path, \"r\", encoding=\"utf-8\") as f:
            data = json.load(f)"""

new_load = """def validate_feature_cache_path(
    feature_path: Union[str, Path],
    allow_legacy: bool = False,
) -> Path:
    \"\"\"
    Validate feature/embedding cache path against legacy 261-corpus isolation rules.
    \"\"\"
    p = Path(feature_path)
    if not allow_legacy and is_fresh_production_mode_active():
        posix_path = p.as_posix()
        if "features/mert/" in posix_path or ("features/" in posix_path and "_beat.npy" in posix_path):
            raise LegacyArtifactError(
                f"Legacy feature cache '{p.name}' from retired 261 corpus cannot be loaded in fresh-production mode."
            )
    return p


def load_remote_index(
    index_file: Optional[Path] = None,
    allow_legacy: bool = False,
    strict: bool = False,
) -> Dict[str, Any]:
    \"\"\"
    Load precomputed remote index from disk.

    In fresh-production mode:
    - Strictly isolates and rejects legacy 261-record corpus index (features/remote_index.json).
    - If strict=True, raises LegacyArtifactError.
    - If strict=False, logs warning and returns an unconfigured fresh production skeleton.
    - Newly generated indexes with valid fresh manifests load successfully.

    In legacy mode (MUSIC_SIMILARITY_MODE="legacy" or allow_legacy=True):
    - Loads historical 261-record index for benchmark evaluation.
    \"\"\"
    path = Path(index_file) if index_file is not None else REMOTE_INDEX_FILE
    if not path.is_file():
        return {
            \"version\": \"1.0\",
            \"dataset_name\": DEFAULT_REMOTE_DATASET,
            \"split\": DEFAULT_REMOTE_SPLIT,
            \"created_at\": None,
            \"updated_at\": None,
            \"total_indexed\": 0,
            \"records_scanned\": 0,
            \"duplicate_records\": 0,
            \"scan_status\": \"NOT_STARTED\",
            \"is_complete\": False,
            \"unresolved_records\": {},
            \"tracks\": {},
        }

    # Detect if this path is the legacy 261-record corpus index
    is_legacy_index = False
    if path.resolve() == REMOTE_INDEX_FILE.resolve():
        is_legacy_index = True
    else:
        try:
            import hashlib
            h = hashlib.sha256(path.read_bytes()).hexdigest()
            if h == LEGACY_ARTIFACT_HASHES.get("remote_index"):
                is_legacy_index = True
        except Exception:
            pass

    if is_legacy_index:
        if not allow_legacy and is_fresh_production_mode_active():
            msg = (
                f"Legacy 261-record corpus index ({path.name}) cannot be loaded in fresh-production mode "
                f"(mode='{get_system_mode()}'). Explicit mode='legacy' or allow_legacy=True is required."
            )
            if strict:
                raise LegacyArtifactError(msg)
            logger.warning(f"REJECTED LEGACY ARTIFACT: {msg}")
            return {
                \"version\": \"1.0\",
                \"dataset_name\": \"fresh_production_unconfigured\",
                \"split\": DEFAULT_REMOTE_SPLIT,
                \"created_at\": None,
                \"updated_at\": None,
                \"total_indexed\": 0,
                \"records_scanned\": 0,
                \"duplicate_records\": 0,
                \"scan_status\": \"UNCONFIGURED_FRESH_PRODUCTION\",
                \"is_complete\": False,
                \"unresolved_records\": {},
                \"tracks\": {},
            }

    try:
        with open(path, \"r\", encoding=\"utf-8\") as f:
            data = json.load(f)"""

old_load_crlf = old_load.replace("\r\n", "\n").replace("\n", "\r\n")
old_load_lf = old_load.replace("\r\n", "\n")

if old_load_crlf in content:
    content = content.replace(old_load_crlf, new_load.replace("\n", "\r\n"))
    print("Replaced load_remote_index CRLF")
elif old_load_lf in content:
    content = content.replace(old_load_lf, new_load)
    print("Replaced load_remote_index LF")
else:
    print("old_load not found!")

with open("src/remote_dataset.py", "wb") as f:
    f.write(content.encode("utf-8"))
print("Done patching src/remote_dataset.py")

