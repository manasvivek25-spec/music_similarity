with open("src/config.py", "rb") as f:
    content = f.read().decode("utf-8")

target = """# Audio Analysis Parameters
SAMPLE_RATE = int(os.getenv("MUSIC_SAMPLE_RATE", "22050"))
MIN_UPLOAD_DURATION_SEC = float(os.getenv("MUSIC_MIN_UPLOAD_DURATION_SEC", "0.0"))  # 0.0 = unlimited
MAX_UPLOAD_SIZE_BYTES = int(os.getenv("MUSIC_MAX_UPLOAD_SIZE_BYTES", str(100 * 1024 * 1024)))  # 100 MB"""

replacement = """# Audio Analysis Parameters
SAMPLE_RATE = int(os.getenv("MUSIC_SAMPLE_RATE", "22050"))
MIN_UPLOAD_DURATION_SEC = float(os.getenv("MUSIC_MIN_UPLOAD_DURATION_SEC", "0.0"))  # 0.0 = unlimited
MAX_UPLOAD_SIZE_BYTES = int(os.getenv("MUSIC_MAX_UPLOAD_SIZE_BYTES", str(100 * 1024 * 1024)))  # 100 MB

# System Operating Mode & Legacy Isolation Boundaries
class LegacyArtifactError(RuntimeError):
    \"\"\"
    Raised when an attempt is made to load or utilize data-dependent artifacts
    from the retired 261-record corpus in fresh-production mode without explicit authorization.
    \"\"\"
    pass

# Operating Mode: "fresh_production" (default, strict isolation) or "legacy" (historical evaluation)
MUSIC_SIMILARITY_MODE = os.getenv("MUSIC_SIMILARITY_MODE", "fresh_production").strip().lower()

# Known legacy artifact signatures (SHA-256 fingerprints of retired 261-record corpus)
LEGACY_ARTIFACT_HASHES = {
    "origin_classifier": "c52583b591132bf72416506699ce0b0307ae2d85ee2c75e110b454d74150c357",
    "remote_index": "c4990a42a64ff024d9f43fb7a44d1191cdc713a857661b368a98e39a613f85f4",
    "null_distribution": "d80616e58b729f3b1d08aaf604f8b5a2ffe5fb41f4a44ca6050a8c672602948a",
    "mert_manifest": "e3c6da56bca5d764cd9581f10978162d442ce6d4ff446bbf5655d4a649319b0e",
    "dataset_manifest": "b36bf0a2f9f0a01c2dcad4eaec43a4efa9c13c5246db9c88ecbe2e2c6c270a6c",
}

# Approved external pretrained foundation models exempt from legacy isolation
APPROVED_EXTERNAL_ENCODERS = {
    "m-a-p/MERT-v1-95M",
    "m-a-p/MERT-v1-330M",
}

def get_system_mode() -> str:
    \"\"\"Return the current operating mode ('fresh_production' vs 'legacy').\"\"\"
    return os.getenv("MUSIC_SIMILARITY_MODE", "fresh_production").strip().lower()

def is_fresh_production_mode_active() -> bool:
    \"\"\"Return True if running in fresh-production mode (strict legacy artifact rejection).\"\"\"
    return get_system_mode() not in ("legacy", "historical")

def is_legacy_mode_active() -> bool:
    \"\"\"Return True if running in explicitly designated legacy/historical mode.\"\"\"
    return get_system_mode() in ("legacy", "historical")"""

target_crlf = target.replace("\r\n", "\n").replace("\n", "\r\n")
target_lf = target.replace("\r\n", "\n")

if target_crlf in content:
    content = content.replace(target_crlf, replacement.replace("\n", "\r\n"))
    print("Replaced CRLF")
elif target_lf in content:
    content = content.replace(target_lf, replacement)
    print("Replaced LF")
else:
    print("Target not found!")

# Also add system_mode and legacy_isolated to get_runtime_config
runtime_target = """        "llm_enabled": bool(LLM_API_KEY),
    }"""
runtime_replacement = """        "llm_enabled": bool(LLM_API_KEY),
        "system_mode": get_system_mode(),
        "legacy_artifacts_isolated": is_fresh_production_mode_active(),
    }"""

if runtime_target.replace("\r\n", "\n").replace("\n", "\r\n") in content:
    content = content.replace(runtime_target.replace("\r\n", "\n").replace("\n", "\r\n"), runtime_replacement.replace("\n", "\r\n"))
    print("Updated get_runtime_config CRLF")
elif runtime_target in content:
    content = content.replace(runtime_target, runtime_replacement)
    print("Updated get_runtime_config LF")

with open("src/config.py", "wb") as f:
    f.write(content.encode("utf-8"))
print("Done")

