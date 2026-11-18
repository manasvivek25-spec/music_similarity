import hashlib
import json
from pathlib import Path

files = [
    "models/origin_classifier.joblib",
    "features/remote_index.json",
    "features/null_distribution.json",
    "features/mert_manifest.json",
    "features/dataset_manifest.json",
]

for f in files:
    p = Path(f)
    if p.is_file():
        data = p.read_bytes()
        h = hashlib.sha256(data).hexdigest()
        print(f"{f}: size={len(data)}, sha256={h}")
    else:
        print(f"{f}: NOT FOUND")

