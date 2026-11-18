import joblib
import os
import hashlib
from pathlib import Path

path = Path("models/origin_classifier.joblib")
print(f"File exists: {path.exists()}")
if path.exists():
    data = path.read_bytes()
    print(f"Size: {len(data)} bytes")
    print(f"SHA-256: {hashlib.sha256(data).hexdigest()}")
    m = joblib.load(str(path))
    print(f"Type: {type(m)}")
    print(f"Steps: {getattr(m, 'named_steps', None)}")
    print(f"Classes: {getattr(m, 'classes_', None)}")
    if hasattr(m, 'named_steps') and 'scaler' in m.named_steps:
        scaler = m.named_steps['scaler']
        print(f"Scaler n_features_in_: {getattr(scaler, 'n_features_in_', None)}")
        print(f"Scaler mean_ shape: {getattr(scaler, 'mean_', None).shape if getattr(scaler, 'mean_', None) is not None else None}")

