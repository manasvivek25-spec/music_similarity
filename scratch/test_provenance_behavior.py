import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.origin_classifier import predict_origin

audio_file = "temp_uploads/short_clip_7s.wav"

# 1. Without env var (pure fresh production mode, no test fixture)
if "MUSIC_ORIGIN_TEST_FIXTURE_PATH" in os.environ:
    del os.environ["MUSIC_ORIGIN_TEST_FIXTURE_PATH"]
res_unconfigured = predict_origin(audio_file)
print("1. Without test fixture (Pure Fresh Production):")
print(f"   predicted_origin: {res_unconfigured.predicted_origin}")
print(f"   model_status:     {res_unconfigured.model_status}")
print(f"   probability_ai:   {res_unconfigured.probability_ai}")

# 2. With test fixture
os.environ["MUSIC_ORIGIN_TEST_FIXTURE_PATH"] = "models/test_fixture_origin_classifier.joblib"
res_fixture = predict_origin(audio_file)
print("\n2. With test fixture:")
print(f"   predicted_origin: {res_fixture.predicted_origin}")
print(f"   model_status:     {res_fixture.model_status}")
print(f"   probability_ai:   {res_fixture.probability_ai}")
print(f"   confidence:       {res_fixture.confidence}")
print(f"   model_version:    {res_fixture.model_version}")

