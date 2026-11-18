import os
import sys
from pathlib import Path
import uvicorn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ["MUSIC_ORIGIN_TEST_FIXTURE_PATH"] = str(Path("models/test_fixture_origin_classifier.joblib").resolve())
os.environ["MUSIC_SIMILARITY_MODE"] = "legacy"

if __name__ == "__main__":
    uvicorn.run("src.api:app", host="127.0.0.1", port=8001, log_level="info")

