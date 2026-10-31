import io
import time
import numpy as np
import scipy.io.wavfile as wav
from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)

print("Starting debug test...", flush=True)
sr = 22050
t = np.linspace(0, 5.0, int(sr * 5.0), endpoint=False)
audio = (np.sin(2 * np.pi * 440 * t) * 16000).astype(np.int16)

buf = io.BytesIO()
wav.write(buf, sr, audio)
buf.seek(0)

files = {"query_audio": ("short_track.wav", buf, "audio/wav")}
t0 = time.time()
print("Sending POST /api/analyze...", flush=True)
response = client.post("/api/analyze", files=files)
print(f"Response status: {response.status_code} in {time.time()-t0:.2f}s", flush=True)
if response.status_code != 200:
    print(response.text[:300])
else:
    print("Success! Keys in response:", list(response.json().keys()))

