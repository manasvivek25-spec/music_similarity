import urllib.request
import json
import time

with open("temp_uploads/short_clip_7s.wav", "rb") as f:
    audio_bytes = f.read()

boundary = "----WebKitFormBoundaryTest123"
body = (
    f"--{boundary}\r\n"
    f'Content-Disposition: form-data; name="query_audio"; filename="short_clip_7s.wav"\r\n'
    f"Content-Type: audio/wav\r\n\r\n"
).encode("utf-8") + audio_bytes + f"\r\n--{boundary}--\r\n".encode("utf-8")

req = urllib.request.Request(
    "http://127.0.0.1:8001/api/analyze",
    data=body,
    headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}
)

t0 = time.time()
with urllib.request.urlopen(req) as resp:
    res = json.loads(resp.read().decode())
t1 = time.time()

print(f"Completed in {t1 - t0:.2f}s")
print("Similarity Score:", res.get("similarity_score"))
print("P-value:", res.get("p_value"))
print("Origin:", res.get("origin_classification"))
print("Creative Score Broad %:", res.get("creative_score", {}).get("broad_score_percent"))
print("Creative Score Tier:", res.get("creative_score", {}).get("score_tier"))
print("Creative Score Normalized:", res.get("creative_score", {}).get("broad_score_normalized"))

