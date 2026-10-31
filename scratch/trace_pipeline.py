import sys, os, logging, time
from pathlib import Path

# Add repo root
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

from src.pipeline import run_pipeline

wav_file = REPO_ROOT / "temp_uploads" / "short_clip_7s.wav"
print(f"Testing pipeline on {wav_file}...", flush=True)

def progress_cb(stage, stage_name, pct, cur, tot, msg):
    print(f"Stage {stage} ({stage_name}) {pct}%: {msg}", flush=True)

t0 = time.time()
res = run_pipeline(
    query_audio_path=wav_file,
    query_metadata={"title": "Short Clip Test"},
    progress_callback=progress_cb
)
print(f"Pipeline complete in {time.time()-t0:.2f}s! Final verdict: {res.final_verdict.three_indicator_verdict}", flush=True)

