import json
from pathlib import Path
import numpy as np

mert_dir = Path("features/mert")
all_mert_files = sorted(list(mert_dir.glob("*_mert.npy")))
print(f"Total files in features/mert: {len(all_mert_files)}")

with open("features/remote_index.json", "r", encoding="utf-8") as f:
    idx = json.load(f)

tracks = idx["tracks"]
print(f"Total tracks in remote_index.json: {len(tracks)}")

cached_track_ids = set()
invalid_files = []
test_files = []

for p in all_mert_files:
    tid = p.name.replace("_mert.npy", "")
    if tid in tracks:
        arr = np.load(str(p))
        if arr.shape == (768,) and not np.isnan(arr).any():
            cached_track_ids.add(tid)
        else:
            invalid_files.append((tid, arr.shape))
    elif "test" in tid or "clip" in tid or "synth" in tid or "fixture" in tid:
        test_files.append(tid)
    else:
        invalid_files.append((tid, "not in index"))

print(f"Valid production cached track IDs count: {len(cached_track_ids)}")
print(f"Test files: {test_files}")
print(f"Other/Invalid files: {invalid_files}")

# Detailed breakdown by category in remote_index.json
categories = {}
for tid, info in tracks.items():
    raw_m = info.get("raw_metadata", {})
    source = info.get("source") or raw_m.get("source", "")
    
    if "haim" in tid:
        if "ai" in tid:
            cat = "HAIM AI"
        elif "human" in tid:
            cat = "HAIM Human"
        elif "hybrid" in tid:
            cat = "HAIM Hybrid"
        elif "temporal" in tid:
            cat = "HAIM Temporal Mix"
        else:
            cat = "HAIM Other"
    elif "jamendo" in tid:
        cat = "MTG-Jamendo"
    elif "fma" in tid.lower() or "free-music-archive" in source.lower() or tid.startswith("track_") or tid.startswith("remote_"):
        cat = "FMA"
    else:
        cat = "Other"
        
    categories.setdefault(cat, {"total": 0, "cached": 0, "missing": []})
    categories[cat]["total"] += 1
    if tid in cached_track_ids:
        categories[cat]["cached"] += 1
    else:
        categories[cat]["missing"].append(tid)

print("\nExact Breakdown:")
sum_cached = 0
for cat, data in sorted(categories.items()):
    c_count = data["cached"]
    t_count = data["total"]
    m_count = len(data["missing"])
    print(f"  {cat}: total={t_count}, cached={c_count}, missing={m_count}")
    sum_cached += c_count

print(f"Sum of category cached: {sum_cached}")

# List all missing track IDs
missing_all = []
for cat, data in categories.items():
    missing_all.extend(data["missing"])
print(f"\nTotal Missing Tracks ({len(missing_all)}):")
for m in sorted(missing_all):
    print(f"  - {m}")

