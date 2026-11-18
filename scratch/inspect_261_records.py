import json
from pathlib import Path

with open("features/remote_index.json", "r", encoding="utf-8") as f:
    remote_idx = json.load(f)
tracks = remote_idx.get("tracks", {})

with open("features/mert_manifest.json", "r", encoding="utf-8") as f:
    mert_manifest = json.load(f)

mert_embeds = mert_manifest.get("embeddings", {})
skipped = mert_manifest.get("skipped", [])

print(f"Total tracks in remote_index: {len(tracks)}")
print(f"MERT manifest total embeddings: {len(mert_embeds)}")
print(f"MERT manifest skipped count: {len(skipped)}")
print(f"MERT manifest skipped details: {skipped}")

missing = [tid for tid in tracks.keys() if tid not in mert_embeds]
print(f"\nTracks in remote_index not in mert_manifest.embeddings ({len(missing)}):")
for tid in missing:
    tinfo = tracks[tid]
    local_p = tinfo.get("local_audio_path")
    exists = Path(local_p).is_file() if local_p else False
    print(f" - ID: {tid}")
    print(f"   origin: {tinfo.get('origin_label')}")
    print(f"   source: {tinfo.get('source')}")
    print(f"   local_path: {local_p} (exists: {exists})")
    print(f"   metadata_only: {tinfo.get('metadata_only')}")
    print(f"   has_audio: {tinfo.get('has_audio')}")

