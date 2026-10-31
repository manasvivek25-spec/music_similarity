from huggingface_hub import HfApi
import json

api = HfApi()

print("=== HAIM Subdirectory Audio Inventory ===")
# Let's inspect A2_fake, B_hybrid, C_mixing
for folder in ["A_full_generation/A2_fake", "B_hybrid", "C_mixing", "A1_real_MTG_audio", "A2_fake_audio", "B_hybrid_audio", "C_mixing_audio"]:
    try:
        items = list(api.list_repo_tree("mippia/HAIM", repo_type="dataset", path_in_repo=folder, recursive=False))
        mp3s = [it for it in items if it.path.endswith(".mp3")]
        dirs = [it for it in items if not it.path.endswith(".mp3") and not it.path.endswith(".txt") and not it.path.endswith(".csv")]
        print(f"\n{folder}: {len(items)} items total, {len(mp3s)} direct MP3s, subdirectories: {[d.path for d in dirs[:10]]}")
        
        # If there are subdirectories, inspect their contents
        for d in dirs[:5]:
            sub_items = list(api.list_repo_tree("mippia/HAIM", repo_type="dataset", path_in_repo=d.path, recursive=False))
            sub_mp3s = sum(1 for it in sub_items if it.path.endswith(".mp3"))
            sub_subdirs = [it.path for it in sub_items if not it.path.endswith(".mp3")]
            print(f"  {d.path}: {len(sub_items)} items ({sub_mp3s} MP3s), sub-subdirs: {sub_subdirs[:5]}")
            for sd in [it for it in sub_items if not it.path.endswith(".mp3") and not it.path.endswith(".txt")][:3]:
                ssd_items = list(api.list_repo_tree("mippia/HAIM", repo_type="dataset", path_in_repo=sd.path, recursive=False))
                ssd_mp3s = sum(1 for it in ssd_items if it.path.endswith(".mp3"))
                print(f"    {sd.path}: {len(ssd_items)} items ({ssd_mp3s} MP3s)")
    except Exception as e:
        print(f"Error checking {folder}: {e}")

print("\n=== SONICS Detailed Inventory ===")
try:
    fake_items = list(api.list_repo_tree("awsaf49/sonics", repo_type="dataset", path_in_repo="fake_songs", recursive=False))
    print(f"fake_songs in SONICS: {len(fake_items)} items, first 10: {[it.path for it in fake_items[:10]]}")
    for it in fake_items[:5]:
        sub = list(api.list_repo_tree("awsaf49/sonics", repo_type="dataset", path_in_repo=it.path, recursive=False))
        print(f"  {it.path}: {len(sub)} items, first 3: {[s.path for s in sub[:3]]}")
except Exception as e:
    print(f"Error checking SONICS: {e}")

