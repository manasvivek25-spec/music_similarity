from pathlib import Path

path = Path("tests/test_full_dataset_scan.py")
content = path.read_text(encoding="utf-8")

old = 'with patch("src.pipeline.NULL_DIST_CACHE_FILE", temp_dir / "mock_null.json"), patch("src.remote_dataset.load_remote_index", return_value=idx_data):'
new = 'with patch("src.pipeline.NULL_DIST_CACHE_FILE", temp_dir / "mock_null.json"), patch("src.config.NULL_DIST_CACHE_FILE", temp_dir / "mock_null.json"), patch("src.remote_dataset.load_remote_index", return_value=idx_data):'

count = content.count(old)
print(f"Occurrences found: {count}")
assert count == 2
content = content.replace(old, new)

path.write_text(content, encoding="utf-8")
print("Updated successfully")

