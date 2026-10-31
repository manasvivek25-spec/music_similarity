file_path = "tests/test_full_dataset_scan.py"
with open(file_path, "r", encoding="utf-8") as f:
    text = f.read()

# Replace both occurrences of with patch("src.remote_dataset.load_remote_index", return_value=idx_data):
target = '        with patch("src.remote_dataset.load_remote_index", return_value=idx_data):'
replacement = '        with patch("src.pipeline.NULL_DIST_CACHE_FILE", temp_dir / "mock_null.json"), patch("src.remote_dataset.load_remote_index", return_value=idx_data):'

text_norm = text.replace('\r\n', '\n')
target_norm = target.replace('\r\n', '\n')

if target_norm in text_norm:
    count = text_norm.count(target_norm)
    text_patched = text_norm.replace(target_norm, replacement)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(text_patched)
    print(f"Successfully replaced {count} occurrences in tests/test_full_dataset_scan.py!")
else:
    print("Target not found!")

