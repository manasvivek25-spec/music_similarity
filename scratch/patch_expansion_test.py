with open("tests/test_full_dataset_expansion.py", "rb") as f:
    content = f.read().decode("utf-8")

old_line = "model = load_origin_classifier()"
new_line = "model = load_origin_classifier(allow_legacy=True)"

content = content.replace(old_line, new_line)

with open("tests/test_full_dataset_expansion.py", "wb") as f:
    f.write(content.encode("utf-8"))
print("Done")

