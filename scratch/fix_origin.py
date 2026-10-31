from pathlib import Path

path = Path("src/origin_classifier.py")
text = path.read_text(encoding="utf-8")
target = '    status_str = result.get("status", "unconfigured")'
replacement = '    group_eval_section = ""\n    status_str = result.get("status", "unconfigured")'
if target in text:
    text = text.replace(target, replacement, 1)
    path.write_text(text, encoding="utf-8")
    print("Fixed initialization.")
else:
    print("Target not found.")

