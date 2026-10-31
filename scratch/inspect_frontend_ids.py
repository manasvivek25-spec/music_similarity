import re
from pathlib import Path

text = Path("frontend/index.html").read_text(encoding="utf-8")
ids = re.findall(r'id=["\']([^"\']+)["\']', text)
print(f"Total element IDs: {len(ids)}")
for i in sorted(set(ids)):
    print(f" - {i}")

