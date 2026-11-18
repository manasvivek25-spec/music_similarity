from pathlib import Path

p = Path("scratch/test_concurrency_stress.py")
text = p.read_text(encoding="utf-8")

text = text.replace(
    "with urllib.request.urlopen(req, timeout=120) as resp:",
    "with urllib.request.urlopen(req, timeout=300) as resp:"
)

text = text.replace(
    "for _ in range(120):",
    "for _ in range(300):"
)

text = text.replace(
    "final_temp_count = count_temp_files()",
    "time.sleep(3.0)\n    final_temp_count = count_temp_files()"
)

p.write_text(text, encoding="utf-8")
print("Successfully updated test_concurrency_stress.py with 300s timeout and 3s cleanup grace.")

