from pathlib import Path

p = Path("scratch/test_concurrency_stress.py")
lines = p.read_text(encoding="utf-8").splitlines()
for i, line in enumerate(lines):
    if "return len([f for f in temp_dir.iterdir()" in line:
        lines[i] = '    return len([f for f in temp_dir.iterdir() if f.is_file() and f.name.startswith("tmp")])'
        break

p.write_text("\n".join(lines) + "\n", encoding="utf-8")
print("Updated count_temp_files cleanly")

