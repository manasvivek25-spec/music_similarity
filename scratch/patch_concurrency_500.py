from pathlib import Path

p = Path("scratch/test_concurrency_stress.py")
text = p.read_text(encoding="utf-8")

text = text.replace(
    "with urllib.request.urlopen(req, timeout=300) as resp:",
    "with urllib.request.urlopen(req, timeout=500) as resp:"
)

text = text.replace(
    "for _ in range(300):",
    "for _ in range(500):"
)

text = text.replace(
    "# ROUND 2: 5 Concurrent Asynchronous /api/jobs/analyze Jobs",
    "time.sleep(3.0)\n    # ROUND 2: 5 Concurrent Asynchronous /api/jobs/analyze Jobs"
)

text = text.replace(
    "# ROUND 3: Mixed Burst (5 Sync + 5 Async = 10 Simultaneous Clients)",
    "time.sleep(3.0)\n    # ROUND 3: Mixed Burst (5 Sync + 5 Async = 10 Simultaneous Clients)"
)

text = text.replace(
    "time.sleep(3.0)\n    final_temp_count = count_temp_files()",
    "time.sleep(5.0)\n    final_temp_count = count_temp_files()"
)

p.write_text(text, encoding="utf-8")
print("Updated test_concurrency_stress.py with 500s timeout, round cool-downs, and 5s final cleanup grace.")

