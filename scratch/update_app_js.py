from pathlib import Path

target_file = Path("frontend/app.js")
text = target_file.read_text(encoding="utf-8")

norm_text = text.replace("\r\n", "\n")

target1 = "    const topMimic = data.mimic_analysis.top_mimic_candidates[0];"
replacement1 = (
    "    const topCandidates = (data.mimic_analysis?.top_mimic_candidates || data.mimic_analysis?.candidates || []);\n"
    "    const topMimic = topCandidates[0];"
)

target2 = "countBadge.innerText = `${data.mimic_analysis.top_mimic_candidates.length} ${labelSuffix}`;"
replacement2 = "countBadge.innerText = `${(topCandidates || []).length} ${labelSuffix}`;"

target3 = "if (data.mimic_analysis.top_mimic_candidates.length > 0) {"
replacement3 = "if ((topCandidates || []).length > 0) {"

target4 = "data.mimic_analysis.top_mimic_candidates.forEach((cand, idx) => {"
replacement4 = "topCandidates.forEach((cand, idx) => {"

assert target1 in norm_text, "target1 not found"
assert target2 in norm_text, "target2 not found"
assert target3 in norm_text, "target3 not found"
assert target4 in norm_text, "target4 not found"

norm_text = norm_text.replace(target1, replacement1, 1)
norm_text = norm_text.replace(target2, replacement2, 1)
norm_text = norm_text.replace(target3, replacement3, 1)
norm_text = norm_text.replace(target4, replacement4, 1)

target_file.write_text(norm_text.replace("\n", "\r\n"), encoding="utf-8")
print("SUCCESS: app.js updated defensively.")

