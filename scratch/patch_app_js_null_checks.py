from pathlib import Path

target_file = Path("frontend/app.js")
text = target_file.read_text(encoding="utf-8")
norm_text = text.replace("\r\n", "\n")

old_cr = '    if (broadEl) broadEl.innerText = cr.broad_score_percent !== null ? cr.broad_score_percent.toFixed(2) : "\u2014";\n\n    if (tierBadgeEl) tierBadgeEl.innerText = cr.is_available ? "Score Ready" : "Unavailable";\n\n    if (noveltyTierEl) noveltyTierEl.innerText = cr.score_tier;\n\n    if (narrowEl) narrowEl.innerText = `Narrow: ${cr.narrow_score_percent !== null ? cr.narrow_score_percent.toFixed(2) + "%" : "N/A"}`;\n\n    if (normEl) normEl.innerText = `Norm: ${cr.broad_score_normalized !== null ? cr.broad_score_normalized.toFixed(4) : "\u2014"}`;'

new_cr = '    if (broadEl) broadEl.innerText = (cr.broad_score_percent != null) ? cr.broad_score_percent.toFixed(2) : "\u2014";\n\n    if (tierBadgeEl) tierBadgeEl.innerText = cr.is_available ? "Score Ready" : "Unavailable";\n\n    if (noveltyTierEl) noveltyTierEl.innerText = cr.score_tier || cr.novelty_tier || "\u2014";\n\n    if (narrowEl) narrowEl.innerText = `Narrow: ${(cr.narrow_score_percent != null) ? cr.narrow_score_percent.toFixed(2) + "%" : "N/A"}`;\n\n    if (normEl) normEl.innerText = `Norm: ${(cr.broad_score_normalized != null) ? cr.broad_score_normalized.toFixed(4) : "\u2014"}`;'

old_oc = '    if (ocConfEl) ocConfEl.innerText = oc.confidence !== null ? `Confidence: ${(oc.confidence * 100).toFixed(1)}%` : oc.model_status;\n\n    if (probAiEl) probAiEl.innerText = oc.probability_ai !== null ? `AI Prob: ${(oc.probability_ai * 100).toFixed(1)}%` : "AI Prob: \u2014";'

new_oc = '    if (ocConfEl) ocConfEl.innerText = (oc.confidence != null) ? `Confidence: ${(oc.confidence * 100).toFixed(1)}%` : (oc.confidence_label || oc.model_status || "\u2014");\n\n    if (probAiEl) probAiEl.innerText = (oc.probability_ai != null) ? `AI Prob: ${(oc.probability_ai * 100).toFixed(1)}%` : ((oc.prob_ai != null) ? `AI Prob: ${(oc.prob_ai * 100).toFixed(1)}%` : "AI Prob: \u2014");'

assert old_cr in norm_text
assert old_oc in norm_text

norm_text = norm_text.replace(old_cr, new_cr, 1)
norm_text = norm_text.replace(old_oc, new_oc, 1)

target_file.write_text(norm_text.replace("\n", "\r\n"), encoding="utf-8")
print("SUCCESS: null checks patched cleanly.")
