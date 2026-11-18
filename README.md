# Statistical Analysis of Melodic Similarity and Creative Originality in Human- and AI-Generated Music

**Platform Version:** 2.0.0 (`fresh_production` default)  
**License:** MIT

An end-to-end research platform and interactive diagnostic suite that measures harmonic/melodic similarity against a balanced, bias-controlled real-audio reference collection, evaluates statistical unusualness against an empirical human null distribution, computes population-relative creative novelty, independently estimates Human-vs-AI origin (strictly decoupled from similarity), extracts musical context, evaluates structural mimic evidence, localizes matching timeline segments with synchronized playback, and presents run-derived offline evaluation and dataset audits.

> **Research & Scope Disclaimer:**  
> System parameters focus exclusively on harmonic and melodic pitch space. Rhythmic-only mimicry, timbral similarity, production style, and lyrical content fall outside active analytical boundaries. Results indicate statistical similarity — not legal plagiarism, copyright infringement, or AI training-data memorization.

---

## Quickstart

See **[RUN.md](RUN.md)** for the full one-command runbook and **[FINAL_REPORT.md](FINAL_REPORT.md)** for the complete technical delivery report.

```powershell
# 1. Start the FastAPI backend & interactive frontend
.\venv\Scripts\uvicorn src.api:app --host 127.0.0.1 --port 8001

# 2. Open the dashboard in your browser
#    http://127.0.0.1:8001/         (Analysis Dashboard + Evaluation & Dataset Audit Tab)
#    http://127.0.0.1:8001/docs     (OpenAPI Interactive Docs)

# 3. Run the automated test suite (156 unit & integration tests)
.\venv\Scripts\pytest -q

# 4. Re-run the real held-out offline evaluation suite (Features 1-7 + PNG figures)
.\venv\Scripts\python -u -m evaluation.run_evaluation
```

---

## 1. Seven-Feature Pipeline Architecture

Core dependency chain: **F1 (Similarity) → F2 (Statistical Unusualness) → F3 (Creative Novelty) → F6 (Structural Mimic Evidence) → F7 (Timeline Localization & Deterministic Suggestions)**.  
**F4 (Human vs. AI Origin)** is **strictly independent** — AI origin is never inferred from similarity.

| Feature | Role | Key Methods & Safeguards |
| :--- | :--- | :--- |
| **F1: Similarity Engine** | Key-invariant harmonic/melodic alignment | 22,050 Hz mono, beat-synchronized 12-bin Chroma CQT, Stage-1 mean-chroma cosine prefilter (≥ 0.30), 12-shift FastDTW + ±1 shift boundary verification, $S = 1 / (1 + d_{\text{norm}})$, floor 0.45. |
| **F2: Statistical Gateway** | Tests whether similarity is statistically unusual | Unsmoothed empirical $p = k/N$ (`MIN_NULL_PAIRS = 5`), operational confidence tiers ($N \ge 1000$ / $300$ / $5$), cutoffs $0.10$ / $0.01$, cap 15, genre fallback below 30 pairs, Benjamini-Hochberg FDR reported separately, research-only smoothed $(k+1)/(N+1)$. |
| **F3: Creative Novelty** | Population-relative pitch-space distinctiveness | $(1 - \Phi(z)) \times 100$ computed against full eligible human reference population (not survivors alone); Case A cross-genre scores; explicitly framed as a statistical proxy, not an aesthetic judgment. |
| **F4: Origin Classifier** | Independent Human vs. AI origin estimate | Combined 824-dim representation (56 acoustic + 768 MERT-v1-95M) + tuned `LogisticRegression(C=0.01)`, threshold 0.50 locked on validation. Scoped strictly to represented generators (`MusicSet`, `SunoCaps`, `Udio`). |
| **F5: Musical Context** | Key, tempo & acoustic descriptors | Krumhansl-Schmuckler 24-profile key, librosa tempo/beat tracking, HPSS harmonic/percussive ratios, spectral rolloff, rhythm regularity. Explicitly badged `Estimated (Unverified)` when not backed by curated metadata. |
| **F6: Mimic Evidence** | Coherent structural overlap & candidate ranking | Reuses F1's DTW path (zero recomputation) to compute `alignment_coverage`, `significant_region_count`, `region_continuity`, `sequential_consistency`, `local_similarity_concentration`, and `mimic_evidence_level` (`High` / `Moderate` / `Low` / `Not applicable`). Same-artist matches suppressed (`Consistent Artist Style`); suppressed in Case C/U. See [`docs/F6_mimic_evidence.md`](docs/F6_mimic_evidence.md). |
| **F7: Timeline & Suggestor** | Timestamped matching segments & studio actions | Top-3 regions ranked by severity with click-to-play audio for both uploaded query and matched reference track (`/api/audio/reference/{track_id}`) plus deterministic studio recommendations. |

### Authoritative Case Routing & Three-Indicator Verdict
- **Case A (Cross-Genre):** Qualifying match survives F2 in a verified genre distinct from the query's native genre.
- **Case B (Same-Genre / Undetermined Genre):** Qualifying match survives F2 in the same or undetermined genre.
- **Case C (No Statistically Unusual Match):** Valid null ($N \ge 5$) with no candidate passing F2 (`verdict = clear`, F6 mimic evidence = `Not applicable`).
- **Case U (Unresolved / Insufficient Null):** $N < 5$ or missing empirical $p$-value (`verdict = incomplete`).

Three-indicator verdict gate (`Similarity ≥ 0.85`, `Empirical p ≤ 0.01`, `Normalized Creative Score ≤ 0.24`): `Critical` (3/3), `Warning` (2/3), `Advisory` (1/3), `Clear` (0/3), or `Incomplete` (missing null evidence).

---

## 2. v2.0 Balanced Real-Audio Corpus & Bias Controls

All v2 artifacts live under `data/v2/` and `models/origin_classifier_v2_fresh.joblib`, leaving the 4 frozen legacy 261-record files (`features/remote_index.json`, `features/null_distribution.json`, `features/mert_manifest.json`, `features/dataset_manifest.json`) untouched byte-for-byte (verified via SHA-256).

- **Verified Real Audio Sources:** `benjamin-paine/free-music-archive-small` (CC Human, genuine FMA genre metadata) + `SleepyJesse/ai_music_large` (Human `free-music-archive-full` + 3 distinct AI generators: `MusicSet`, `SunoCaps`, `Udio` CC0).
- **Clean Verified Pool:** **2,112 tracks** (1,186 Human / 926 AI; 26 skipped for short/silent audio, 0 decode failures, 7 near-duplicates removed at MERT cosine > 0.98 or chroma > 0.9999; 0.0% missing MERT embeddings).
- **Identical Confound-Controlled Normalization:** 22,050 Hz mono, exact center `10.0s` window (`INDEX_MAX_SECONDS = 10.0`), 4th-order zero-phase `7,500 Hz` Butterworth low-pass filter, `RMS = 0.10`.
- **Metadata Confound Probes (5-fold StratifiedGroupKFold):**
  - Pre-normalization metadata (`orig_dur`, `orig_sr`, `orig_ch`, `orig_bytes`, `raw_silence`): **LR AUC = 0.9912 / RF AUC = 1.0000**
  - Post-normalization fixed physical features: **LR AUC = 0.4802 / RF AUC = 0.5050**
  - Post-normalization + silence ratio: **LR AUC = 0.4785 / RF AUC = 0.5004** (pure chance)
- **Six Strictly Disjoint Group-Aware Splits (`PASSED_ZERO_LEAKAGE` — 0 track/group/hash overlap across all 15 split pairs):**
  - `reference_index`: 386 (193 H / 193 AI, 270 groups)
  - `null_population`: 200 Human (104 groups → `N = 2,000` global pairs, `mean = 0.49672`, `std = 0.05219`, plus 6 genuine FMA genres with ≥ 30 pairs)
  - `f4_train` / `f4_val` / `f4_test`: 702 / 232 / 232 (50/50 balanced)
  - `held_out`: 300 (150 H / 150 AI, 211 groups)

---

## 3. Real Held-Out Evaluation Summary

Every metric below is computed directly from `evaluation/run_evaluation.py` and saved in [`reports/evaluation_results.json`](reports/evaluation_results.json) and [`reports/evaluation_table.md`](reports/evaluation_table.md):

| Feature | Key Metric (Real Held-Out Audio) | Ours (95% CI) | Baseline | n |
| :--- | :--- | :--- | :--- | :---: |
| **F1: Similarity** | Top-1 / Top-3 Retrieval | **90.00%** [83.33%, 94.19%] / **93.33%** | 52.50% (Cosine) / 57.50% (DTW-0) | 120 |
| **F1: Similarity** | Semitone Shift Recovery (±1..6) | **100.00%** [92.59%, 100.00%] | 0.00% (DTW-0) | 48 |
| **F1: Similarity** | Related vs. Unrelated ROC-AUC | **0.9892** (PR-AUC 0.9840) | 0.5000 (Random) | 480 |
| **F2: Significance** | Observed FPR at α=0.10 / 0.05 / 0.01 | **10.00%** / **4.50%** / **1.50%** | 0.00% (Fixed S≥0.65) | 200 |
| **F2: Significance** | Power (α=0.05) / Null Q-Q R² | **94.17%** / **R² = 0.9792** | 83.33% (Fixed S≥0.65) | 120 |
| **F3: Novelty** | Monotonicity (Spearman ρ) | **ρ = -0.9759** (Guards OK) | 0.0 (Unordered) | 200 |
| **F4: Origin** | Held-Out Balanced Acc / ROC-AUC | **71.00%** / **0.7756** (ECE 0.0357) | 50.00% Maj / 0.5140 Confound | 300 |
| **F4: Origin** | Independent Human Specificity | **72.00%** [64.33%, 78.57%] | 50.00% (Chance) | 150 |
| **F5: Context** | Key & Tempo Extraction | **Estimated, unverified** (100% cov.) | No curated ground truth | 300 |
| **F6: Mimic** | Source in Top-1 / Top-3 | **90.00%** / **93.33%** | 2.78% / 8.33% (Random) | 120 |
| **F7: Timeline** | Spliced Segment IoU / Boundary MAE | **0.3439** / **3.33s** | 0.2727 / 8.00s (Whole-Track) | 12 |

---

## 4. Honest Scope & Known Limitations

1. **Pitch-Space Boundary:** F1/F2/F3/F6/F7 operate strictly on 12-bin beat-synchronized chroma (harmonic/melodic pitch space). Rhythmic-only similarity, vocals/lyrics, and production timbre are out of scope.
2. **F4 Generator Generalization (LOGO):** Leave-One-Generator-Out testing shows partial transfer to unseen `SunoCaps` (76.7% recall) and `Udio` (45.1% recall), but near-zero recall on unseen `MusicSet` (2.9%) when `MusicSet` is excluded from training. F4 is therefore scoped strictly to represented generators.
3. **F5 Unverified Estimates:** Neither FMA-small nor `SleepyJesse/ai_music_large` provides human-verified BPM/key labels; F5 outputs are explicitly badged `Estimated (Unverified)` rather than claiming fabricated accuracy.
4. **Global DTW vs. Partial Splices & Beat-Tracker Octave Sensitivity:** Global DTW aligns full sequences, causing coverage inflation on partial splices and occasional half-/double-tempo beat-tracking variance on sparse ambient tracks.
5. **Pairwise Null Non-Independence:** Null pairs share underlying human tracks; empirical `p = 0.0` reflects finite resolution (`< 1/2000`), and research-only smoothed `(k+1)/(N+1)` is reported alongside unsmoothed `k/N`.
