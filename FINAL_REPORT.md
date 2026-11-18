# Final Technical Delivery Report — Music Similarity & Creative Originality Platform (v2.0)

**Project Title:** *Statistical Analysis of Melodic Similarity and Creative Originality in Human- and AI-Generated Music*  
**Delivery Branch:** `final-delivery`  
**Delivery Date:** 2026-11-18 (IST)  
**Scope Disclaimer:** *System parameters focus exclusively on harmonic and melodic pitch space. Rhythmic-only mimicry, timbral similarity, production style, and lyrical content fall outside active analytical boundaries. Results indicate statistical similarity — not legal plagiarism, copyright infringement, or AI training-data memorization.*

---

## 1. Executive Summary & Phase-by-Phase Delivery

The platform was completed end-to-end across Phases P0 through P10 on branch `final-delivery` without touching `main` or modifying any of the 4 frozen legacy 261-record benchmark files.

| Phase | Scope Delivered | Key Artifacts | Elapsed / Timing |
| :---: | :--- | :--- | :--- |
| **P0** | Audit, branch `final-delivery`, hardware/GPU check, backup of 4 frozen legacy files, baseline `pytest` (146/146) | `ENVIRONMENT.md`, `STATE.json`, `PROGRESS.md`, `DECISIONS.md`, `BLOCKERS.md` | ~4 min (19:01–19:05 IST) |
| **P1** | Dataset schema/license/polarity probing & acquisition of 10 verified real-audio shards (2,145 rows) | `evaluation/probe_datasets_p1.py`, `evaluation/download_v2_shards.py`, `data/v2/downloaded_shards.json` | ~11 min (19:05–19:16 IST) |
| **P3** | Resumable v2 indexer with identical confound-controlled normalization, `.npy` caches, and RTX 3050 CUDA FP16 MERT-v1-95M extraction (1,314.5 tracks/min; 0% missing) | `src/v2_indexer.py`, `data/v2/index_state.json` | ~13 min (19:16–19:29 IST) |
| **P2** | Near-duplicate removal, pre- vs. post-normalization confound probes, 6 strictly disjoint group-aware splits, zero-leakage audit | `src/v2_cleaning_audit.py`, `reports/dataset_audit.md`, `reports/dataset_audit.json`, `reports/label_migration_report.json` | ~4 min (19:29–19:33 IST) |
| **P4** | Empirical null distribution rebuild (`N = 2,000` global pairs + 6 genuine FMA genres ≥ 30 pairs) with strict cache invalidation key | `src/v2_null_builder.py`, `data/v2/v2_null_distribution.json`, `src/stats_test.py` | ~3 min (19:33–19:36 IST) |
| **P5** | F4 origin classifier 5-fold group-stratified CV + RandomizedSearch across 3 feature sets × 5 model families, validation calibration/threshold lock, LOGO & 8-perturbation robustness stress test | `src/v2_f4_trainer.py`, `models/origin_classifier_v2_fresh.joblib`, `models/origin_classifier_v2_metrics.json` | ~13 min (19:36–19:49 IST) |
| **P6** | F6 structural mimic evidence (`docs/F6_mimic_evidence.md`), Case C verdict bug fix, v2 pipeline & API endpoints, content-hash Truth Panel lookup, new unit tests | `src/mimic.py`, `src/pipeline.py`, `src/api.py`, `src/truth_lookup.py`, `tests/test_v2_engineering_and_f6.py` | ~7 min (19:49–19:56 IST) |
| **P7** | Replaced synthetic evaluation with real held-out audio evaluation (300 held-out tracks + 120 real controlled derivations), 4 PNG diagnostic figures, plain-language truth table | `evaluation/real_eval.py`, `evaluation/evaluation_manifest.csv`, `reports/evaluation_results.json`, `reports/evaluation_table.md`, `reports/figures/*.png` | ~5 min (19:56–20:01 IST) |
| **P8** | Complete 7-feature UI, Truth Panel, Evaluation & Dataset Audit tab, dark glassmorphism contrast polish, synchronized query + reference playback, Headless Chrome CDP verification (0 errors) | `frontend/index.html`, `frontend/app.js`, `frontend/style.css`, `evaluation/verify_ui_cdp.py`, `reports/ui_checks/*.png` | ~7 min (20:01–20:08 IST) |
| **P9** | Bug fixes (corrupt WAV 400 handler, Case badge consistency), endpoint smoke, edge-case validation, 24-request concurrency stress test, 50-query consistency sweep, two consecutive clean passes | `evaluation/p9_stress_and_consistency.py`, `reports/p9_verification_report.json` | ~12 min (20:08–20:20 IST) |
| **P10** | Pinned `requirements.txt`, `RUN.md`, `README.md`, `FINAL_REPORT.md`, frozen legacy SHA-256 verification, clean commit & git tag | `FINAL_REPORT.md`, `RUN.md`, `README.md`, `requirements.txt` | ~5 min |

---

## 2. Dataset Provenance, Cleaning, Bias Controls & Leakage Audit

### 2.1 Verified Real Audio Sources & Polarity Verification
Every track's `origin_label` was programmatically verified against its source provenance (`src/v2_indexer.py::verify_label_and_provenance`):
- **Human (`1,186` clean tracks):**
  - `benjamin-paine/free-music-archive-small` shards 0–1 (CC license, genuine 164-class FMA genre & artist metadata): **1,052** clean tracks.
  - `SleepyJesse/ai_music_large` shard 0 (`free-music-archive-full`, `ai_generated=False`): **134** clean tracks.
- **AI (`926` clean tracks across 3 distinct verified generators):**
  - `SleepyJesse/ai_music_large` shards 76–80 (`ManzhenWei/MusicSet`, `ai_generated=True`): **766** clean tracks.
  - `SleepyJesse/ai_music_large` shard 75 (`SunoCaps`, `ai_generated=True`): **48** clean tracks.
  - `SleepyJesse/ai_music_large` shard 111 (`blanchon/udio_dataset`, CC0, `ai_generated=True`): **112** clean tracks.
- **Exclusions & Deduplication:**
  - Total raw shard rows processed: **2,145**
  - Skipped for short/silent audio (`< 9.5s` or effectively silent): **26**
  - Decode failures: **0** (`0.0%`)
  - Near-duplicates pruned (`MERT cosine > 0.98` or `chroma cosine > 0.9999`): **7**
  - Polarity conflicts / unknown origin excluded: **0**
  - Clean verified pool: **2,112 tracks** (`0.0%` missing MERT, `0.05%` missing tempo, `50.19%` missing genre — kept honestly as `"Unknown"`, never invented).

### 2.2 Confound-Controlled Normalization & Confound Probes
Raw MusicSet tracks are 10.00s FLAC at mixed 16 kHz / 44.1 kHz whereas FMA/Suno/Udio are 30s+ at 44.1 kHz. Applying identical normalization to every track (**22,050 Hz mono, exact center `10.0s` window, 4th-order zero-phase `7,500 Hz` Butterworth low-pass filter, `RMS = 0.10`**) completely eliminated container/duration/bandwidth confounds (`reports/dataset_audit.json`):

| Confound Probe Feature Set (5-Fold Group CV, n = 2,112) | Logistic Regression ROC-AUC | Random Forest ROC-AUC | Interpretation |
| :--- | :---: | :---: | :--- |
| Pre-normalization metadata (`orig_dur`, `orig_sr`, `orig_ch`, `orig_bytes`, `raw_silence`) | **0.9912** | **1.0000** | Massive source/codec confound before normalization |
| Post-normalization fixed physical (`norm_dur`, `norm_sr`, `norm_ch`, `norm_rms`, `peak`, `dc`) | **0.4802** | **0.5050** | Indistinguishable from random chance (0.50) |
| Post-normalization + silence ratio | **0.4785** | **0.5004** | Confounds neutralized |

### 2.3 Six Strictly Disjoint Group-Aware Splits (`PASSED_ZERO_LEAKAGE`)
Verified **0 track_id overlaps**, **0 artist/cluster group_id overlaps**, and **0 SHA-256 content-hash overlaps** across all 15 pairwise combinations of splits:

| Split | Total | Human | AI | Unique Groups | Purpose |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `reference_index` (`data/v2/v2_index.json`) | **386** | 193 | 193 | 270 | Active F1/F6/F7 search corpus |
| `null_population` (`data/v2/v2_null_population.json`) | **200** | 200 | 0 | 104 | Disjoint human null pool (`N = 2,000` global pairs, `mean = 0.49672`, `std = 0.05219`, 6 FMA genres ≥ 30 pairs) |
| `f4_train` | **702** | 351 | 351 | 519 | F4 CV & training |
| `f4_val` | **232** | 116 | 116 | 174 | F4 model, calibration & threshold selection |
| `f4_test` | **232** | 116 | 116 | 155 | Untouched F4 internal test |
| `held_out` (`data/v2/v2_held_out.json`) | **300** | 150 | 150 | 211 | Untouched offline evaluation & Truth Panel |

---

## 3. Feature 4 Origin Classifier — Tuning, Calibration, LOGO & Robustness

All hyperparameter searches (5-fold group-stratified CV across `acoustic_56`, `mert_768`, and `combined_824` × 5 model families: `LogisticRegression`, `LinearSVC`, `RBF-SVM`, `RandomForest`, `HistGradientBoosting`), probability calibration checks (`uncalibrated`, `sigmoid`, `isotonic`), and decision-threshold selection were performed strictly on `f4_train` + `f4_val` (`models/origin_classifier_v2_metrics.json`):

- **Selected Model:** `combined_824` (56 acoustic + 768 MERT-v1-95M) + `LogisticRegression(C=0.01)`, `calibration_method = uncalibrated` (lowest validation Brier `0.2176`), `decision_threshold = 0.50` (`val_roc_auc = 0.7235`, `val_bal_acc = 0.6810`).
- **Untouched `f4_test` (n = 232):** ROC-AUC **0.7398** `[0.6529, 0.8140]`, PR-AUC **0.7251**, Balanced Acc **66.38%** `[59.37%, 73.40%]`, Human Specificity **64.66%**, ECE **0.1052**, Brier **0.2123**.
- **Untouched `held_out` (n = 300: 150 H / 150 AI):**
  - ROC-AUC: **0.7756** `[0.6993, 0.8580]` (vs. Metadata Confound Probe `0.5140`, LR acoustic-only baseline `0.7540`, RF `0.7259`)
  - PR-AUC: **0.7620** `[0.6109, 0.8896]` | F1: **0.7071** `[0.5962, 0.7986]`
  - Balanced Accuracy: **71.00%** `[63.89%, 79.00%]` (McNemar vs. Majority `p < 0.0001`; McNemar vs. Confound Probe `p = 1e-6`)
  - Independent Human Specificity: **72.00%** `[64.33%, 78.57%]` (108 TN, 42 FP) | EER: **29.33%**
  - Calibration: **ECE = 0.0357**, **Brier = 0.1945** (paired bootstrap Brier Δ vs. confound = `-0.0595`, `p < 0.0001`)
  - Per-Generator Held-Out Recall: `MusicSet` **67.74%** (n=124), `SunoCaps` **75.00%** (n=8), `Udio` **83.33%** (n=18)
- **Leave-One-Generator-Out (LOGO) Generalization (Honest Weak Spot):**
  - Unseen `SunoCaps` (n=30): **76.67%** recall `[59.07%, 88.21%]`
  - Unseen `Udio` (n=71): **45.07%** recall `[34.05%, 56.60%]`
  - Unseen `MusicSet` (n=482): **2.90%** recall `[1.74%, 4.82%]` — confirms why F4 is explicitly scoped in the UI and API to represented generator families only.
- **8-Perturbation Robustness Stress Test (n = 40 untouched test subset, Clean ROC-AUC = 0.7850):**
  - `reencode_mp3_128k`: **0.7775** (Δ -0.0075) | `reencode_mp3_64k`: **0.7625** (Δ -0.0225)
  - `resample_16k_roundtrip`: **0.7750** (Δ -0.0100) | `additive_noise_20db`: **0.7825** (Δ -0.0025)
  - `additive_noise_15db`: **0.7800** (Δ -0.0050) | `trim_center_6s`: **0.7375** (Δ -0.0475)
  - `gain_minus_6db`: **0.7825** (Δ -0.0025) | `gain_plus_3db`: **0.7825** (Δ -0.0025)

---

## 4. Master Offline Evaluation Results (Real Held-Out Audio, Seed = 42)

All metrics traced directly to [`reports/evaluation_results.json`](reports/evaluation_results.json) and [`reports/evaluation_table.md`](reports/evaluation_table.md):

| Feature | Metric | Ours (95% CI) | Baseline | n | Truth Source |
| :--- | :--- | :--- | :--- | :---: | :--- |
| Feature 1: Similarity Engine | Top-1 Retrieval Accuracy | **90.00% [83.33%, 94.19%]** | 52.50% (Cosine) / 57.50% (DTW-0) | 120 | Real Held-Out Derived Queries |
| Feature 1: Similarity Engine | Top-3 Retrieval Accuracy | **93.33% [87.39%, 96.58%]** | 56.67% (Mean-Chroma Cosine) | 120 | Real Held-Out Derived Queries |
| Feature 1: Similarity Engine | MAP / MR1 | **0.9218 [0.8980, 0.9476]** | 0.5659 (Cosine) / 0.6296 (DTW-0) | 120 | Real Held-Out Derived Queries |
| Feature 1: Similarity Engine | Shift Recovery (`k in ±1..6`) | **100.00% [92.59%, 100.00%]** | 0.00% (DTW Shift-0 Only) | 48 | Real Transposed Queries (±1..6) |
| Feature 1: Similarity Engine | Related vs. Unrelated ROC-AUC / PR-AUC | **0.9892 / 0.9840** | 0.5000 (Random) | 480 | Real Related vs. Unrelated Pairs |
| Feature 2: Statistical Gateway | Observed FPR (`α=0.10` / `0.05` / `0.01`) | **10.00%** / **4.50%** / **1.50%** | 0.00% (Fixed S≥0.65) | 200 | Real Unrelated Held-Out Pairs |
| Feature 2: Statistical Gateway | Statistical Power (`α=0.05`) | **94.17% [88.45%, 97.15%]** | 83.33% (Fixed S≥0.65) | 120 | Real Derived Related Pairs |
| Feature 2: Statistical Gateway | Null p-value Q-Q R² / KS p-val | **R² = 0.9792 / KS p = 0.0004** | N/A (Fixed Threshold) | 200 | Uniform(0,1) Calibration |
| Feature 3: Creative Novelty | Monotonicity (Spearman ρ) & Guards | **ρ = -0.9759 (Guards OK = True)** | Unordered (ρ = 0.0) | 200 | Disjoint Real Human Population |
| Feature 4: Origin Classifier | Held-Out Balanced Accuracy | **71.00% [63.89%, 79.00%]** | 50.00% Maj / 70.33% LR / 69.33% RF | 300 | Real Held-Out Origin Labels |
| Feature 4: Origin Classifier | Held-Out ROC-AUC / PR-AUC | **0.7756 [0.6993, 0.8580] / 0.7620** | 0.5140 Confound / 0.7540 LR | 300 | Real Held-Out Origin Labels |
| Feature 4: Origin Classifier | Human Specificity / EER | **72.00% [64.33%, 78.57%] / 29.33%** | 50.00% EER (Chance) | 150 | Real Independent Human Set |
| Feature 4: Origin Classifier | Calibration ECE / Brier | **ECE = 0.0357 / Brier = 0.1945** | 0.2500 Brier (Constant 0.5) | 300 | Real Held-Out Origin Labels |
| Feature 5: Musical Context | Key & Tempo Coverage | **Key 100.0% / Tempo 100.0% (Estimated, unverified)** | N/A (No curated ground truth) | 300 | Unverified (No Curated Labels) |
| Feature 6: Candidate Identification | Source in Top-1 / Top-3 | **90.00% [83.33%, 94.19%] / 93.33%** | 2.78% / 8.33% (Random) | 120 | Real Derived Source Provenance |
| Feature 6: Candidate Identification | Coverage Error & High/Mod Rate | **Err = 0.5621 [0.5125, 0.6091] \| 29.2%** | Err = 0.1318 (Naive 100%) | 120 | Real Overlap Ratio & Truth |
| Feature 7: Timeline Localization | Spliced Segment IoU / Boundary MAE | **0.3439 [0.1367, 0.5648] / 3.33s** | 0.2727 / 8.00s (Whole-Track) | 12 | Real Spliced Audio Intervals |

### Plain-Language Summary: How Well Do Our Results Match the Truth?

| Feature | Truth Source | Predicted vs. True Agreement | n | 95% CI | Known Limitations |
| :--- | :--- | :--- | :---: | :--- | :--- |
| F1: Similarity Engine | Controlled real-audio derivations (pitch ±1..6, stretch 0.9/1.1x, noise 15/20dB, excerpt, splice) | **90.00% Top-1 / 93.33% Top-3 (100.00% shift recovery)** | 120 | [83.33%, 94.19%] | Pitch-space only; severe splices/noise lower global DTW similarity vs. pure transpositions. |
| F2: Statistical Significance | 200 unrelated cross-group real human pairs vs. 2,000-pair null | **95.50% specificity at α=0.05 (FPR 4.50%, Power 94.17%)** | 200 | FPR [2.39%, 8.33%] | Pairwise null scores share tracks; finite `N=2000` implies `p=0.0` means `< 0.0005`. |
| F3: Creative Novelty | Disjoint verified human population | **100.0% monotonic ordering (ρ=-0.9759); guards passed** | 200 | KS p=0.0002 | Statistical pitch-space distinctiveness proxy; not an aesthetic creativity judgment. |
| F4: Origin Classifier | Untouched real held-out tracks (Human vs. 3 AI generators) | **71.00% Balanced Acc (ROC-AUC 0.7756, Human Spec. 72.00%)** | 300 | [63.89%, 79.00%] | Scoped to represented generators (`MusicSet`, `SunoCaps`, `Udio`); unseen MusicSet LOGO recall is 2.9%. |
| F5: Musical Context | No curated human-verified BPM/key labels in corpus | **Estimated, unverified (accuracy intentionally unreported)** | 300 | N/A | Requires curated GTZAN/GiantSteps annotations for formal BPM/key accuracy. |
| F6: Mimic Evidence | Known source track & overlap ratio on real derivations | **90.00% Top-1 / 93.33% Top-3 (High/Mod on 29.2%)** | 120 | [83.33%, 94.19%] | Global DTW stretches across unrelated frames on partial splices, raising coverage error. |
| F7: Timeline Localization | Real 6.0s excerpts spliced into 20–22s real tracks at known `[start, end]` | **Mean Segment IoU 34.4% (Boundary MAE 3.33s vs. 8.00s baseline)** | 12 | IoU [0.1367, 0.5648] | Beat-synchronous frame quantization (~0.5s/beat) limits sub-beat boundary precision; IoU gain over whole-track is not statistically significant at n=12 (`p=0.523`). |

---

## 5. Verification, UI Audit, Frozen Corpus Integrity & Honest Weak Spots

- **Automated Test Suite (`pytest`):** **156 passed, 0 failed** across **two consecutive clean full passes** (`55.80s` Pass 1; `56.53s` Pass 2), including 146 original tests + 10 new v2/F6/corrupt-upload regression tests in `tests/test_v2_engineering_and_f6.py`.
- **Static Compile Check:** `python -m compileall -q src evaluation tests` exited cleanly (`0`) on both passes.
- **Headless Chrome CDP UI Verification (`evaluation/verify_ui_cdp.py`):** **PASSED** on **two consecutive clean passes** with **0 console errors, 0 page errors, 0 failed local requests**, verifying Dashboard home (`386` indexed tracks), invalid-file client validation, real held-out upload triggering the **Ground Truth Verification Panel (`MATCH`)** + F1–F7 cards, and the Evaluation & Dataset Audit tab (`21` master rows, `7` truth rows, `Leakage Audit: PASSED`, all 4 PNG diagnostic figures loaded). Screenshots saved under `reports/ui_checks/`.
- **Stress, Concurrency & Consistency Verification (`reports/p9_verification_report.json`, 161.58s):** **PASSED** — `33` OpenAPI paths discovered; 11/11 core GET endpoints HTTP `200`; malformed upload checks (non-audio `400`, empty 0-byte `400`, corrupt WAV `400`); E2E User Rehearsal (identical reference repeat `S = 0.9890 == 0.9890`, held-out Truth Panel matched `true`, spliced derivation Truth Panel matched `true`, unrelated track `false`); **24/24 mixed concurrent requests HTTP `200`**; **50-query consistency sweep with 0 contradictions** (`Case B: 47, Case C: 3`; `advisory: 28, warning: 19, critical: 3`); **0 leaked files** in `temp_uploads/`; resumable index state verified `COMPLETED`.
- **Frozen Legacy 261-Record Corpus Integrity (SHA-256 Verified Unchanged):**
  - `features/remote_index.json`: `c4990a42a64ff024d9f43fb7a44d1191cdc713a857661b368a98e39a613f85f4` (**MATCH**)
  - `features/null_distribution.json`: `d80616e58b729f3b1d08aaf604f8b5a2ffe5fb41f4a44ca6050a8c672602948a` (**MATCH**)
  - `features/mert_manifest.json`: `e3c6da56bca5d764cd9581f10978162d442ce6d4ff446bbf5655d4a649319b0e` (**MATCH**)
  - `features/dataset_manifest.json`: `b36bf0a2f9f0a01c2dcad4eaec43a4efa9c13c5246db9c88ecbe2e2c6c270a6c` (**MATCH**)

### Honest Weak Spots & Known Limitations
1. **Unseen-Generator Generalization (F4 LOGO):** Excluding `MusicSet` during LOGO collapses unseen `MusicSet` recall to `2.9%` (`n=482`), proving AI acoustic fingerprints are generator-specific and must never be claimed universal.
2. **Global DTW Coverage Error on Splices (F6/F7):** Global DTW must align every frame from start to end, producing mean coverage error `0.5621` on derived queries and spliced IoU `0.3439` (`n=12`, paired bootstrap `p=0.523` vs. whole-track baseline). Subsequence DTW would improve partial-splice IoU but was intentionally not altered because Feature 1's global DTW algorithm is locked.
3. **Beat-Tracker Octave Sensitivity:** On sparse/ambient tracks, `librosa.beat.beat_track` can occasionally switch between half-time and double-time across PCM_16 quantization boundaries, reducing self-similarity from `~0.99` to `~0.76` while preserving rank-1 identity and `shift = 0`.
4. **F5 Unverified Key/Tempo:** No human-verified BPM/key labels exist in the acquired corpora; accuracy is honestly unreported.

---

## 6. Engineering Decisions & Blockers

- **Decisions:** All 12 architectural and scientific decisions (with rationale and rejected alternatives) are recorded in [`DECISIONS.md`](DECISIONS.md).
- **Blockers:** **Zero hard blockers** encountered (`BLOCKERS.md`). All real audio datasets downloaded cleanly from Hugging Face, GPU CUDA accelerated MERT extraction, and headless Chrome CDP verified the live UI.
