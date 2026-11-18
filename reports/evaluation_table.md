# Master Offline Evaluation Results (Real Held-Out Audio, v2)

**Evaluation Date:** 2026-11-18T14:27:25Z  
**Dataset Provenance:** Strictly disjoint REAL held-out tracks (`data/v2/v2_held_out.json`, 0% overlap with reference index, null, and training)  
**Random Seed:** 42 | **Runtime:** 149.34s  

## 1. Master Metrics Table
| Feature | Metric | Ours (95% CI) | Baseline | n | Truth source |
| :--- | :--- | :--- | :--- | :---: | :--- |
| Feature 1: Similarity Engine | Top-1 Retrieval Accuracy | **90.00% [83.33%, 94.19%]** | 52.50% (Cosine) / 57.50% (DTW-0) | 120 | Real Held-Out Derived Queries |
| Feature 1: Similarity Engine | Top-3 Retrieval Accuracy | **93.33% [87.39%, 96.58%]** | 56.67% (Mean-Chroma Cosine) | 120 | Real Held-Out Derived Queries |
| Feature 1: Similarity Engine | MAP / MR1 | **0.9218 [0.8980, 0.9476]** | 0.5659 (Cosine) / 0.6296 (DTW-0) | 120 | Real Held-Out Derived Queries |
| Feature 1: Similarity Engine | Shift Recovery (k in ±1..6) | **100.00% [92.59%, 100.00%]** | 0.00% (DTW Shift-0 Only) | 48 | Real Transposed Queries (±1..6) |
| Feature 1: Similarity Engine | Related vs Unrelated ROC-AUC / PR-AUC | **0.9892 / 0.9840** | 0.5000 (Random) | 480 | Real Related vs Unrelated Pairs |
| Feature 2: Statistical Gateway | Observed FPR (α=0.10) | **10.00% [6.57%, 14.94%]** | 0.00% (Fixed S>=0.65) | 200 | Real Unrelated Held-Out Pairs |
| Feature 2: Statistical Gateway | Observed FPR (α=0.05) | **4.50% [2.39%, 8.33%]** | 0.00% (Fixed S>=0.65) | 200 | Real Unrelated Held-Out Pairs |
| Feature 2: Statistical Gateway | Observed FPR (α=0.01) | **1.50% [0.51%, 4.32%]** | 0.00% (Fixed S>=0.65) | 200 | Real Unrelated Held-Out Pairs |
| Feature 2: Statistical Gateway | Statistical Power (α=0.05) | **94.17% [88.45%, 97.15%]** | 83.33% (Fixed S>=0.65) | 120 | Real Derived Related Pairs |
| Feature 2: Statistical Gateway | Null p-value QQ R² / KS p-val | **R²=0.9792 / KS p=0.0004** | N/A (Fixed Threshold) | 200 | Uniform(0,1) Calibration |
| Feature 3: Creative Novelty | Monotonicity (Spearman ρ) & Guards | **ρ=-0.9759 (Guards OK=True)** | Unordered (ρ=0.0) | 200 | Disjoint Real Human Population |
| Feature 4: Origin Classifier | Held-Out Balanced Accuracy | **71.00% [63.89%, 79.00%]** | 50.00% (Maj) / 70.33% (LR) / 69.33% (RF) | 300 | Real Held-Out Origin Labels |
| Feature 4: Origin Classifier | Held-Out ROC-AUC | **0.7756 [0.6993, 0.8580]** | 0.5140 (Confound) / 0.7540 (LR) / 0.7259 (RF) | 300 | Real Held-Out Origin Labels |
| Feature 4: Origin Classifier | Held-Out PR-AUC / F1 | **0.7620 [0.6109, 0.8896] / 0.7071 [0.5962, 0.7986]** | 0.5000 (Random) | 300 | Real Held-Out Origin Labels |
| Feature 4: Origin Classifier | Human Specificity / EER | **72.00% [64.33%, 78.57%] / EER=29.33%** | 50.00% EER (Chance) | 150 | Real Independent Human Set |
| Feature 4: Origin Classifier | Calibration ECE / Brier | **ECE=0.0357 / Brier=0.1945** | 0.2500 Brier (Constant 0.5) | 300 | Real Held-Out Origin Labels |
| Feature 5: Musical Context | Key & Tempo Coverage | **Key 100.0% / Tempo 100.0% (Estimated, unverified)** | N/A (No curated ground truth) | 300 | Unverified (No Curated BPM/Key Labels) |
| Feature 6: Candidate Identification | Source in Top-1 / Top-3 | **90.00% [83.33%, 94.19%] / 93.33% [87.39%, 96.58%]** | 2.78% / 8.33% (Random) | 120 | Real Derived Source Provenance |
| Feature 6: Candidate Identification | Coverage Error & Evidence Agreement | **Err=0.5621 [0.5125, 0.6091] | High/Mod=29.2%** | Err=0.1318 (Naive 100%) | 120 | Real Overlap Ratio & Controlled Truth |
| Feature 7: Timeline Localization | Spliced Segment IoU | **0.3439 [0.1367, 0.5648]** | 0.2727 (Whole-Track Guess) | 12 | Real Spliced Audio Intervals |
| Feature 7: Timeline Localization | Timestamp Boundary Error | **3.33s [1.90s, 4.76s]** | 8.00s (Whole-Track Guess) | 12 | Real Spliced Audio Intervals |

## 2. How Well Do Our Results Match the Truth? (Plain-Language Summary)
| Feature | Truth Source | Predicted vs. True Agreement | n | 95% CI | Known Limitations |
| :--- | :--- | :--- | :---: | :--- | :--- |
| Feature 1: Similarity Engine | Controlled real-audio derivations (pitch ±1..6, stretch 0.9/1.1x, noise 15/20dB, excerpt, splice) | **90.00% Top-1 / 93.33% Top-3 (Shift recovery 100.00%)** | 120 | [83.33%, 94.19%] | Operates strictly in 12-bin chroma pitch space; severe splices/noise reduce global DTW alignment relative to pure transpositions. |
| F2: Statistical Significance | 200 genuinely unrelated cross-group real human pairs vs 2,000-pair disjoint null | **95.50% specificity at α=0.05 (Observed FPR 4.50%, Power 94.17%)** | 200 | FPR [2.39%, 8.33%] | Pairwise null scores share tracks (not strictly independent); empirical p=0.0 means < 1/N due to finite N=2000. |
| Feature 3: Creative Novelty | Disjoint verified human reference population (monotonicity + uniform sanity) | **100.0% monotonic ordering (ρ=-0.9759); unavailable guards passed** | 200 | KS p=0.0002 | Statistical proxy for pitch-space distinctiveness relative to corpus; not an aesthetic judgment of artistic creativity. |
| Feature 4: Origin Classifier | Untouched real held-out tracks (balanced Human FMA vs AI MusicSet/SunoCaps/Udio) | **71.00% Balanced Accuracy (ROC-AUC 0.7756, Human Spec. 72.00%)** | 300 | [63.89%, 79.00%] | Generalizes only to represented generators (MusicSet, SunoCaps, Udio); see LOGO recall & compression drops in evaluation_results.json. |
| Feature 5: Musical Context | No curated human-verified BPM/key labels in FMA-small/SleepyJesse | **Estimated, unverified (accuracy intentionally not reported)** | 300 | N/A | Estimated via librosa tempogram & Krumhansl-Schmuckler; requires curated GTZAN/GiantSteps key/tempo ground truth for formal accuracy. |
| Feature 6: Candidate Identification | Known source track & true overlap ratio on real controlled derivations | **90.00% Top-1 / 93.33% Top-3 (High/Mod evidence on 29.2% of related)** | 120 | [83.33%, 94.19%] | Global DTW stretches alignment across unrelated frames on partial splices, increasing coverage error vs local subsequence matching. |
| Feature 7: Timeline Localization | Real 6.0s audio excerpts spliced into unrelated 20–22s real tracks at known [start, end] | **Mean Segment IoU 34.4% (Boundary MAE 3.33s vs 8.00s baseline)** | 12 | IoU [0.1367, 0.5648] | Beat-synchronous frame quantization (~0.5s/beat) limits sub-beat boundary precision. |
