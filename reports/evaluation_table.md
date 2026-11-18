# Master Offline Evaluation Results: Canonical Seven Features

**Evaluation Date:** 2026-11-17 08:05:56 UTC  
**Dataset Provenance:** Strictly held-out items only (never in reference index or training)  
**Random Seed:** 42 (fully reproducible, seeded execution)  
**Runtime Duration:** 25.48 seconds  

| Feature | Metric | Ours (95% CI) | Baseline | n | Truth source |
| :--- | :--- | :--- | :--- | :---: | :--- |
| Feature 1: Similarity Engine | Top-1 Retrieval Accuracy | **52.22% [44.96%, 59.40%]** | 37.78% (Mean-Chroma Cosine) | 180 | Held-Out Controlled Variations |
| Feature 1: Similarity Engine | Top-3 Retrieval Accuracy | **78.89% [72.36%, 84.22%]** | 48.89% (Mean-Chroma Cosine) | 180 | Held-Out Controlled Variations |
| Feature 1: Similarity Engine | Mean Average Precision (MAP) | **0.6721 [0.6040, 0.7447]** | 0.5009 (Mean-Chroma Cosine) | 180 | Held-Out Controlled Variations |
| Feature 1: Similarity Engine | Mean Reciprocal Rank (MR1) | **0.6721 [0.6040, 0.7447]** | 0.5009 (Mean-Chroma Cosine) | 180 | Held-Out Controlled Variations |
| Feature 1: Similarity Engine | Shift Recovery Rate (k in ±1..6) | **14.17% [9.04%, 21.52%]** | 0.00% (DTW Shift 0 Only) | 120 | Known Transposition k in ±1..6 |
| Feature 2: Statistical Gateway | Observed FPR (alpha = 0.10) | **8.89% [3.51%, 20.73%]** | 97.78% (Fixed S >= 0.65) | 45 | Unrelated Held-Out Null Pairs |
| Feature 2: Statistical Gateway | Observed FPR (alpha = 0.05) | **4.44% [1.23%, 14.83%]** | 97.78% (Fixed S >= 0.65) | 45 | Unrelated Held-Out Null Pairs |
| Feature 2: Statistical Gateway | Observed FPR (alpha = 0.01) | **0.00% [0.00%, 7.87%]** | 97.78% (Fixed S >= 0.65) | 45 | Unrelated Held-Out Null Pairs |
| Feature 2: Statistical Gateway | Statistical Power (alpha = 0.05) | **16.67% [11.93%, 22.80%]** | 72.78% (Fixed S >= 0.65) | 180 | Controlled Related Pairs |
| Feature 2: Statistical Gateway | p-value QQ Uniformity (R^2) | **1.0000** | N/A (Fixed Threshold) | 45 | Theoretical Uniform Quantiles |
| Feature 2: Statistical Gateway | Kolmogorov-Smirnov Uniformity p-val | **p = 1.0000 (stat=0.0222)** | p < 0.0001 (Non-uniform) | 45 | Null Pair Empirical p-values |
| Feature 3: Creative Novelty | Duplicate Rank on Held-Out Humans | **1.00** | 2.50 (Random Chance) | 5 | Independent Held-Out Humans |
| Feature 3: Creative Novelty | KS Uniformity on Held-Out Humans | **p = 0.0000 (stat=1.0000)** | p < 0.0001 (Degenerate Clustering) | 5 | Independent Held-Out Humans |
| Feature 4: Origin Classifier | Balanced Accuracy | **18.75% [11.06%, 29.97%]** | 50.00% (Majority) | 64 | Ground-Truth Origin Labels |
| Feature 4: Origin Classifier | ROC AUC | **0.1689 [0.0733, 0.2485]** | 1.0000 (LR) / 1.0000 (RF) | 64 | Ground-Truth Origin Labels |
| Feature 4: Origin Classifier | PR AUC | **0.3371 [0.1124, 0.7337]** | 0.5000 (Random Guess) | 64 | Ground-Truth Origin Labels |
| Feature 4: Origin Classifier | Equal Error Rate (EER) | **81.25%** | 50.00% (Majority Baseline) | 64 | Ground-Truth Origin Labels |
| Feature 4: Origin Classifier | TPR at 1% FPR | **0.00%** | 1.00% (Diagonal) | 64 | Ground-Truth Origin Labels |
| Feature 4: Origin Classifier | TPR at 5% FPR | **0.00%** | 5.00% (Diagonal) | 64 | Ground-Truth Origin Labels |
| Feature 4: Origin Classifier | Per-Generator Recall (Average) | **18.22%** | 50.00% (Majority) | 32 | Generator Architecture Sidecars |
| Feature 4: Origin Classifier | Specificity on Independent Human Set | **18.75% [8.89%, 35.31%]** | 100.00% (Majority) | 32 | Held-Out Independent Humans |
| Feature 4: Origin Classifier | Expected Calibration Error (ECE) | **0.6919** | 0.2500 (Uncalibrated) | 64 | Ground-Truth Origin Labels |
| Feature 4: Origin Classifier | Brier Score | **0.6357** | 0.0003 (Logistic Regression) | 64 | Ground-Truth Origin Labels |
| Feature 5: Musical Context | MIREX Key Score | **0.7584 [0.6984, 0.8153]** | 0.1158 (Majority C Major) | 190 | Sidecar Musical Key Truth |
| Feature 5: Musical Context | Tempo Acc1 (Tolerance 8%) | **99.47% [97.08%, 99.91%]** | 20.00% (Mean 120 BPM) | 190 | Sidecar BPM Truth |
| Feature 5: Musical Context | Tempo Acc2 (Harmonics Multiples) | **100.00% [98.02%, 100.00%]** | 40.00% (Mean 120 BPM) | 190 | Sidecar BPM Truth |
| Feature 5: Musical Context | BPM Mean Absolute Error (MAE) | **2.19 BPM [1.66, 3.15]** | 18.56 BPM (Mean 120 BPM) | 190 | Sidecar BPM Truth |
| Feature 6: Candidate Identification | Source in Top-1 Candidates | **52.22% [44.96%, 59.40%]** | 10.00% (Random Ranker) | 180 | Candidate Provenance Track ID |
| Feature 6: Candidate Identification | Source in Top-3 Candidates | **78.89% [72.36%, 84.22%]** | 30.00% (Random Ranker) | 180 | Candidate Provenance Track ID |
| Feature 6: Candidate Identification | Alignment Coverage Error | **0.7432 [0.7075, 0.7789]** | 0.0500 (Naive 100% Guess) | 180 | True Overlap Ratio |
| Feature 7: Timeline Localization | Segment Intersection over Union (IoU) | **0.6680 [0.5577, 0.7668]** | 0.3333 (Whole-Track Guess) | 10 | Spliced Query Interval Truth |
| Feature 7: Timeline Localization | Timestamp Boundary Error | **0.66s [0.47s, 0.88s]** | 4.00s (Whole-Track Guess) | 10 | Spliced Query Interval Truth |
