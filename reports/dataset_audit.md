# Dataset v2 Acquisition, Cleaning, Bias Control & Leakage Audit

**Generated:** 2026-11-18T14:14:57Z  
**Preprocessing Standard:** `22050 Hz` mono · Center `10.0s` window (`INDEX_MAX_SECONDS`) · 4th-order zero-phase `7,500 Hz` Butterworth low-pass · RMS `0.10` loudness normalization  
**Feature Version:** `v2.0-chroma_cqt_beatsync-lp7500-rms010` | **Algorithm Version:** `f1_12shift_fastdtw_v1`  

## 1. Sources & Label Polarity Verification
- **Human (Verified CC):** `benjamin-paine/free-music-archive-small` (shards 0–1; genuine artist, album, and 164-class FMA genre labels) + `SleepyJesse/ai_music_large` shard 0 (`free-music-archive-full`, verified `ai_generated=False`).
- **AI (Verified Multi-Generator):** `SleepyJesse/ai_music_large` shards 76–80 (`ManzhenWei/MusicSet`, verified `ai_generated=True`), shard 75 (`SunoCaps`, verified `ai_generated=True`), and shard 111 (`blanchon/udio_dataset`, CC0, verified `ai_generated=True`).
- **Unknown / Conflicting Policy:** Zero invented genres (tracks without genuine genre metadata remain `Unknown`). Any conflicting polarity is assigned `unknown` and excluded from training, null, and evaluation.

## 2. Deduplication & Cleaning Summary
- Exact-hash unique indexed tracks: **2119**
- Exact content-hash duplicates removed: **0**
- Near-duplicates removed (MERT cosine > 0.98 or Chroma > 0.9999): **7**
- Skipped (short/silent): **26** | Failed decode: **0**
- Clean verified pool: **2112** (Human: 1186, AI: 926)

## 3. Disjoint Group-Aware Partition Counts
| Partition | Total | Human | AI | Unique Groups | Purpose |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `reference_index` | 386 | 193 | 193 | 270 | Live F1/F3/F6 Reference Catalog (`data/v2/v2_index.json`) |
| `null_population` | 200 | 200 | 0 | 104 | F2 Pairwise Human-Human Null Distribution (Human-only) |
| `f4_train` | 702 | 351 | 351 | 519 | F4 Origin Classifier Training & Nested/Grouped CV |
| `f4_val` | 232 | 116 | 116 | 174 | F4 Probability Calibration & Threshold Selection Only |
| `f4_test` | 232 | 116 | 116 | 155 | F4 Untouched Test Set (Evaluated Once) |
| `held_out` | 300 | 150 | 150 | 211 | Section 8 Real Held-Out Evaluation (Disjoint from All Above) |

## 4. Leakage Audit (Zero-Overlap Verification)
- **Status:** `PASSED_ZERO_LEAKAGE`
- **Total Track-ID / Group-ID / Content-Hash Cross-Split Overlaps:** **0** (all 15 split pairs verified = `0`).

## 5. Confound Probes (Metadata & Non-Musical Features Only)
| Probe | Features | LR ROC-AUC | RF ROC-AUC | LR Bal. Acc | RF Bal. Acc |
| :--- | :--- | :---: | :---: | :---: | :---: |
| Majority Baseline | Constant majority class | 0.5000 | 0.5000 | 0.5000 | 0.5000 |
| Pre-Normalization Metadata | orig_dur, orig_sr, orig_ch, orig_bytes, raw_silence | 0.9912 | 1.0000 | 0.9462 | 0.9979 |
| Post-Norm Physical/Container | norm_dur, norm_sr, norm_ch, norm_rms, peak, dc_offset | 0.4802 | 0.5050 | 0.5008 | 0.4978 |
| Post-Norm + Silence Ratio | Physical + post-norm silence ratio | 0.4785 | 0.5004 | 0.5009 | 0.4921 |

> **Interpretation:** Pre-normalization metadata strongly separates sources (different original durations/SRs/file sizes). Identical normalization (10.0s center window, 22,050 Hz mono, 7,500 Hz Butterworth low-pass, RMS 0.10) eliminates duration/SR/channel/bitrate confounds; any residual signal in post_normalization_with_silence_ratio is attributable to genuine acoustic dynamic-range / silence distribution differences and is explicitly documented.

## 6. Generator & Verified Genre Breakdown (Clean Pool)
- **Generators:** `{'MusicSet': 766, 'SunoCaps': 48, 'Udio': 112, 'human_fma_full': 134, 'human_fma_small': 1052}`
- **Top Verified Genres (unlabelled stays Unknown):** `{'Unknown': 1060, 'Folk': 167, 'Hip-Hop': 162, 'Rock': 79, 'Pop': 66, 'International': 62, 'Experimental Pop': 56, 'Latin America': 47, 'Audio Collage': 38, 'Soundtrack': 29, 'Avant-Garde': 24, 'Reggae - Dub': 23}`
- **Missing-Value Rates:** `{'genre_missing_rate': 0.5019, 'artist_missing_rate': 0.5019, 'tempo_missing_rate': 0.0005, 'mert_missing_rate': 0.0}`

## 7. Known Limitations
- Genuine genre labels exist only for FMA-small human tracks; AI tracks from SleepyJesse (MusicSet, SunoCaps, Udio) do not carry verified genre tags and are honestly retained as `Unknown`.
- Center 10.0s normalization matches MusicSet's native 10.0s clip duration so duration cannot confound F4, at the cost of evaluating a 10.0s center excerpt rather than full multi-minute arrangements during corpus indexing.
- AI tracks represent 3 specific generators (MusicSet, SunoCaps, Udio); F4 generalization is scoped strictly to represented generators.

## 8. Versioned Artifact SHA-256 Hashes
- `data\v2\v2_index.json`: `6c87b2bc05df70ee2086ff61ee504a4ff8d3881e5f10c5dfdfaef0c31cd58443`
- `data\v2\v2_null_population.json`: `78c32cdcbe3db727c5a3390c99d1bdfa768721a2b525e5ec06aa213a115137b6`
- `data\v2\v2_f4_splits.json`: `01dd56b35f43a36e144802a2efe9085ca83aa489a014516398c3b561cb8b24ec`
- `data\v2\v2_held_out.json`: `12f67470de4cbd7f1325f04029d0eb57313daec26a0d923120755df39a700f85`
