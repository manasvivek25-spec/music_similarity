# Runbook — Music Similarity & Creative Originality Platform (v2.0)

All commands are Windows PowerShell-safe and assume the workspace root `C:\Users\len\music_similarity\music_similarity` with the project virtual environment `.\venv\`.

---

## 1. Start the Backend Server & Open the Frontend

```powershell
.\venv\Scripts\uvicorn src.api:app --host 127.0.0.1 --port 8001
```

Then open the dashboard in your browser:
- **Analysis Dashboard & Evaluation/Audit UI:** `http://127.0.0.1:8001/`
- **Interactive FastAPI OpenAPI Docs:** `http://127.0.0.1:8001/docs`
- **Live Health JSON:** `http://127.0.0.1:8001/health`

---

## 2. Run the Full Automated Test Suite

```powershell
.\venv\Scripts\pytest -q
```

Static Python compile check:
```powershell
.\venv\Scripts\python -m compileall -q src evaluation tests
```

---

## 3. Re-Run the Offline Real Held-Out Evaluation Suite (Features 1–7 + Figures)

Recomputes `evaluation/evaluation_manifest.csv`, `reports/evaluation_results.json`, `reports/evaluation_table.md`, and the 4 PNG figures under `reports/figures/` (seeded with `SEED=42`):

```powershell
.\venv\Scripts\python -u -m evaluation.run_evaluation
```

---

## 4. Re-Run the Resumable v2 Dataset Pipeline (Optional Full Rebuild)

The prebuilt v2 index, disjoint splits, null distribution (`N=2,000`), and tuned F4 origin classifier (`models/origin_classifier_v2_fresh.joblib`) are already materialized. To re-run any stage cleanly:

1. **Download verified real audio shards (Human FMA-small + SleepyJesse Human/MusicSet/SunoCaps/Udio):**
   ```powershell
   .\venv\Scripts\python -u evaluation/download_v2_shards.py
   ```
2. **Resumable normalized feature extraction & GPU/CPU MERT embedding cache (safe to interrupt & resume):**
   ```powershell
   .\venv\Scripts\python -u -m src.v2_indexer
   ```
3. **Cleaning, near-duplicate pruning, confound probes, 6 disjoint splits & zero-leakage audit:**
   ```powershell
   .\venv\Scripts\python -u -m src.v2_cleaning_audit
   ```
4. **Empirical null distribution rebuild (`N=2,000` global + 6 verified FMA genres):**
   ```powershell
   .\venv\Scripts\python -u -m src.v2_null_builder
   ```
5. **F4 Origin Classifier tuning, calibration, LOGO generalization & 8-perturbation robustness stress test:**
   ```powershell
   .\venv\Scripts\python -u -m src.v2_f4_trainer
   ```

---

## 5. Run Live Browser UI (CDP) & Stress/Consistency Verification

With the FastAPI server running on `http://127.0.0.1:8001`:

```powershell
# Headless Chrome DevTools UI check + 4 screenshots under reports/ui_checks/
.\venv\Scripts\python -u evaluation/verify_ui_cdp.py

# Endpoint smoke, malformed upload checks, 24-request concurrency & 50-query consistency sweep
.\venv\Scripts\python -u evaluation/p9_stress_and_consistency.py
```

---

## 6. Legacy 261-Record Corpus Mode (Read-Only Benchmark)

The 4 historical 261-record files (`features/remote_index.json`, `features/null_distribution.json`, `features/mert_manifest.json`, `features/dataset_manifest.json`) are frozen and untouched. Fresh-production v2 mode is the default (`MUSIC_SIMILARITY_MODE=fresh_production`). To explicitly inspect the legacy benchmark in read-only mode:

```powershell
$env:MUSIC_SIMILARITY_MODE="legacy"
.\venv\Scripts\pytest -q tests/test_legacy_artifact_isolation.py
Remove-Item Env:MUSIC_SIMILARITY_MODE
```
