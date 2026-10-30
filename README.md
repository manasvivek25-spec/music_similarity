# Statistical Analysis of Similarity in Human- and AI-Generated Music

**Course:** Introduction to Data Science (IDSE)  
**Platform Version:** 1.0.0  
**License:** MIT

An end-to-end data science research platform and diagnostic suite designed to evaluate harmonic and melodic similarity between musical compositions, determine whether observed structural convergence is statistically anomalous relative to genre populations, measure creative novelty, independently classify track origin (Human vs. AI), detect candidate mimicry, and generate evidence-grounded studio suggestions.

> **Research & Diagnostic Disclaimer:**  
> System parameters focus exclusively on harmonic and melodic pitch space. Rhythmic-only mimicry, timbral similarity, production style, and lyrical content fall outside active analytical boundaries. Results indicate statistical similarity — not legal plagiarism, copyright infringement, or AI training-data memorization.

---

## 1. System Architecture & Methodology

The pipeline evaluates musical works through a deterministic, multistage scientific process:

```
[Query Audio] (min 60s)
      │
      ▼
1. Feature Extraction (12-bin Chroma CQT & Beat-Synchronous Matrix via librosa)
      │
      ▼
2. Feature 5: Metadata Validation & Musical Context (BPM & Krumhansl-Schmuckler Key)
      │
      ▼
3. Stage 1: Coarse Cosine Prefilter (Mean chroma cosine cutoff ≥ 0.30)
      │
      ▼
4. Stage 2: Global 12-Shift Key-Invariant DTW Sweep (Similarity floor ≥ 0.45, ±1 shift boundary check)
      │
      ▼
5. Stage 3: Genre Influence Map (Percentage distribution & Cross-genre detection)
      │
      ▼
6. Feature 2: Statistical Gateway (Empirical p-value vs null distribution, max 15 survivors)
      │
      ▼
7. Feature 3: Creative Novelty Score (Broad & narrow z-score: (1 - CDF(z)) * 100)
      │
      ▼
8. Case Routing (Case A: Cross-Genre | Case B: Same-Genre | Case C: No Match)
      │
      ▼
9. Feature 4: Independent Acoustic Origin Classifier (Decoupled 56-feature Random Forest)
      │
      ▼
10. Feature 6: Mimic Identification (P-value ranking & same-artist warning suppression)
      │
      ▼
11. Feature 7: Suggestor Box & Evidence Timeline (DTW path reuse & localized timestamps)
      │
      ▼
12. Final Three-Indicator Verdict (Critical | Warning | Advisory | Clear | Incomplete)
```

### Case Routing Logic
* **Case A — Cross-Genre:** A qualifying match survives Feature 2 in a known genre different from the query track's declared native genre. Evaluates broad and narrow novelty scores across up to two target genres.
* **Case B — Same-Genre:** A qualifying match survives Feature 2 in the same native genre as the query track. Evaluates native broad score and narrow score (when $\ge 5$ survivors exist).
* **Case C — No Match:** All candidates fall below the $0.45$ similarity floor or fail the statistical gateway ($p > 0.10$). Mimic warnings are not generated; a broad creative score is calculated if reference population permits.

### Final Verdict Formulation
The final research verdict evaluates three independent indicators:
1. **Similarity Index:** $\ge 0.85$
2. **Empirical P-Value:** $\le 0.01$
3. **Normalized Creative Score:** $\le 0.24$

* **3 Indicators Triggered:** `Critical — All indicators confirm structural imitation`
* **3 Indicators Triggered:** `Critical — All indicators confirm structural imitation`
* **2 Indicators Triggered:** `Warning — Multiple indicators suggest structural similarity`
* **1 Indicator Triggered:** `Advisory — Single indicator, likely genre convention`
* **0 Indicators Triggered:** `Clear — No significant similarity detected`
* **Unavailable P-Value (N < 5):** `Incomplete — Insufficient Null Population` (does not evaluate as a false/cleared verdict; provisional similarity and creative indicators are displayed with prominent explanation).

### Empirical P-Value Specification & Minimum-Null-Pair Guard (Authoritative Decision 1)
* **Authoritative Empirical Formula:**
  $$\text{p-value} = \frac{k}{N} = \frac{\sum \mathbb{I}(S_{\text{null}} \ge S_{\text{query}})}{N_{\text{valid\_null}}}$$
  The platform retains the unsmoothed empirical formula without pseudo-count $+1$ smoothing.
* **Minimum-Null-Pair Guard (`MIN_NULL_PAIRS = 5`):**
  In `src/stats_test.py`, when the number of valid finite null scores is $N < 5$, the statistical gateway returns an explicit unavailable result rather than calculating a p-value:
  - `p_value = None`
  - `confidence = "insufficient"`
  - Actual valid null sample count and required minimum (5).
  - Clear explanation that the empirical null population is insufficient.
* **Operational Sample-Size Confidence Tiers:**
  - **High:** $N_{\text{null}} \ge 1000$ valid pairs.
  - **Moderate:** $300 \le N_{\text{null}} < 1000$ valid pairs.
  - **Low:** $5 \le N_{\text{null}} < 300$ valid pairs.
  - **Insufficient:** $N_{\text{null}} < 5$ valid pairs ($p$-value unavailable).
  *Note:* These operational sample-size labels are practical heuristic engineering tiers indicating sample volume, not formal mathematical guarantees of statistical confidence or population independence.

### Feature 6: Alignment Coverage & Candidate Ranking (Authoritative Decision 5)
* **Metric Definition:**
  $$\text{alignment\_coverage} = \frac{\text{matched\_query\_frames}}{\text{total\_query\_frames}}$$
  Reuses Feature 1's retained DTW warping path and local step costs without rerunning DTW.
* **Local Match Criteria & Path Inflation Prevention:**
  - `matched_query_frames` counts unique query frames covered by qualifying strong-local-similarity regions ($\text{cost} \le \text{mean\_cost} \times 0.70$).
  - Repeated path steps where a single query frame maps to multiple reference frames are counted exactly once to prevent coverage inflation.
  - Missing alignment data returns an explicit unavailable state (`None`).
* **Interpretation Tiers:**
  - **$0\%\text{--}20\%$:** Isolated passage
  - **$>20\%\text{--}50\%$:** Partial structural match
  - **$>50\%\text{--}100\%$:** Broad structural overlap
  *Note:* Coverage is an analytical localization metric, not proof of intentional copying.
* **Candidate Ranking:**
  Candidates are ranked by ascending valid empirical $p$-value, using descending DTW similarity as the tie-breaker. Candidates with unavailable $p$-values are ordered after valid $p$-values. Case-insensitive same-artist suppression (`Consistent Artist Style`) remains strictly active.

### Distinct Catalog & Analysis Count Fields (Authoritative Decision 4)
The API and frontend expose five distinct catalog quantities without double-counting:
1. `total_indexed_tracks`: Exact count of distinct indexed audio tracks in the catalog.
2. `human_tracks`: Tracks with human provenance origin.
3. `ai_tracks`: Tracks with AI generative origin.
4. `tracks_with_estimated_genre`: Tracks whose genre was inferred by a genuine classifier model.
5. `null_pairs_global`: Valid pairs in the global null distribution (or `null` if uncomputed).

In individual analysis responses, six distinct quantities are distinguished:
- **Indexed Corpus Size:** Total candidate pool in catalog.
- **Candidates Considered:** Tracks undergoing initial evaluation.
- **Candidates Retained by Prefilter:** Tracks passing Stage 1 mean-chroma cosine filtering ($\ge 0.30$).
- **Candidates Evaluated by DTW:** Shortlisted tracks aligned via 12-shift circular DTW.
- **Feature 2 Survivors:** Tracks exceeding similarity floor ($\ge 0.45$) and passing statistical gateway.
- **Valid Null-Distribution Pairs:** Pairs used to evaluate empirical statistical significance.

### Normalized Distance & Similarity Relationship
* **Exact Transformation:**
  $$\text{Similarity} = \frac{1}{1 + D_{\text{norm}}} \iff D_{\text{norm}} = \frac{1}{\text{Similarity}} - 1$$
  where $D_{\text{norm}} = \frac{\min \text{DTW accumulated cost}}{\max(\text{warping path length}, 1)}$.
* **Example:** For an observed similarity of $0.547$, the corresponding normalized distance is $D_{\text{norm}} \approx 0.828$.

### Creative Score Units & Novelty Tiers
* **Creative Score Percentage (`creative_score_percent`):** Range $[0.0, 100.0]\%$, computed as $(1 - \Phi(z)) \times 100$. Novelty tiers evaluate this percentage ($< 25\%$ High Structural Imitation, $25\text{--}50\%$ Algorithmic Saturation Risk, $50\text{--}75\%$ Standard Compositional Variance, $\ge 75\%$ High Creative Novelty).
* **Creative Score Normalized (`creative_score_normalized`):** Range $[0.0, 1.0]$, representing $\frac{\text{creative\_score\_percent}}{100}$. Evaluated against the verdict cutoff ($\le 0.24$). Precision is consistently displayed in the dashboard (2 decimals for percentage, 4 decimals for normalized).

---

## 2. Repository Structure

```text
music_similarity/
├── data/                          # Raw audio repositories (git-ignored audio files)
│   ├── human_songs/               # Ground-truth human audio corpus (.gitkeep)
│   ├── ai_generated/              # AI-generated audio corpus (.gitkeep)
│   ├── reference_pool/            # Null distribution reference pool (.gitkeep)
│   └── metadata_template.csv      # Template for audio catalog metadata
├── features/                      # Cached numpy feature matrices (.npy)
│   └── .gitkeep
├── models/                        # Serialized scikit-learn models (.joblib)
│   └── .gitkeep
├── notebooks/                     # Exploratory research notebooks
│   └── .gitkeep
├── results/                       # Exported figures and statistics
│   └── .gitkeep
├── src/                           # Central Python pipeline package
│   ├── __init__.py                # Package version definition
│   ├── config.py                  # Central configuration and thresholds
│   ├── schemas.py                 # Pydantic data schemas for typed validation
│   ├── extract.py                 # Chroma CQT extraction & caching
│   ├── similarity.py              # 12-shift DTW alignment & cosine prefilter
│   ├── stats_test.py              # Null distribution construction & empirical p-values
│   ├── genre.py                   # Stage 3 Genre Influence Map & metadata loader
│   ├── creative_score.py          # Broad & narrow creative z-score computation
│   ├── origin_classifier.py       # Independent Human/AI origin classification
│   ├── feature5.py                # Metadata validation, key correlation & tempo tracking
│   ├── mimic.py                   # Mimic ranking & same-artist suppression
│   ├── suggestor.py               # Localized segment timeline & studio suggestions
│   ├── pipeline.py                # Central pipeline coordinator & verdict engine
│   ├── utils.py                   # Audio file discovery & duration helpers
│   └── api.py                     # FastAPI REST API backend & static server
├── frontend/                      # Web dashboard
│   ├── index.html                 # Research dashboard UI
│   ├── styles.css                 # Glassmorphism & WebGL shader styles
│   └── app.js                     # Frontend API client & interactive visualizations
├── tests/                         # Pytest test suite
│   ├── test_similarity.py         # Unit tests for DTW, shift invariance, and tiers
│   ├── test_stats.py              # Unit tests for p-values, creative scores, and verdict
│   └── test_api.py                # REST API endpoint integration tests
├── .env.example                   # Environment variable template
├── .gitignore                     # Git exclusion rules
├── metadata_schema.csv            # Field documentation for metadata CSVs
├── pytest.ini                     # Pytest configuration
├── requirements.txt               # Pinned Python dependencies
└── README.md                      # Project documentation
```

---

## 3. Installation & Setup

### Requirements
* **Python 3.11** (64-bit recommended)
* A modern browser with WebGL support

### Setup Steps

1. **Activate the Virtual Environment:**

   *Windows PowerShell:*
   ```powershell
   venv\Scripts\Activate.ps1
   ```

   *Windows Command Prompt (cmd.exe):*
   ```cmd
   venv\Scripts\activate.bat
   ```

   *(Optional) If creating a new virtual environment:*
   ```powershell
   python -m venv venv
   venv\Scripts\Activate.ps1
   ```

2. **Install Dependencies:**
   ```powershell
   python -m pip install -r requirements.txt
   ```

3. **Configure Environment Variables (Optional):**
   ```powershell
   Copy-Item .env.example .env
   ```

---

## 4. Running the Application

Start the unified FastAPI server:

```powershell
python -m uvicorn src.api:app --reload
```

The application will start on `http://127.0.0.1:8000`:
* **Interactive Web Dashboard:** `http://localhost:8000/`
* **Interactive API Documentation (Swagger UI):** `http://localhost:8000/docs`
* **Health Diagnostic Probe:** `http://localhost:8000/health`
* **Runtime Capabilities:** `http://localhost:8000/api/config`

---

## 5. Dataset & Metadata Ingestion

### Folder Conventions
Place uncommitted audio files (`.wav`, `.mp3`, `.flac`, `.ogg`) into their respective directories:
* `data/human_songs/`: Songs by human artists.
* `data/ai_generated/`: Generative tracks from Suno, Udio, MusicGen, etc.
* `data/reference_pool/`: Unrelated songs for null distribution generation.

### Metadata CSV Schema
Create or update `data/metadata.csv` using the format defined in [`metadata_schema.csv`](file:///c:/Users/len/music_similarity/music_similarity/metadata_schema.csv):

| Field | Type | Description |
| :--- | :--- | :--- |
| `track_id` | string | Unique track identifier (e.g. `human_001`) |
| `file_path` | string | Path relative to repository root (`data/human_songs/song1.wav`) |
| `title` | string | Composition title |
| `artist` | string | Artist / composer (used for Feature 6 same-artist check) |
| `genre` | string | Musical genre label |
| `source_type` | string | `acoustic`, `electronic`, `generative`, `field_recording` |
| `origin_label`| string | `Human` or `AI` |
| `key` | string | Known musical key (e.g. `C Minor`) |
| `tempo` | float | Known tempo in BPM |

---

## 6. Training the Origin Classifier

When labeled audio files are placed into `data/human_songs/` and `data/ai_generated/`, train the model:

```powershell
python -c "from src.origin_classifier import train_origin_classifier; print(train_origin_classifier('data/human_songs', 'data/ai_generated'))"
```

The fitted model is saved to `models/origin_classifier.joblib`. If audio files are absent, the platform reports `Unconfigured / Missing Labeled Data` without crashing.

---

## 7. Automated Testing & Verification

Run the test suite across all modules:

```powershell
python -m pytest -v
```

### Covered Test Cases
* **Chroma Alignment:** Identical matrices ($\approx 1.0$), transposed keys (4 semitones $\approx 1.0$), random noise decay ($< 0.65$), partial corruption monotonicity.
* **Prefilter & Boundary:** Coarse cosine filtering at $0.30$, zero-vector handling, non-finite input guards.
* **Statistical Gateway:** Empirical p-value calculation, genre fallback below 30 samples, confidence tiers ($N \ge 1000, 300-999, < 300$).
* **Creative Novelty:** Broad and narrow scores, zero-variance standard deviation handling, score tiers.
* **Mimic & Verdict:** P-value ranking, same-artist warning suppression (`Consistent Artist Style`), 3-indicator verdict matrix.
* **API Endpoints:** 60-second upload duration validation ($422$), candidate count checks ($400$), `/health`, `/api/config`, `/api/analyze`, `/api/mimic`.
* **Remote Dataset Streaming:** Dynamic schema validation, in-memory incremental decoding, compact JSON indexing, cache hits verification, two-stage prefiltering and DTW alignment.

---

## 8. Remote Dataset Streaming & Stable Incremental Indexing

### Active Research Dataset (Authoritative Decision 2)
The platform integrates the Hugging Face audio corpus:
* **Active Dataset:** [`SleepyJesse/ai_music_large`](https://huggingface.co/datasets/SleepyJesse/ai_music_large) (split: `train`)
* **Prior Research Citation:** The **SONICS** dataset is cited in the literature as prior research motivation for evaluating human-vs-AI acoustic divergence, but `SleepyJesse/ai_music_large` is the active dataset used for the implemented empirical pipeline.
* **Corpus Schema & Provenance:**
  - `audio`: Embedded audio record (in-memory bytes / array).
  - `ai_generated`: Boolean/numeric flag indicating generative AI origin.
  - `source`: Provenance tracking field.
  - *Genre & Artist:* The inspected records contain no authentic metadata for genre or artist. The platform does not assume `"ambient"` or any hardcoded genre as a ground truth default.

### Acoustic Genre Estimation
When raw audio is indexed, an acoustic genre estimation module (`src/genre_classifier.py` and `src/genre.py`) provides GTZAN-compatible 10-class genre estimation:
- **Feature Extraction:** 20 MFCC coefficients extracted from audio waveforms with temporal mean and standard deviation aggregation (40-dimensional feature vector).
- **Artifact Format:** Serialized scikit-learn classifier (`models/genre_classifier.joblib`).
- **Unconfigured State:** If no trained model artifact exists, genre analysis gracefully falls back to an explicit, honest unconfigured state (`estimated_genre: null`, `genre_source: "unknown"`).
- **Record Schema:**
  - `genre`: Verified metadata only, or `null`.
  - `estimated_genre`: Predicted genre label, or `null`.
  - `genre_confidence`: Valid model probability score, or `null`.
  - `genre_source`: `"metadata"`, `"estimated"`, or `"unknown"`.

### Stable Content-Derived Identifiers (Authoritative Decision 3)
Sequential IDs (e.g. `remote_000000`) have been replaced with content-derived identifiers:
$$\text{track\_id} = \text{track\_}\langle 12\text{ hex characters}\rangle$$
- **Derivation:** Derived from the first 4096 bytes of audio via `hashlib.sha256(audio_bytes[:4096]).hexdigest()[:12]`, or from a deterministic contiguous waveform slice when only decoded arrays are present.
- **Provenance Retention:** Stable source IDs from the remote corpus (such as `id` or `source`) are retained as separate provenance metadata (`source_id`).
- **Collision Handling:** The 12-character hex string is a truncated prefix hash. A collision check compares candidate hashes against existing catalog entries; if an identical prefix collides with an unrelated track, the prefix length is dynamically extended.
- **Atomic Persistence:** Remote index updates write to a unique temporary file (`.tmp`) and use atomic filesystem replacement (`os.replace`) to prevent file corruption in case of unexpected termination.
- **Safe Incremental Indexing:** Re-indexing identical tracks reuses existing entries without duplication or overwriting unrelated records.

### Methodological & Research Limitations
1. **Empirical Null Distribution Overlap:**
   Pairwise similarity scores in the empirical null distribution are derived from combinations within the available reference pool. When sample sizes are small ($N < 300$), pairwise comparisons share underlying tracks and are not fully independent observations.
2. **Operational Confidence Labels:**
   Operational confidence labels (`High`, `Moderate`, `Low`, `Insufficient`) represent heuristic engineering sample-volume tiers. They do not constitute formal asymptotic mathematical guarantees of population representation or statistical independence.
3. **Decoupled Acoustic Origin Classifier (Feature 4):**
   The 56-feature Random Forest architecture is fully implemented, but remains in an unconfigured state awaiting suitable labeled training datasets. No synthetic or fabricated labels are generated.

### CLI Indexing Commands

* **Small Development Sample (Default 10 records):**
  ```powershell
  python -m src.remote_dataset
  # or explicitly:
  python -m src.remote_dataset --sample 10
  ```

* **Inspect Remote Schema Only:**
  ```powershell
  python -m src.remote_dataset --inspect-schema
  ```

* **Inspect Metadata Without Reading Audio:**
  ```powershell
  python -m src.remote_dataset --inspect-metadata --sample 5
  ```

* **Full-Corpus Pass (Requires explicit `--full` flag):**
  ```powershell
  python -m src.remote_dataset --full
  ```

* **Force Cache Re-indexing:**
  ```powershell
  python -m src.remote_dataset --sample 10 --force
  ```

* **JSON Output for Scripting/Pipelines:**
  ```powershell
  python -m src.remote_dataset --sample 5 --json
  ```

### Running Tests Without Network Access
All test suites in `tests/` run 100% offline using synthesized audio waveforms and mocked record streams. Normal test execution does not depend on network access or downloading the real Hugging Face corpus.

