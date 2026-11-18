# Feature 6: Mimic Evidence Extension Specification

## 1. Scope & Non-Infringement Disclaimer
Feature 6 quantifies **structural melodic/harmonic mimic patterns** strictly within 12-pitch-class chroma space. It does **not** measure lyrics, vocal/instrument timbre, mastering style, or copyright infringement, and produces statistical evidence rather than legal conclusions.

## 2. Preserved Baseline Behavior (Locked)
- **Similarity floor:** `SIMILARITY_FLOOR = 0.45`.
- **Primary ranking:** Ascending raw empirical $p$-value ($p = k/N$), with descending Feature 1 DTW similarity as tie-breaker. Candidates with unavailable $p$ (`None`) sort after valid $p$-values and are **never** classified as statistical survivors.
- **Same-artist suppression:** Warning suppression (`Consistent Artist Style`) applies **only** when both query and candidate artist fields are known (non-empty) and case-insensitively equal.
- **Zero DTW recalculation:** All F6 structural evidence fields strictly reuse the optimal warping path $P = [(a_0, b_0), \dots, (a_{L-1}, b_{L-1})]$ and per-step Euclidean local costs $c_0, \dots, c_{L-1}$ retained from Feature 1.
- **Alignment coverage:**
  $$\text{coverage} = \frac{|\{a_t : (a_t, b_t) \in P,\ c_t \le 0.70 \cdot \bar{c}\}|}{T_{\text{query}}} \in [0, 1]$$
  where $\bar{c} = \frac{1}{L}\sum_{t=0}^{L-1} c_t$ and $T_{\text{query}}$ is the total number of query frames.

## 3. Structural Evidence Extensions (Path-Derived)
Let $Q = \{t \in \{0, \dots, L-1\} : c_t \le 0.70 \cdot \bar{c}\}$ be the set of strong-local-similarity steps along the DTW path, and partition $Q$ into maximal contiguous index runs $R_1, R_2, \dots, R_m$. A run $R_j$ is a **distinct significant matching region** if $|R_j| \ge 4$ steps (`MIN_REGION_STEPS = 4`).

1. **Significant Region Count (`significant_region_count`)**
   $$M = |\{R_j : |R_j| \ge 4\}|$$
2. **Region Continuity / Coherence (`region_continuity`)**
   Measures whether qualifying matches form coherent sustained passages rather than scattered isolated frames:
   $$\text{continuity} = \begin{cases} 0.0 & \text{if } |Q| = 0 \\ \frac{\sum_{|R_j| \ge 4} |R_j|}{|Q|} & \text{otherwise} \end{cases} \in [0.0, 1.0]$$
3. **Sequential Consistency (`sequential_consistency`)**
   Measures whether matched regions progress in consistent temporal order across both tracks:
   - If $M \ge 2$: let $(a_j^{\text{start}}, b_j^{\text{start}})$ be the starting frame pair of region $R_j$ ordered by $a_j^{\text{start}}$. Then:
     $$\text{seq\_consistency} = \frac{1}{M - 1}\sum_{j=1}^{M-1} \mathbb{I}(a_{j+1}^{\text{start}} \ge a_j^{\text{start}} \land b_{j+1}^{\text{start}} \ge b_j^{\text{start}})$$
   - If $M = 1$: computed as the fraction of internal step transitions within $R_1$ exhibiting strictly positive joint progress ($a_{t+1} > a_t$ and $b_{t+1} > b_t$, or non-degenerate monotonic advance) divided by $\max(|R_1|-1, 1)$, clipped to $[0.0, 1.0]$.
   - If $M = 0$: `0.0`.
4. **Local-Similarity Concentration (`local_similarity_concentration`)**
   Let local step similarity be $s_t = \frac{1}{1 + c_t}$. Concentration is the fraction of total path similarity mass residing inside qualifying matching steps:
   $$\text{concentration} = \begin{cases} 0.0 & \text{if } \sum_{t=0}^{L-1} s_t = 0 \\ \frac{\sum_{t \in Q} s_t}{\sum_{t=0}^{L-1} s_t} & \text{otherwise} \end{cases} \in [0.0, 1.0]$$

## 4. Mimic Evidence Level Classification (`mimic_evidence_level`)
Combines the structural path evidence metrics with empirical $p$-value strictly as supporting evidence (F6 is never reduced to $p$-sorting alone):

| Level | Exact Condition |
| :--- | :--- |
| **`Not applicable`** | `p_value is None`, OR `similarity < 0.45`, OR `artist_relationship == "Consistent Artist Style"`, OR Case C / Case U context, OR (path provided and `significant_region_count == 0` with `p_value > 0.10`). |
| **`High`** | `p_value <= 0.01` AND `significant_region_count >= 1` AND `alignment_coverage >= 0.30` AND `region_continuity >= 0.45` AND `sequential_consistency >= 0.70` AND `local_similarity_concentration >= 0.35`. |
| **`Moderate`** | Not `High`, AND `p_value <= 0.10` AND `significant_region_count >= 1` AND `alignment_coverage >= 0.15` AND `sequential_consistency >= 0.50`. |
| **`Low`** | Remaining eligible candidates with `similarity >= 0.45` and `p_value is not None` whose regional evidence is fragmented, brief, or non-significant (`p_value > 0.10`). |

When DTW path/cost arrays are omitted by a legacy matrix-only caller, structural path metrics return `None` (never fabricated) and `mimic_evidence_level` degrades honestly to `"Not applicable"` when `p_value is None` or `"Low"`/`"Moderate"` only with explicit note.
