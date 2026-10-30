/**
 * Statistical Analysis of Audio Similarity in Human- and AI-Generated Music
 * Central Frontend Dashboard Controller
 */

(function () {
  const API_BASE = window.location.origin;

  // State
  let serverConfig = null;

  // -------------------------------------------------------------------------
  // 1. Health & Dataset Status
  // -------------------------------------------------------------------------
  async function fetchHealthAndStatus() {
    const statusDot = document.getElementById("backendStatusDot");
    const statusText = document.getElementById("backendStatusText");
    const classifierText = document.getElementById("classifierStatusText");
    const invHuman = document.getElementById("invHumanCount");
    const invAi = document.getElementById("invAiCount");
    const invRef = document.getElementById("invRefCount");

    try {
      const res = await fetch(`${API_BASE}/health`);
      if (res.ok) {
        const data = await res.json();
        if (statusDot) statusDot.className = "w-2 h-2 rounded-full bg-emerald-500 animate-pulse";
        if (statusText) {
          const total = data.total_indexed_tracks !== undefined
            ? data.total_indexed_tracks
            : (data.reference_track_count + data.human_track_count + data.ai_track_count);
          statusText.innerText = `FastAPI Online · ${total} Audio Tracks Indexed`;
        }
        if (classifierText) {
          classifierText.innerText = `Classifier: ${data.origin_classifier_status}`;
        }
        const invTotal = document.getElementById("invTotalIndexed");
        const invHuman = document.getElementById("invHumanCount");
        const invAi = document.getElementById("invAiCount");
        const invEstGenre = document.getElementById("invEstGenreCount");
        const invNullPairs = document.getElementById("invNullPairsCount");

        if (invTotal) invTotal.innerText = data.total_indexed_tracks !== undefined ? data.total_indexed_tracks : 0;
        if (invHuman) invHuman.innerText = data.human_tracks !== undefined ? data.human_tracks : (data.human_track_count || 0);
        if (invAi) invAi.innerText = data.ai_tracks !== undefined ? data.ai_tracks : (data.ai_track_count || 0);
        if (invEstGenre) invEstGenre.innerText = data.tracks_with_estimated_genre !== undefined ? data.tracks_with_estimated_genre : 0;
        if (invNullPairs) invNullPairs.innerText = data.null_pairs_global !== null && data.null_pairs_global !== undefined ? data.null_pairs_global : "Unavailable";

        try {
          const mertRes = await fetch(`${API_BASE}/api/mert/status`);
          if (mertRes.ok) {
            const mertData = await mertRes.json();
            const mertBadge = document.getElementById("mertStatusBadge");
            if (mertBadge) {
              if (mertData.is_supported) {
                mertBadge.className = "px-2.5 py-1 rounded-full text-[10px] font-mono font-bold bg-emerald-500/20 text-emerald-300 border border-emerald-500/30";
                mertBadge.innerText = `Active (${mertData.device.toUpperCase()})`;
              } else {
                mertBadge.className = "px-2.5 py-1 rounded-full text-[10px] font-mono font-bold bg-amber-500/20 text-amber-300 border border-amber-500/30";
                mertBadge.innerText = "Optional (Offline)";
              }
            }
          }
        } catch (_) {}

        // Query rich index status
        try {
          const idxRes = await fetch(`${API_BASE}/api/index/status`);
          if (idxRes.ok) {
            const idxData = await idxRes.json();
            if (statusText) {
              const compStr = idxData.is_complete ? "Complete" : (idxData.status || "Partial");
              statusText.innerText = `FastAPI Online · ${idxData.total_indexed} Tracks Indexed (${compStr})`;
            }
            if (invTotal) invTotal.innerText = idxData.total_indexed;
            if (invHuman) invHuman.innerText = idxData.human_tracks;
            if (invAi) invAi.innerText = idxData.ai_tracks;
            if (invEstGenre && idxData.estimated_genre_tracks !== undefined) invEstGenre.innerText = idxData.estimated_genre_tracks;
            if (invNullPairs && idxData.null_pairs_cached !== undefined && idxData.null_pairs_cached !== null) invNullPairs.innerText = idxData.null_pairs_cached;

            updateDatasetManifestUI(idxData);
          }
        } catch (_) {}
      } else {
        throw new Error(`HTTP ${res.status}`);
      }
    } catch (err) {
      if (statusDot) statusDot.className = "w-2 h-2 rounded-full bg-amber-500";
      if (statusText) statusText.innerText = "Backend Offline · Run: uvicorn src.api:app";
      if (classifierText) classifierText.innerText = "Classifier: Offline";
    }

    try {
      const cfgRes = await fetch(`${API_BASE}/api/config`);
      if (cfgRes.ok) {
        serverConfig = await cfgRes.json();
      }
    } catch (_) {}
  }

  // -------------------------------------------------------------------------
  // 2. Query File Selection
  // -------------------------------------------------------------------------
  function setupFileInput() {
    const fileInput = document.getElementById("mainQueryFileInput");
    const label = document.getElementById("mainQueryFileLabel");
    const titleInput = document.getElementById("queryTitleInput");

    if (!fileInput) return;

    fileInput.addEventListener("change", () => {
      if (fileInput.files && fileInput.files[0]) {
        const file = fileInput.files[0];
        const sizeMb = (file.size / (1024 * 1024)).toFixed(2);
        if (label) label.innerText = `Selected: ${file.name} (${sizeMb} MB)`;
        if (titleInput && !titleInput.value) {
          titleInput.value = file.name.replace(/\.[^/.]+$/, "").replace(/_/g, " ");
        }
      } else {
        if (label) label.innerText = "Select or Drag Query Audio File";
      }
    });
  }

  // -------------------------------------------------------------------------
  // 3. Render Pipeline Analysis Results
  // -------------------------------------------------------------------------
  function renderPipelineResults(data, isDemo = false) {
    const verdictBanner = document.getElementById("verdictBannerSection");
    const verdictCodeBadge = document.getElementById("verdictCodeBadge");
    const caseRoutingBadge = document.getElementById("caseRoutingBadge");
    const verdictLabelText = document.getElementById("verdictLabelText");
    const verdictExplanationText = document.getElementById("verdictExplanationText");

    const indSimBadge = document.getElementById("indSimBadge");
    const indPvalBadge = document.getElementById("indPvalBadge");
    const indCreatBadge = document.getElementById("indCreatBadge");

    const v = data.final_verdict;
    const c = data.case_routing;

    // Verdict Styling
    let borderClass = "border-white/10";
    let bgBadgeClass = "bg-white/10 text-white";

    if (v.verdict_code === "critical") {
      borderClass = "border-rose-500/80 bg-rose-950/20";
      bgBadgeClass = "bg-rose-500 text-black";
    } else if (v.verdict_code === "warning") {
      borderClass = "border-amber-500/80 bg-amber-950/20";
      bgBadgeClass = "bg-amber-500 text-black";
    } else if (v.verdict_code === "advisory") {
      borderClass = "border-blue-500/80 bg-blue-950/20";
      bgBadgeClass = "bg-blue-500 text-black";
    } else if (v.verdict_code === "clear") {
      borderClass = "border-emerald-500/80 bg-emerald-950/20";
      bgBadgeClass = "bg-emerald-500 text-black";
    } else if (v.verdict_code === "incomplete") {
      borderClass = "border-purple-500/80 bg-purple-950/20";
      bgBadgeClass = "bg-purple-500 text-white font-bold";
    }

    if (verdictBanner) verdictBanner.className = `glass-card rounded-2xl p-6 border-2 ${borderClass} shadow-2xl relative overflow-hidden transition-all duration-500`;
    if (verdictCodeBadge) {
      verdictCodeBadge.className = `px-3 py-1 rounded-full text-xs font-mono font-bold uppercase tracking-wider ${bgBadgeClass}`;
      verdictCodeBadge.innerText = (isDemo ? "[DEMO] " : "") + `Verdict: ${v.verdict_code}`;
    }
    if (caseRoutingBadge) {
      caseRoutingBadge.innerText = `${c.case_type} · ${c.case_name}`;
      if (c.case_type === "Case U" || c.case_name.toLowerCase().includes("unresolved")) {
        caseRoutingBadge.className = "px-3 py-1 rounded-full text-xs font-mono font-semibold bg-purple-500/20 text-purple-300 border border-purple-500/40";
      } else {
        caseRoutingBadge.className = "px-3 py-1 rounded-full text-xs font-mono font-semibold bg-white/10 text-[#d4ff00] border border-[#d4ff00]/30";
      }
    }
    if (verdictLabelText) verdictLabelText.innerText = v.verdict_label;
    if (verdictExplanationText) verdictExplanationText.innerText = v.explanation;

    // Corpus Coverage & Accounting Breakdown Banner
    const corpusContainer = document.getElementById("corpusCoverageContainer");
    const corpusBadge = document.getElementById("corpusStatusBadge");
    const corpusSummaryText = document.getElementById("corpusSummaryText");
    const corpusNote = document.getElementById("corpusCoverageNote");
    const accTotal = document.getElementById("accTotalCorpus");
    const accChecked = document.getElementById("accPrefilterChecked");
    const accRetained = document.getElementById("accPrefilterRetained");
    const accDiscarded = document.getElementById("accPrefilterDiscarded");
    const accDtw = document.getElementById("accDtwEvaluated");
    const accFailures = document.getElementById("accLoadFailures");
    const accFailuresSub = document.getElementById("accLoadFailuresSub");
    const failureWarning = document.getElementById("corpusFailureWarning");
    const failureDetails = document.getElementById("corpusFailureDetails");

    if (corpusContainer) {
      corpusContainer.classList.remove("hidden");
      const isComplete = Boolean(data.is_index_complete);
      const totalIndexed = data.candidates_considered !== undefined && data.candidates_considered > 0
        ? data.candidates_considered
        : (data.total_corpus_size || 0);
      const retained = data.candidates_retained_prefilter !== undefined ? data.candidates_retained_prefilter : 0;
      const discarded = data.candidates_discarded_prefilter !== undefined
        ? data.candidates_discarded_prefilter
        : Math.max(0, totalIndexed - retained);
      const dtw = data.candidates_evaluated !== undefined ? data.candidates_evaluated : retained;
      const failed = data.candidates_failed_to_load !== undefined ? data.candidates_failed_to_load : 0;

      if (corpusBadge) {
        if (isComplete) {
          corpusBadge.className = "px-2.5 py-1 rounded-full text-[11px] font-bold uppercase tracking-wider bg-emerald-500/20 text-emerald-400 border border-emerald-500/40";
          corpusBadge.innerText = "Corpus: Complete (100% Indexed)";
        } else {
          corpusBadge.className = "px-2.5 py-1 rounded-full text-[11px] font-bold uppercase tracking-wider bg-amber-500/20 text-amber-300 border border-amber-500/40";
          corpusBadge.innerText = `Corpus: Partial (${totalIndexed} Tracks Indexed)`;
        }
      }

      if (corpusSummaryText) {
        corpusSummaryText.innerText = isComplete
          ? `Full dataset scan completed across all ${totalIndexed} tracks in SleepyJesse/ai_music_large.`
          : `Partial corpus scan: Evaluated against ${totalIndexed} tracks currently indexed.`;
      }

      if (corpusNote) {
        corpusNote.innerText = `Split: train · Dataset: SleepyJesse/ai_music_large`;
      }

      if (accTotal) accTotal.innerText = totalIndexed;
      if (accChecked) accChecked.innerText = totalIndexed;
      if (accRetained) accRetained.innerText = retained;
      if (accDiscarded) accDiscarded.innerText = discarded;
      if (accDtw) accDtw.innerText = dtw;
      if (accFailures) {
        accFailures.innerText = failed;
        if (failed > 0) {
          accFailures.className = "text-rose-400 font-bold text-sm block mt-0.5";
          if (accFailuresSub) accFailuresSub.innerText = `${failed} missing/corrupted`;
        } else {
          accFailures.className = "text-emerald-400 font-bold text-sm block mt-0.5";
          if (accFailuresSub) accFailuresSub.innerText = "Zero missing";
        }
      }

      if (failureWarning && failureDetails) {
        if (failed > 0) {
          failureWarning.classList.remove("hidden");
          const names = (data.failed_candidate_details || []).map(f => f.candidate_id || f.error).slice(0, 5).join(", ");
          failureDetails.innerText = `${failed} candidate feature file(s) failed to load (${names}${failed > 5 ? '...' : ''}). They were accounted for and excluded from DTW alignment.`;
        } else {
          failureWarning.classList.add("hidden");
        }
      }
    }

    // Feature 2 Data Reference for Null Count
    const topMimic = data.mimic_analysis.top_mimic_candidates[0];
    const topPval = topMimic && topMimic.p_value !== null ? topMimic.p_value : null;
    const nullPairsCount = data.null_distribution_sample_count !== undefined
      ? data.null_distribution_sample_count
      : (topMimic && topMimic.null_sample_count !== undefined ? topMimic.null_sample_count : 0);

    // Prominent Low Confidence / Insufficient Notice Banner
    const confNoticeEl = document.getElementById("verdictConfidenceNotice");
    const confNoticeTextEl = document.getElementById("verdictConfidenceText");
    const isInsufficient = v.verdict_code === "incomplete" || topPval === null;
    const isLowConf = isInsufficient ||
      (topMimic && topMimic.confidence === "Low") ||
      (data.candidates_evaluated !== undefined && data.candidates_evaluated < 30) ||
      Boolean(data.confidence_note);

    if (confNoticeEl) {
      if (isLowConf) {
        confNoticeEl.classList.remove("hidden");
        if (confNoticeTextEl) {
          if (isInsufficient) {
            confNoticeTextEl.innerText = data.confidence_note ||
              `Statistical Notice: Insufficient null population (N=${nullPairsCount}, minimum 5 required). Empirical p-value cannot be calculated; statistical gateway and completed verdict are incomplete.`;
          } else {
            confNoticeTextEl.innerText = data.confidence_note ||
              `Low Statistical Confidence Notice: Evaluated against a small candidate pool (${data.candidates_evaluated || 2} candidates, ${nullPairsCount} null comparisons). While analytical criteria triggered this provisional status, small sample sizes carry high sampling error and are insufficient for reliable population-level conclusions.`;
          }
        }
      } else {
        confNoticeEl.classList.add("hidden");
      }
    }

    // 3 Indicators
    function renderInd(el, label, val) {
      if (!el) return;
      if (val === true) {
        el.className = "flex items-center justify-between px-3 py-1.5 rounded-lg bg-rose-500/20 border border-rose-500/40 text-xs font-mono text-rose-300 font-bold";
        el.innerHTML = `<span>${label}</span><span>TRIGGERED (TRUE)</span>`;
      } else if (val === false) {
        el.className = "flex items-center justify-between px-3 py-1.5 rounded-lg bg-emerald-500/20 border border-emerald-500/40 text-xs font-mono text-emerald-300";
        el.innerHTML = `<span>${label}</span><span>NOT TRIGGERED (FALSE)</span>`;
      } else {
        el.className = "flex items-center justify-between px-3 py-1.5 rounded-lg bg-amber-500/20 border border-amber-500/40 text-xs font-mono text-amber-300";
        el.innerHTML = `<span>${label}</span><span>UNAVAILABLE (N &lt; 5)</span>`;
      }
    }

    renderInd(indSimBadge, "1. Similarity ≥ 0.85", v.indicator_similarity);
    renderInd(indPvalBadge, "2. Empirical p ≤ 0.01", v.indicator_p_value);
    renderInd(indCreatBadge, "3. Creative Score ≤ 0.24", v.indicator_creative);

    // Hero Cards
    // Feature 1
    const topSim = topMimic ? topMimic.similarity : 0.0;
    const topShift = topMimic ? topMimic.best_shift : 0;
    const topTier = topMimic ? topMimic.similarity_tier : "Negligible";

    const simScoreEl = document.getElementById("card1SimScore");
    const shiftBadgeEl = document.getElementById("card1ShiftBadge");
    const tierLabelEl = document.getElementById("card1TierLabel");
    const filterLabelEl = document.getElementById("card1FilterLabel");
    const card1DistLabel = document.getElementById("card1DistLabel");

    if (simScoreEl) simScoreEl.innerText = topSim.toFixed(3);
    if (shiftBadgeEl) shiftBadgeEl.innerText = `Shift: ${topShift > 0 ? "+" : ""}${topShift} semi`;
    if (tierLabelEl) tierLabelEl.innerText = topTier;

    // Prefilter reduction display
    const pct = data.cosine_prefilter ? Number(data.cosine_prefilter.percentage_removed) : 0.0;
    const clampedPct = Math.abs(pct) < 1e-4 ? 0.0 : Math.max(0.0, pct);
    if (filterLabelEl) {
      if (data.candidates_considered !== undefined && data.candidates_retained_prefilter !== undefined) {
        filterLabelEl.innerText = `Prefilter: ${data.candidates_retained_prefilter}/${data.candidates_considered} retained (${clampedPct === 0.0 ? "0%" : `-${clampedPct.toFixed(1)}%`})`;
      } else {
        filterLabelEl.innerText = clampedPct === 0.0 ? "Prefilter: 0% reduction" : `Prefilter: -${clampedPct.toFixed(1)}%`;
      }
    }

    // Normalized Distance display
    if (card1DistLabel) {
      if (topMimic && topMimic.normalized_distance !== null && topMimic.normalized_distance !== undefined) {
        card1DistLabel.innerText = `Norm Dist: ${Number(topMimic.normalized_distance).toFixed(3)}`;
      } else if (topSim > 0.0) {
        const derivedDist = (1.0 / topSim) - 1.0;
        card1DistLabel.innerText = `Norm Dist: ${derivedDist.toFixed(3)}`;
      } else {
        card1DistLabel.innerText = "Norm Dist: —";
      }
    }

    // Feature 2
    const pvalEl = document.getElementById("card2Pval");
    const confBadgeEl = document.getElementById("card2ConfidenceBadge");
    const sevLabelEl = document.getElementById("card2SeverityLabel");
    const sampleCountEl = document.getElementById("card2SampleCount");
    const fallbackSrcEl = document.getElementById("card2FallbackSource");

    const topConf = topMimic ? topMimic.confidence : (nullPairsCount < 5 ? "Insufficient" : "Low");

    if (pvalEl) pvalEl.innerText = topPval !== null ? `p = ${topPval.toFixed(4)}` : "Unavailable";
    const adjPvalEl = document.getElementById("card2AdjPval");
    if (adjPvalEl) {
      if (topMimic && topMimic.adjusted_p_value !== null && topMimic.adjusted_p_value !== undefined) {
        adjPvalEl.innerText = `FDR q: ${Number(topMimic.adjusted_p_value).toFixed(4)}`;
      } else {
        adjPvalEl.innerText = "FDR q: —";
      }
    }
    if (confBadgeEl) {
      confBadgeEl.innerText = `Conf: ${topConf}`;
      confBadgeEl.className = topConf.toLowerCase() === "insufficient"
        ? "px-2 py-0.5 rounded text-[10px] font-mono font-bold bg-amber-500/20 text-amber-300 border border-amber-500/30"
        : "px-2 py-0.5 rounded text-[10px] font-mono font-bold bg-white/10 text-white border border-white/10";
    }
    if (sevLabelEl) {
      if (topPval === null) {
        sevLabelEl.innerText = `Insufficient null population (N=${nullPairsCount}, minimum 5 required)`;
        sevLabelEl.className = "text-xs font-semibold text-amber-400 mt-1";
      } else {
        sevLabelEl.innerText = topMimic ? topMimic.severity_label : "No Survivors";
        sevLabelEl.className = topPval <= 0.01 ? "text-xs font-semibold text-rose-400 mt-1" : "text-xs font-semibold text-[#d4ff00] mt-1";
      }
    }
    if (sampleCountEl) sampleCountEl.innerText = `Null Pairs: ${nullPairsCount}`;
    if (fallbackSrcEl) fallbackSrcEl.innerText = `Source: ${topMimic ? (topMimic.fallback_source || "global_null") : "global_null"}`;

    // Feature 3: Precision consistency (Requirement 7)
    const cr = data.creative_score;
    const broadEl = document.getElementById("card3BroadScore");
    const tierBadgeEl = document.getElementById("card3TierBadge");
    const noveltyTierEl = document.getElementById("card3NoveltyTier");
    const narrowEl = document.getElementById("card3NarrowScore");
    const normEl = document.getElementById("card3NormScore");

    if (broadEl) broadEl.innerText = cr.broad_score_percent !== null ? cr.broad_score_percent.toFixed(2) : "—";
    if (tierBadgeEl) tierBadgeEl.innerText = cr.is_available ? "Score Ready" : "Unavailable";
    if (noveltyTierEl) noveltyTierEl.innerText = cr.score_tier;
    if (narrowEl) narrowEl.innerText = `Narrow: ${cr.narrow_score_percent !== null ? cr.narrow_score_percent.toFixed(2) + "%" : "N/A"}`;
    if (normEl) normEl.innerText = `Norm: ${cr.broad_score_normalized !== null ? cr.broad_score_normalized.toFixed(4) : "—"}`;

    // Feature 4
    const oc = data.origin_classifier;
    const predEl = document.getElementById("card4PredictedOrigin");
    const ocConfEl = document.getElementById("card4ConfidenceLabel");
    const probAiEl = document.getElementById("card4ProbAi");
    const ocStatusTag = document.getElementById("card4StatusTag");

    if (predEl) predEl.innerText = oc.predicted_origin;
    if (ocConfEl) ocConfEl.innerText = oc.confidence !== null ? `Confidence: ${(oc.confidence * 100).toFixed(1)}%` : oc.model_status;
    if (probAiEl) probAiEl.innerText = oc.probability_ai !== null ? `AI Prob: ${(oc.probability_ai * 100).toFixed(1)}%` : "AI Prob: —";
    if (ocStatusTag) ocStatusTag.innerText = oc.model_status.includes("Active") ? "Active" : "Unconfigured";

    // Stage 3 Genre Influence Map
    const genreContainer = document.getElementById("genreDistributionContainer");
    const genreCrossBadge = document.getElementById("genreCrossBadge");

    if (genreCrossBadge) {
      genreCrossBadge.innerText = data.genre_map.is_cross_genre ? "Cross-Genre Detected" : "Native Genre Only";
      genreCrossBadge.className = data.genre_map.is_cross_genre
        ? "px-2.5 py-0.5 rounded-full text-[10px] font-mono font-bold bg-amber-500/20 text-amber-300 border border-amber-500/30"
        : "px-2.5 py-0.5 rounded-full text-[10px] font-mono font-bold bg-white/10 text-white border border-white/10";
    }

    if (genreContainer) {
      if (data.genre_map.distribution && data.genre_map.distribution.length > 0) {
        let gHtml = "";
        if (data.candidates_evaluated !== undefined && data.candidates_evaluated < 30) {
          gHtml += `
            <div class="px-2.5 py-1.5 rounded-lg bg-amber-500/10 border border-amber-500/20 text-amber-300 text-[10px] font-mono mb-2">
              ⚠️ Small Sample Notice: Evaluated on ${data.candidates_evaluated} candidates. Insufficient evidence to infer population-level genre distribution.
            </div>
          `;
        }
        data.genre_map.distribution.forEach((item) => {
          gHtml += `
            <div class="space-y-1">
              <div class="flex items-center justify-between text-xs font-mono">
                <span class="text-white font-medium">${item.genre} (${item.count} tracks)</span>
                <span class="text-[#d4ff00] font-bold">${item.percentage.toFixed(1)}%</span>
              </div>
              <div class="w-full bg-white/10 rounded-full h-2 overflow-hidden">
                <div class="bg-[#d4ff00] h-full rounded-full transition-all duration-700" style="width: ${item.percentage}%"></div>
              </div>
            </div>
          `;
        });
        genreContainer.innerHTML = gHtml;
      } else {
        genreContainer.innerHTML = `<div class="text-xs font-mono text-[#9ba1a6] text-center py-4">No genre metadata identified among survivors.</div>`;
      }
    }

    // Feature 5 Musical Context
    const metaKeyEl = document.getElementById("ctxEstimatedKey");
    const metaTempoEl = document.getElementById("ctxEstimatedTempo");
    const metaDurEl = document.getElementById("ctxDuration");
    const metaCosEl = document.getElementById("ctxCosineReduction");

    if (metaKeyEl) metaKeyEl.innerText = data.query_track.estimated_key || "Unknown";
    if (metaTempoEl) metaTempoEl.innerText = data.query_track.estimated_tempo ? `${data.query_track.estimated_tempo} BPM` : "— BPM";
    if (metaDurEl) metaDurEl.innerText = `${data.query_track.duration_sec.toFixed(1)}s`;
    if (metaCosEl) {
      metaCosEl.innerText = clampedPct === 0.0 ? "0% removed" : `-${clampedPct.toFixed(1)}% removed`;
    }

    // Section 4B: Extra Audio Features & MERT Representation
    const extraH = document.getElementById("extraHarmonicRatio");
    const extraP = document.getElementById("extraPercussiveRatio");
    const extraRoll = document.getElementById("extraRolloff");
    const extraRhythm = document.getElementById("extraRhythmRegularity");

    if (data.extra_features) {
      if (extraH && data.extra_features.hpss_harmonic_ratio !== undefined && data.extra_features.hpss_harmonic_ratio !== null) {
        extraH.innerText = Number(data.extra_features.hpss_harmonic_ratio).toFixed(3);
      }
      if (extraP && data.extra_features.hpss_percussive_ratio !== undefined && data.extra_features.hpss_percussive_ratio !== null) {
        extraP.innerText = Number(data.extra_features.hpss_percussive_ratio).toFixed(3);
      }
      if (extraRoll && data.extra_features.spectral_rolloff_mean !== undefined && data.extra_features.spectral_rolloff_mean !== null) {
        extraRoll.innerText = `${Math.round(data.extra_features.spectral_rolloff_mean)} Hz`;
      }
      if (extraRhythm && data.extra_features.rhythm_regularity !== undefined && data.extra_features.rhythm_regularity !== null) {
        extraRhythm.innerText = Number(data.extra_features.rhythm_regularity).toFixed(3);
      }
    }

    const mertBadge = document.getElementById("mertStatusBadge");
    if (mertBadge && data.mert_status) {
      if (data.mert_status.is_supported) {
        mertBadge.innerText = `Active (${(data.mert_status.device || 'CPU').toUpperCase()})`;
        mertBadge.className = "px-2.5 py-1 rounded-full text-[10px] font-mono font-bold bg-emerald-500/20 text-emerald-300 border border-emerald-500/30";
      } else {
        mertBadge.innerText = data.mert_status.status || "Optional (Offline)";
        mertBadge.className = "px-2.5 py-1 rounded-full text-[10px] font-mono font-bold bg-amber-500/20 text-amber-300 border border-amber-500/30";
      }
    }

    // Feature 6 Mimic Candidates Table
    const tableBody = document.getElementById("mimicCandidateTableBody");
    const countBadge = document.getElementById("mimicCandidateCountBadge");

    if (countBadge) countBadge.innerText = `${data.mimic_analysis.top_mimic_candidates.length} Evaluated Survivors`;

    if (tableBody) {
      if (data.mimic_analysis.top_mimic_candidates.length > 0) {
        let tHtml = "";
        data.mimic_analysis.top_mimic_candidates.forEach((cand, idx) => {
          const shiftStr = (cand.best_shift > 0 ? "+" : "") + cand.best_shift + " semi";
          const pvalStr = cand.p_value !== null ? cand.p_value.toFixed(4) : "Unavailable";

          let relBadge = `<span class="px-2 py-0.5 rounded bg-white/10 text-white">${cand.artist_relationship || "Unknown Artist"}</span>`;
          if (cand.artist_relationship === "Consistent Artist Style") {
            relBadge = `<span class="px-2 py-0.5 rounded bg-emerald-500/20 text-emerald-300 font-bold border border-emerald-500/30">Consistent Artist Style</span>`;
          } else if (cand.artist_relationship === "Different Artist") {
            relBadge = `<span class="px-2 py-0.5 rounded bg-rose-500/20 text-rose-300 font-bold border border-rose-500/30">Different Artist</span>`;
          } else {
            relBadge = `<span class="px-2 py-0.5 rounded bg-white/10 text-[#9ba1a6]">Unknown Artist</span>`;
          }

          let warnBadge = "";
          if (cand.mimic_warning_active) {
            warnBadge = `<span class="px-2 py-0.5 rounded bg-rose-500/20 text-rose-400 font-bold border border-rose-500/40">Active Warning</span>`;
          } else if (cand.p_value === null) {
            warnBadge = `<span class="px-2 py-0.5 rounded bg-amber-500/20 text-amber-300 font-mono text-[10px]">Statistical Gateway Unavailable</span>`;
          } else if (cand.artist_relationship === "Consistent Artist Style") {
            warnBadge = `<span class="px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-400">Suppressed (Same Artist)</span>`;
          } else if (!cand.artist || cand.artist === "Unknown" || cand.artist_relationship === "Unknown Artist") {
            warnBadge = `<span class="px-2 py-0.5 rounded bg-white/10 text-[#9ba1a6]">Unknown — relationship not assessed</span>`;
          } else {
            warnBadge = `<span class="px-2 py-0.5 rounded bg-white/10 text-[#9ba1a6]">No Warning</span>`;
          }

          const candDist = cand.normalized_distance !== null && cand.normalized_distance !== undefined
            ? Number(cand.normalized_distance)
            : (cand.similarity > 0 ? (1.0 / cand.similarity - 1.0) : null);
          const distLabel = candDist !== null ? ` · Dist: ${candDist.toFixed(3)}` : " · Dist: —";

          // Duration disparity warning
          let disparityBadge = "";
          if (cand.duration_disparity_warning) {
            disparityBadge = `
              <div class="mt-1 px-2 py-0.5 rounded bg-amber-500/20 text-amber-300 border border-amber-500/30 text-[10px] font-mono flex items-center gap-1">
                <span class="text-xs">⚠️</span>
                <span>${cand.disparity_explanation || 'Duration disparity ≥ 2.5x: DTW path stretching may distort similarity.'}</span>
              </div>
            `;
          }

          // Alignment Coverage cell
          let covCell = `<span class="text-[#9ba1a6]">Unavailable</span>`;
          if (cand.alignment_coverage !== null && cand.alignment_coverage !== undefined) {
            const pctVal = cand.coverage_percent !== null && cand.coverage_percent !== undefined
              ? cand.coverage_percent.toFixed(1)
              : (cand.alignment_coverage * 100).toFixed(1);
            const interp = cand.coverage_interpretation || "Coverage evaluated";
            let interpColor = "text-[#9ba1a6]";
            if (cand.alignment_coverage > 0.50) interpColor = "text-rose-400 font-bold";
            else if (cand.alignment_coverage > 0.20) interpColor = "text-amber-400";
            else interpColor = "text-blue-300";

            covCell = `
              <span class="font-bold text-white">${pctVal}%</span>
              <span class="text-[10px] ${interpColor} block">${interp}</span>
            `;
          }

          const candDurStr = cand.duration_sec !== null && cand.duration_sec !== undefined
            ? ` · Dur: ${cand.duration_sec.toFixed(1)}s`
            : "";

          tHtml += `
            <tr class="hover:bg-white/5 transition-colors">
              <td class="py-3 px-3">
                <div class="font-bold text-white">#${idx + 1} · ${cand.title || cand.candidate_id}</div>
                <div class="text-[10px] text-[#9ba1a6]">Artist: ${cand.artist || "Unknown"} · Genre: ${cand.genre || "General"}${candDurStr}</div>
                ${disparityBadge}
              </td>
              <td class="py-3 px-3">
                <span class="font-bold text-white">${(cand.similarity * 100).toFixed(1)}%</span>
                <span class="text-[10px] text-[#9ba1a6] block">${cand.similarity_tier}${distLabel}</span>
              </td>
              <td class="py-3 px-3">
                ${covCell}
              </td>
              <td class="py-3 px-3">
                <span class="font-bold ${cand.p_value !== null && cand.p_value <= 0.01 ? "text-rose-400" : (cand.p_value === null ? "text-amber-400" : "text-white")}">${pvalStr}</span>
                <span class="text-[10px] text-[#9ba1a6] block">${cand.severity_label}</span>
                ${cand.adjusted_p_value !== null && cand.adjusted_p_value !== undefined ? `<span class="text-[10px] text-cyan-300 block font-mono">FDR q: ${Number(cand.adjusted_p_value).toFixed(4)}</span>` : ""}
              </td>
              <td class="py-3 px-3 text-[#9ba1a6]">${shiftStr}</td>
              <td class="py-3 px-3">${relBadge}</td>
              <td class="py-3 px-3">${warnBadge}</td>
            </tr>
          `;
        });
        tableBody.innerHTML = tHtml;
      } else {
        tableBody.innerHTML = `<tr><td colspan="7" class="text-center py-6 text-[#9ba1a6]">No candidates survived the statistical gateway (Case C - No Match).</td></tr>`;
      }
    }

    // Feature 7 Evidence Timeline & Studio Suggestions
    const timelineContainer = document.getElementById("evidenceTimelineContainer");
    if (timelineContainer) {
      if (data.evidence_timeline && data.evidence_timeline.length > 0) {
        let tlHtml = "";
        const isProvisional = topPval === null || nullPairsCount < 5 || (c && c.case_type === "Case U");
        if (isProvisional) {
          tlHtml += `
            <div class="p-3 rounded-lg bg-amber-500/10 border border-amber-500/30 text-amber-300 text-xs font-mono mb-3 flex items-start gap-2">
              <span class="material-symbols-outlined text-sm mt-0.5 flex-shrink-0">info</span>
              <span>Provisional Analysis: Statistical gateway is unavailable (N &lt; 5). Identified high-similarity intervals and studio recommendations are advisory estimates.</span>
            </div>
          `;
        }
        data.evidence_timeline.forEach((item, idx) => {
          let segHtml = "";
          if (item.flagged_segments && item.flagged_segments.length > 0) {
            item.flagged_segments.forEach((seg) => {
              segHtml += `
                <div class="bg-white/5 border border-white/10 rounded-lg p-2.5 font-mono text-[11px] flex flex-col justify-between">
                  <div class="flex items-center justify-between text-[#d4ff00]">
                    <span>Interval [${seg.start_time_a.toFixed(1)}s – ${seg.end_time_a.toFixed(1)}s]</span>
                    <span class="font-bold">S = ${(seg.segment_similarity * 100).toFixed(1)}%</span>
                  </div>
                  <div class="text-[10px] text-[#9ba1a6] mt-1">Matched in Candidate: [${seg.start_time_b.toFixed(1)}s – ${seg.end_time_b.toFixed(1)}s]</div>
                </div>
              `;
            });
          } else {
            segHtml = `<div class="text-xs font-mono text-[#9ba1a6]">No concentrated high-similarity segments exceeded threshold.</div>`;
          }

          let sugHtml = "";
          item.studio_suggestions.forEach((sug) => {
            sugHtml += `
              <li class="flex items-start gap-2 text-xs text-[#f0f1f2]">
                <span class="material-symbols-outlined text-[16px] text-[#d4ff00] mt-0.5 flex-shrink-0">check_circle</span>
                <span>${sug}</span>
              </li>
            `;
          });

          tlHtml += `
            <div class="bg-white/5 border border-white/10 rounded-xl p-4 space-y-3">
              <div class="flex items-center justify-between border-b border-white/10 pb-2">
                <div>
                  <span class="font-bold text-sm text-white">Target: ${item.candidate_title}</span>
                  <span class="text-xs text-[#9ba1a6] font-mono ml-2">(Overall S = ${(item.similarity * 100).toFixed(1)}%, Shift: +${item.best_shift})</span>
                </div>
                <span class="px-2 py-0.5 rounded text-[10px] font-mono font-bold bg-rose-500/20 text-rose-400 border border-rose-500/30">${item.severity}</span>
              </div>

              <div>
                <span class="text-[10px] font-mono uppercase tracking-wider text-[#9ba1a6] block mb-2">High-Overlap Time Segments</span>
                <div class="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-2">
                  ${segHtml}
                </div>
              </div>

              <div class="pt-2">
                <span class="text-[10px] font-mono uppercase tracking-wider text-[#9ba1a6] block mb-2">Evidence-Grounded Studio Recommendations</span>
                <ul class="space-y-1.5">
                  ${sugHtml}
                </ul>
              </div>
            </div>
          `;
        });
        timelineContainer.innerHTML = tlHtml;
      } else {
        timelineContainer.innerHTML = `<div class="text-xs font-mono text-[#9ba1a6] text-center py-6">No matching candidate timeline generated (Case C - No Match).</div>`;
      }
    }
  }

  // -------------------------------------------------------------------------
  // 4. Run Analysis Action (Asynchronous with 13-Stage SSE & Polling Progress)
  // -------------------------------------------------------------------------
  function setupAnalysisAction() {
    const btn = document.getElementById("runFullAnalysisBtn");
    const fileInput = document.getElementById("mainQueryFileInput");
    const titleInput = document.getElementById("queryTitleInput");
    const artistInput = document.getElementById("queryArtistInput");
    const genreInput = document.getElementById("queryGenreInput");
    const feedback = document.getElementById("pipelineStatusFeedback");
    const btnText = document.getElementById("runAnalysisBtnText");
    const btnIcon = document.getElementById("runAnalysisBtnIcon");

    // Live progress elements
    const progressContainer = document.getElementById("liveProgressContainer");
    const progressSpinner = document.getElementById("liveProgressSpinner");
    const progressStageName = document.getElementById("liveProgressStageName");
    const progressPercent = document.getElementById("liveProgressPercent");
    const progressBar = document.getElementById("liveProgressBar");
    const progressMessage = document.getElementById("liveProgressMessage");
    const progressCounts = document.getElementById("liveProgressCounts");

    if (!btn) return;

    function resetBtn() {
      btn.disabled = false;
      if (btnText) btnText.innerText = "Execute Pipeline Analysis";
      if (btnIcon) btnIcon.className = "material-symbols-outlined text-[18px]";
    }

    function setBtnLoading(msg) {
      btn.disabled = true;
      if (btnText) btnText.innerText = msg || "Running Analysis Pipeline...";
      if (btnIcon) btnIcon.className = "material-symbols-outlined text-[18px] animate-spin";
      if (progressSpinner) {
        progressSpinner.className = "material-symbols-outlined text-base text-[#d4ff00] animate-spin";
        progressSpinner.innerText = "progress_activity";
      }
    }

    function updateProgressUI(stageNum, stageName, pct, msg, processed, total) {
      if (progressContainer) progressContainer.classList.remove("hidden");
      if (progressStageName) progressStageName.innerText = `Stage ${stageNum}/13: ${stageName}`;
      if (progressPercent) progressPercent.innerText = `${pct.toFixed(1)}%`;
      if (progressBar) progressBar.style.width = `${pct}%`;
      if (progressMessage) progressMessage.innerText = msg || "Processing...";
      if (progressCounts) {
        progressCounts.innerText = total > 0 ? `${processed} / ${total} items` : `${stageNum} / 13 stages`;
      }
    }

    btn.addEventListener("click", async () => {
      if (!fileInput || !fileInput.files || !fileInput.files[0]) {
        if (feedback) {
          feedback.innerText = "Please select a query audio file first.";
          feedback.className = "text-xs font-mono min-h-[20px] text-center text-rose-400 font-bold pt-1";
        }
        return;
      }

      const file = fileInput.files[0];
      const formData = new FormData();
      formData.append("query_audio", file);
      if (titleInput && titleInput.value.trim()) formData.append("title", titleInput.value.trim());
      if (artistInput && artistInput.value.trim()) formData.append("artist", artistInput.value.trim());
      if (genreInput && genreInput.value.trim()) formData.append("genre", genreInput.value.trim());

      setBtnLoading("Starting Analysis Worker...");
      if (feedback) {
        feedback.innerText = "Initializing 13-stage pipeline against SleepyJesse/ai_music_large corpus...";
        feedback.className = "text-xs font-mono min-h-[20px] text-center text-[#d4ff00] font-semibold pt-1";
      }

      updateProgressUI(1, "Intake & Audio Decoding", 0.0, "Uploading query audio file...", 0, 0);

      // Attempt async job route first with real-time SSE progress
      try {
        const jobRes = await fetch(`${API_BASE}/api/jobs/analyze`, {
          method: "POST",
          body: formData,
        });

        if (jobRes.ok) {
          const jobData = await jobRes.json();
          const jobId = jobData.job_id;
          let jobFinished = false;
          let pollTimer = null;

          function handleJobCompletion(result) {
            if (jobFinished) return;
            jobFinished = true;
            if (pollTimer) clearInterval(pollTimer);
            if (progressSpinner) {
              progressSpinner.className = "material-symbols-outlined text-base text-emerald-400 font-bold";
              progressSpinner.innerText = "check_circle";
            }
            updateProgressUI(13, "Pipeline Complete", 100.0, "Analysis complete!", 13, 13);
            renderPipelineResults(result, false);
            if (feedback) {
              const considered = result.candidates_considered || result.total_corpus_size || 0;
              feedback.innerText = `Analysis complete! Evaluated ${result.candidates_evaluated} candidates across ${considered} indexed dataset tracks.`;
              feedback.className = "text-xs font-mono min-h-[20px] text-center text-emerald-400 font-semibold pt-1";
            }
            resetBtn();
          }

          function handleJobFailure(errMessage) {
            if (jobFinished) return;
            jobFinished = true;
            if (pollTimer) clearInterval(pollTimer);
            if (progressSpinner) {
              progressSpinner.className = "material-symbols-outlined text-base text-rose-400 font-bold";
              progressSpinner.innerText = "error";
            }
            console.error("Job failed:", errMessage);
            if (feedback) {
              feedback.innerText = `Pipeline Error: ${errMessage}`;
              feedback.className = "text-xs font-mono min-h-[20px] text-center text-rose-400 font-bold pt-1";
            }
            resetBtn();
          }

          // Polling fallback
          function startPolling() {
            if (pollTimer || jobFinished) return;
            pollTimer = setInterval(async () => {
              if (jobFinished) {
                clearInterval(pollTimer);
                return;
              }
              try {
                const sRes = await fetch(`${API_BASE}/api/jobs/${jobId}/status`);
                if (!sRes.ok) return;
                const s = await sRes.json();
                const pct = Math.min(100, Math.max(0, s.progress_pct || 0));
                updateProgressUI(s.stage || 1, s.stage_name || "Running", pct, s.message || "", s.processed_count || 0, s.total_count || 0);

                if (s.status === "completed" && s.result) {
                  handleJobCompletion(s.result);
                } else if (s.status === "failed") {
                  handleJobFailure(s.error || "Analysis pipeline failed.");
                }
              } catch (_) {}
            }, 600);
          }

          // Open SSE stream
          try {
            const sse = new EventSource(`${API_BASE}/api/jobs/${jobId}/events`);
            sse.onmessage = (evt) => {
              try {
                const ev = JSON.parse(evt.data);
                const stageNum = ev.stage || 1;
                const stageName = ev.stage_name || `Stage ${stageNum}`;
                const pct = Math.min(100, Math.max(0, ev.progress_pct || 0));
                const msg = ev.message || "";
                const processed = ev.processed_count || 0;
                const total = ev.total_count || 0;

                updateProgressUI(stageNum, stageName, pct, msg, processed, total);

                if (ev.done || ev.status === "completed") {
                  sse.close();
                  if (ev.result) {
                    handleJobCompletion(ev.result);
                  } else {
                    startPolling();
                  }
                } else if (ev.status === "failed") {
                  sse.close();
                  handleJobFailure(ev.error || "Analysis pipeline failed.");
                }
              } catch (e) {
                console.error("Error processing SSE event:", e);
              }
            };

            sse.onerror = () => {
              sse.close();
              if (!jobFinished) {
                startPolling();
              }
            };
          } catch (_) {
            startPolling();
          }

          return;
        }
      } catch (_) {
        // Fall back to synchronous endpoint
      }

      // Synchronous fallback
      try {
        setBtnLoading("Analyzing Harmonic Space & DTW Alignment...");
        const res = await fetch(`${API_BASE}/api/analyze`, {
          method: "POST",
          body: formData,
        });

        if (!res.ok) {
          const errData = await res.json().catch(() => ({}));
          throw new Error(errData.detail || `Server error (HTTP ${res.status})`);
        }

        const data = await res.json();
        if (progressSpinner) {
          progressSpinner.className = "material-symbols-outlined text-base text-emerald-400 font-bold";
          progressSpinner.innerText = "check_circle";
        }
        updateProgressUI(13, "Pipeline Complete", 100.0, "Analysis complete!", 13, 13);
        renderPipelineResults(data, false);

        if (feedback) {
          const considered = data.candidates_considered || data.total_corpus_size || 0;
          feedback.innerText = `Analysis complete! Evaluated ${data.candidates_evaluated} candidates across ${considered} indexed dataset tracks.`;
          feedback.className = "text-xs font-mono min-h-[20px] text-center text-emerald-400 font-semibold pt-1";
        }
      } catch (err) {
        console.error("Pipeline analysis failure:", err);
        if (progressSpinner) {
          progressSpinner.className = "material-symbols-outlined text-base text-rose-400 font-bold";
          progressSpinner.innerText = "error";
        }
        if (feedback) {
          feedback.innerText = `Analysis Error: ${err.message}`;
          feedback.className = "text-xs font-mono min-h-[20px] text-center text-rose-400 font-bold pt-1";
        }
      } finally {
        resetBtn();
      }
    });
  }

  // -------------------------------------------------------------------------
  // 5. Dataset Streaming & Scanner Controller
  // -------------------------------------------------------------------------
  let scanEventSource = null;

  function updateDatasetManifestUI(data) {
    if (!data) return;
    const statusBadge = document.getElementById("dsScanStatusBadge");
    const currentOpText = document.getElementById("dsCurrentOpText");
    const scannedCount = document.getElementById("dsScannedCount");
    const audioIndexedCount = document.getElementById("dsAudioIndexedCount");
    const reusedCount = document.getElementById("dsReusedCount");
    const duplicateCount = document.getElementById("dsDuplicateCount");
    const failedCount = document.getElementById("dsFailedCount");
    const originBreakdown = document.getElementById("dsOriginBreakdown");
    const genreBreakdown = document.getElementById("dsGenreBreakdown");
    const elapsedText = document.getElementById("dsElapsedText");
    const scanStopBtn = document.getElementById("dsScanStopBtn");
    const scanSampleBtn = document.getElementById("dsScanSampleBtn");
    const scanFullBtn = document.getElementById("dsScanFullBtn");
    const progressBarContainer = document.getElementById("dsProgressBarContainer");
    const scanMsgText = document.getElementById("dsScanMsgText");
    const scanStatsText = document.getElementById("dsScanStatsText");

    if (scannedCount) scannedCount.innerText = data.records_scanned || 0;
    if (audioIndexedCount) audioIndexedCount.innerText = data.total_indexed || data.audio_indexed || 0;
    if (reusedCount) reusedCount.innerText = data.existing_reused || 0;
    if (duplicateCount) duplicateCount.innerText = data.duplicate_records || 0;
    if (failedCount) failedCount.innerText = data.failed_records || data.records_awaiting_retry || 0;
    if (originBreakdown) originBreakdown.innerText = `${data.human_tracks || 0} / ${data.ai_tracks || 0}`;
    if (genreBreakdown) genreBreakdown.innerText = `${data.verified_genre_tracks || 0} / ${data.estimated_genre_tracks || 0} / ${data.unknown_genre_tracks || 0}`;
    if (elapsedText) elapsedText.innerText = `${(data.elapsed_sec || 0).toFixed(1)}s`;

    if (currentOpText && data.current_operation) {
      currentOpText.innerText = data.current_operation;
    }

    const isScanning = data.is_scanning || data.status === "Running";

    if (statusBadge) {
      statusBadge.innerText = `Status: ${data.status || (isScanning ? "Running" : "Ready")}`;
      if (isScanning) {
        statusBadge.className = "px-2.5 py-1 rounded-full text-xs font-bold uppercase tracking-wider bg-amber-500/20 text-amber-300 border border-amber-500/40 animate-pulse";
      } else if (data.status === "Complete") {
        statusBadge.className = "px-2.5 py-1 rounded-full text-xs font-bold uppercase tracking-wider bg-emerald-500/20 text-emerald-300 border border-emerald-500/40";
      } else if (data.status === "Completed with Errors") {
        statusBadge.className = "px-2.5 py-1 rounded-full text-xs font-bold uppercase tracking-wider bg-rose-500/20 text-rose-300 border border-rose-500/40";
      } else {
        statusBadge.className = "px-2.5 py-1 rounded-full text-xs font-bold uppercase tracking-wider bg-white/10 text-white";
      }
    }

    if (progressBarContainer) {
      if (isScanning) {
        progressBarContainer.classList.remove("hidden");
        if (scanMsgText) scanMsgText.innerText = data.current_operation || "Streaming dataset...";
        if (scanStatsText) scanStatsText.innerText = `${data.audio_indexed || 0} indexed · ${data.records_scanned || 0} scanned`;
      } else {
        progressBarContainer.classList.add("hidden");
      }
    }

    if (scanStopBtn) {
      if (isScanning) scanStopBtn.classList.remove("hidden");
      else scanStopBtn.classList.add("hidden");
    }
    if (scanSampleBtn) scanSampleBtn.disabled = isScanning;
    if (scanFullBtn) scanFullBtn.disabled = isScanning;
  }

  function setupDatasetScanner() {
    const scanSampleBtn = document.getElementById("dsScanSampleBtn");
    const scanFullBtn = document.getElementById("dsScanFullBtn");
    const scanStopBtn = document.getElementById("dsScanStopBtn");

    function listenScanEvents() {
      if (scanEventSource) scanEventSource.close();
      try {
        scanEventSource = new EventSource(`${API_BASE}/api/dataset/scan/events`);
        scanEventSource.onmessage = (evt) => {
          try {
            const ev = JSON.parse(evt.data);
            if (ev.type === "complete" || ev.type === "finished" || ev.type === "error") {
              scanEventSource.close();
              scanEventSource = null;
              fetchHealthAndStatus();
            } else if (ev.type === "progress") {
              const currentOpText = document.getElementById("dsCurrentOpText");
              const scanMsgText = document.getElementById("dsScanMsgText");
              const scanStatsText = document.getElementById("dsScanStatsText");
              const scannedCount = document.getElementById("dsScannedCount");
              const audioIndexedCount = document.getElementById("dsAudioIndexedCount");

              if (scanMsgText && ev.current_operation) scanMsgText.innerText = ev.current_operation;
              if (currentOpText && ev.current_operation) currentOpText.innerText = ev.current_operation;
              if (scanStatsText) scanStatsText.innerText = `${ev.processed_records || 0} indexed · ${ev.records_scanned || 0} scanned`;
              if (scannedCount && ev.records_scanned !== undefined) scannedCount.innerText = ev.records_scanned;
              if (audioIndexedCount && ev.processed_records !== undefined) audioIndexedCount.innerText = ev.processed_records;
            }
          } catch (_) {}
        };
        scanEventSource.onerror = () => {
          if (scanEventSource) scanEventSource.close();
          scanEventSource = null;
          fetchHealthAndStatus();
        };
      } catch (_) {}
    }

    if (scanSampleBtn) {
      scanSampleBtn.addEventListener("click", async () => {
        try {
          scanSampleBtn.disabled = true;
          const res = await fetch(`${API_BASE}/api/dataset/scan/start`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ full: false, sample_size: 20, resume: true }),
          });
          listenScanEvents();
          setTimeout(fetchHealthAndStatus, 500);
        } catch (err) {
          console.error("Failed to start sample scan:", err);
          scanSampleBtn.disabled = false;
        }
      });
    }

    if (scanFullBtn) {
      scanFullBtn.addEventListener("click", async () => {
        try {
          scanFullBtn.disabled = true;
          const res = await fetch(`${API_BASE}/api/dataset/scan/start`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ full: true, resume: true }),
          });
          listenScanEvents();
          setTimeout(fetchHealthAndStatus, 500);
        } catch (err) {
          console.error("Failed to start full scan:", err);
          scanFullBtn.disabled = false;
        }
      });
    }

    if (scanStopBtn) {
      scanStopBtn.addEventListener("click", async () => {
        try {
          await fetch(`${API_BASE}/api/dataset/scan/stop`, { method: "POST" });
          setTimeout(fetchHealthAndStatus, 500);
        } catch (err) {
          console.error("Failed to stop scan:", err);
        }
      });
    }
  }

  // -------------------------------------------------------------------------
  // 6. Refresh Button Listener
  // -------------------------------------------------------------------------
  function setupRefresh() {
    const btn = document.getElementById("refreshHealthBtn");
    if (btn) btn.addEventListener("click", fetchHealthAndStatus);
  }

  // -------------------------------------------------------------------------
  // Initialization
  // -------------------------------------------------------------------------
  document.addEventListener("DOMContentLoaded", () => {
    fetchHealthAndStatus();
    setupFileInput();
    setupAnalysisAction();
    setupDatasetScanner();
    setupRefresh();
  });
})();
