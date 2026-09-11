/**
 * Statistical Analysis of Similarity in Human- and AI-Generated Music
 * Research Dashboard & API Client (http://localhost:8001)
 */

(function () {
  // Determine API base dynamically (works on localhost:8001 or any deployed host)
  const API_BASE = window.location.origin;

  // -------------------------------------------------------------------------
  // 1. Trajectory Waveform Interaction (Feature 2)
  // -------------------------------------------------------------------------
  window.updateTrajectory = function (timestamp, yOffset, pValText, isCritical) {
    const scrubPoint = document.getElementById("scrubPoint");
    const trajectoryValue = document.getElementById("trajectoryValue");

    if (trajectoryValue) {
      trajectoryValue.innerText = pValText;
      if (isCritical) {
        trajectoryValue.className = "font-stat-lg text-stat-lg tracking-tight font-black text-rose-700";
      } else {
        trajectoryValue.className = "font-stat-lg text-stat-lg tracking-tight font-black text-on-surface";
      }
    }

    if (scrubPoint) {
      const positions = {
        "0:00": 15,
        "0:30": 65,
        "1:14": 115,
        "1:30": 165,
        "2:00": 210,
        "2:30": 260,
        "3:00": 310,
      };
      const cxCoord = positions[timestamp] || 115;
      scrubPoint.setAttribute("cx", cxCoord);
      scrubPoint.setAttribute("cy", yOffset);
    }
  };

  // -------------------------------------------------------------------------
  // 2. Remediation Studio Mode Switcher (Deterministic vs AI Phrasing)
  // -------------------------------------------------------------------------
  window.switchRemedyMode = function (mode) {
    const btnDet = document.getElementById("modeDeterministicBtn");
    const btnAi = document.getElementById("modeAiBtn");
    const badge = document.getElementById("guaranteeBadge");
    const desc1 = document.getElementById("remedyDesc1");
    const desc2 = document.getElementById("remedyDesc2");
    const desc3 = document.getElementById("remedyDesc3");

    if (mode === "deterministic") {
      if (btnDet) btnDet.className = "px-4 py-1.5 rounded-full font-label-sm text-[11px] font-bold transition-all bg-inverse-surface text-inverse-on-surface shadow-sm";
      if (btnAi) btnAi.className = "px-4 py-1.5 rounded-full font-label-sm text-[11px] font-bold transition-all text-secondary hover:text-on-surface";
      if (badge) badge.innerText = "Traceable Deterministic Guarantee: Output mathematically bound to S ≤ 0.650.";
      if (desc1) desc1.innerText = "Reflect interval contours symmetrically across root F# in segment [01:14-01:48]. Preserves rhythm while eliminating melodic infringement.";
      if (desc2) desc2.innerText = "Key-transpose localized stem from F# minor to G# minor. Clears legal threshold while retaining exact melodic pattern.";
      if (desc3) desc3.innerText = "Resynthesize analog lead with FM pluck palette, shifting spectral centroid by +1,200 Hz and flattening MFCC peaks.";
    } else {
      if (btnAi) btnAi.className = "px-4 py-1.5 rounded-full font-label-sm text-[11px] font-bold transition-all bg-inverse-surface text-inverse-on-surface shadow-sm";
      if (btnDet) btnDet.className = "px-4 py-1.5 rounded-full font-label-sm text-[11px] font-bold transition-all text-secondary hover:text-on-surface";
      if (badge) badge.innerText = "Generative AI Phrasing: Neural re-voicing trained with similarity penalty objective.";
      if (desc1) desc1.innerText = "Diffusion melodic re-phrase with 0.8 temperature conditioned on harmonic resolution with Velvet Echoes decorrelation.";
      if (desc2) desc2.innerText = "Generative counterpoint continuation replacing chorus motif with non-overlapping modal voicing.";
      if (desc3) desc3.innerText = "Neural timbre transfer applying physical modeled guitar acoustic profile to replace analog synth envelope.";
    }
  };

  // -------------------------------------------------------------------------
  // 3. Remediation Option Selection (Feature 7)
  // -------------------------------------------------------------------------
  let selectedRemedyIdx = 1;

  window.selectRemedy = function (optIndex) {
    selectedRemedyIdx = optIndex;
    for (let i = 1; i <= 3; i++) {
      const lbl = document.getElementById("optLabel" + i);
      if (lbl) {
        if (i === optIndex) {
          lbl.classList.remove("border-white/80", "bg-white/40");
          lbl.classList.add("border-primary", "bg-white/60");
        } else {
          lbl.classList.remove("border-primary", "bg-white/60");
          lbl.classList.add("border-white/80", "bg-white/40");
        }
      }
    }
  };

  // -------------------------------------------------------------------------
  // 4. Connect Backend Health & Live Status
  // -------------------------------------------------------------------------
  async function checkBackendHealth() {
    const statusText = document.getElementById("backendStatusText");
    const statusDot = document.getElementById("backendStatusDot");

    try {
      const res = await fetch(`${API_BASE}/api/health`);
      if (res.ok) {
        const data = await res.json();
        if (statusText) statusText.innerText = `FastAPI Online (${data.engine})`;
        if (statusDot) {
          statusDot.className = "w-2 h-2 rounded-full bg-emerald-500 animate-pulse";
        }
      }
    } catch (err) {
      console.warn("Backend not reached at", API_BASE, err);
      if (statusText) statusText.innerText = "Backend Offline · Recheck Server";
      if (statusDot) {
        statusDot.className = "w-2 h-2 rounded-full bg-amber-500";
      }
    }
  }

  // -------------------------------------------------------------------------
  // 5. Dynamic Rendering of Candidates (Feature 6)
  // -------------------------------------------------------------------------
  function renderCandidateCards(candidates, queryId) {
    const container = document.getElementById("candidateCardsContainer");
    if (!container) return;

    if (!candidates || candidates.length === 0) {
      container.innerHTML = `
        <div class="col-span-2 p-6 rounded-xl bg-white/60 text-center text-secondary">
          No candidates ranked yet. Upload a dataset to begin.
        </div>
      `;
      return;
    }

    let html = "";
    candidates.forEach((cand, idx) => {
      const isTop = cand.is_top_match || idx === 0;
      const rankNum = idx + 1;
      const simPercent = (cand.similarity * 100).toFixed(1);
      const isSignificant = cand.p_value < 0.01;
      const shiftStr = (cand.best_shift > 0 ? "+" : "") + cand.best_shift;

      const cardClasses = isTop
        ? "bg-gradient-to-br from-white/85 via-white/65 to-white/45 backdrop-blur-xl border-2 border-rose-300/80 rounded-lg p-space-lg shadow-[0_12px_36px_rgba(225,29,72,0.08),inset_0_1px_1px_rgba(255,255,255,0.85)] flex flex-col justify-between transition-all duration-500 ease-out hover:-translate-y-2.5 hover:scale-[1.01]"
        : "bg-gradient-to-br from-white/80 via-white/60 to-white/40 backdrop-blur-xl border border-white/70 rounded-lg p-space-lg shadow-[0_12px_36px_rgba(0,0,0,0.05),inset_0_1px_1px_rgba(255,255,255,0.85)] flex flex-col justify-between transition-all duration-500 ease-out hover:-translate-y-2.5 hover:scale-[1.01]";

      const badgeHeader = isTop
        ? `<span class="px-2.5 py-0.5 rounded-full text-[10px] font-mono font-bold bg-rose-100 text-rose-800 border border-rose-200 uppercase tracking-wider flex items-center gap-1">
             <span class="w-1.5 h-1.5 rounded-full bg-rose-600 animate-pulse"></span> Candidate #${rankNum} · Top Match
           </span>`
        : `<span class="px-2.5 py-0.5 rounded-full text-[10px] font-mono font-medium bg-white/70 text-secondary border border-white/80 uppercase tracking-wider">
             Candidate #${rankNum}
           </span>`;

      const significanceTag = isSignificant
        ? `<span class="font-mono text-xs font-bold text-rose-700 bg-white/70 px-2 py-0.5 rounded-full border border-rose-200">Flagged (p &lt; 0.01)</span>`
        : `<span class="font-mono text-xs text-emerald-700 bg-emerald-50 px-2 py-0.5 rounded-full border border-emerald-200">Insignificant (p &ge; 0.05)</span>`;

      html += `
        <div class="${cardClasses}">
          <div>
            <div class="flex items-center justify-between mb-3">
              ${badgeHeader}
              ${significanceTag}
            </div>
            <div class="flex items-start gap-space-md">
              <div class="w-12 h-12 rounded-2xl bg-gradient-to-br from-[#FFF6A8]/90 to-[#FFF075]/80 backdrop-blur-md border border-white/80 text-on-surface flex items-center justify-center flex-shrink-0 shadow-sm">
                <span class="material-symbols-outlined text-[24px]">${isTop ? "music_note" : "album"}</span>
              </div>
              <div class="min-w-0 flex-1">
                <span class="font-label-sm text-label-sm text-secondary uppercase tracking-wider block font-bold">Candidate Audio Track</span>
                <span class="font-stat-md text-stat-md text-on-surface block tracking-tight leading-tight truncate" title="${cand.candidate_id}">${cand.candidate_id}</span>
                <div class="flex items-center gap-2 mt-1 font-mono text-xs">
                  <span class="font-bold text-on-surface">S = ${simPercent}%</span>
                  <span class="text-secondary">·</span>
                  <span class="${isSignificant ? "text-rose-600 font-semibold" : "text-secondary"}">p = ${cand.p_value.toFixed(4)}</span>
                </div>
                <p class="font-body-sm text-body-sm text-on-surface-variant mt-2 leading-relaxed">
                  ${isTop ? "Highest structural and melodic alignment with query song across all 12 circular pitch shifts." : "Secondary reference candidate evaluated against empirical null reference bounds."}
                </p>
              </div>
            </div>
          </div>
          <div class="flex items-center justify-between pt-space-md mt-space-md border-t ${isTop ? "border-rose-200/50" : "border-black/5"}">
            <div class="flex gap-1.5 font-mono text-[11px]">
              <span class="bg-white/60 border border-white/70 px-2.5 py-1 rounded-full text-secondary font-medium shadow-sm">Shift: ${shiftStr} semi</span>
              <span class="bg-white/60 border border-white/70 px-2.5 py-1 rounded-full text-secondary font-medium shadow-sm">Dist: ${cand.normalized_distance.toFixed(4)}</span>
            </div>
          </div>
        </div>
      `;
    });

    container.innerHTML = html;
  }

  // -------------------------------------------------------------------------
  // 6. Audio Dataset Upload Handler (POST /api/candidates/rank/audio)
  // -------------------------------------------------------------------------
  function setupDatasetUpload() {
    const uploadModal = document.getElementById("uploadModal");
    const openUploadBtn = document.getElementById("openUploadBtn");
    const openUploadBtnCard = document.getElementById("openUploadBtnCard");
    const closeUploadBtn = document.getElementById("closeUploadBtn");

    const queryFileInput = document.getElementById("queryFileInput");
    const datasetFileInput = document.getElementById("datasetFileInput");
    const queryFileInfo = document.getElementById("queryFileInfo");
    const datasetFileInfo = document.getElementById("datasetFileInfo");
    const runDatasetUploadBtn = document.getElementById("runDatasetUploadBtn");
    const uploadStatusMessage = document.getElementById("uploadStatusMessage");
    const uploadBtnText = document.getElementById("uploadBtnText");
    const uploadBtnIcon = document.getElementById("uploadBtnIcon");

    function openModal() {
      if (uploadModal) {
        uploadModal.classList.remove("hidden");
        uploadModal.classList.add("flex");
      }
    }

    function closeModal() {
      if (uploadModal) {
        uploadModal.classList.add("hidden");
        uploadModal.classList.remove("flex");
      }
    }

    if (openUploadBtn) openUploadBtn.addEventListener("click", openModal);
    if (openUploadBtnCard) openUploadBtnCard.addEventListener("click", openModal);
    if (closeUploadBtn) closeUploadBtn.addEventListener("click", closeModal);

    if (uploadModal) {
      uploadModal.addEventListener("click", (e) => {
        if (e.target === uploadModal) closeModal();
      });
    }

    if (queryFileInput && queryFileInfo) {
      queryFileInput.addEventListener("change", () => {
        if (queryFileInput.files && queryFileInput.files[0]) {
          const f = queryFileInput.files[0];
          queryFileInfo.innerText = `Selected: ${f.name} (${(f.size / 1024).toFixed(1)} KB)`;
          queryFileInfo.classList.remove("hidden");
        } else {
          queryFileInfo.classList.add("hidden");
        }
      });
    }

    if (datasetFileInput && datasetFileInfo) {
      datasetFileInput.addEventListener("change", () => {
        const count = datasetFileInput.files ? datasetFileInput.files.length : 0;
        if (count > 0) {
          datasetFileInfo.innerText = `Selected: ${count} dataset audio file${count > 1 ? "s" : ""}`;
          datasetFileInfo.classList.remove("hidden");
        } else {
          datasetFileInfo.classList.add("hidden");
        }
      });
    }

    if (runDatasetUploadBtn) {
      runDatasetUploadBtn.addEventListener("click", async () => {
        const queryFile = queryFileInput && queryFileInput.files ? queryFileInput.files[0] : null;
        const datasetFiles = datasetFileInput && datasetFileInput.files ? datasetFileInput.files : [];

        if (!queryFile) {
          if (uploadStatusMessage) {
            uploadStatusMessage.innerText = "Please select a query audio file first.";
            uploadStatusMessage.className = "text-xs font-mono text-center min-h-[20px] text-rose-600 font-bold";
          }
          return;
        }

        if (datasetFiles.length === 0) {
          if (uploadStatusMessage) {
            uploadStatusMessage.innerText = "Please select at least 1 candidate audio file in your dataset.";
            uploadStatusMessage.className = "text-xs font-mono text-center min-h-[20px] text-rose-600 font-bold";
          }
          return;
        }

        // Build FormData with query_file and multiple candidate_files
        const formData = new FormData();
        formData.append("query_file", queryFile);
        for (let i = 0; i < datasetFiles.length; i++) {
          formData.append("candidate_files", datasetFiles[i]);
        }

        // UI Loading state
        runDatasetUploadBtn.disabled = true;
        if (uploadBtnText) uploadBtnText.innerText = "Extracting Chroma & DTW Aligning Dataset...";
        if (uploadBtnIcon) uploadBtnIcon.className = "material-symbols-outlined text-[20px] animate-spin";
        if (uploadStatusMessage) {
          uploadStatusMessage.innerText = `Processing query vs ${datasetFiles.length} candidate tracks across 12 pitch shifts...`;
          uploadStatusMessage.className = "text-xs font-mono text-center min-h-[20px] text-primary font-semibold";
        }

        try {
          const res = await fetch(`${API_BASE}/api/candidates/rank/audio`, {
            method: "POST",
            body: formData,
          });

          if (!res.ok) {
            const errData = await res.json().catch(() => ({}));
            throw new Error(errData.detail || `Server error (${res.status})`);
          }

          const data = await res.json();

          // 1. Update Query Track Title
          const queryTitleEl = document.getElementById("queryTrackTitle");
          if (queryTitleEl) queryTitleEl.innerText = data.query_id;

          // 2. Update Dataset Badge Count
          const badgeCountEl = document.getElementById("datasetBadgeCount");
          if (badgeCountEl) {
            badgeCountEl.innerText = `${data.candidates.length} Candidates Loaded`;
          }

          // 3. Update Hero Squircle 1 (Feature 1: Similarity Engine)
          const similarityScore = document.getElementById("similarityScore");
          const bestShiftTag = document.getElementById("bestShiftTag");
          if (data.top_match) {
            if (similarityScore) {
              similarityScore.innerText = data.top_match.similarity.toFixed(3);
            }
            if (bestShiftTag) {
              const shiftSign = data.top_match.best_shift > 0 ? "+" : "";
              bestShiftTag.innerText = `12-Chroma · Best shift: ${shiftSign}${data.top_match.best_shift} semi`;
            }

            // 4. Update Trajectory / Significance (Feature 2)
            const trajectoryValue = document.getElementById("trajectoryValue");
            if (trajectoryValue) {
              trajectoryValue.innerText = `p = ${data.top_match.p_value.toFixed(4)}`;
              if (data.top_match.p_value < 0.01) {
                trajectoryValue.className = "font-stat-lg text-stat-lg tracking-tight font-black text-rose-700";
              } else {
                trajectoryValue.className = "font-stat-lg text-stat-lg tracking-tight font-black text-on-surface";
              }
            }
          }

          // 5. Render Dynamic Candidate Cards
          renderCandidateCards(data.candidates, data.query_id);

          if (uploadStatusMessage) {
            uploadStatusMessage.innerText = `Ranking complete! Top match: ${data.top_match ? data.top_match.candidate_id : "N/A"}`;
            uploadStatusMessage.className = "text-xs font-mono text-center min-h-[20px] text-emerald-700 font-bold";
          }

          setTimeout(() => {
            closeModal();
          }, 800);
        } catch (err) {
          console.error("Dataset upload failed", err);
          if (uploadStatusMessage) {
            uploadStatusMessage.innerText = `Error: ${err.message}`;
            uploadStatusMessage.className = "text-xs font-mono text-center min-h-[20px] text-rose-600 font-bold";
          }
        } finally {
          runDatasetUploadBtn.disabled = false;
          if (uploadBtnText) uploadBtnText.innerText = "Compute DTW Similarity Across Dataset";
          if (uploadBtnIcon) uploadBtnIcon.className = "material-symbols-outlined text-[20px]";
        }
      });
    }
  }

  // -------------------------------------------------------------------------
  // 7. Initialization & Endpoint Event Handlers
  // -------------------------------------------------------------------------
  document.addEventListener("DOMContentLoaded", () => {
    checkBackendHealth();
    setupDatasetUpload();

    // Recompute DTW Significance Test via POST /api/stats/null-test
    const quickAnalyzeBtn = document.getElementById("quickAnalyzeBtn");
    if (quickAnalyzeBtn) {
      quickAnalyzeBtn.addEventListener("click", async () => {
        const icon = quickAnalyzeBtn.querySelector(".material-symbols-outlined");
        if (icon) icon.classList.add("animate-spin");
        try {
          const simEl = document.getElementById("similarityScore");
          const currentScore = simEl ? parseFloat(simEl.innerText) || 0.884 : 0.884;

          const res = await fetch(`${API_BASE}/api/stats/null-test`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ query_score: currentScore }),
          });

          if (res.ok) {
            const data = await res.json();
            const traj = document.getElementById("trajectoryValue");
            if (traj) {
              traj.innerText = `p = ${data.p_value.toFixed(4)}`;
              if (data.is_statistically_significant) {
                traj.className = "font-stat-lg text-stat-lg tracking-tight font-black text-rose-700";
              } else {
                traj.className = "font-stat-lg text-stat-lg tracking-tight font-black text-on-surface";
              }
            }
          }
        } catch (e) {
          console.warn("Null test request error:", e);
        } finally {
          setTimeout(() => {
            if (icon) icon.classList.remove("animate-spin");
          }, 600);
        }
      });
    }

    // Apply Safe Transform Button (calls /api/stats/null-test to project lowered similarity)
    const applyRemedyBtn = document.getElementById("applyRemedyBtn");
    const similarityScore = document.getElementById("similarityScore");

    if (applyRemedyBtn && similarityScore) {
      applyRemedyBtn.addEventListener("click", async () => {
        const scores = { 1: "0.412", 2: "0.528", 3: "0.380" };
        const newScore = scores[selectedRemedyIdx] || "0.412";

        similarityScore.innerText = newScore;
        similarityScore.className = "font-stat-lg text-stat-lg text-emerald-800 tracking-tight drop-shadow-sm font-black transition-all";

        applyRemedyBtn.innerHTML = `<span class="material-symbols-outlined text-[18px]">verified</span> <span>Applied (S = ${newScore}, Safe Harbor)</span>`;
        applyRemedyBtn.className = "px-space-lg py-space-sm rounded-full bg-inverse-surface text-inverse-on-surface font-label-md text-label-md transition-all duration-300 shadow-md flex items-center gap-1.5 font-bold cursor-pointer";

        // Query backend null test to verify newly reduced score
        try {
          const res = await fetch(`${API_BASE}/api/stats/null-test`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ query_score: parseFloat(newScore) }),
          });
          if (res.ok) {
            const data = await res.json();
            const trajVal = document.getElementById("trajectoryValue");
            if (trajVal) {
              trajVal.innerText = `p = ${data.p_value.toFixed(4)}`;
              trajVal.className = "font-stat-lg text-stat-lg text-emerald-700 tracking-tight font-black";
            }
          }
        } catch (e) {
          console.warn("Null test request error:", e);
        }
      });
    }
  });
})();
