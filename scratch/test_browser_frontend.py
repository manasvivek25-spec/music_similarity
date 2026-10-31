import asyncio
import json
import os
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path
import websockets

CHROME_PATH = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
DEBUG_PORT = 9444
TARGET_URL = "http://127.0.0.1:8001/"
OUTPUT_REPORT = Path("reports/frontend_browser_verification.json")
PROFILE_DIR = Path("temp_uploads/chrome_test_profile")


async def run_browser_verification():
    if PROFILE_DIR.exists():
        shutil.rmtree(PROFILE_DIR, ignore_errors=True)
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)

    cmd = [
        CHROME_PATH,
        "--headless=new",
        f"--remote-debugging-port={DEBUG_PORT}",
        f"--user-data-dir={PROFILE_DIR.resolve()}",
        "--disable-gpu",
        "--disable-extensions",
        "--no-first-run",
        "--no-default-browser-check",
        TARGET_URL,
    ]

    print(f"Launching Chrome headless on port {DEBUG_PORT}...")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    console_logs = []
    exceptions = []

    try:
        # Wait for debugging port to become available
        ws_url = None
        for _ in range(30):
            await asyncio.sleep(0.5)
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{DEBUG_PORT}/json", timeout=2) as resp:
                    targets = json.loads(resp.read().decode("utf-8"))
                    for t in targets:
                        if t.get("type") == "page" and "127.0.0.1:8001" in t.get("url", ""):
                            ws_url = t.get("webSocketDebuggerUrl")
                            break
                    if ws_url:
                        break
            except Exception:
                pass

        assert ws_url, f"Failed to acquire WebSocket debugger URL from Chrome on port {DEBUG_PORT}"
        print(f"Connected to CDP page: {ws_url}")

        async with websockets.connect(ws_url, max_size=10_000_000) as ws:
            msg_id = 0

            async def send_cmd(method, params=None):
                nonlocal msg_id
                msg_id += 1
                payload = {"id": msg_id, "method": method, "params": params or {}}
                await ws.send(json.dumps(payload))
                while True:
                    raw = await ws.recv()
                    data = json.loads(raw)
                    if "method" in data:
                        # Handle background events
                        if data["method"] == "Runtime.consoleAPICalled":
                            console_logs.append(data["params"])
                        elif data["method"] == "Runtime.exceptionThrown":
                            exceptions.append(data["params"])
                    elif data.get("id") == payload["id"]:
                        return data.get("result", {})

            # Enable Runtime, Page, DOM
            await send_cmd("Runtime.enable")
            await send_cmd("Page.enable")
            await send_cmd("DOM.enable")

            # Wait 3 seconds for initial fetchHealthAndStatus to execute
            await asyncio.sleep(3.0)

            # 1. Inspect Page Title
            title_res = await send_cmd("Runtime.evaluate", {
                "expression": "document.title",
                "returnByValue": True
            })
            page_title = title_res.get("result", {}).get("value")
            print(f"Page Title: {page_title}")

            # 2. Check initial Health & Corpus Status rendered in DOM
            eval_initial_script = """
            (() => {
                const getVal = id => {
                    const el = document.getElementById(id);
                    return el ? (el.innerText || el.textContent || '').trim() : null;
                };
                return {
                    backendStatus: getVal('backendStatusText'),
                    classifierStatus: getVal('classifierStatusText'),
                    mertBadge: getVal('mertStatusBadge'),
                    invTotal: getVal('invTotalIndexed'),
                    invHuman: getVal('invHumanCount'),
                    invAi: getVal('invAiCount'),
                    invNullPairs: getVal('invNullPairsCount'),
                    invEstGenre: getVal('invEstGenreCount')
                };
            })()
            """
            init_res = await send_cmd("Runtime.evaluate", {
                "expression": eval_initial_script,
                "returnByValue": True
            })
            initial_dom_state = init_res.get("result", {}).get("value", {})
            print("Initial DOM State:", json.dumps(initial_dom_state, indent=2))

            # 3. Check structural presence of all 7 Feature Section elements
            check_features_script = """
            (() => {
                const check = id => Boolean(document.getElementById(id));
                return {
                    feature1_similarity: {
                        card1SimScore: check('card1SimScore'),
                        card1TierLabel: check('card1TierLabel'),
                        card1DistLabel: check('card1DistLabel'),
                        card1ShiftBadge: check('card1ShiftBadge'),
                        card1FilterLabel: check('card1FilterLabel')
                    },
                    feature2_significance: {
                        card2Pval: check('card2Pval'),
                        card2AdjPval: check('card2AdjPval'),
                        card2SeverityLabel: check('card2SeverityLabel'),
                        card2SampleCount: check('card2SampleCount'),
                        card2ConfidenceBadge: check('card2ConfidenceBadge')
                    },
                    feature3_novelty: {
                        card3NormScore: check('card3NormScore'),
                        card3BroadScore: check('card3BroadScore'),
                        card3NarrowScore: check('card3NarrowScore'),
                        card3NoveltyTier: check('card3NoveltyTier'),
                        card3TierBadge: check('card3TierBadge')
                    },
                    feature4_classifier: {
                        card4PredictedOrigin: check('card4PredictedOrigin'),
                        card4ConfidenceLabel: check('card4ConfidenceLabel'),
                        card4ProbAi: check('card4ProbAi'),
                        card4StatusTag: check('card4StatusTag'),
                        card4Version: check('card4Version')
                    },
                    feature5_context: {
                        ctxEstimatedTempo: check('ctxEstimatedTempo'),
                        ctxEstimatedKey: check('ctxEstimatedKey'),
                        ctxDuration: check('ctxDuration'),
                        ctxCosineReduction: check('ctxCosineReduction'),
                        extraHarmonicRatio: check('extraHarmonicRatio'),
                        extraPercussiveRatio: check('extraPercussiveRatio'),
                        extraRhythmRegularity: check('extraRhythmRegularity'),
                        extraRolloff: check('extraRolloff')
                    },
                    feature6_mimic: {
                        mimicCandidateCountBadge: check('mimicCandidateCountBadge'),
                        mimicCandidateTableBody: check('mimicCandidateTableBody')
                    },
                    feature7_suggestor_and_timeline: {
                        suggestorBox: check('suggestor-box'),
                        evidenceTimeline: check('evidenceTimelineContainer')
                    },
                    verdict_banner: {
                        verdictBannerSection: check('verdictBannerSection'),
                        verdictCodeBadge: check('verdictCodeBadge'),
                        verdictLabelText: check('verdictLabelText'),
                        verdictConfidenceText: check('verdictConfidenceText'),
                        verdictExplanationText: check('verdictExplanationText'),
                        caseRoutingBadge: check('caseRoutingBadge')
                    }
                };
            })()
            """
            feat_res = await send_cmd("Runtime.evaluate", {
                "expression": check_features_script,
                "returnByValue": True
            })
            feature_elements = feat_res.get("result", {}).get("value", {})
            print("Feature Elements Checked:", json.dumps(feature_elements, indent=2))

            # 4. Execute renderPipelineResults with a complete synthetic pipeline analysis payload
            # This verifies the complete UI rendering loop across all 7 features and verdict
            render_test_payload = {
                "query_track": {
                    "track_id": "test_query_001",
                    "title": "Acoustic Test Symphony",
                    "artist": "Test Ensemble",
                    "genre": "Classical",
                    "origin_label": "Human",
                    "duration_sec": 45.2,
                    "estimated_tempo": 124.0,
                    "estimated_key": "C Major",
                    "raw_features_available": True,
                    "beat_features_available": True
                },
                "cosine_prefilter": {
                    "total_candidates": 261,
                    "retained_candidates": 85,
                    "discarded_candidates": 176,
                    "threshold": 0.40,
                    "reduction_percent": 67.43
                },
                "genre_map": {
                    "source_genre": "Classical",
                    "target_genre": "Classical",
                    "is_cross_genre": False,
                    "mapping_confidence": 1.0
                },
                "total_corpus_size": 261,
                "candidates_considered": 261,
                "candidates_retained_prefilter": 85,
                "candidates_discarded_prefilter": 176,
                "candidates_evaluated": 85,
                "candidates_surviving_floor": 12,
                "candidates_surviving_pvalue": 4,
                "candidates_failed_to_load": 0,
                "failed_candidate_details": [],
                "null_distribution_sample_count": 6105,
                "confidence_note": "Rigorous baseline with 6,105 empirical null pairs.",
                "is_index_complete": True,
                "corpus_status": "Complete",
                "duration_disparity_detected": False,
                "case_routing": {
                    "case_type": "Case 1",
                    "case_name": "Significant Similarity with Low Novelty",
                    "routing_explanation": "Query exhibits high structural similarity and low creative novelty against canonical human references."
                },
                "creative_score": {
                    "broad_score_percent": 18.5,
                    "broad_score_normalized": 0.185,
                    "narrow_score_percent": 21.0,
                    "narrow_score_normalized": 0.210,
                    "score_tier": "Low Novelty",
                    "novelty_tier": "Low Novelty",
                    "is_available": True,
                    "explanation": "Query novelty falls in the lower quartile of the empirical human baseline."
                },
                "origin_classifier": {
                    "predicted_origin": "AI-Generated",
                    "confidence": 0.884,
                    "probability_ai": 0.884,
                    "prob_ai": 0.884,
                    "prob_human": 0.116,
                    "confidence_label": "High Confidence",
                    "model_status": "Active",
                    "status_tag": "Active (LinearSVC + MERT 768-dim)",
                    "model_version": "2.0-mert-enhanced",
                    "is_available": True
                },
                "mimic_analysis": {
                    "top_mimic_candidates": [
                        {
                            "candidate_id": "haim_ai_ace_step_0001",
                            "title": "Chopin Prelude Derivative",
                            "artist": "ACE-Step Generator",
                            "genre": "Classical",
                            "similarity": 0.784,
                            "similarity_tier": "Moderate Similarity",
                            "p_value": 0.0024,
                            "adjusted_p_value": 0.0096,
                            "best_shift": 0,
                            "normalized_distance": 0.275,
                            "confidence": "High",
                            "severity_label": "High Similarity",
                            "mimic_warning_active": True,
                            "coverage_percent": 68.5,
                            "coverage_interpretation": "Broad structural overlap",
                            "artist_relationship": "Different Artist",
                            "origin_prediction": "AI-Generated"
                        },
                        {
                            "candidate_id": "haim_human_06001",
                            "title": "Nocturne in E-flat",
                            "artist": "Frederic Chopin",
                            "genre": "Classical",
                            "similarity": 0.692,
                            "similarity_tier": "Moderate Similarity",
                            "p_value": 0.0381,
                            "adjusted_p_value": 0.0762,
                            "best_shift": 2,
                            "normalized_distance": 0.445,
                            "confidence": "Moderate",
                            "severity_label": "Moderate Similarity",
                            "mimic_warning_active": False,
                            "coverage_percent": 42.0,
                            "coverage_interpretation": "Partial structural match",
                            "artist_relationship": "Different Artist",
                            "origin_prediction": "Human"
                        }
                    ],
                    "evaluated_count": 85,
                    "survivors_count": 12,
                    "total_candidates_analyzed": 12,
                    "top_similarity": 0.784,
                    "min_p_value": 0.0024,
                    "has_anomalous_candidate": True
                },
                "evidence_timeline": [
                    {
                        "candidate_id": "haim_ai_ace_step_0001",
                        "candidate_title": "Chopin Prelude Derivative",
                        "candidate_artist": "ACE-Step Generator",
                        "similarity": 0.784,
                        "p_value": 0.0024,
                        "severity": "High Similarity",
                        "best_shift": 0,
                        "flagged_segments": [
                            {
                                "start_time_a": 0.0,
                                "end_time_a": 15.2,
                                "start_time_b": 2.5,
                                "end_time_b": 17.7,
                                "segment_similarity": 0.812
                            }
                        ],
                        "studio_suggestions": [
                            "Transpose harmonic progression by 3 semitones to decouple from reference tonic.",
                            "Introduce syncopated rhythmic displacement in the primary thematic phrase."
                        ]
                    }
                ],
                "extra_features": {
                    "harmonic_ratio": 0.742,
                    "percussive_ratio": 0.258,
                    "rhythm_regularity": 3.84,
                    "spectral_rolloff_hz": 2840.5
                },
                "mert_status": {
                    "is_supported": True,
                    "device": "cuda",
                    "device_name": "NVIDIA GeForce RTX 3050 Laptop GPU"
                },
                "final_verdict": {
                    "verdict_code": "warning",
                    "verdict_label": "High Statistical Similarity Alert",
                    "confidence": "High Confidence",
                    "explanation": "Multiple independent statistical, alignment, and origin indicators point to substantial structural similarity with known catalog items."
                }
            }

            render_js = f"""
            (() => {{
                const payload = {json.dumps(render_test_payload)};
                // Call renderPipelineResults from app.js
                if (typeof renderPipelineResults === 'function') {{
                    renderPipelineResults(payload, false);
                    return {{
                        success: true,
                        renderedVerdict: document.getElementById('verdictCodeBadge')?.innerText,
                        renderedCard1Score: document.getElementById('card1SimScore')?.innerText,
                        renderedCard2Pval: document.getElementById('card2Pval')?.innerText,
                        renderedCard3Score: document.getElementById('card3NormScore')?.innerText,
                        renderedCard4Origin: document.getElementById('card4PredictedOrigin')?.innerText,
                        renderedCard5Tempo: document.getElementById('ctxEstimatedTempo')?.innerText,
                        renderedCard6Count: document.getElementById('mimicCandidateCountBadge')?.innerText,
                        renderedTimelineLength: document.getElementById('evidenceTimelineContainer')?.children?.length || 0
                    }};
                }} else {{
                    return {{ success: false, error: 'renderPipelineResults function not found in window scope' }};
                }}
            }})()
            """
            render_res = await send_cmd("Runtime.evaluate", {
                "expression": render_js,
                "returnByValue": True
            })
            print("RAW RENDER_RES:", json.dumps(render_res, indent=2))
            rendered_state = render_res.get("result", {}).get("value", {})
            print("Rendered State:", json.dumps(rendered_state, indent=2))

            # Check if any errors occurred
            error_logs = [log for log in console_logs if log.get("type") == "error"]

            report_data = {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "target_url": TARGET_URL,
                "browser": "Google Chrome Headless (CDP)",
                "page_title": page_title,
                "initial_dom_state": initial_dom_state,
                "feature_elements_checked": feature_elements,
                "rendered_state": rendered_state,
                "console_log_count": len(console_logs),
                "console_error_count": len(error_logs),
                "exception_count": len(exceptions),
                "console_errors": error_logs,
                "exceptions": exceptions,
                "all_feature_sections_present": all(
                    all(vals.values()) for vals in feature_elements.values()
                ),
                "status": "PASSED" if len(exceptions) == 0 and len(error_logs) == 0 and rendered_state.get("success") else "FAILED"
            }

            OUTPUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
            with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
                json.dump(report_data, f, indent=2)

            print(f"Browser verification report saved to {OUTPUT_REPORT}")
            print(f"Status: {report_data['status']}")
            assert report_data["status"] == "PASSED", f"Browser test failed with {len(exceptions)} exceptions and {len(error_logs)} console errors."

    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        if PROFILE_DIR.exists():
            shutil.rmtree(PROFILE_DIR, ignore_errors=True)
        print("Chrome closed and test profile cleaned up.")


if __name__ == "__main__":
    asyncio.run(run_browser_verification())
