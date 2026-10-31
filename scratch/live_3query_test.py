import requests
import json
import time
import os

BASE_URL = "http://127.0.0.1:8001"

test_files = [
    {
        "name": "Query 1: Short Audio Clip (7s)",
        "path": "temp_uploads/short_clip_7s.wav",
        "description": "Standard short audio query (< 30s) testing default pipeline"
    },
    {
        "name": "Query 2: Key Transposed Clip (+3 Semitones)",
        "path": "temp_uploads/test_transformed_3semitones.wav",
        "description": "Pitch-shifted query evaluating key-invariant chromagram similarity"
    },
    {
        "name": "Query 3: Long Audio Clip (65s)",
        "path": "temp_uploads/test_65s.wav",
        "description": "Unclipped full-duration audio (> 60s) verifying duration cap removal"
    }
]

print("=== Starting 3-Query Live API Verification ===")
results = []

for item in test_files:
    file_path = item["path"]
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        continue
    
    print(f"\n--- Submitting {item['name']} ({file_path}) ---")
    start_t = time.time()
    with open(file_path, "rb") as f:
        files = {"query_audio": (os.path.basename(file_path), f, "audio/wav")}
        data_form = {"threshold": 0.65}
        resp = requests.post(f"{BASE_URL}/api/analyze", files=files, data=data_form)
    elapsed = time.time() - start_t
    
    status = resp.status_code
    print(f"Status Code: {status} in {elapsed:.2f}s")
    
    if status == 200:
        d = resp.json()
        query_track = d.get("query_track", {})
        case_routing = d.get("case_routing", {})
        creative_score = d.get("creative_score", {})
        origin_classifier = d.get("origin_classifier", {})
        mimic_analysis = d.get("mimic_analysis", {})
        evidence_timeline = d.get("evidence_timeline", [])
        final_verdict = d.get("final_verdict", {})
        top_candidates = mimic_analysis.get("top_mimic_candidates", [])
        top_c = top_candidates[0] if top_candidates else {}
        
        parsed_timeline = []
        for ev in evidence_timeline:
            parsed_timeline.append({
                "candidate_id": ev.get("candidate_id"),
                "candidate_title": ev.get("candidate_title"),
                "candidate_artist": ev.get("candidate_artist"),
                "similarity": ev.get("similarity"),
                "p_value": ev.get("p_value"),
                "severity": ev.get("severity"),
                "best_shift": ev.get("best_shift"),
                "flagged_segments": [
                    {
                        "start_time_a": seg.get("start_time_a"),
                        "end_time_a": seg.get("end_time_a"),
                        "start_time_b": seg.get("start_time_b"),
                        "end_time_b": seg.get("end_time_b"),
                        "mean_step_cost": seg.get("mean_step_cost"),
                        "segment_similarity": seg.get("segment_similarity")
                    }
                    for seg in ev.get("flagged_segments", [])
                ],
                "studio_suggestions": ev.get("studio_suggestions", [])
            })

        summary = {
            "query_name": item["name"],
            "file": file_path,
            "status_code": status,
            "latency_s": round(elapsed, 2),
            "feature1_similarity_engine": {
                "total_corpus_size": d.get("total_corpus_size"),
                "candidates_considered": d.get("candidates_considered"),
                "candidates_retained_prefilter": d.get("candidates_retained_prefilter"),
                "candidates_evaluated_dtw": d.get("candidates_evaluated"),
                "candidates_surviving_floor": d.get("candidates_surviving_floor"),
                "candidates_surviving_pvalue": d.get("candidates_surviving_pvalue"),
                "top_similarity": top_c.get("similarity"),
                "top_candidate_id": top_c.get("candidate_id"),
                "best_shift": top_c.get("best_shift")
            },
            "feature2_statistical_significance": {
                "p_value": top_c.get("p_value"),
                "k_exceedance_count": top_c.get("exceedance_count"),
                "n_null_samples": top_c.get("null_sample_size") or d.get("null_distribution_sample_count"),
                "adjusted_p_value": top_c.get("adjusted_p_value"),
                "confidence": top_c.get("confidence"),
                "severity_label": top_c.get("severity_label")
            },
            "feature3_creative_novelty": {
                "case_classification": case_routing.get("case_type"),
                "case_name": case_routing.get("case_name"),
                "case_description": case_routing.get("description"),
                "broad_novelty_score_percent": creative_score.get("broad_score_percent"),
                "narrow_novelty_score_percent": creative_score.get("narrow_score_percent"),
                "score_tier": creative_score.get("score_tier")
            },
            "feature4_origin_classifier": {
                "predicted_origin": origin_classifier.get("predicted_origin"),
                "confidence": origin_classifier.get("confidence"),
                "probability_ai": origin_classifier.get("probability_ai"),
                "model_status": origin_classifier.get("model_status")
            },
            "feature5_musical_context": {
                "estimated_tempo_bpm": query_track.get("tempo"),
                "estimated_key": query_track.get("estimated_key"),
                "meter": query_track.get("meter"),
                "duration_seconds": query_track.get("duration_sec")
            },
            "feature6_mimic_identification": {
                "top_mimic_candidate_id": top_c.get("candidate_id"),
                "top_mimic_similarity": top_c.get("similarity"),
                "mimic_warning_active": top_c.get("mimic_warning_active"),
                "coverage_percent": top_c.get("coverage_percent"),
                "coverage_interpretation": top_c.get("coverage_interpretation"),
                "duration_disparity_warning": top_c.get("duration_disparity_warning"),
                "disparity_explanation": top_c.get("disparity_explanation"),
                "ranked_candidates_count": len(top_candidates),
                "top_3_candidates": [
                    {
                        "candidate_id": c.get("candidate_id"),
                        "similarity": round(c.get("similarity", 0), 4),
                        "p_value": round(c.get("p_value", 1.0), 5) if c.get("p_value") is not None else None,
                        "origin": c.get("origin_prediction")
                    }
                    for c in top_candidates[:3]
                ]
            },
            "feature7_suggestor_box_and_timeline": {
                "timeline_items_count": len(evidence_timeline),
                "timeline_items": parsed_timeline,
                "final_verdict": final_verdict.get("verdict_label"),
                "true_indicators_count": final_verdict.get("true_indicators_count"),
                "verdict_explanation": final_verdict.get("explanation")
            }
        }
        results.append(summary)
        print("  Summary:")
        print(json.dumps(summary, indent=2))
    else:
        print(f"  Error: {resp.text}")
        results.append({
            "query_name": item["name"],
            "status_code": status,
            "error": resp.text
        })

output_path = "reports/live_api_retest_results.json"
os.makedirs("reports", exist_ok=True)
with open(output_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nSaved live API test results to {output_path}")
