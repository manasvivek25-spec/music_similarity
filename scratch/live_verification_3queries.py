import json
import time
import requests
from pathlib import Path

API_URL = "http://127.0.0.1:8001/api/analyze"

queries = [
    ("Query 1 (7s Short Clip)", Path("temp_uploads/short_clip_7s.wav")),
    ("Query 2 (Pitch Shifted 3 Semitones)", Path("temp_uploads/test_transformed_3semitones.wav")),
    ("Query 3 (65s Full Audio)", Path("temp_uploads/test_65s.wav")),
]

results = []

for label, path in queries:
    print(f"\n==========================================")
    print(f"RUNNING LIVE VERIFICATION: {label}")
    print(f"File: {path} (exists: {path.is_file()})")
    print(f"==========================================")

    t0 = time.time()
    with open(path, "rb") as f:
        files = {"query_audio": (path.name, f, "audio/wav")}
        data = {"artist": "Live Test Artist", "title": label, "genre": "Electronic"}
        resp = requests.post(API_URL, files=files, data=data, timeout=300)

    elapsed = round(time.time() - t0, 3)
    print(f"Status Code: {resp.status_code} in {elapsed}s")

    if resp.status_code == 200:
        res = resp.json()

        # Feature 1
        cand_eval = res.get("candidates_evaluated", 0)
        best_sim = max([c["similarity"] for c in res.get("mimic_analysis", {}).get("top_mimic_candidates", [])], default=0.0)

        # Feature 2
        top_cand = res.get("mimic_analysis", {}).get("top_mimic_candidates", [{}])[0] if res.get("mimic_analysis", {}).get("top_mimic_candidates") else {}
        p_val = top_cand.get("p_value")
        k_val = top_cand.get("exceedance_count")
        n_val = top_cand.get("null_sample_size") or res.get("null_distribution_sample_count")
        conf = top_cand.get("confidence")

        # Feature 3
        cs = res.get("creative_score", {})
        broad = cs.get("broad_score_percent")
        tier = cs.get("score_tier")

        # Feature 4
        orig = res.get("origin_classifier", {})
        orig_pred = orig.get("origin_prediction")
        orig_conf = orig.get("confidence")

        # Feature 5
        qt = res.get("query_track", {})
        key = qt.get("estimated_key")
        tempo = qt.get("estimated_tempo")

        # Feature 6
        top_cands = res.get("mimic_analysis", {}).get("top_mimic_candidates", [])
        top_cid = top_cand.get("candidate_id")
        cov_pct = top_cand.get("coverage_percent")
        cov_interp = top_cand.get("coverage_interpretation")

        # Feature 7
        tl = res.get("evidence_timeline", [])

        # Verdict
        verdict = res.get("final_verdict", {})
        v_code = verdict.get("verdict_code")
        v_label = verdict.get("verdict_label")

        out = {
            "query_label": label,
            "elapsed_sec": elapsed,
            "feature1": {
                "total_corpus_considered": res.get("candidates_considered"),
                "candidates_retained_prefilter": res.get("candidates_retained_prefilter"),
                "candidates_evaluated": cand_eval,
                "candidates_surviving_floor": res.get("candidates_surviving_floor"),
                "best_similarity": best_sim,
            },
            "feature2": {
                "p_value": p_val,
                "exceedance_count_k": k_val,
                "null_sample_size_N": n_val,
                "confidence": conf,
                "null_distribution_sample_count": res.get("null_distribution_sample_count"),
            },
            "feature3": {
                "broad_score_percent": broad,
                "score_tier": tier,
                "case_type": res.get("case_routing", {}).get("case_type"),
                "case_name": res.get("case_routing", {}).get("case_name"),
            },
            "feature4": {
                "origin_prediction": orig_pred,
                "confidence": orig_conf,
                "model_status": orig.get("model_status"),
            },
            "feature5": {
                "estimated_key": key,
                "estimated_tempo": tempo,
                "duration_sec": qt.get("duration_sec"),
            },
            "feature6": {
                "top_candidate_count": len(top_cands),
                "top_candidate_id": top_cid,
                "top_similarity": top_cand.get("similarity"),
                "normalized_distance": top_cand.get("normalized_distance"),
                "best_shift": top_cand.get("best_shift"),
                "alignment_coverage_pct": cov_pct,
                "coverage_interpretation": cov_interp,
            },
            "feature7": {
                "timeline_items_count": len(tl),
            },
            "verdict": {
                "verdict_code": v_code,
                "verdict_label": v_label,
            },
            "mert_status": res.get("mert_status", {}),
        }
        results.append(out)
        print(json.dumps(out, indent=2))
    else:
        print(f"Error {resp.status_code}: {resp.text}")

with open("scratch/live_verification_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nAll 3 queries completed and saved to scratch/live_verification_results.json!")
