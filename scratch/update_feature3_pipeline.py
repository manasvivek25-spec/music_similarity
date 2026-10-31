from pathlib import Path

pipe_path = Path("src/pipeline.py")
text = pipe_path.read_text(encoding="utf-8")

target = """            # Reuse DTW similarity if already computed during Stage 5 candidate evaluation
            matched_rec = next((r for r in evaluated_records if r.get("candidate_id") == cid), None)
            if matched_rec is not None:
                sim = float(matched_rec["similarity"])
            else:
                # Candidate was excluded by coarse cosine prefilter (< 0.30); bounded by prefilter cosine
                cand_mean = c_desc.get("mean_chroma")
                if cand_mean is not None and len(cand_mean) == 12:
                    sim = float(cosine_similarity_vectors(query_mean_chroma, cand_mean))
                else:
                    sim = 0.0"""

replacement = """            # Reuse DTW similarity if already computed during Stage 5 candidate evaluation
            matched_rec = next((r for r in evaluated_records if r.get("candidate_id") == cid), None)
            if matched_rec is not None:
                sim = float(matched_rec["similarity"])
            else:
                # Compute genuine DTW similarity for human reference excluded by coarse prefilter
                # Ensures Feature 3 broad novelty reference population strictly consists of comparable DTW observations
                sim = None
                beat_ref = meta.get("beat_cache") or meta.get("beat_feature_path")
                if beat_ref:
                    bp = Path(beat_ref)
                    beat_file = bp if bp.is_absolute() else (REPO_ROOT / bp)
                else:
                    beat_file = FEATURES_DIR / f"{cid}_beat.npy"
                if beat_file.is_file():
                    try:
                        c_mat = np.load(str(beat_file))
                        sub_mat = c_mat[:, :300] if (c_mat.ndim == 2 and c_mat.shape[1] > 300) else c_mat
                        sim = float(dtw_similarity(chroma_beat, sub_mat))
                    except Exception as e:
                        logger.debug(f"Failed to compute DTW for human reference {cid}: {e}")
                        sim = None"""

assert target in text, "target block not found in src/pipeline.py"
text = text.replace(target, replacement, 1)
pipe_path.write_text(text, encoding="utf-8")
print("Updated src/pipeline.py successfully!")

