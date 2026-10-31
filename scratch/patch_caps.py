from pathlib import Path

# 1. Patch src/stats_test.py
p_stats = Path("src/stats_test.py")
text = p_stats.read_text(encoding="utf-8")
old_stat = """    null_scores: List[float] = []
    for i, j in pairs:
        score = similarity_func(chroma_dict[track_ids[i]], chroma_dict[track_ids[j]])
        if np.isfinite(score):
            null_scores.append(float(score))"""

new_stat = """    null_scores: List[float] = []
    for i, j in pairs:
        c_i = chroma_dict[track_ids[i]]
        c_j = chroma_dict[track_ids[j]]
        c_i_eval = c_i[:, :300] if (c_i.ndim == 2 and c_i.shape[1] > 300) else c_i
        c_j_eval = c_j[:, :300] if (c_j.ndim == 2 and c_j.shape[1] > 300) else c_j
        score = similarity_func(c_i_eval, c_j_eval)
        if np.isfinite(score):
            null_scores.append(float(score))"""

if old_stat in text:
    text = text.replace(old_stat, new_stat)
    p_stats.write_text(text, encoding="utf-8")
    print("Patched stats_test.py")
else:
    print("stats_test.py target not found")

# 2. Patch src/pipeline.py
p_pipe = Path("src/pipeline.py")
text_pipe = p_pipe.read_text(encoding="utf-8")
old_pipe = """                beat_file = FEATURES_DIR / f"{cid}_beat.npy"
                if beat_file.is_file():
                    try:
                        human_chroma[cid] = np.load(str(beat_file))
                        human_meta[cid] = meta
                    except Exception:
                        pass"""

new_pipe = """                beat_file = FEATURES_DIR / f"{cid}_beat.npy"
                if beat_file.is_file():
                    try:
                        mat = np.load(str(beat_file))
                        human_chroma[cid] = mat[:, :300] if (mat.ndim == 2 and mat.shape[1] > 300) else mat
                        human_meta[cid] = meta
                    except Exception:
                        pass"""

if old_pipe in text_pipe:
    text_pipe = text_pipe.replace(old_pipe, new_pipe)
    p_pipe.write_text(text_pipe, encoding="utf-8")
    print("Patched pipeline.py")
else:
    print("pipeline.py target not found")

