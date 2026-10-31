import re

file_path = "src/origin_classifier.py"
with open(file_path, "r", encoding="utf-8") as f:
    text = f.read()

target = '        results["temporal_c1"] = {\n            "sample_size": len(tmp_feats),\n            "predicted_human": hum_count,\n            "predicted_ai": ai_count,\n            "ai_detection_rate": round(ai_count / len(tmp_feats), 4),\n            "mean_prob_ai": round(float(np.mean(probs_tmp)), 4),\n            "conclusion": "Synthetic passages shift spectral features toward AI class; mean probability reflects balanced mixture."\n        }\n        \n    return results'

replacement = '''        results["temporal_c1"] = {
            "sample_size": len(tmp_feats),
            "predicted_human": hum_count,
            "predicted_ai": ai_count,
            "ai_detection_rate": round(ai_count / len(tmp_feats), 4),
            "mean_prob_ai": round(float(np.mean(probs_tmp)), 4),
            "conclusion": "Synthetic passages shift spectral features toward AI class; mean probability reflects balanced mixture."
        }

    # 4. Generator-Specific Evaluations across Independent AI Families
    for gen_name in ["lyria-pro3", "musicgen", "acestep"]:
        g_feats = []
        for tid, t in tracks_dict.items():
            if t.get("generator") == gen_name and str(t.get("origin_label", "")).lower() in ("ai", "ai_generated", "synthetic"):
                f = t.get("origin_features_56")
                if f and len(f) == 56:
                    g_feats.append((tid, np.array(f, dtype=np.float32)))
        if g_feats:
            X_g = np.array([x[1] for x in g_feats])
            preds_g = model.predict(X_g)
            probs_g = model.predict_proba(X_g)[:, 1] if hasattr(model, "predict_proba") else [0.0]*len(preds_g)
            ai_detected = int(sum(1 for p in preds_g if p == 1))
            results[f"generator_{gen_name.replace('-', '_')}"] = {
                "generator": gen_name,
                "sample_size": len(g_feats),
                "predicted_ai": ai_detected,
                "predicted_human": len(g_feats) - ai_detected,
                "ai_recall": round(ai_detected / len(g_feats), 4),
                "mean_prob_ai": round(float(np.mean(probs_g)), 4),
                "conclusion": f"Evaluation on independent {gen_name} generator architecture."
            }

    # 5. Genuine Generator-Held-Out (Leave-One-Generator-Out) Validation
    logo_results = {}
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression

    all_human_feats = [np.array(t.get("origin_features_56"), dtype=np.float32) for t in tracks_dict.values() if t.get("origin_label") == "human" and t.get("origin_features_56")]
    generators = ["acestep", "lyria-pro3", "musicgen"]
    gen_data = {
        g: [np.array(t.get("origin_features_56"), dtype=np.float32) for t in tracks_dict.values() if t.get("generator") == g and t.get("origin_features_56")]
        for g in generators
    }

    for held_out_gen in generators:
        test_ai = gen_data.get(held_out_gen, [])
        if not test_ai or len(test_ai) < 2:
            continue

        train_ai = []
        for train_gen in generators:
            if train_gen != held_out_gen:
                train_ai.extend(gen_data.get(train_gen, []))

        if not train_ai or len(train_ai) < 2 or not all_human_feats:
            continue

        n_hum_train = int(len(all_human_feats) * 0.75)
        train_hum = all_human_feats[:n_hum_train]
        test_hum = all_human_feats[n_hum_train:]

        X_tr = np.array(train_hum + train_ai)
        y_tr = np.array([0]*len(train_hum) + [1]*len(train_ai))

        X_te = np.array(test_hum + test_ai)
        y_te = np.array([0]*len(test_hum) + [1]*len(test_ai))

        try:
            logo_clf = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=1000, random_state=42))])
            logo_clf.fit(X_tr, y_tr)
            ai_test_preds = logo_clf.predict(np.array(test_ai))
            ai_recall = float(np.mean(ai_test_preds == 1))
            hum_test_preds = logo_clf.predict(np.array(test_hum))
            hum_recall = float(np.mean(hum_test_preds == 0))

            logo_results[held_out_gen] = {
                "held_out_generator": held_out_gen,
                "train_generators": [g for g in generators if g != held_out_gen],
                "train_sample_size": len(X_tr),
                "test_sample_size": len(X_te),
                "unseen_generator_recall": round(ai_recall, 4),
                "held_out_human_recall": round(hum_recall, 4),
                "balanced_accuracy": round((ai_recall + hum_recall) / 2.0, 4),
            }
        except Exception:
            pass

    if logo_results:
        results["leave_one_generator_out"] = logo_results
        
    return results'''

# Normalize newlines for replacement
text_normalized = text.replace('\r\n', '\n')
if target in text_normalized:
    text_patched = text_normalized.replace(target, replacement)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(text_patched)
    print("Successfully patched src/origin_classifier.py!")
else:
    print("Target string not found in normalized text!")

