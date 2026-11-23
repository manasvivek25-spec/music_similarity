"""
Analysis 4 (Parallel & Fast): F4 Diagnostics on 56-dim and 133-dim features.
Parallel extraction of 133-dim features across CPU cores with disk caching.
Confusion matrix, Wilson 95% CIs, balanced accuracy, ROC-AUC, EER,
Leave-One-Generator-Out recall, majority/metadata-only/genre-only baselines.
Seed=42. Output: reports/analysis4_f4_diagnostics.json
"""

import json, numpy as np, sys, time, os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from sklearn.metrics import (confusion_matrix, balanced_accuracy_score,
                              roc_auc_score, roc_curve, precision_score, f1_score)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              HistGradientBoostingClassifier, VotingClassifier)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

SEED = 42
OUT = BASE / "reports" / "analysis4_f4_diagnostics.json"
CACHE_133 = BASE / "features" / "origin_features_133.npz"

def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p_hat = k / n
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denom
    spread = z * np.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2)) / denom
    return (round(float(max(0, center - spread)), 4), round(float(min(1, center + spread)), 4))

def make_ensemble():
    rf = RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1)
    et = ExtraTreesClassifier(n_estimators=100, random_state=SEED, n_jobs=-1)
    hgb = HistGradientBoostingClassifier(random_state=SEED)
    ens = VotingClassifier(estimators=[("rf", rf), ("et", et), ("hgb", hgb)], voting="soft")
    return Pipeline([("scaler", StandardScaler()), ("classifier", ens)])

def compute_metrics(y_true, y_pred, y_prob, dim_name):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    n = len(y_true)
    
    acc = float((tp + tn) / n)
    sens = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    auc = float(roc_auc_score(y_true, y_prob))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    
    fpr_arr, tpr_arr, _ = roc_curve(y_true, y_prob)
    fnr_arr = 1 - tpr_arr
    eer_idx = np.argmin(np.abs(fpr_arr - fnr_arr))
    eer = float((fpr_arr[eer_idx] + fnr_arr[eer_idx]) / 2.0)
    
    ci_acc = wilson_ci(int(tp + tn), n)
    ci_sens = wilson_ci(int(tp), int(tp + fn))
    ci_spec = wilson_ci(int(tn), int(tn + fp))
    
    return {
        "feature_dim_label": dim_name,
        "n_samples": int(n),
        "confusion_matrix": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)},
        "accuracy": round(acc, 4),
        "accuracy_wilson_95ci": ci_acc,
        "sensitivity_ai_recall": round(sens, 4),
        "sensitivity_wilson_95ci": ci_sens,
        "specificity_human_recall": round(spec, 4),
        "specificity_wilson_95ci": ci_spec,
        "balanced_accuracy": round(bal_acc, 4),
        "roc_auc": round(auc, 4),
        "eer": round(eer, 4),
        "precision": round(prec, 4),
        "f1": round(f1, 4),
    }

def _extract_worker(task):
    audio_path, tid, label, genre = task
    from src.origin_classifier import extract_origin_features_from_audio
    try:
        f = extract_origin_features_from_audio(audio_path)
        return tid, f, label, genre, True
    except Exception as exc:
        return tid, None, label, genre, False

def main():
    print("=== Analysis 4: F4 Diagnostics on 56-dim and 133-dim Features ===")
    ri = json.load(open(BASE / "features" / "remote_index.json"))
    tracks = ri["tracks"]
    
    # ── 1. 56-dim Feature Evaluation ──
    print("\n--- Evaluating 56-dim Features (cached in remote_index.json) ---")
    X56_list, y56_list, genres_56, tids_56 = [], [], [], []
    for tid, t in tracks.items():
        if any(x in tid for x in ("ref_", "tmp")): continue
        lbl = str(t.get("origin_label", "")).lower()
        feats = t.get("origin_features_56")
        if feats and len(feats) == 56 and np.all(np.isfinite(feats)):
            if lbl == "human":
                y_val = 0
            elif lbl in ("ai", "ai_generated", "synthetic"):
                y_val = 1
            else:
                continue
            X56_list.append(np.array(feats, dtype=np.float32))
            y56_list.append(y_val)
            genres_56.append(str(t.get("genre", "Unknown")))
            tids_56.append(tid)
            
    X56 = np.array(X56_list)
    y56 = np.array(y56_list)
    print(f"56-dim: {len(X56)} samples (Human={np.sum(y56==0)}, AI={np.sum(y56==1)})")
    
    pipe56 = make_ensemble()
    skf56 = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    y56_pred = cross_val_predict(pipe56, X56, y56, cv=skf56, method="predict")
    y56_prob = cross_val_predict(pipe56, X56, y56, cv=skf56, method="predict_proba")[:, 1]
    
    metrics_56 = compute_metrics(y56, y56_pred, y56_prob, "56-dim (Acoustic Descriptors)")
    print(f"56-dim CV: Accuracy={metrics_56['accuracy']} (95% CI: {metrics_56['accuracy_wilson_95ci']})")
    print(f"           BalAcc={metrics_56['balanced_accuracy']}, ROC-AUC={metrics_56['roc_auc']}, EER={metrics_56['eer']}")
    print(f"           Confusion Matrix: {metrics_56['confusion_matrix']}")
    
    # ── 2. 133-dim Feature Evaluation (with Parallel Extraction / Caching) ──
    print("\n--- Evaluating 133-dim Features (Rich Acoustic Feature Set) ---")
    if CACHE_133.exists():
        print(f"Loading cached 133-dim features from {CACHE_133} ...")
        cached_data = np.load(str(CACHE_133))
        X133 = cached_data["X"]
        y133 = cached_data["y"]
        tids_133 = cached_data["tids"].tolist()
        genres_133 = cached_data["genres"].tolist()
    else:
        tasks = []
        for tid, t in tracks.items():
            if any(x in tid for x in ("ref_", "tmp")): continue
            lbl = str(t.get("origin_label", "")).lower()
            if lbl == "human":
                p = BASE / "data" / "human_songs" / f"{tid}.wav"
                label = 0
            elif lbl in ("ai", "ai_generated", "synthetic"):
                p = BASE / "data" / "ai_generated" / f"{tid}.wav"
                label = 1
            else:
                continue
            if p.exists():
                tasks.append((str(p), tid, label, str(t.get("genre", "Unknown"))))
        
        print(f"Extracting 133-dim features for {len(tasks)} audio files using 10 parallel processes...")
        t0 = time.time()
        results = []
        with ProcessPoolExecutor(max_workers=min(10, os.cpu_count() or 4)) as executor:
            for res in executor.map(_extract_worker, tasks):
                results.append(res)
        print(f"Parallel feature extraction completed in {time.time() - t0:.1f}s.")
        
        X133_list, y133_list, tids_133, genres_133 = [], [], [], []
        for tid, feat, label, genre, success in results:
            if success and feat is not None and len(feat) == 133 and np.all(np.isfinite(feat)):
                X133_list.append(feat)
                y133_list.append(label)
                tids_133.append(tid)
                genres_133.append(genre)
                
        X133 = np.array(X133_list)
        y133 = np.array(y133_list)
        np.savez_compressed(str(CACHE_133), X=X133, y=y133, tids=np.array(tids_133), genres=np.array(genres_133))
        print(f"Cached {len(X133)} 133-dim feature vectors to {CACHE_133}.")

    print(f"133-dim: {len(X133)} samples (Human={np.sum(y133==0)}, AI={np.sum(y133==1)})")
    pipe133 = make_ensemble()
    skf133 = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    y133_pred = cross_val_predict(pipe133, X133, y133, cv=skf133, method="predict")
    y133_prob = cross_val_predict(pipe133, X133, y133, cv=skf133, method="predict_proba")[:, 1]
    
    metrics_133 = compute_metrics(y133, y133_pred, y133_prob, "133-dim (Full Rich Acoustic Vectors)")
    print(f"133-dim CV: Accuracy={metrics_133['accuracy']} (95% CI: {metrics_133['accuracy_wilson_95ci']})")
    print(f"            BalAcc={metrics_133['balanced_accuracy']}, ROC-AUC={metrics_133['roc_auc']}, EER={metrics_133['eer']}")
    print(f"            Confusion Matrix: {metrics_133['confusion_matrix']}")

    # ── 3. Baselines ──
    print("\n--- Evaluating Diagnostic Baselines ---")
    maj_class = int(np.bincount(y56).argmax())
    maj_preds = np.full_like(y56, maj_class)
    maj_acc = float(np.mean(maj_preds == y56))
    maj_bal = float(balanced_accuracy_score(y56, maj_preds))
    
    # Genre-only rule: predict AI if genre in {"Rock", "Jazz"} (the top AI genres in corpus)
    ai_heavy = {"Rock", "Jazz"}
    genre_preds = np.array([1 if g in ai_heavy else 0 for g in genres_56])
    genre_acc = float(np.mean(genre_preds == y56))
    genre_bal = float(balanced_accuracy_score(y56, genre_preds))
    
    # Metadata empirical prior: predict AI if genre AI-rate > 0.50
    genre_counts = defaultdict(lambda: [0, 0])
    for g, label in zip(genres_56, y56):
        genre_counts[g][label] += 1
    meta_preds = np.array([1 if (genre_counts[g][1] / max(sum(genre_counts[g]), 1)) > 0.5 else 0 for g in genres_56])
    meta_acc = float(np.mean(meta_preds == y56))
    meta_bal = float(balanced_accuracy_score(y56, meta_preds))
    
    baselines = {
        "majority_class": {
            "feature_dim_label": "0-d (No features)",
            "accuracy": round(maj_acc, 4),
            "balanced_accuracy": round(maj_bal, 4),
            "rule": "Predict majority class (balanced 50/50)",
        },
        "genre_heuristic_rock_jazz": {
            "feature_dim_label": "Metadata-only (Genre label)",
            "accuracy": round(genre_acc, 4),
            "balanced_accuracy": round(genre_bal, 4),
            "rule": "Predict AI if genre is Rock or Jazz, else Human",
        },
        "metadata_empirical_prior": {
            "feature_dim_label": "Metadata-only (Corpus genre empirical base rate)",
            "accuracy": round(meta_acc, 4),
            "balanced_accuracy": round(meta_bal, 4),
            "rule": "Predict AI if genre AI-rate in corpus > 0.50",
        }
    }
    print(f"Baselines: Majority Acc={maj_acc:.4f}, Genre Heuristic Acc={genre_acc:.4f}, Meta Prior Acc={meta_acc:.4f}")

    # ── 4. Leave-One-Generator-Out (LOGO) Generalization ──
    print("\n--- Evaluating Leave-One-Generator-Out (LOGO) ---")
    gen_names = []
    for tid in tids_56:
        t = tracks[tid]
        rm = t.get("raw_metadata", {}) or {}
        g = rm.get("generator") or rm.get("source") or t.get("dataset_name", "")
        if "lyria" in str(g).lower() or "lyria" in tid.lower():
            gen_names.append("lyria-pro3")
        elif "musicgen" in str(g).lower() or "musicgen" in tid.lower():
            gen_names.append("musicgen")
        elif "acestep" in str(g).lower() or "acestep" in tid.lower():
            gen_names.append("acestep")
        else:
            gen_names.append("multi-gen-ai")
            
    logo_results = {}
    unique_gens = sorted(list(set(gen_names)))
    ai_mask = (y56 == 1)
    human_indices = np.where(~ai_mask)[0]
    
    for held_gen in unique_gens:
        test_ai_indices = [i for i in np.where(ai_mask)[0] if gen_names[i] == held_gen]
        train_ai_indices = [i for i in np.where(ai_mask)[0] if gen_names[i] != held_gen]
        if len(test_ai_indices) < 2 or len(train_ai_indices) < 2:
            continue
            
        n_h_tr = int(len(human_indices) * 0.8)
        train_idx = np.concatenate([human_indices[:n_h_tr], train_ai_indices])
        test_idx = np.concatenate([human_indices[n_h_tr:], test_ai_indices])
        
        clf = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=1000, random_state=SEED))])
        clf.fit(X56[train_idx], y56[train_idx])
        preds = clf.predict(X56[test_idx])
        
        test_ai_subset = (y56[test_idx] == 1)
        ai_rec = float(np.mean(preds[test_ai_subset] == 1))
        h_rec = float(np.mean(preds[~test_ai_subset] == 0))
        logo_results[held_gen] = {
            "feature_dim_label": "56-dim",
            "held_out_generator": held_gen,
            "test_ai_samples": int(len(test_ai_indices)),
            "unseen_generator_ai_recall": round(ai_rec, 4),
            "held_out_human_recall": round(h_rec, 4),
            "balanced_accuracy": round(float((ai_rec + h_rec) / 2.0), 4),
        }
        print(f"LOGO [{held_gen}]: AI Recall={ai_rec:.4f}, Human Recall={h_rec:.4f}, BalAcc={(ai_rec+h_rec)/2:.4f}")

    diagnostics = {
        "analysis": "F4_Origin_Classifier_Diagnostics",
        "seed": SEED,
        "metrics_56dim": metrics_56,
        "metrics_133dim": metrics_133,
        "baselines": baselines,
        "leave_one_generator_out": logo_results,
    }
    
    with open(OUT, "w") as f:
        json.dump(diagnostics, f, indent=2)
    print(f"\n✓ Saved complete F4 diagnostics to {OUT}")

if __name__ == "__main__":
    main()
