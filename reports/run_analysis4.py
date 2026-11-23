"""
Analysis 4: F4 Diagnostics — Full classifier evaluation on 56-dim and 133-dim features.
Confusion matrix, Wilson 95% CIs, balanced accuracy, ROC-AUC, EER,
leave-one-generator-out recall, majority/metadata-only/genre-only baselines.
Seed=42. Output: reports/analysis4_f4_diagnostics.json
"""
import json, numpy as np, sys, time
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import joblib
from sklearn.metrics import (confusion_matrix, balanced_accuracy_score,
                              roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, LeaveOneGroupOut
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier, HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from scipy.stats import norm as scipy_norm

SEED = 42
BASE = Path(__file__).resolve().parent.parent
OUT  = BASE / "reports" / "analysis4_f4_diagnostics.json"
FEAT = BASE / "features"

print("Loading 56-dim features from remote_index.json...")
ri = json.load(open(BASE / "features" / "remote_index.json"))
tracks = ri["tracks"]

X56_list, y_list, tids_list, genre_list, gen_list = [], [], [], [], []
for tid, t in tracks.items():
    if any(x in tid for x in ("ref_","tmp")): continue
    lbl = str(t.get("origin_label","")).lower()
    feats = t.get("origin_features_56")
    if feats and len(feats)==56 and np.all(np.isfinite(feats)):
        y_val = 0 if lbl=="human" else 1 if lbl in ("ai","ai_generated","synthetic") else None
        if y_val is not None:
            X56_list.append(np.array(feats, dtype=np.float32))
            y_list.append(y_val)
            tids_list.append(tid)
            genre_list.append(str(t.get("genre","Unknown")))
            # generator: infer from filename pattern for AI tracks
            gen = "human"
            if y_val == 1:
                rm = t.get("raw_metadata",{}) or {}
                gen = str(rm.get("generator","") or rm.get("source","") or "unknown_ai")
                if not gen or gen=="unknown_ai":
                    # Try to infer from track_id or dataset_name
                    dn = str(t.get("dataset_name",""))
                    if "lyria" in dn.lower() or "lyria" in tid.lower():
                        gen = "lyria"
                    elif "musicgen" in dn.lower() or "musicgen" in tid.lower():
                        gen = "musicgen"
                    elif "acestep" in dn.lower() or "acestep" in tid.lower():
                        gen = "acestep"
                    else:
                        gen = "unknown_ai"
            gen_list.append(gen)

X56 = np.array(X56_list)
y    = np.array(y_list)
print(f"56-dim: {len(X56)} samples (human={sum(y==0)}, ai={sum(y==1)})")

# ── Helper: Wilson 95% CI ─────────────────────────────────────────────────────
def wilson_ci(k, n, z=1.96):
    if n == 0: return (0.0, 0.0)
    p_hat = k / n
    denom = 1 + z**2/n
    center = (p_hat + z**2/(2*n)) / denom
    spread = z * np.sqrt(p_hat*(1-p_hat)/n + z**2/(4*n**2)) / denom
    return (round(max(0, center-spread), 4), round(min(1, center+spread), 4))

# ── Build main ensemble (same as production) ──────────────────────────────────
def make_ensemble():
    rf  = RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1)
    et  = ExtraTreesClassifier(n_estimators=100, random_state=SEED, n_jobs=-1)
    hgb = HistGradientBoostingClassifier(random_state=SEED)
    ens = VotingClassifier(estimators=[("rf",rf),("et",et),("hgb",hgb)], voting="soft")
    return Pipeline([("scaler",StandardScaler()),("classifier",ens)])

# ── 5-fold CV predictions ─────────────────────────────────────────────────────
print("Running 5-fold CV on 56-dim features...")
pipe56 = make_ensemble()
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
y_pred_cv = cross_val_predict(pipe56, X56, y, cv=skf, method="predict")
y_prob_cv = cross_val_predict(pipe56, X56, y, cv=skf, method="predict_proba")[:,1]

cm = confusion_matrix(y, y_pred_cv)
tn, fp, fn, tp = cm.ravel()
n = len(y); n_h = int(sum(y==0)); n_a = int(sum(y==1))

accuracy     = float((tp+tn)/n)
sensitivity  = float(tp / (tp+fn))  # AI recall
specificity  = float(tn / (tn+fp))  # Human recall
bal_acc      = float(balanced_accuracy_score(y, y_pred_cv))
roc_auc      = float(roc_auc_score(y, y_prob_cv))
precision    = float(tp / (tp+fp)) if (tp+fp)>0 else 0.0
f1           = float(2*tp / (2*tp+fp+fn)) if (2*tp+fp+fn)>0 else 0.0

# Wilson CIs
ci_acc  = wilson_ci(int(tp+tn), n)
ci_sens = wilson_ci(int(tp), int(tp+fn))
ci_spec = wilson_ci(int(tn), int(tn+fp))

# EER: equal error rate (where FPR = FNR)
fpr_arr, tpr_arr, thresholds = roc_curve(y, y_prob_cv)
fnr_arr = 1 - tpr_arr
eer_idx = np.argmin(np.abs(fpr_arr - fnr_arr))
eer = float((fpr_arr[eer_idx] + fnr_arr[eer_idx]) / 2)

print(f"56-dim CV: Acc={accuracy:.4f} BalAcc={bal_acc:.4f} ROC-AUC={roc_auc:.4f} EER={eer:.4f}")
print(f"  Confusion: TN={tn} FP={fp} FN={fn} TP={tp}")
print(f"  Sensitivity={sensitivity:.4f} Specificity={specificity:.4f}")

# ── 133-dim features: extract fresh from audio ────────────────────────────────
print("\nExtracting 133-dim features from audio files...")
from src.origin_classifier import extract_origin_features_from_audio

X133_list = []; y133_list = []; tids133 = []
audio_base_dirs = {
    "human": BASE / "data" / "human_songs",
    "ai": BASE / "data" / "ai_generated",
}
for tid, t in tracks.items():
    if any(x in tid for x in ("ref_","tmp")): continue
    lbl = str(t.get("origin_label","")).lower()
    if lbl == "human":
        p = audio_base_dirs["human"] / f"{tid}.wav"
        y_val = 0
    elif lbl in ("ai","ai_generated","synthetic"):
        p = audio_base_dirs["ai"] / f"{tid}.wav"
        y_val = 1
    else:
        continue
    if p.exists():
        try:
            feats = extract_origin_features_from_audio(str(p))
            if feats is not None and len(feats)==133 and np.all(np.isfinite(feats)):
                X133_list.append(feats)
                y133_list.append(y_val)
                tids133.append(tid)
        except Exception as e:
            pass

X133 = np.array(X133_list); y133 = np.array(y133_list)
print(f"133-dim: {len(X133)} samples (human={sum(y133==0)}, ai={sum(y133==1)})")

cv_res_133 = {}
if len(X133) >= 10 and len(set(y133)) == 2:
    pipe133 = make_ensemble()
    skf133  = StratifiedKFold(n_splits=min(5, len(X133)//2), shuffle=True, random_state=SEED)
    yp133   = cross_val_predict(pipe133, X133, y133, cv=skf133, method="predict")
    yprob133= cross_val_predict(pipe133, X133, y133, cv=skf133, method="predict_proba")[:,1]
    cm133   = confusion_matrix(y133, yp133)
    tn3, fp3, fn3, tp3 = cm133.ravel()
    cv_res_133 = {
        "n_samples": len(X133),
        "feature_dim": 133,
        "accuracy": round(float((tp3+tn3)/len(X133)),4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y133, yp133)),4),
        "roc_auc": round(float(roc_auc_score(y133, yprob133)),4),
        "confusion_matrix": cm133.tolist(),
        "sensitivity_ai_recall": round(float(tp3/(tp3+fn3)) if (tp3+fn3)>0 else 0,4),
        "specificity_human_recall": round(float(tn3/(tn3+fp3)) if (tn3+fp3)>0 else 0,4),
    }
    print(f"133-dim CV: Acc={cv_res_133['accuracy']} BalAcc={cv_res_133['balanced_accuracy']} AUC={cv_res_133['roc_auc']}")

# ── Leave-One-Generator-Out (LOGO) on AI tracks ───────────────────────────────
print("\nRunning Leave-One-Generator-Out...")
ai_idxs = [i for i,v in enumerate(y_list) if v == 1]
h_idxs  = [i for i,v in enumerate(y_list) if v == 0]
generators_ai = [gen_list[i] for i in ai_idxs]
unique_gens = [g for g in set(generators_ai) if g != "human" and g != "unknown_ai"]
print(f"Generator labels: {set(generators_ai)}")

logo_results = {}
for held_gen in unique_gens:
    test_ai_idx  = [i for i in ai_idxs if gen_list[i] == held_gen]
    train_ai_idx = [i for i in ai_idxs if gen_list[i] != held_gen]
    if len(test_ai_idx) < 2 or len(train_ai_idx) < 2: continue
    # Train on all human + non-held AI, test on held AI
    n_h_tr = int(len(h_idxs)*0.8)
    train_idx = h_idxs[:n_h_tr] + train_ai_idx
    test_idx  = h_idxs[n_h_tr:] + test_ai_idx
    X_tr = X56[train_idx]; y_tr = y[train_idx]
    X_te = X56[test_idx];  y_te = y[test_idx]
    try:
        lr = Pipeline([("sc",StandardScaler()),("clf",LogisticRegression(max_iter=1000,random_state=SEED))])
        lr.fit(X_tr, y_tr)
        y_te_pred = lr.predict(X_te)
        ai_test_mask = np.array([y_list[i] for i in test_idx]) == 1
        ai_recall = float(np.mean(y_te_pred[ai_test_mask] == 1))
        h_recall  = float(np.mean(y_te_pred[~ai_test_mask] == 0))
        logo_results[held_gen] = {
            "n_train": len(train_idx), "n_test_ai": int(ai_test_mask.sum()),
            "ai_recall": round(ai_recall,4), "human_recall": round(h_recall,4),
            "balanced_acc": round((ai_recall+h_recall)/2,4),
        }
        print(f"  LOGO {held_gen}: AI recall={ai_recall:.3f}, H recall={h_recall:.3f}")
    except Exception as e:
        print(f"  LOGO {held_gen} ERROR: {e}")

# ── Baselines ─────────────────────────────────────────────────────────────────
# 1. Majority class
majority_class = int(np.bincount(y).argmax())
majority_pred  = np.full_like(y, majority_class)
maj_acc = float(np.mean(majority_pred == y))
maj_balacc = float(balanced_accuracy_score(y, majority_pred))

# 2. Genre-only baseline (predict AI if genre in AI-heavy genres)
ai_heavy_genres = {"Rock", "Jazz"}  # from corpus distribution
genre_pred = np.array([1 if genre_list[i] in ai_heavy_genres else 0 for i in range(len(y))])
genre_acc  = float(np.mean(genre_pred == y))
genre_balacc = float(balanced_accuracy_score(y, genre_pred))

# 3. Metadata-only (predict AI if genre appears with many AI tracks)
# Compute per-genre AI fraction
genre_ai_frac = defaultdict(list)
for i in range(len(y)):
    genre_ai_frac[genre_list[i]].append(y[i])
genre_ai_rate = {g: np.mean(v) for g, v in genre_ai_frac.items()}
meta_pred = np.array([1 if genre_ai_rate.get(genre_list[i], 0.5) > 0.5 else 0 for i in range(len(y))])
meta_acc  = float(np.mean(meta_pred == y))
meta_balacc = float(balanced_accuracy_score(y, meta_pred))

print(f"Baselines: majority_acc={maj_acc:.3f} genre_acc={genre_acc:.3f} meta_acc={meta_acc:.3f}")

# ── Save ──────────────────────────────────────────────────────────────────────
result = {
    "analysis": "F4_Origin_Classifier_Diagnostics",
    "seed": SEED,
    "56dim_cv": {
        "feature_dim": 56,
        "n_samples": n,
        "cv_folds": 5,
        "confusion_matrix": {"TN":int(tn),"FP":int(fp),"FN":int(fn),"TP":int(tp)},
        "accuracy": round(accuracy,4),
        "accuracy_wilson_95ci": ci_acc,
        "sensitivity_ai_recall": round(sensitivity,4),
        "sensitivity_wilson_95ci": ci_sens,
        "specificity_human_recall": round(specificity,4),
        "specificity_wilson_95ci": ci_spec,
        "balanced_accuracy": round(bal_acc,4),
        "roc_auc": round(roc_auc,4),
        "eer": round(eer,4),
        "precision": round(precision,4),
        "f1": round(f1,4),
    },
    "133dim_cv": cv_res_133,
    "leave_one_generator_out": logo_results,
    "baselines": {
        "majority_class": {"predicted_class":majority_class,"accuracy":round(maj_acc,4),"balanced_accuracy":round(maj_balacc,4)},
        "genre_only": {"accuracy":round(genre_acc,4),"balanced_accuracy":round(genre_balacc,4),"rule":"predict AI if genre in Rock/Jazz"},
        "metadata_genre_rate": {"accuracy":round(meta_acc,4),"balanced_accuracy":round(meta_balacc,4),"genre_ai_rates":{g:round(v,3) for g,v in genre_ai_rate.items()}},
    },
}

OUT.parent.mkdir(parents=True, exist_ok=True)
json.dump(result, open(OUT,"w"), indent=2)
print(f"\n✓ Saved to {OUT}")
