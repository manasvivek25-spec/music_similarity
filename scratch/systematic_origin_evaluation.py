"""
Systematic Origin Classifier Generalization Evaluation.
Runs Leave-One-Generator-Out (LOGO) and Stratified K-Fold CV across multiple model families,
feature spaces, regularizations, and threshold tuning strategies.
Saves comprehensive results to reports/classifier_generalization_advanced.json.
"""
import json
import warnings
from pathlib import Path
import numpy as np

warnings.filterwarnings("ignore")

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    confusion_matrix, accuracy_score, balanced_accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
)

REPO_ROOT = Path(".").resolve()
with open(REPO_ROOT / "features/remote_index.json", "r", encoding="utf-8") as f:
    tracks = json.load(f)["tracks"]

mert_dir = REPO_ROOT / "features/mert"

human_items = []
ai_items = {"acestep": [], "lyria-pro3": [], "musicgen": []}

for tid, t in sorted(tracks.items()):
    f56 = t.get("origin_features_56")
    if f56 is None or len(f56) != 56:
        continue
    f_arr = np.array(f56, dtype=np.float32)
    m_p = mert_dir / f"{tid}_mert.npy"
    if not m_p.is_file():
        continue
    m_raw = np.load(str(m_p))
    m_vec = m_raw.mean(axis=0) if m_raw.ndim == 2 else m_raw
    m_vec = np.array(m_vec, dtype=np.float32)
    
    item = {
        "tid": tid,
        "feat_56": f_arr,
        "mert": m_vec,
        "fused": np.concatenate([f_arr, m_vec]),
        "origin_label": t.get("origin_label"),
        "generator": t.get("generator")
    }
    cat = str(t.get("origin_label", "")).lower()
    gen = t.get("generator")
    if cat in ("human", "0", "false"):
        human_items.append(item)
    elif gen in ai_items:
        ai_items[gen].append(item)

print(f"Loaded corpus: Human={len(human_items)}, ACE-Step={len(ai_items['acestep'])}, Lyria-Pro3={len(ai_items['lyria-pro3'])}, MusicGen={len(ai_items['musicgen'])}")

def compute_metrics(y_true, y_pred, y_score=None):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    bal_acc = (spec + rec) / 2.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    
    roc_auc = None
    pr_auc = None
    if y_score is not None:
        try:
            roc_auc = float(roc_auc_score(y_true, y_score))
        except Exception:
            roc_auc = None
        try:
            pr_auc = float(average_precision_score(y_true, y_score))
        except Exception:
            pr_auc = None
            
    return {
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "accuracy": float(acc),
        "balanced_accuracy": float(bal_acc),
        "precision": float(prec),
        "recall": float(rec),
        "specificity": float(spec),
        "f1": float(f1),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc
    }

# 1. Stratified 5-Fold Cross Validation across entire corpus (Standard supervised baseline)
def evaluate_stratified_cv(feature_key="mert"):
    all_data = []
    for h in human_items:
        all_data.append((h[feature_key], 0, "human"))
    for gen, items in ai_items.items():
        for a in items:
            all_data.append((a[feature_key], 1, gen))
            
    X = np.array([d[0] for d in all_data])
    y = np.array([d[1] for d in all_data])
    gens = [d[2] for d in all_data]
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_preds = np.zeros(len(y))
    y_scores = np.zeros(len(y))
    
    for tr_idx, te_idx in skf.split(X, y):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[tr_idx])
        X_te = scaler.transform(X[te_idx])
        clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        clf.fit(X_tr, y[tr_idx])
        scores = clf.decision_function(X_te)
        y_scores[te_idx] = scores
        y_preds[te_idx] = (scores > 0).astype(int)
        
    metrics = compute_metrics(y, y_preds, y_scores)
    per_gen = {}
    for gen in ["acestep", "lyria-pro3", "musicgen"]:
        g_idx = [i for i, g in enumerate(gens) if g == gen]
        rec_g = float(np.mean(y_preds[g_idx])) if g_idx else 0.0
        per_gen[gen] = {"count": len(g_idx), "recall": rec_g}
    h_idx = [i for i, g in enumerate(gens) if g == "human"]
    per_gen["human_specificity"] = float(1.0 - np.mean(y_preds[h_idx])) if h_idx else 0.0
    
    return {"overall": metrics, "per_generator": per_gen}

# 2. LOGO Cross Validation
def evaluate_logo(holdout_gen, model_factory, feature_key="mert", select_concordant=False):
    other_gens = [g for g in ai_items if g != holdout_gen]
    train_ai = []
    for g in other_gens:
        train_ai.extend(ai_items[g])
    test_ai = ai_items[holdout_gen]
    
    rng = np.random.RandomState(42)
    n_splits = 5
    split_metrics = []
    split_gen_recalls = []
    
    perm = rng.permutation(len(human_items))
    fold_size = len(human_items) // n_splits
    
    for k in range(n_splits):
        te_h_idx = perm[k * fold_size : (k + 1) * fold_size]
        tr_h_idx = np.setdiff1d(perm, te_h_idx)
        
        tr_h = [human_items[i] for i in tr_h_idx]
        te_h = [human_items[i] for i in te_h_idx]
        
        X_train_raw = np.vstack([r[feature_key] for r in tr_h] + [r[feature_key] for r in train_ai])
        y_train = np.array([0] * len(tr_h) + [1] * len(train_ai))
        
        X_test_raw = np.vstack([r[feature_key] for r in te_h] + [r[feature_key] for r in test_ai])
        y_test = np.array([0] * len(te_h) + [1] * len(test_ai))
        
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_train_raw)
        X_te = scaler.transform(X_test_raw)
        
        if select_concordant:
            H_tr = X_tr[y_train == 0]
            n_tr_h = len(tr_h)
            g1_items = ai_items[other_gens[0]]
            g2_items = ai_items[other_gens[1]]
            g1_scaled = X_tr[n_tr_h : n_tr_h + len(g1_items)]
            g2_scaled = X_tr[n_tr_h + len(g1_items) :]
            
            diff1 = np.mean(g1_scaled, axis=0) - np.mean(H_tr, axis=0)
            diff2 = np.mean(g2_scaled, axis=0) - np.mean(H_tr, axis=0)
            concordant_mask = (np.sign(diff1) == np.sign(diff2)) & (np.abs(diff1) > 0.2) & (np.abs(diff2) > 0.2)
            if np.sum(concordant_mask) >= 5:
                X_tr = X_tr[:, concordant_mask]
                X_te = X_te[:, concordant_mask]
                
        clf = model_factory()
        clf.fit(X_tr, y_train)
        
        if hasattr(clf, "decision_function"):
            scores = clf.decision_function(X_te)
        elif hasattr(clf, "predict_proba"):
            scores = clf.predict_proba(X_te)[:, 1]
        else:
            scores = clf.predict(X_te).astype(float)
            
        preds = clf.predict(X_te)
        
        m = compute_metrics(y_test, preds, scores)
        split_metrics.append(m)
        rec_ai = np.mean(preds[len(te_h):] == 1)
        split_gen_recalls.append(rec_ai)
        
    avg_m = {}
    for key in ["accuracy", "balanced_accuracy", "precision", "recall", "specificity", "f1", "roc_auc", "pr_auc"]:
        vals = [sm[key] for sm in split_metrics if sm[key] is not None]
        avg_m[key] = float(np.mean(vals)) if vals else None
        
    avg_m["target_generator_recall"] = float(np.mean(split_gen_recalls))
    return avg_m

models = {
    "LogisticRegression_C1": lambda: LogisticRegression(C=1.0, max_iter=1000, random_state=42),
    "LogisticRegression_C0.1": lambda: LogisticRegression(C=0.1, max_iter=1000, random_state=42),
    "LogisticRegression_L1": lambda: LogisticRegression(penalty="l1", solver="liblinear", C=0.5, random_state=42),
    "LinearSVC_C0.01": lambda: LinearSVC(C=0.01, random_state=42),
    "LinearSVC_C0.1": lambda: LinearSVC(C=0.1, random_state=42),
    "RBF_SVC_C1": lambda: SVC(kernel="rbf", C=1.0, random_state=42),
    "RandomForest": lambda: RandomForestClassifier(n_estimators=50, max_depth=6, random_state=42),
    "ExtraTrees": lambda: ExtraTreesClassifier(n_estimators=50, max_depth=6, random_state=42),
    "MLP_32": lambda: MLPClassifier(hidden_layer_sizes=(32,), max_iter=300, alpha=0.1, random_state=42),
}

print("\n--- Running LOGO CV on Held-Out Generators ---")
results = {
    "stratified_5fold_cv": {
        "mert": evaluate_stratified_cv("mert"),
        "feat_56": evaluate_stratified_cv("feat_56"),
        "fused": evaluate_stratified_cv("fused")
    },
    "logo_cv": {}
}

print(f"\nStratified 5-Fold CV (MERT):")
print(f"  Accuracy: {results['stratified_5fold_cv']['mert']['overall']['accuracy']:.3f}")
print(f"  Balanced Accuracy: {results['stratified_5fold_cv']['mert']['overall']['balanced_accuracy']:.3f}")
print(f"  Human Specificity: {results['stratified_5fold_cv']['mert']['per_generator']['human_specificity']:.3f}")
print(f"  ACE-Step Recall:   {results['stratified_5fold_cv']['mert']['per_generator']['acestep']['recall']:.3f}")
print(f"  MusicGen Recall:   {results['stratified_5fold_cv']['mert']['per_generator']['musicgen']['recall']:.3f}")
print(f"  Lyria-Pro3 Recall: {results['stratified_5fold_cv']['mert']['per_generator']['lyria-pro3']['recall']:.3f}")

for holdout in ["acestep", "musicgen", "lyria-pro3"]:
    results["logo_cv"][holdout] = {}
    print(f"\n================ LOGO: Held-Out Generator = {holdout} ================")
    for m_name, m_factory in models.items():
        res_mert = evaluate_logo(holdout, m_factory, "mert")
        res_56 = evaluate_logo(holdout, m_factory, "feat_56")
        res_fused = evaluate_logo(holdout, m_factory, "fused")
        results["logo_cv"][holdout][m_name] = {
            "mert": res_mert,
            "feat_56": res_56,
            "fused": res_fused
        }
        print(f"Model: {m_name:<24} | MERT: Rec={res_mert['target_generator_recall']:.3f}, Spec={res_mert['specificity']:.3f}, BalAcc={res_mert['balanced_accuracy']:.3f} | Fused: Rec={res_fused['target_generator_recall']:.3f}, Spec={res_fused['specificity']:.3f}")

print("\n--- Testing Concordance Feature Selection on Lyria-Pro3 LOGO ---")
concordance_results = {}
for m_name in ["LogisticRegression_C0.1", "LinearSVC_C0.01", "RBF_SVC_C1", "RandomForest"]:
    c_res = evaluate_logo("lyria-pro3", models[m_name], "mert", select_concordant=True)
    concordance_results[m_name] = c_res
    print(f"Concordant MERT + {m_name:<24}: Rec={c_res['target_generator_recall']:.3f}, Spec={c_res['specificity']:.3f}, BalAcc={c_res['balanced_accuracy']:.3f}")

results["concordance_selection_logo_lyria"] = concordance_results

print("\n--- Empirical Trade-off Curve for Lyria-Pro3 Under LOGO ---")
train_ai = ai_items["acestep"] + ai_items["musicgen"]
test_ai = ai_items["lyria-pro3"]

threshold_sweep = []
for thresh in np.linspace(-2.0, 2.0, 41):
    specs = []
    recs = []
    for k in range(5):
        rng = np.random.RandomState(k + 100)
        perm = rng.permutation(len(human_items))
        n_te = len(human_items) // 5
        te_h = [human_items[i] for i in perm[:n_te]]
        tr_h = [human_items[i] for i in perm[n_te:]]
        
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(np.vstack([r["mert"] for r in tr_h] + [r["mert"] for r in train_ai]))
        y_tr = np.array([0] * len(tr_h) + [1] * len(train_ai))
        
        clf = LogisticRegression(C=0.1, max_iter=1000, random_state=42)
        clf.fit(X_tr, y_tr)
        
        df_h = clf.decision_function(scaler.transform(np.vstack([r["mert"] for r in te_h])))
        df_lyria = clf.decision_function(scaler.transform(np.vstack([r["mert"] for r in test_ai])))
        
        spec = np.mean(df_h <= thresh)
        rec = np.mean(df_lyria > thresh)
        specs.append(spec)
        recs.append(rec)
        
    avg_spec = float(np.mean(specs))
    avg_rec = float(np.mean(recs))
    threshold_sweep.append({
        "threshold": float(thresh),
        "human_specificity": avg_spec,
        "lyria_recall": avg_rec,
        "balanced_accuracy": float((avg_spec + avg_rec) / 2.0)
    })

results["lyria_threshold_tradeoff"] = threshold_sweep

valid_tradeoffs = [t for t in threshold_sweep if t["human_specificity"] >= 0.90]
best_at_90_spec = max(valid_tradeoffs, key=lambda x: x["lyria_recall"]) if valid_tradeoffs else None
if best_at_90_spec:
    print(f"Best Lyria Recall achievable at >=90% Human Specificity: {best_at_90_spec['lyria_recall']*100:.1f}% (Threshold={best_at_90_spec['threshold']:.2f}, Specificity={best_at_90_spec['human_specificity']*100:.1f}%)")

reach_70_rec = [t for t in threshold_sweep if t["lyria_recall"] >= 0.70]
best_spec_at_70_rec = max(reach_70_rec, key=lambda x: x["human_specificity"]) if reach_70_rec else None
if best_spec_at_70_rec:
    print(f"Human Specificity when Lyria Recall is forced to >=70%: {best_spec_at_70_rec['human_specificity']*100:.1f}% (Recall={best_spec_at_70_rec['lyria_recall']*100:.1f}%, Threshold={best_spec_at_70_rec['threshold']:.2f}) -> VIOLATES >=90% SPECIFICITY CONSTRAINT")

out_path = REPO_ROOT / "reports/classifier_generalization_advanced.json"
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print(f"\nSaved advanced generalization results to: {out_path}")
