import json
from pathlib import Path
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.pipeline import Pipeline
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score

from src.config import FEATURES_DIR, REMOTE_INDEX_FILE

with open(REMOTE_INDEX_FILE, "r", encoding="utf-8") as f:
    tracks = json.load(f)["tracks"]

# Collect clean labeled data
human_data = []
ai_by_gen = {"acestep": [], "lyria-pro3": [], "musicgen": []}
mert_by_id = {}

mert_dir = FEATURES_DIR / "mert"

for tid, t in tracks.items():
    feat_56 = t.get("origin_features_56")
    if not feat_56 or len(feat_56) != 56:
        continue
    f_arr = np.array(feat_56, dtype=np.float32)
    
    # Check MERT
    mert_p = mert_dir / f"{tid}_mert.npy"
    mert_arr = None
    if mert_p.is_file():
        m_raw = np.load(str(mert_p))
        mert_arr = m_raw.mean(axis=0) if m_raw.ndim == 2 else m_raw
        
    lbl = str(t.get("origin_label", "")).lower()
    gen = t.get("generator")
    
    if lbl in ("human", "0", "false"):
        human_data.append((tid, f_arr, mert_arr, "human"))
    elif gen in ai_by_gen:
        ai_by_gen[gen].append((tid, f_arr, mert_arr, gen))

print(f"Collected Clean Data:")
print(f"  Human: {len(human_data)} tracks")
for g, recs in ai_by_gen.items():
    print(f"  AI [{g}]: {len(recs)} tracks")

generators = ["acestep", "lyria-pro3", "musicgen"]

def run_logo(pipeline_builder, use_mert=False):
    logo_results = {}
    for held_out in generators:
        test_ai = ai_by_gen[held_out]
        train_ai = []
        for g in generators:
            if g != held_out:
                train_ai.extend(ai_by_gen[g])
                
        # Split humans 75/25 deterministically
        n_h_tr = int(len(human_data) * 0.75)
        train_h = human_data[:n_h_tr]
        test_h = human_data[n_h_tr:]
        
        def get_x(recs):
            if use_mert:
                return np.array([r[2] for r in recs if r[2] is not None])
            return np.array([r[1] for r in recs])
            
        X_tr = np.vstack([get_x(train_h), get_x(train_ai)])
        y_tr = np.array([0]*len(train_h) + [1]*len(train_ai))
        
        X_test_ai = get_x(test_ai)
        X_test_h = get_x(test_h)
        
        clf = pipeline_builder()
        clf.fit(X_tr, y_tr)
        
        preds_ai = clf.predict(X_test_ai)
        preds_h = clf.predict(X_test_h)
        
        ai_rec = float(np.mean(preds_ai == 1))
        h_rec = float(np.mean(preds_h == 0))
        logo_results[held_out] = {
            "unseen_ai_recall": round(ai_rec, 4),
            "human_recall": round(h_rec, 4),
            "balanced_acc": round((ai_rec + h_rec)/2, 4)
        }
    return logo_results

print("\n--- EXPERIMENT 1: Baseline 56-dim with StandardScaler + LogisticRegression ---")
exp1 = run_logo(lambda: Pipeline([("s", StandardScaler()), ("c", LogisticRegression(max_iter=1000, random_state=42))]))
print(json.dumps(exp1, indent=2))

print("\n--- EXPERIMENT 2: Baseline 56-dim with RobustScaler + LogisticRegression(C=0.1, class_weight='balanced') ---")
exp2 = run_logo(lambda: Pipeline([("s", RobustScaler()), ("c", LogisticRegression(C=0.1, class_weight="balanced", max_iter=1000, random_state=42))]))
print(json.dumps(exp2, indent=2))

print("\n--- EXPERIMENT 3: 56-dim with RobustScaler + LinearSVC(C=0.05, class_weight='balanced') ---")
exp3 = run_logo(lambda: Pipeline([("s", RobustScaler()), ("c", LinearSVC(C=0.05, class_weight="balanced", random_state=42))]))
print(json.dumps(exp3, indent=2))

print("\n--- EXPERIMENT 4: 56-dim with SelectKBest(25) + RobustScaler + RidgeClassifier ---")
exp4 = run_logo(lambda: Pipeline([
    ("s", RobustScaler()),
    ("sel", SelectKBest(f_classif, k=25)),
    ("c", RidgeClassifier(class_weight="balanced", random_state=42))
]))
print(json.dumps(exp4, indent=2))

print("\n--- EXPERIMENT 5: Supplementary MERT Foundation Embeddings (768-dim) alone ---")
exp5 = run_logo(lambda: Pipeline([("s", StandardScaler()), ("c", LogisticRegression(C=0.1, max_iter=1000, random_state=42))]), use_mert=True)
print(json.dumps(exp5, indent=2))

print("\n--- EXPERIMENT 6: MERT Embeddings with LinearSVC(C=0.01) ---")
exp6 = run_logo(lambda: Pipeline([("s", StandardScaler()), ("c", LinearSVC(C=0.01, random_state=42))]), use_mert=True)
print(json.dumps(exp6, indent=2))

