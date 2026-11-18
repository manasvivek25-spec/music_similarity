import json
from pathlib import Path
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler, normalize, QuantileTransformer
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score

REPO_ROOT = Path(".").resolve()
with open(REPO_ROOT / "features/remote_index.json", "r", encoding="utf-8") as f:
    tracks = json.load(f)["tracks"]

mert_dir = REPO_ROOT / "features/mert"

human_haim = []
human_jamendo = []
ai_clean = {"acestep": [], "lyria-pro3": [], "musicgen": []}

for tid, t in tracks.items():
    f56 = t.get("origin_features_56")
    if f56 is None or len(f56) != 56:
        continue
    f_arr = np.array(f56, dtype=np.float32)
    m_p = mert_dir / f"{tid}_mert.npy"
    m_vec = None
    if m_p.is_file():
        m_raw = np.load(str(m_p))
        m_vec = m_raw.mean(axis=0) if m_raw.ndim == 2 else m_raw
    item = {"tid": tid, "feat_56": f_arr, "mert": m_vec, "meta": t}
    cat = str(t.get("origin_label", "")).lower()
    gen = t.get("generator")
    if cat in ("human", "0", "false"):
        if "jamendo" in tid:
            human_jamendo.append(item)
        else:
            human_haim.append(item)
    elif gen in ai_clean:
        ai_clean[gen].append(item)

all_human = human_haim + human_jamendo
print(f"Data Breakdown: HAIM Human={len(human_haim)}, Jamendo Human={len(human_jamendo)}, ACE={len(ai_clean['acestep'])}, Lyria={len(ai_clean['lyria-pro3'])}, MusicGen={len(ai_clean['musicgen'])}")

# Define feature extractors
def extract_X(items, mode):
    if mode == "acoustic_56":
        return np.array([r["feat_56"] for r in items])
    elif mode == "mert_768":
        return np.array([r["mert"] for r in items if r["mert"] is not None])
    elif mode == "fusion_824":
        return np.array([np.concatenate([r["feat_56"], r["mert"]]) for r in items if r["mert"] is not None])
    elif mode == "mert_l2":
        X = np.array([r["mert"] for r in items if r["mert"] is not None])
        return normalize(X)
    elif mode == "fusion_l2":
        # normalize both separately before concatenation
        X_56 = normalize(np.array([r["feat_56"] for r in items]))
        X_m = normalize(np.array([r["mert"] for r in items if r["mert"] is not None]))
        return np.hstack([X_56, X_m])

# Model candidates to evaluate
def get_candidate_models():
    return {
        "LinearSVC(C=0.01)": lambda: Pipeline([("s", StandardScaler()), ("c", LinearSVC(C=0.01, random_state=42))]),
        "LinearSVC(C=0.1)": lambda: Pipeline([("s", StandardScaler()), ("c", LinearSVC(C=0.1, random_state=42))]),
        "LinearSVC(C=1.0)": lambda: Pipeline([("s", StandardScaler()), ("c", LinearSVC(C=1.0, random_state=42))]),
        "LinearSVC_balanced": lambda: Pipeline([("s", StandardScaler()), ("c", LinearSVC(C=0.1, class_weight="balanced", random_state=42))]),
        "LogisticRegression(C=0.1)": lambda: Pipeline([("s", StandardScaler()), ("c", LogisticRegression(C=0.1, max_iter=1000, random_state=42))]),
        "LogisticRegression(C=1.0)": lambda: Pipeline([("s", StandardScaler()), ("c", LogisticRegression(C=1.0, max_iter=1000, random_state=42))]),
        "LogisticRegression_balanced": lambda: Pipeline([("s", StandardScaler()), ("c", LogisticRegression(C=0.1, class_weight="balanced", max_iter=1000, random_state=42))]),
        "RBF_SVC(C=1.0, gamma=scale)": lambda: Pipeline([("s", StandardScaler()), ("c", SVC(C=1.0, kernel="rbf", probability=True, random_state=42))]),
        "RBF_SVC(C=10.0, gamma=scale)": lambda: Pipeline([("s", StandardScaler()), ("c", SVC(C=10.0, kernel="rbf", probability=True, random_state=42))]),
        "RBF_SVC(C=1.0, gamma=auto)": lambda: Pipeline([("s", StandardScaler()), ("c", SVC(C=1.0, kernel="rbf", gamma="auto", probability=True, random_state=42))]),
        "RandomForest(n=100)": lambda: Pipeline([("s", StandardScaler()), ("c", RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42))]),
        "RandomForest_balanced": lambda: Pipeline([("s", StandardScaler()), ("c", RandomForestClassifier(n_estimators=100, max_depth=6, class_weight="balanced", random_state=42))]),
        "HistGradientBoosting": lambda: Pipeline([("s", StandardScaler()), ("c", HistGradientBoostingClassifier(max_iter=100, random_state=42))]),
        "kNN(k=5, cosine)": lambda: Pipeline([("s", StandardScaler()), ("c", KNeighborsClassifier(n_neighbors=5, metric="cosine"))]),
        "kNN(k=3, cosine)": lambda: Pipeline([("s", StandardScaler()), ("c", KNeighborsClassifier(n_neighbors=3, metric="cosine"))]),
        "MLP(64,32)": lambda: Pipeline([("s", StandardScaler()), ("c", MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42))])
    }

generators = ["acestep", "lyria-pro3", "musicgen"]
feature_modes = ["acoustic_56", "mert_768", "fusion_824", "mert_l2", "fusion_l2"]

print("\nRunning systematic Leave-One-Generator-Out sweep...")
results_grid = []

# Fixed outer split: 75% human for train, 25% human for test
np.random.seed(42)
idx_h = np.random.permutation(len(all_human))
n_tr_h = int(len(all_human) * 0.75)
tr_h_fixed = [all_human[i] for i in idx_h[:n_tr_h]]
te_h_fixed = [all_human[i] for i in idx_h[n_tr_h:]]

for feat_mode in feature_modes:
    for model_name, model_fn in get_candidate_models().items():
        logo_recalls = {}
        human_recalls = {}
        bal_accs = {}
        
        for held_out in generators:
            test_ai = ai_clean[held_out]
            train_ai = []
            for g in generators:
                if g != held_out:
                    train_ai.extend(ai_clean[g])
            
            # Filter tracks with valid representations
            tr_h_sub = [r for r in tr_h_fixed if r["mert"] is not None or "56" in feat_mode]
            te_h_sub = [r for r in te_h_fixed if r["mert"] is not None or "56" in feat_mode]
            train_ai_sub = [r for r in train_ai if r["mert"] is not None or "56" in feat_mode]
            test_ai_sub = [r for r in test_ai if r["mert"] is not None or "56" in feat_mode]
            
            X_tr = extract_X(tr_h_sub + train_ai_sub, feat_mode)
            y_tr = np.array([0]*len(tr_h_sub) + [1]*len(train_ai_sub))
            
            X_te_h = extract_X(te_h_sub, feat_mode)
            X_te_a = extract_X(test_ai_sub, feat_mode)
            
            try:
                clf = model_fn()
                clf.fit(X_tr, y_tr)
                
                preds_h = clf.predict(X_te_h)
                preds_a = clf.predict(X_te_a)
                
                h_rec = float(np.mean(preds_h == 0))
                a_rec = float(np.mean(preds_a == 1))
                b_acc = (h_rec + a_rec) / 2.0
                
                logo_recalls[held_out] = a_rec
                human_recalls[held_out] = h_rec
                bal_accs[held_out] = b_acc
            except Exception as e:
                logo_recalls[held_out] = 0.0
                human_recalls[held_out] = 0.0
                bal_accs[held_out] = 0.0

        mean_ai_recall = np.mean(list(logo_recalls.values()))
        mean_bal_acc = np.mean(list(bal_accs.values()))
        mean_h_spec = np.mean(list(human_recalls.values()))
        
        results_grid.append({
            "feature_mode": feat_mode,
            "model_name": model_name,
            "acestep_recall": logo_recalls.get("acestep", 0),
            "musicgen_recall": logo_recalls.get("musicgen", 0),
            "lyria_recall": logo_recalls.get("lyria-pro3", 0),
            "mean_human_spec": mean_h_spec,
            "mean_ai_recall": mean_ai_recall,
            "mean_bal_acc": mean_bal_acc
        })

# Sort by Lyria recall descending, then mean AI recall
results_grid.sort(key=lambda x: (x["lyria_recall"], x["mean_ai_recall"]), reverse=True)

print("\n--- TOP 15 CONFIGURATIONS RANKED BY LYRIA RECALL ---")
print(f"{'Feature Mode':<14} | {'Model Name':<28} | {'ACE':<6} | {'MGen':<6} | {'Lyria':<6} | {'Human':<6} | {'BalAcc':<6}")
print("-" * 85)
for r in results_grid[:15]:
    print(f"{r['feature_mode']:<14} | {r['model_name']:<28} | {r['acestep_recall']*100:5.1f}% | {r['musicgen_recall']*100:5.1f}% | {r['lyria_recall']*100:5.1f}% | {r['mean_human_spec']*100:5.1f}% | {r['mean_bal_acc']*100:5.1f}%")

print("\n--- TOP 10 CONFIGURATIONS RANKED BY MEAN BALANCED ACCURACY ---")
results_grid_bal = sorted(results_grid, key=lambda x: x["mean_bal_acc"], reverse=True)
print(f"{'Feature Mode':<14} | {'Model Name':<28} | {'ACE':<6} | {'MGen':<6} | {'Lyria':<6} | {'Human':<6} | {'BalAcc':<6}")
print("-" * 85)
for r in results_grid_bal[:10]:
    print(f"{r['feature_mode']:<14} | {r['model_name']:<28} | {r['acestep_recall']*100:5.1f}% | {r['musicgen_recall']*100:5.1f}% | {r['lyria_recall']*100:5.1f}% | {r['mean_human_spec']*100:5.1f}% | {r['mean_bal_acc']*100:5.1f}%")

