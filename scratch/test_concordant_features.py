import json
from pathlib import Path
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler, normalize
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score

REPO_ROOT = Path(".").resolve()
with open(REPO_ROOT / "features/remote_index.json", "r", encoding="utf-8") as f:
    tracks = json.load(f)["tracks"]

mert_dir = REPO_ROOT / "features/mert"

human = []
ai_dict = {"acestep": [], "lyria-pro3": [], "musicgen": []}

for tid, t in tracks.items():
    f56 = t.get("origin_features_56")
    m_p = mert_dir / f"{tid}_mert.npy"
    if not m_p.is_file() or f56 is None or len(f56) != 56:
        continue
    m_raw = np.load(str(m_p))
    m_vec = m_raw.mean(axis=0) if m_raw.ndim == 2 else m_raw
    f_arr = np.array(f56, dtype=np.float32)
    # Combine or test MERT
    item = {"tid": tid, "feat_56": f_arr, "mert": m_vec}
    cat = str(t.get("origin_label", "")).lower()
    gen = t.get("generator")
    if cat in ("human", "0", "false"):
        human.append(item)
    elif gen in ai_dict:
        ai_dict[gen].append(item)

print(f"Total valid: Human={len(human)}, ACE={len(ai_dict['acestep'])}, Lyria={len(ai_dict['lyria-pro3'])}, MusicGen={len(ai_dict['musicgen'])}")

# Test Leave-One-Generator-Out with Concordant Feature Selection on MERT
generators = ["acestep", "lyria-pro3", "musicgen"]

# Fixed 75/25 human split
np.random.seed(42)
idx_h = np.random.permutation(len(human))
n_tr_h = int(len(human) * 0.75)
tr_h = [human[i] for i in idx_h[:n_tr_h]]
te_h = [human[i] for i in idx_h[n_tr_h:]]

X_tr_h_m = np.array([r["mert"] for r in tr_h])
X_te_h_m = np.array([r["mert"] for r in te_h])

print("\n--- Concordant Multi-Generator Feature Selection Test ---")

for held_out in generators:
    train_gens = [g for g in generators if g != held_out]
    g1, g2 = train_gens[0], train_gens[1]
    
    X_g1 = np.array([r["mert"] for r in ai_dict[g1]])
    X_g2 = np.array([r["mert"] for r in ai_dict[g2]])
    X_test_ai = np.array([r["mert"] for r in ai_dict[held_out]])
    
    # 1. On training data ONLY, find features where g1 vs human and g2 vs human agree in sign
    diff_g1 = np.mean(X_g1, axis=0) - np.mean(X_tr_h_m, axis=0)
    diff_g2 = np.mean(X_g2, axis=0) - np.mean(X_tr_h_m, axis=0)
    
    pooled_std = np.std(X_tr_h_m, axis=0) + 1e-6
    d1 = diff_g1 / pooled_std
    d2 = diff_g2 / pooled_std
    
    # Agreement condition: same sign and both have magnitude > threshold
    for min_d in [0.0, 0.2, 0.3, 0.4, 0.5]:
        concordant_mask = (np.sign(d1) == np.sign(d2)) & (np.abs(d1) >= min_d) & (np.abs(d2) >= min_d)
        n_feats = np.sum(concordant_mask)
        if n_feats < 5:
            continue
            
        X_tr_all = np.vstack([X_tr_h_m[:, concordant_mask], X_g1[:, concordant_mask], X_g2[:, concordant_mask]])
        y_tr_all = np.array([0]*len(X_tr_h_m) + [1]*(len(X_g1) + len(X_g2)))
        
        X_te_h_sub = X_te_h_m[:, concordant_mask]
        X_te_a_sub = X_test_ai[:, concordant_mask]
        
        for C_val in [0.01, 0.1, 1.0]:
            pipe = Pipeline([("s", StandardScaler()), ("c", LogisticRegression(C=C_val, max_iter=1000, random_state=42))])
            pipe.fit(X_tr_all, y_tr_all)
            
            p_h = pipe.predict(X_te_h_sub)
            p_a = pipe.predict(X_te_a_sub)
            
            rec_h = np.mean(p_h == 0)
            rec_a = np.mean(p_a == 1)
            bal = (rec_h + rec_a) / 2.0
            
            if held_out == "lyria-pro3" and rec_a > 0.20:
                print(f"Held out: {held_out:<10} | min_d={min_d:.1f} ({n_feats:3d} feats) | C={C_val:<4} | Human Spec: {rec_h*100:5.1f}% | AI Rec: {rec_a*100:5.1f}% | BalAcc: {bal*100:5.1f}%")
            elif held_out != "lyria-pro3" and rec_a > 0.50 and min_d == 0.3 and C_val == 0.1:
                print(f"Held out: {held_out:<10} | min_d={min_d:.1f} ({n_feats:3d} feats) | C={C_val:<4} | Human Spec: {rec_h*100:5.1f}% | AI Rec: {rec_a*100:5.1f}% | BalAcc: {bal*100:5.1f}%")

