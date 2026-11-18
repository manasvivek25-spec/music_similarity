import json
from pathlib import Path
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler, normalize
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score

REPO_ROOT = Path(".").resolve()
with open(REPO_ROOT / "features/remote_index.json", "r", encoding="utf-8") as f:
    tracks = json.load(f)["tracks"]

mert_dir = REPO_ROOT / "features/mert"

human_clean = []
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
        human_clean.append(item)
    elif gen in ai_clean:
        ai_clean[gen].append(item)

print(f"Human: {len(human_clean)}, ACE-Step: {len(ai_clean['acestep'])}, Lyria-Pro3: {len(ai_clean['lyria-pro3'])}, MusicGen: {len(ai_clean['musicgen'])}")

# Analyze centroids and pairwise cosine distances between generators and human
def get_matrix(items, feat_key):
    return np.array([r[feat_key] for r in items if r[feat_key] is not None])

H_m = get_matrix(human_clean, "mert")
A_m = get_matrix(ai_clean["acestep"], "mert")
L_m = get_matrix(ai_clean["lyria-pro3"], "mert")
M_m = get_matrix(ai_clean["musicgen"], "mert")

# Normalize
H_norm = normalize(H_m)
A_norm = normalize(A_m)
L_norm = normalize(L_m)
M_norm = normalize(M_m)

H_center = np.mean(H_norm, axis=0)
A_center = np.mean(A_norm, axis=0)
L_center = np.mean(L_norm, axis=0)
M_center = np.mean(M_norm, axis=0)

def cos_sim(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

print("\n--- MERT Centroid Cosine Similarities ---")
print(f"Human vs ACE-Step:  {cos_sim(H_center, A_center):.4f}")
print(f"Human vs MusicGen:  {cos_sim(H_center, M_center):.4f}")
print(f"Human vs Lyria:     {cos_sim(H_center, L_center):.4f}")
print(f"ACE-Step vs MusicGen:{cos_sim(A_center, M_center):.4f}")
print(f"ACE-Step vs Lyria:   {cos_sim(A_center, L_center):.4f}")
print(f"MusicGen vs Lyria:   {cos_sim(M_center, L_center):.4f}")

# Now let's test a baseline classifier on Lyria-Pro3 held out and inspect decision function values
train_ai = ai_clean["acestep"] + ai_clean["musicgen"]
test_ai = ai_clean["lyria-pro3"]

# 75% human for train, 25% for test (fixed seed)
np.random.seed(42)
idx_h = np.random.permutation(len(human_clean))
n_tr = int(len(human_clean) * 0.75)
tr_h = [human_clean[i] for i in idx_h[:n_tr]]
te_h = [human_clean[i] for i in idx_h[n_tr:]]

print(f"\nTrain: {len(tr_h)} Human, {len(train_ai)} AI (ACE+MusicGen). Test: {len(te_h)} Human, {len(test_ai)} Lyria.")

# Compare decision values of Lyria vs Human test vs Train AI
clf = Pipeline([("scaler", StandardScaler()), ("svc", LinearSVC(C=0.01, random_state=42))])
X_tr = np.vstack([r["mert"] for r in tr_h] + [r["mert"] for r in train_ai])
y_tr = np.array([0]*len(tr_h) + [1]*len(train_ai))
clf.fit(X_tr, y_tr)

df_h_test = clf.decision_function(np.vstack([r["mert"] for r in te_h]))
df_lyria = clf.decision_function(np.vstack([r["mert"] for r in test_ai]))
df_acestep = clf.decision_function(np.vstack([r["mert"] for r in ai_clean["acestep"]]))
df_musicgen = clf.decision_function(np.vstack([r["mert"] for r in ai_clean["musicgen"]]))

print("\n--- LinearSVC Decision Function Distributions (C=0.01) ---")
print(f"Human test (target <= 0): mean = {np.mean(df_h_test):.3f}, std = {np.std(df_h_test):.3f}, range = [{np.min(df_h_test):.3f}, {np.max(df_h_test):.3f}]")
print(f"ACE-Step   (target > 0):  mean = {np.mean(df_acestep):.3f}, std = {np.std(df_acestep):.3f}, range = [{np.min(df_acestep):.3f}, {np.max(df_acestep):.3f}]")
print(f"MusicGen   (target > 0):  mean = {np.mean(df_musicgen):.3f}, std = {np.std(df_musicgen):.3f}, range = [{np.min(df_musicgen):.3f}, {np.max(df_musicgen):.3f}]")
print(f"Lyria-Pro3 (target > 0):  mean = {np.mean(df_lyria):.3f}, std = {np.std(df_lyria):.3f}, range = [{np.min(df_lyria):.3f}, {np.max(df_lyria):.3f}]")
print(f"Lyria tracks with df > 0: {sum(df_lyria > 0)} / {len(df_lyria)} ({np.mean(df_lyria > 0)*100:.1f}%)")
print(f"Lyria tracks with df > -0.5: {sum(df_lyria > -0.5)} / {len(df_lyria)} ({np.mean(df_lyria > -0.5)*100:.1f}%)")

