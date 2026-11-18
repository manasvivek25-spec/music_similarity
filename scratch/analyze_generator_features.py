import json
from pathlib import Path
import numpy as np
from scipy.stats import ttest_ind, mannwhitneyu

REPO_ROOT = Path(".").resolve()
with open(REPO_ROOT / "features/remote_index.json", "r", encoding="utf-8") as f:
    tracks = json.load(f)["tracks"]

mert_dir = REPO_ROOT / "features/mert"

human = []
acestep = []
musicgen = []
lyria = []

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
    item = {"tid": tid, "feat_56": f_arr, "mert": m_vec}
    cat = str(t.get("origin_label", "")).lower()
    gen = t.get("generator")
    if cat in ("human", "0", "false"):
        human.append(item)
    elif gen == "acestep":
        acestep.append(item)
    elif gen == "musicgen":
        musicgen.append(item)
    elif gen == "lyria-pro3":
        lyria.append(item)

# Feature names in 56-dim vector:
# 0-12: mfcc_mean (13)
# 13-25: mfcc_std (13)
# 26: cent_mean, 27: cent_std
# 28: rolloff_mean, 29: rolloff_std
# 30: zcr_mean, 31: zcr_std
# 32-43: chroma_mean (12)
# 44-55: chroma_std (12)
feat_names_56 = (
    [f"mfcc_mean_{i}" for i in range(13)] +
    [f"mfcc_std_{i}" for i in range(13)] +
    ["cent_mean", "cent_std", "rolloff_mean", "rolloff_std", "zcr_mean", "zcr_std"] +
    [f"chroma_mean_{i}" for i in range(12)] +
    [f"chroma_std_{i}" for i in range(12)]
)

H_56 = np.array([r["feat_56"] for r in human])
A_56 = np.array([r["feat_56"] for r in acestep])
M_56 = np.array([r["feat_56"] for r in musicgen])
L_56 = np.array([r["feat_56"] for r in lyria])

print(f"Sample sizes: Human={len(H_56)}, ACE={len(A_56)}, MusicGen={len(M_56)}, Lyria={len(L_56)}")

# Compute Cohen's d effect size between Human and each generator
# d = (mean_ai - mean_human) / pooled_std
def cohens_d(x_ai, x_human):
    n_a, n_h = len(x_ai), len(x_human)
    var_a, var_h = np.var(x_ai, ddof=1, axis=0), np.var(x_human, ddof=1, axis=0)
    pooled_sd = np.sqrt(((n_a - 1) * var_a + (n_h - 1) * var_h) / (n_a + n_h - 2) + 1e-9)
    return (np.mean(x_ai, axis=0) - np.mean(x_human, axis=0)) / pooled_sd

d_ace = cohens_d(A_56, H_56)
d_mgen = cohens_d(M_56, H_56)
d_lyria = cohens_d(L_56, H_56)

print("\n--- Acoustic 56-dim Effect Sizes (Cohen's d vs Human) ---")
print(f"{'Feature':<16} | {'d(ACE)':<8} | {'d(MGen)':<8} | {'d(Lyria)':<8} | {'Consistent Sign?':<16}")
print("-" * 65)

consistent_feats = []
for i in range(56):
    s_ace = np.sign(d_ace[i])
    s_mgen = np.sign(d_mgen[i])
    s_lyria = np.sign(d_lyria[i])
    is_consistent = (s_ace == s_mgen == s_lyria) and (abs(d_ace[i]) > 0.2 and abs(d_mgen[i]) > 0.2 and abs(d_lyria[i]) > 0.2)
    if is_consistent:
        consistent_feats.append(i)
    if abs(d_lyria[i]) > 0.3 or is_consistent:
        print(f"{feat_names_56[i]:<16} | {d_ace[i]:+8.2f} | {d_mgen[i]:+8.2f} | {d_lyria[i]:+8.2f} | {'YES' if is_consistent else 'no'}")

print(f"\nTotal acoustic features with consistent effect across ALL 3 generators: {len(consistent_feats)} of 56")
print("Consistent features:", [feat_names_56[i] for i in consistent_feats])

# Now let's do the same for MERT (768 dimensions)
H_m = np.array([r["mert"] for r in human if r["mert"] is not None])
A_m = np.array([r["mert"] for r in acestep if r["mert"] is not None])
M_m = np.array([r["mert"] for r in musicgen if r["mert"] is not None])
L_m = np.array([r["mert"] for r in lyria if r["mert"] is not None])

d_m_ace = cohens_d(A_m, H_m)
d_m_mgen = cohens_d(M_m, H_m)
d_m_lyria = cohens_d(L_m, H_m)

consistent_mert = []
for i in range(768):
    s1, s2, s3 = np.sign(d_m_ace[i]), np.sign(d_m_mgen[i]), np.sign(d_m_lyria[i])
    if (s1 == s2 == s3) and (abs(d_m_ace[i]) > 0.3 and abs(d_m_mgen[i]) > 0.3 and abs(d_m_lyria[i]) > 0.3):
        consistent_mert.append(i)

print(f"\nTotal MERT dimensions with consistent effect across ALL 3 generators (|d| > 0.3): {len(consistent_mert)} of 768")
if consistent_mert:
    top_consistent_mert = sorted(consistent_mert, key=lambda i: min(abs(d_m_ace[i]), abs(d_m_mgen[i]), abs(d_m_lyria[i])), reverse=True)[:10]
    for idx in top_consistent_mert:
        print(f"  MERT dim {idx:3d}: d(ACE)={d_m_ace[idx]:+6.2f}, d(MGen)={d_m_mgen[idx]:+6.2f}, d(Lyria)={d_m_lyria[idx]:+6.2f}")

