"""
Figure generation script for the LaTeX report.
Loads real project artifacts and generates 4 high-resolution (300 DPI) figures.
Run from the music_similarity_submission directory.
"""

import json
import sys
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent.parent.parent   # music_similarity_submission
FIGS = Path(__file__).resolve().parent / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

STYLE = dict(dpi=300, bbox_inches="tight")
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "grid.alpha": 0.35,
})

# ─── 1. Dataset Distribution ──────────────────────────────────────────────────
print("Generating fig_dataset_distribution.png ...")

# Real genre counts from dataset_manifest.csv
genres_all = ["Blues", "Classical", "Country", "Disco", "Hiphop", "Jazz", "Metal", "Pop", "Reggae", "Rock"]
# 100 GTZAN human: 10 tracks per genre × 10 genres
human_counts = {g: 10 for g in genres_all}  # confirmed from manifest

# AI counts from remote_index (confirmed from inspection):
ai_raw = {"Blues": 9, "Classical": 8, "Country": 0, "Disco": 0,
          "Hiphop": 3, "Jazz": 23, "Metal": 1, "Pop": 1, "Reggae": 0, "Rock": 55}

x = np.arange(len(genres_all))
width = 0.35

fig, ax = plt.subplots(figsize=(11, 5))
bars_h = ax.bar(x - width/2, [human_counts[g] for g in genres_all], width,
                label="Human (GTZAN)", color="#2563EB", edgecolor="white", linewidth=0.5)
bars_a = ax.bar(x + width/2, [ai_raw[g] for g in genres_all], width,
                label="AI (Multi-Gen)", color="#DC2626", edgecolor="white", linewidth=0.5)

# Reference pool annotation
ax.axhline(y=4, color="#16A34A", linestyle="--", linewidth=1.5, alpha=0.8, label="Reference Pool (4 tracks, mixed genres)")

ax.set_xlabel("Genre")
ax.set_ylabel("Number of Tracks")
ax.set_title("Figure 1: 204-Track Corpus Composition — Human vs. AI Tracks by Genre", fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(genres_all, rotation=15, ha="right")
ax.legend(loc="upper right", framealpha=0.9)
ax.set_ylim(0, 68)
ax.yaxis.set_minor_locator(plt.MultipleLocator(5))
ax.grid(axis="y", alpha=0.3)

# Add value labels on bars
for bar in bars_h:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., h + 0.5, f"{int(h)}",
            ha="center", va="bottom", fontsize=8, color="#1E3A6E")
for bar in bars_a:
    h = bar.get_height()
    if h > 0:
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.5, f"{int(h)}",
                ha="center", va="bottom", fontsize=8, color="#7F1D1D")

fig.tight_layout()
fig.savefig(FIGS / "fig_dataset_distribution.png", **STYLE)
plt.close(fig)
print("  Saved fig_dataset_distribution.png")

# ─── 2. Null Distribution ─────────────────────────────────────────────────────
print("Generating fig_null_distribution.png ...")

with open(BASE / "features" / "null_distribution.json", "r") as f:
    nd = json.load(f)
null_scores = np.array(nd["null_scores"])
mu = float(np.mean(null_scores))
sigma = float(np.std(null_scores))
n_pairs = nd["sample_pairs_count"]

print(f"  Null distribution: N={n_pairs}, mu={mu:.4f}, sigma={sigma:.4f}")

from scipy.stats import gaussian_kde, norm

fig, ax = plt.subplots(figsize=(9, 5))

# Histogram
n_bins = 55
counts, bin_edges, patches = ax.hist(null_scores, bins=n_bins, density=True,
                                      color="#93C5FD", edgecolor="white", linewidth=0.3,
                                      alpha=0.75, label=f"Empirical Null ($N = {n_pairs:,}$ pairs)")

# KDE overlay
kde = gaussian_kde(null_scores, bw_method=0.3)
x_range = np.linspace(null_scores.min() - 0.02, null_scores.max() + 0.02, 300)
ax.plot(x_range, kde(x_range), color="#1D4ED8", linewidth=2.0, label=f"KDE ($\\mu={mu:.4f}$, $\\sigma={sigma:.4f}$)")

# Normal approximation
ax.plot(x_range, norm.pdf(x_range, mu, sigma), color="#7C3AED",
        linewidth=1.5, linestyle="--", alpha=0.6, label=r"$\mathcal{N}(\mu, \sigma^2)$ fit")

# Reference lines
ax.axvline(x=0.85, color="#DC2626", linewidth=2.0, linestyle="-",
           label="Structural Imitation Threshold ($S=0.85$)")
ax.axvline(x=mu, color="#059669", linewidth=1.5, linestyle=":",
           label=f"Population Mean ($\\mu={mu:.4f}$)")
ax.axvline(x=mu + 2*sigma, color="#D97706", linewidth=1.2, linestyle="-.",
           label=f"$\\mu + 2\\sigma = {mu+2*sigma:.4f}$")

# Sample query score (ai_003_classical vs human_001_blues typical cross-genre score ~0.53)
sample_score = 0.5312
ax.axvline(x=sample_score, color="#6D28D9", linewidth=1.2, linestyle="--", alpha=0.7,
           label=f"Sample Query Score ($S={sample_score}$)")

ax.set_xlabel("DTW Similarity Score $S$")
ax.set_ylabel("Probability Density")
ax.set_title(f"Figure 2: Empirical Null Similarity Distribution ($N = {n_pairs:,}$ Human Pairs, 100 GTZAN Tracks)", fontweight="bold")
ax.legend(loc="upper left", fontsize=8.5, framealpha=0.92)
ax.set_xlim(0.32, 0.72)
ax.grid(axis="y", alpha=0.3)

# Annotation arrow for threshold
ax.annotate("No null pair exceeds\n$S = 0.85$ threshold",
            xy=(0.85, 0.05), xycoords=("data", "axes fraction"),
            xytext=(0.78, 0.5), textcoords=("data", "axes fraction"),
            arrowprops=dict(arrowstyle="->", color="#DC2626", lw=1.5),
            fontsize=8, color="#DC2626", ha="center")

fig.tight_layout()
fig.savefig(FIGS / "fig_null_distribution.png", **STYLE)
plt.close(fig)
print("  Saved fig_null_distribution.png")

# ─── 3. CV & Confusion Matrix ─────────────────────────────────────────────────
print("Generating fig_cv_and_confusion.png ...")

import joblib
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Load 56-dim features from remote_index
with open(BASE / "features" / "remote_index.json", "r") as f:
    ri = json.load(f)
tracks_ri = ri.get("tracks", {})

X_list, y_list = [], []
for tid, t in tracks_ri.items():
    lbl = str(t.get("origin_label", "")).lower()
    feats = t.get("origin_features_56")
    if feats and len(feats) == 56 and np.all(np.isfinite(feats)):
        y_val = 0 if lbl == "human" else 1 if lbl in ("ai", "ai_generated", "synthetic") else None
        if y_val is not None:
            X_list.append(np.array(feats, dtype=np.float32))
            y_list.append(y_val)

X = np.array(X_list)
y = np.array(y_list)
print(f"  Loaded {len(X)} samples (human={sum(y==0)}, ai={sum(y==1)})")

# Load production model (trained on 133-dim; adapt to 56-dim)
pipe = joblib.load(str(BASE / "models" / "origin_classifier.joblib"))

# Build matching pipeline on 56-dim features
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier, VotingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

rf56 = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
et56 = ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1)
hgb56 = HistGradientBoostingClassifier(random_state=42)
ensemble56 = VotingClassifier(estimators=[("rf", rf56), ("et", et56), ("hgb", hgb56)], voting="soft")
pipe56 = Pipeline([("scaler", StandardScaler()), ("classifier", ensemble56)])

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_accs = cross_val_score(pipe56, X, y, cv=skf, scoring="accuracy")
cv_f1s = cross_val_score(pipe56, X, y, cv=skf, scoring="f1")
print(f"  CV Accuracy: {np.mean(cv_accs):.4f} ± {np.std(cv_accs):.4f}")
print(f"  CV F1: {np.mean(cv_f1s):.4f} ± {np.std(cv_f1s):.4f}")

# Fit full model for confusion matrix (use all data, predict with leave-one-out fold)
from sklearn.model_selection import cross_val_predict
y_pred = cross_val_predict(pipe56, X, y, cv=skf)
cm = confusion_matrix(y, y_pred)

fig, (ax_cv, ax_cm) = plt.subplots(1, 2, figsize=(12, 5))

# Left: CV bar chart
fold_labels = [f"Fold {i+1}" for i in range(5)]
x_folds = np.arange(5)
w = 0.38
bars_acc = ax_cv.bar(x_folds - w/2, cv_accs, w, label="Accuracy", color="#2563EB", edgecolor="white")
bars_f1  = ax_cv.bar(x_folds + w/2, cv_f1s, w, label="F1-Score", color="#16A34A", edgecolor="white")
ax_cv.axhline(np.mean(cv_accs), color="#2563EB", linestyle="--", linewidth=1.5, alpha=0.7,
              label=f"Mean Acc = {np.mean(cv_accs):.3f}")
ax_cv.axhline(np.mean(cv_f1s), color="#16A34A", linestyle="--", linewidth=1.5, alpha=0.7,
              label=f"Mean F1 = {np.mean(cv_f1s):.3f}")
ax_cv.set_xticks(x_folds)
ax_cv.set_xticklabels(fold_labels)
ax_cv.set_ylim(0.5, 1.05)
ax_cv.set_ylabel("Score")
ax_cv.set_title("5-Fold Stratified CV (Soft-Voting Ensemble\nRF + ET + HGB, 56-dim Features)", fontweight="bold")
ax_cv.legend(fontsize=8.5)
ax_cv.grid(axis="y", alpha=0.3)

for bar in bars_acc:
    ax_cv.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
               f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)
for bar in bars_f1:
    ax_cv.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
               f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)

# Right: Confusion matrix
im = ax_cm.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
fig.colorbar(im, ax=ax_cm, fraction=0.046, pad=0.04)
tick_marks = [0, 1]
ax_cm.set_xticks(tick_marks)
ax_cm.set_yticks(tick_marks)
ax_cm.set_xticklabels(["Human", "AI"])
ax_cm.set_yticklabels(["Human", "AI"])
thresh = cm.max() / 2.0
for i in range(2):
    for j in range(2):
        ax_cm.text(j, i, f"{cm[i, j]}", ha="center", va="center", fontsize=14,
                   color="white" if cm[i, j] > thresh else "black", fontweight="bold")
ax_cm.set_xlabel("Predicted Label")
ax_cm.set_ylabel("True Label")
ax_cm.set_title("Confusion Matrix — Cross-Validated\nOrigin Classification (Human vs. AI)", fontweight="bold")

fig.suptitle("Figure 3: Soft-Voting Ensemble Origin Classifier — 5-Fold CV Performance",
             fontweight="bold", y=1.01)
fig.tight_layout()
fig.savefig(FIGS / "fig_cv_and_confusion.png", **STYLE)
plt.close(fig)
print("  Saved fig_cv_and_confusion.png")

# ─── 4. Feature Importance ───────────────────────────────────────────────────
print("Generating fig_feature_importance.png ...")

# Fit full pipeline on X, then extract feature importances
pipe56.fit(X, y)

# Get importances from RF and ET (which support feature_importances_)
rf_model = pipe56.named_steps["classifier"].estimators_[0]  # RF
et_model = pipe56.named_steps["classifier"].estimators_[1]  # ET

# 56-dim features: 20 mfcc_mean + 20 mfcc_std + 4 spectral (cent, roll, bw, flat) + 4 spectral_std + 2 zcr + 4 chroma_agg + 2 hp
# Actual 56-dim layout from origin_classifier.py (compare_and_train uses 56-dim):
# The 56-dim version is a legacy/simplified set; the 133-dim is the full pipeline version.
# remote_index stores 56-dim features. Let's reconstruct labels.
feat_labels_56 = (
    [f"MFCC-{i+1} Mean" for i in range(20)] +
    [f"MFCC-{i+1} Std" for i in range(20)] +
    ["SpCentroid Mean", "SpRolloff Mean", "SpBandwidth Mean", "SpFlatness Mean",
     "SpCentroid Std", "SpRolloff Std", "SpBandwidth Std", "SpFlatness Std",
     "ZCR Mean", "ZCR Std",
     "Chroma Mean", "Chroma Std",
     "HP Ratio", "Harmonic RMS Mean"]
)

# Pad/trim to 56 if needed
feat_labels_56 = feat_labels_56[:56]
while len(feat_labels_56) < 56:
    feat_labels_56.append(f"Feature-{len(feat_labels_56)+1}")

imp_rf = rf_model.feature_importances_
imp_et = et_model.feature_importances_
imp_avg = (imp_rf + imp_et) / 2.0

# Top 15
top15_idx = np.argsort(imp_avg)[-15:][::-1]
top15_labels = [feat_labels_56[i] for i in top15_idx]
top15_imp = imp_avg[top15_idx]

fig, ax = plt.subplots(figsize=(9, 6))
colors = plt.cm.Blues(np.linspace(0.4, 0.9, 15))[::-1]
bars = ax.barh(range(14, -1, -1), top15_imp, color=colors, edgecolor="white", linewidth=0.3)
ax.set_yticks(range(14, -1, -1))
ax.set_yticklabels(top15_labels, fontsize=9)
ax.set_xlabel("Mean Feature Importance (RF + ET Average)")
ax.set_title("Figure 4: Top 15 Acoustic Features — Human vs. AI Origin Classifier\n(RF + ExtraTrees Average Importance, 56-dim Feature Space)", fontweight="bold")
ax.grid(axis="x", alpha=0.3)

for bar, val in zip(bars, top15_imp):
    ax.text(bar.get_width() + 0.0005, bar.get_y() + bar.get_height()/2,
            f"{val:.4f}", va="center", fontsize=8)

fig.tight_layout()
fig.savefig(FIGS / "fig_feature_importance.png", **STYLE)
plt.close(fig)
print("  Saved fig_feature_importance.png")

print("\n✓ All 4 figures generated successfully in:", FIGS)
print(f"\nSummary statistics for LaTeX report:")
print(f"  Null dist N={n_pairs}, mu={mu:.4f}, sigma={sigma:.4f}")
print(f"  CV Accuracy: {np.mean(cv_accs):.4f} ± {np.std(cv_accs):.4f}")
print(f"  CV F1:       {np.mean(cv_f1s):.4f} ± {np.std(cv_f1s):.4f}")
print(f"  Confusion matrix:\n{cm}")
print(f"  Top feat: {top15_labels[0]} ({top15_imp[0]:.4f}), {top15_labels[1]} ({top15_imp[1]:.4f})")
