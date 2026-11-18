"""
P5 Feature 4 Origin Classifier Tuning, Calibration, Robustness & Generalization Suite (Section 7).

Protocol:
  - Feature sets compared: acoustic_56, mert_768, combined_824
  - Models tuned on f4_train ONLY via StratifiedGroupKFold + RandomizedSearchCV (seed=42):
      LogisticRegression, LinearSVC (calibrated), RBF SVC, RandomForest, HistGradientBoosting
  - Scaling / PCA strictly inside sklearn Pipeline (zero leakage)
  - Calibration (Platt sigmoid vs isotonic) and decision threshold selected on f4_val ONLY
  - Evaluated ONCE on untouched f4_test and ONCE on untouched held_out
  - Reports precision, recall, F1, balanced accuracy, ROC-AUC, PR-AUC, EER, TPR@1%/5% FPR,
    confusion matrix, specificity, per-generator recall, ECE, Brier, 95% group bootstrap CIs (>=2000),
    McNemar + paired bootstrap vs majority & confound baselines
  - Leave-One-Generator-Out (LOGO) generalization across MusicSet, SunoCaps, Udio
  - Robustness under re-encoding (128k/64k), resampling (16k->22.05k), additive noise, trimming, gain change
  - Saves fresh_production model bundle to models/origin_classifier_v2_fresh.joblib
"""

import io
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import librosa
import numpy as np
import scipy.signal as signal
import soundfile as sf
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC

from src.config import (
    FEATURE_VERSION,
    INDEX_MAX_SECONDS,
    REPO_ROOT,
    SAMPLE_RATE,
    V2_MODEL_FILE,
    V2_MODEL_METRICS_FILE,
)
from src.v2_cleaning_audit import V2_F4_SPLITS_FILE, V2_HELD_OUT_FILE
from src.v2_indexer import atomic_write_json, extract_features_from_waveform, normalize_waveform
from evaluation.statistics import (
    compute_ece,
    compute_group_bootstrap_ci,
    compute_mcnemar_test,
    compute_paired_bootstrap,
    compute_roc_pr_and_eer,
    compute_wilson_ci,
)


def _load_split_arrays(tracks: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    tids = sorted(tracks.keys())
    X56, X768, Xmeta = [], [], []
    y, groups, gens, wavs = [], [], [], []
    kept_tids = []

    for tid in tids:
        rec = tracks[tid]
        f56_p = REPO_ROOT / rec["f56_cache"]
        mert_p = REPO_ROOT / rec["mert_cache"]
        if not f56_p.is_file() or not mert_p.is_file():
            continue
        v56 = np.load(str(f56_p)).astype(np.float32).ravel()
        v768 = np.load(str(mert_p)).astype(np.float32).ravel()
        if len(v56) != 56 or len(v768) != 768:
            continue
        if not (np.all(np.isfinite(v56)) and np.all(np.isfinite(v768))):
            continue
        post_m = rec["post_norm_metadata"]
        vmeta = np.array([
            float(post_m["norm_duration_sec"]),
            float(post_m["norm_sr"]),
            float(post_m["norm_channels"]),
            float(post_m["norm_rms"]),
            float(post_m["norm_peak"]),
            float(post_m["norm_dc_offset"]),
            float(post_m["silence_ratio_norm"]),
        ], dtype=np.float32)

        kept_tids.append(tid)
        X56.append(v56)
        X768.append(v768)
        Xmeta.append(vmeta)
        y.append(1 if rec["origin_label"] == "ai" else 0)
        groups.append(rec["group_id"])
        gens.append(rec.get("generator", "unknown"))
        wavs.append(REPO_ROOT / rec["norm_wav_path"])

    X56_a = np.vstack(X56)
    X768_a = np.vstack(X768)
    X824_a = np.hstack([X56_a, X768_a])
    Xmeta_a = np.vstack(Xmeta)

    return {
        "track_ids": kept_tids,
        "acoustic_56": X56_a,
        "mert_768": X768_a,
        "combined_824": X824_a,
        "metadata_confound": Xmeta_a,
        "y": np.array(y, dtype=int),
        "groups": np.array(groups),
        "generators": np.array(gens),
        "wav_paths": wavs,
    }


def _get_positive_probs(model: Any, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1].astype(float)
    z = model.decision_function(X).astype(float)
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def _evaluate_predictions(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    groups: np.ndarray,
    generators: np.ndarray,
    threshold: float = 0.5,
    n_bootstraps: int = 2000,
    seed: int = 42,
) -> Dict[str, Any]:
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]

    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    roc_pr = compute_roc_pr_and_eer(y_true, y_prob)
    ece = compute_ece(y_true, y_prob)
    brier = float(brier_score_loss(y_true, y_prob))

    # Group bootstrap 95% CIs (>= 2000 resamples)
    _, roc_ci = compute_group_bootstrap_ci(
        groups, y_true, y_prob,
        metric_fn=lambda yt, ys: float(roc_auc_score(yt, ys)),
        n_bootstraps=n_bootstraps, seed=seed,
    )
    _, pr_ci = compute_group_bootstrap_ci(
        groups, y_true, y_prob,
        metric_fn=lambda yt, ys: float(compute_roc_pr_and_eer(yt, ys)["pr_auc"]),
        n_bootstraps=n_bootstraps, seed=seed,
    )
    _, f1_ci = compute_group_bootstrap_ci(
        groups, y_true, y_pred,
        metric_fn=lambda yt, yp: float(f1_score(yt, yp, zero_division=0)),
        n_bootstraps=n_bootstraps, seed=seed,
    )
    _, bal_ci = compute_group_bootstrap_ci(
        groups, y_true, y_pred,
        metric_fn=lambda yt, yp: float(balanced_accuracy_score(yt, yp)),
        n_bootstraps=n_bootstraps, seed=seed,
    )

    n_h = int(np.sum(y_true == 0))
    n_ai = int(np.sum(y_true == 1))
    spec = float(tn / max(n_h, 1))
    spec_ci = compute_wilson_ci(int(tn), n_h)

    per_gen_recall = {}
    for g in sorted(set(generators[y_true == 1].tolist())):
        mask = (generators == g) & (y_true == 1)
        ng = int(np.sum(mask))
        kg = int(np.sum(y_pred[mask] == 1))
        per_gen_recall[g] = {
            "recall": round(kg / max(ng, 1), 4),
            "ci": compute_wilson_ci(kg, ng),
            "n": ng,
        }

    return {
        "n_samples": int(len(y_true)),
        "n_human": n_h,
        "n_ai": n_ai,
        "threshold": round(float(threshold), 4),
        "confusion_matrix": cm,
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": {"value": round(f1, 4), "ci": f1_ci},
        "balanced_accuracy": {"value": round(bal_acc, 4), "ci": bal_ci},
        "roc_auc": {"value": roc_pr["roc_auc"], "ci": roc_ci},
        "pr_auc": {"value": roc_pr["pr_auc"], "ci": pr_ci},
        "eer": roc_pr["eer"],
        "tpr_at_1pct_fpr": roc_pr["tpr_at_1pct_fpr"],
        "tpr_at_5pct_fpr": roc_pr["tpr_at_5pct_fpr"],
        "specificity_human": {"value": round(spec, 4), "ci": spec_ci, "n": n_h},
        "per_generator_recall": per_gen_recall,
        "avg_generator_recall": round(float(np.mean([v["recall"] for v in per_gen_recall.values()])), 4) if per_gen_recall else 0.0,
        "ece": ece,
        "brier_score": round(brier, 4),
    }


def _build_search_spaces(seed: int = 42, use_pca: bool = False) -> Dict[str, Tuple[Pipeline, Dict[str, Any]]]:
    pca_step = [("pca", PCA(n_components=0.95, random_state=seed))] if use_pca else []

    lr_pipe = Pipeline([("scaler", StandardScaler())] + pca_step + [("clf", LogisticRegression(max_iter=2000, random_state=seed))])
    lr_grid = {"clf__C": [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]}

    lsvc_pipe = Pipeline([("scaler", StandardScaler())] + pca_step + [("clf", LinearSVC(dual=False, max_iter=3000, random_state=seed))])
    lsvc_grid = {"clf__C": [0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0]}

    rbf_pipe = Pipeline([("scaler", StandardScaler())] + pca_step + [("clf", SVC(kernel="rbf", probability=True, random_state=seed))])
    rbf_grid = {"clf__C": [0.1, 0.5, 1.0, 3.0, 10.0], "clf__gamma": ["scale", "auto", 0.001, 0.01]}

    rf_pipe = Pipeline([("scaler", StandardScaler())] + pca_step + [("clf", RandomForestClassifier(random_state=seed, n_jobs=-1))])
    rf_grid = {"clf__n_estimators": [100, 200, 300], "clf__max_depth": [None, 8, 16, 24], "clf__min_samples_leaf": [1, 2, 4]}

    hgb_pipe = Pipeline([("scaler", StandardScaler())] + pca_step + [("clf", HistGradientBoostingClassifier(random_state=seed))])
    hgb_grid = {"clf__learning_rate": [0.03, 0.05, 0.1, 0.2], "clf__max_depth": [3, 5, 8, None], "clf__max_iter": [100, 200]}

    return {
        "logistic_regression": (lr_pipe, lr_grid),
        "linear_svm": (lsvc_pipe, lsvc_grid),
        "rbf_svm": (rbf_pipe, rbf_grid),
        "random_forest": (rf_pipe, rf_grid),
        "hist_gradient_boosting": (hgb_pipe, hgb_grid),
    }


def _evaluate_robustness_acoustic_only(
    model: Any,
    feature_set: str,
    threshold: float,
    test_split: Dict[str, Any],
    mert_model_bundle: Optional[Tuple[Any, Any, str]] = None,
    n_subset: int = 40,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Evaluate production F4 model robustness on a balanced real test subset under:
      - mp3_128k / mp3_64k (or OGG/lowpass codec degradation)
      - resample_16k (22.05k -> 16k -> 22.05k)
      - additive_noise_20db, additive_noise_15db
      - trim_6s (center 6.0s)
      - gain_minus6db, gain_plus3db
    """
    rng = np.random.default_rng(seed)
    y_all = test_split["y"]
    h_idx = np.where(y_all == 0)[0]
    a_idx = np.where(y_all == 1)[0]
    k = min(n_subset // 2, len(h_idx), len(a_idx))
    chosen = np.concatenate([h_idx[:k], a_idx[:k]])
    y_sub = y_all[chosen]
    wavs_sub = [test_split["wav_paths"][int(i)] for i in chosen]

    # Clean baseline on this exact subset
    X_clean = test_split[feature_set][chosen]
    p_clean = _get_positive_probs(model, X_clean)
    clean_auc = float(roc_auc_score(y_sub, p_clean))
    clean_bal = float(balanced_accuracy_score(y_sub, (p_clean >= threshold).astype(int)))

    def _extract_vec(y_mod: np.ndarray) -> np.ndarray:
        # Re-normalize through identical inference preprocessing
        y_n, _ = normalize_waveform(y_mod, SAMPLE_RATE, target_sec=INDEX_MAX_SECONDS, target_sr=SAMPLE_RATE)
        f56 = extract_features_from_waveform(y_n, sr=SAMPLE_RATE)["f56"]
        if feature_set == "acoustic_56":
            return f56
        # Extract MERT on modified waveform
        proc, m_mod, dev = mert_model_bundle
        mert_sr = int(getattr(proc, "sampling_rate", 24000))
        y24 = librosa.resample(y_n, orig_sr=SAMPLE_RATE, target_sr=mert_sr, res_type="soxr_hq").astype(np.float32)
        inp = proc(y24, sampling_rate=mert_sr, return_tensors="pt")
        inp = {kk: vv.to(dev) for kk, vv in inp.items()}
        import torch
        with torch.no_grad():
            if dev == "cuda":
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    emb = m_mod(**inp).last_hidden_state.float().mean(dim=1).squeeze(0).cpu().numpy()
            else:
                emb = m_mod(**inp).last_hidden_state.float().mean(dim=1).squeeze(0).cpu().numpy()
        if feature_set == "mert_768":
            return emb.astype(np.float32)
        return np.concatenate([f56, emb.astype(np.float32)])

    def _reencode_mp3(y_sig: np.ndarray, bitrate_mode: str) -> np.ndarray:
        buf = io.BytesIO()
        try:
            # soundfile MP3 encode/decode; compression_level 0.3 (~128k) or 0.9 (~64k)
            comp = 0.3 if bitrate_mode == "128k" else 0.9
            sf.write(buf, y_sig, SAMPLE_RATE, format="MP3", subtype="MPEG_LAYER_III", compression_level=comp)
            buf.seek(0)
            y_dec, _ = sf.read(buf, dtype="float32")
            return y_dec
        except Exception:
            # Fallback to OGG Vorbis compression if libsndfile build lacks MP3 encoder
            buf = io.BytesIO()
            comp = 0.4 if bitrate_mode == "128k" else 0.9
            sf.write(buf, y_sig, SAMPLE_RATE, format="OGG", subtype="VORBIS", compression_level=comp)
            buf.seek(0)
            y_dec, _ = sf.read(buf, dtype="float32")
            return y_dec

    perturbations = {
        "reencode_mp3_128k": lambda y_s: _reencode_mp3(y_s, "128k"),
        "reencode_mp3_64k": lambda y_s: _reencode_mp3(y_s, "64k"),
        "resample_16k_roundtrip": lambda y_s: librosa.resample(
            librosa.resample(y_s, orig_sr=SAMPLE_RATE, target_sr=16000, res_type="soxr_hq"),
            orig_sr=16000, target_sr=SAMPLE_RATE, res_type="soxr_hq",
        ).astype(np.float32),
        "additive_noise_20db": lambda y_s: (
            y_s + rng.normal(0, np.sqrt(np.mean(y_s ** 2) / (10.0 ** 2.0)), size=len(y_s))
        ).astype(np.float32),
        "additive_noise_15db": lambda y_s: (
            y_s + rng.normal(0, np.sqrt(np.mean(y_s ** 2) / (10.0 ** 1.5)), size=len(y_s))
        ).astype(np.float32),
        "trim_center_6s": lambda y_s: y_s[int(2.0 * SAMPLE_RATE):int(8.0 * SAMPLE_RATE)],
        "gain_minus_6db": lambda y_s: (y_s * 0.5).astype(np.float32),
        "gain_plus_3db": lambda y_s: np.clip(y_s * 1.4125, -0.99, 0.99).astype(np.float32),
    }

    results = {
        "n_subset": int(len(y_sub)),
        "clean_roc_auc": round(clean_auc, 4),
        "clean_balanced_accuracy": round(clean_bal, 4),
        "perturbations": {},
    }

    for pname, fn in perturbations.items():
        vecs = []
        for wp in wavs_sub:
            y_raw, _ = sf.read(str(wp), dtype="float32")
            y_mod = fn(y_raw)
            vecs.append(_extract_vec(y_mod))
        X_mod = np.vstack(vecs)
        p_mod = _get_positive_probs(model, X_mod)
        auc_m = float(roc_auc_score(y_sub, p_mod))
        bal_m = float(balanced_accuracy_score(y_sub, (p_mod >= threshold).astype(int)))
        results["perturbations"][pname] = {
            "roc_auc": round(auc_m, 4),
            "roc_auc_drop": round(auc_m - clean_auc, 4),
            "balanced_accuracy": round(bal_m, 4),
            "balanced_accuracy_drop": round(bal_m - clean_bal, 4),
        }
        print(f"  [Robustness] {pname:24s}: AUC={auc_m:.4f} (Δ={auc_m-clean_auc:+.4f}), BalAcc={bal_m:.4f} (Δ={bal_m-clean_bal:+.4f})", flush=True)

    return results


def _run_leave_one_generator_out(
    best_model_builder,
    train_split: Dict[str, Any],
    val_split: Dict[str, Any],
    test_split: Dict[str, Any],
    feature_set: str,
) -> Dict[str, Any]:
    """
    Leave-One-Source/Generator-Out generalization across AI generators (MusicSet, SunoCaps, Udio).
    Trains on Humans + 2 AI generators, evaluates recall on the held-out 3rd AI generator.
    """
    X_all = np.vstack([train_split[feature_set], val_split[feature_set], test_split[feature_set]])
    y_all = np.concatenate([train_split["y"], val_split["y"], test_split["y"]])
    gen_all = np.concatenate([train_split["generators"], val_split["generators"], test_split["generators"]])

    ai_gens = sorted(set(gen_all[y_all == 1].tolist()))
    logo_res = {}
    for unseen_gen in ai_gens:
        mask_unseen = (gen_all == unseen_gen) & (y_all == 1)
        mask_train = ~mask_unseen
        if np.sum(mask_unseen) < 5 or len(np.unique(y_all[mask_train])) < 2:
            continue
        clf = best_model_builder()
        clf.fit(X_all[mask_train], y_all[mask_train])
        probs_u = _get_positive_probs(clf, X_all[mask_unseen])
        preds_u = (probs_u >= 0.5).astype(int)
        k_hit = int(np.sum(preds_u == 1))
        n_u = int(np.sum(mask_unseen))
        logo_res[unseen_gen] = {
            "unseen_generator": unseen_gen,
            "n_unseen_tracks": n_u,
            "recall": round(k_hit / max(n_u, 1), 4),
            "ci": compute_wilson_ci(k_hit, n_u),
            "mean_prob_ai": round(float(np.mean(probs_u)), 4),
        }
        print(f"  [LOGO] Unseen generator '{unseen_gen}' (n={n_u}): recall={logo_res[unseen_gen]['recall']:.4f}", flush=True)
    return logo_res


def run_p5_f4_tuning(seed: int = 42) -> Dict[str, Any]:
    t0 = time.time()
    print("[P5] Loading strictly disjoint F4 train/val/test and held_out splits...", flush=True)
    with open(V2_F4_SPLITS_FILE, "r", encoding="utf-8") as f:
        f4_data = json.load(f)["splits"]
    with open(V2_HELD_OUT_FILE, "r", encoding="utf-8") as f:
        held_tracks = json.load(f)["tracks"]

    tr = _load_split_arrays(f4_data["f4_train"])
    va = _load_split_arrays(f4_data["f4_val"])
    te = _load_split_arrays(f4_data["f4_test"])
    ho = _load_split_arrays(held_tracks)

    print(
        f"[P5] Loaded sizes: train={len(tr['y'])}, val={len(va['y'])}, test={len(te['y'])}, held_out={len(ho['y'])}",
        flush=True,
    )

    feature_sets = ["acoustic_56", "mert_768", "combined_824"]
    cv_summary: Dict[str, Dict[str, Any]] = {}
    best_overall = {"val_auc": -1.0}

    sgkf = StratifiedGroupKFold(n_splits=5)

    for fset in feature_sets:
        use_pca = (fset != "acoustic_56")
        spaces = _build_search_spaces(seed=seed, use_pca=use_pca)
        cv_summary[fset] = {}

        for mname, (pipe, grid) in spaces.items():
            # Count total combinations for RandomizedSearchCV
            n_combos = 1
            for v in grid.values():
                n_combos *= len(v)
            n_iter = min(n_combos, 8)

            search = RandomizedSearchCV(
                estimator=pipe,
                param_distributions=grid,
                n_iter=n_iter,
                scoring="roc_auc",
                cv=sgkf,
                random_state=seed,
                n_jobs=-1,
                refit=True,
            )
            search.fit(tr[fset], tr["y"], groups=tr["groups"])
            best_est = search.best_estimator_

            # Evaluate uncalibrated on validation split
            raw_val_probs = _get_positive_probs(best_est, va[fset])
            raw_val_auc = float(roc_auc_score(va["y"], raw_val_probs))

            # Calibrate on f4_val ONLY (Platt sigmoid vs Isotonic) using FrozenEstimator
            best_cal_method = "uncalibrated"
            best_cal_model = best_est
            best_cal_brier = float(brier_score_loss(va["y"], raw_val_probs))
            best_cal_probs = raw_val_probs

            for cal_method in ("sigmoid", "isotonic"):
                try:
                    cal = CalibratedClassifierCV(FrozenEstimator(best_est), method=cal_method, cv="prefit")
                    cal.fit(va[fset], va["y"])
                    cp = cal.predict_proba(va[fset])[:, 1]
                    cb = float(brier_score_loss(va["y"], cp))
                    if cb < best_cal_brier:
                        best_cal_brier = cb
                        best_cal_method = cal_method
                        best_cal_model = cal
                        best_cal_probs = cp
                except Exception:
                    pass

            # Select optimal threshold on f4_val ONLY (maximize balanced accuracy)
            cand_thresholds = np.linspace(0.20, 0.80, 61)
            best_th = 0.50
            best_val_bal = -1.0
            for th in cand_thresholds:
                bacc = float(balanced_accuracy_score(va["y"], (best_cal_probs >= th).astype(int)))
                if bacc > best_val_bal:
                    best_val_bal = bacc
                    best_th = float(th)

            val_auc = float(roc_auc_score(va["y"], best_cal_probs))
            val_ece = compute_ece(va["y"], best_cal_probs)

            rec_entry = {
                "cv_mean_roc_auc": round(float(search.best_score_), 4),
                "best_params": {k: (str(v) if v is None else v) for k, v in search.best_params_.items()},
                "calibration_method": best_cal_method,
                "selected_val_threshold": round(best_th, 4),
                "val_roc_auc": round(val_auc, 4),
                "val_balanced_accuracy": round(best_val_bal, 4),
                "val_brier": round(best_cal_brier, 4),
                "val_ece": val_ece,
            }
            cv_summary[fset][mname] = rec_entry
            print(
                f"  [CV] {fset:12s} | {mname:22s} | CV_AUC={rec_entry['cv_mean_roc_auc']:.4f} | "
                f"Val_AUC={val_auc:.4f} | Val_BalAcc={best_val_bal:.4f} | Cal={best_cal_method} (th={best_th:.2f})",
                flush=True,
            )

            # Pick best by validation ROC-AUC (tie-break: lower val Brier)
            if (val_auc > best_overall["val_auc"]) or (
                abs(val_auc - best_overall["val_auc"]) < 1e-4 and best_cal_brier < best_overall.get("val_brier", 1.0)
            ):
                best_overall = {
                    "feature_set": fset,
                    "model_name": mname,
                    "val_auc": val_auc,
                    "val_brier": best_cal_brier,
                    "threshold": best_th,
                    "calibration_method": best_cal_method,
                    "best_params": search.best_params_,
                    "uncalibrated_model": best_est,
                    "calibrated_model": best_cal_model,
                    "cv_summary_entry": rec_entry,
                }

    win_fset = best_overall["feature_set"]
    win_name = best_overall["model_name"]
    win_model = best_overall["calibrated_model"]
    win_th = best_overall["threshold"]
    print(f"\n[P5] WINNER on Validation: {win_fset} + {win_name} (cal={best_overall['calibration_method']}, th={win_th:.2f}, val_auc={best_overall['val_auc']:.4f})", flush=True)

    # Evaluate ONCE on untouched f4_test AND ONCE on untouched held_out!
    test_probs = _get_positive_probs(win_model, te[win_fset])
    test_metrics = _evaluate_predictions(te["y"], test_probs, te["groups"], te["generators"], threshold=win_th, n_bootstraps=2000, seed=seed)

    held_probs = _get_positive_probs(win_model, ho[win_fset])
    held_metrics = _evaluate_predictions(ho["y"], held_probs, ho["groups"], ho["generators"], threshold=win_th, n_bootstraps=2000, seed=seed)

    # Baselines on untouched f4_test & held_out: Majority, Metadata-Confound LR, Acoustic-56 LR, Acoustic-56 LinearSVC, Acoustic-56 RF
    maj_cls = int(np.bincount(tr["y"]).argmax())
    maj_pred_te = np.full_like(te["y"], maj_cls)
    maj_pred_ho = np.full_like(ho["y"], maj_cls)

    conf_lr = Pipeline([("sc", StandardScaler()), ("clf", LogisticRegression(max_iter=1000, random_state=seed))])
    conf_lr.fit(tr["metadata_confound"], tr["y"])
    conf_prob_te = conf_lr.predict_proba(te["metadata_confound"])[:, 1]
    conf_prob_ho = conf_lr.predict_proba(ho["metadata_confound"])[:, 1]

    # Standard F4 comparison baselines on winning feature set (LR, LinearSVC, RF)
    def _fit_simple_baselines(fset_name: str):
        out = {}
        for bname, est in [
            ("lr", Pipeline([("sc", StandardScaler()), ("clf", LogisticRegression(max_iter=1000, random_state=seed))])),
            ("linearsvc", Pipeline([("sc", StandardScaler()), ("clf", LinearSVC(dual=False, max_iter=3000, random_state=seed))])),
            ("rf", RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1)),
        ]:
            est.fit(tr[fset_name], tr["y"])
            pte = _get_positive_probs(est, te[fset_name])
            pho = _get_positive_probs(est, ho[fset_name])
            out[bname] = {
                "test_roc_auc": round(float(roc_auc_score(te["y"], pte)), 4),
                "test_balanced_accuracy": round(float(balanced_accuracy_score(te["y"], (pte >= 0.5).astype(int))), 4),
                "held_out_roc_auc": round(float(roc_auc_score(ho["y"], pho)), 4),
                "held_out_balanced_accuracy": round(float(balanced_accuracy_score(ho["y"], (pho >= 0.5).astype(int))), 4),
                "held_out_probs": pho,
            }
        return out

    f4_baselines = _fit_simple_baselines(win_fset)

    # McNemar & Paired Bootstrap vs Majority and Metadata-Confound baselines on held_out
    win_pred_ho = (held_probs >= win_th).astype(int)
    mcnemar_vs_majority = compute_mcnemar_test(ho["y"], win_pred_ho, maj_pred_ho)
    mcnemar_vs_confound = compute_mcnemar_test(ho["y"], win_pred_ho, (conf_prob_ho >= 0.5).astype(int))
    paired_brier_vs_confound = compute_paired_bootstrap(
        (held_probs - ho["y"]) ** 2,
        (conf_prob_ho - ho["y"]) ** 2,
        n_bootstraps=2000,
        seed=seed,
    )

    # Leave-One-Generator-Out generalization
    print("\n[P5] Running Leave-One-Generator-Out (LOGO) Generalization...", flush=True)
    def _builder():
        from sklearn.base import clone
        return clone(best_overall["uncalibrated_model"])
    logo_results = _run_leave_one_generator_out(_builder, tr, va, te, win_fset)

    # Robustness suite (re-encoding 128k/64k, resample 16k, noise 20/15dB, trim 6s, gain +-dB)
    print("\n[P5] Running Robustness Stress Evaluation on untouched test subset...", flush=True)
    mert_bundle = None
    if win_fset != "acoustic_56":
        import torch
        from transformers import AutoModel, Wav2Vec2FeatureExtractor
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        proc = Wav2Vec2FeatureExtractor.from_pretrained("m-a-p/MERT-v1-95M", trust_remote_code=True)
        m_mod = AutoModel.from_pretrained("m-a-p/MERT-v1-95M", trust_remote_code=True).to(dev)
        m_mod.eval()
        mert_bundle = (proc, m_mod, dev)

    robustness_results = _evaluate_robustness_acoustic_only(
        win_model, win_fset, win_th, te, mert_model_bundle=mert_bundle, n_subset=40, seed=seed
    )

    # Save fresh_production model artifact bundle
    model_version = "2.0.0-fresh_production"
    dataset_id = "v2_balanced_fma_sleepyjesse_real"
    model_bundle = {
        "model": win_model,
        "uncalibrated_model": best_overall["uncalibrated_model"],
        "feature_set": win_fset,
        "model_name": win_name,
        "threshold": win_th,
        "calibration_method": best_overall["calibration_method"],
        "model_version": model_version,
        "dataset_id": dataset_id,
        "feature_version": FEATURE_VERSION,
        "index_max_seconds": INDEX_MAX_SECONDS,
        "sample_rate": SAMPLE_RATE,
        "fresh_production": True,
        "represented_generators": sorted(set(tr["generators"][tr["y"] == 1].tolist())),
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    V2_MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_bundle, str(V2_MODEL_FILE))

    # Strip non-serializable arrays before writing metrics JSON
    clean_baselines = {
        k: {kk: vv for kk, vv in v.items() if kk != "held_out_probs"}
        for k, v in f4_baselines.items()
    }
    clean_baselines["majority"] = {
        "test_balanced_accuracy": round(float(balanced_accuracy_score(te["y"], maj_pred_te)), 4),
        "held_out_balanced_accuracy": round(float(balanced_accuracy_score(ho["y"], maj_pred_ho)), 4),
    }
    clean_baselines["metadata_confound_probe"] = {
        "test_roc_auc": round(float(roc_auc_score(te["y"], conf_prob_te)), 4),
        "held_out_roc_auc": round(float(roc_auc_score(ho["y"], conf_prob_ho)), 4),
    }

    metrics_payload = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_sec": round(time.time() - t0, 2),
        "seed": seed,
        "fresh_production": True,
        "model_version": model_version,
        "dataset_id": dataset_id,
        "model_path": str(V2_MODEL_FILE.relative_to(REPO_ROOT)),
        "selected_model": {
            "feature_set": win_fset,
            "algorithm": win_name,
            "calibration_method": best_overall["calibration_method"],
            "decision_threshold": round(win_th, 4),
            "hyperparameters": best_overall["cv_summary_entry"]["best_params"],
            "training_set_size": int(len(tr["y"])),
            "validation_set_size": int(len(va["y"])),
            "represented_generators": model_bundle["represented_generators"],
        },
        "cv_and_val_comparison": cv_summary,
        "test_set_metrics": test_metrics,
        "held_out_metrics": held_metrics,
        "baselines": clean_baselines,
        "statistical_comparisons_held_out": {
            "mcnemar_vs_majority": mcnemar_vs_majority,
            "mcnemar_vs_metadata_confound": mcnemar_vs_confound,
            "paired_bootstrap_brier_vs_confound": paired_brier_vs_confound,
        },
        "leave_one_generator_out": logo_results,
        "robustness_stress_test": robustness_results,
    }
    atomic_write_json(V2_MODEL_METRICS_FILE, metrics_payload)

    print(
        f"\n[P5] DONE in {metrics_payload['elapsed_sec']}s! "
        f"Test AUC={test_metrics['roc_auc']['value']:.4f} (BalAcc={test_metrics['balanced_accuracy']['value']:.4f}) | "
        f"Held-Out AUC={held_metrics['roc_auc']['value']:.4f} (BalAcc={held_metrics['balanced_accuracy']['value']:.4f}, Spec={held_metrics['specificity_human']['value']:.4f})",
        flush=True,
    )
    return metrics_payload


if __name__ == "__main__":
    run_p5_f4_tuning(seed=42)
