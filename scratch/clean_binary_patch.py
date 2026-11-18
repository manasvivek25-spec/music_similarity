from pathlib import Path

p = Path("src/origin_classifier.py")
text = p.read_text(encoding="utf-8")  # python normalizes to \n on read if newline not specified, so let's open in binary

raw = p.read_bytes()

target = b"""        "model_path": str(target_path),\r
        "model_version": MODEL_VERSION,\r
    }\r
\r
\r
def run_origin_classifier_suite("""

insertion = b"""        "model_path": str(target_path),\r
        "model_version": MODEL_VERSION,\r
    }\r
\r
\r
def validate_production_dataset_isolation(\r
    dataset_path,\r
    allow_legacy: bool = False\r
) -> bool:\r
    \"\"\"\r
    Ensure the production classifier training pipeline never accidentally trains\r
    on or leaks from the retired 261-record corpus.\r
    \"\"\"\r
    p = Path(dataset_path).resolve()\r
    if not allow_legacy:\r
        if "remote_index.json" in p.name or "261" in str(p):\r
            raise ValueError(\r
                "Legacy 261-record corpus is retired from production origin classifier optimization. "\r
                "Production training requires a fresh, unconfounded dataset portfolio."\r
            )\r
    return True\r
\r
\r
def build_group_disjoint_partitions(\r
    groups: np.ndarray,\r
    y: Optional[np.ndarray] = None,\r
    test_size: float = 0.2,\r
    val_size: float = 0.2,\r
    random_state: int = 42\r
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:\r
    \"\"\"\r
    Partition sample indices into Train, Validation, and Test subsets such that\r
    NO group (e.g. generator family, artist ID, recording session) appears in\r
    more than one partition. Guarantees zero group leakage. If labels `y` are\r
    provided, partitions are stratified across classes.\r
\r
    Returns\r
    -------\r
    train_idx : np.ndarray\r
    val_idx : np.ndarray\r
    test_idx : np.ndarray\r
    \"\"\"\r
    unique_groups = np.unique(groups)\r
    if len(unique_groups) < 3:\r
        raise ValueError(\r
            f"At least 3 distinct groups required for train/val/test disjoint partitioning, found {len(unique_groups)}"\r
        )\r
\r
    rng = np.random.RandomState(random_state)\r
\r
    if y is not None:\r
        group_labels = {g: int(np.round(np.mean(y[groups == g]))) for g in unique_groups}\r
        c0_groups = rng.permutation([g for g, lbl in group_labels.items() if lbl == 0])\r
        c1_groups = rng.permutation([g for g, lbl in group_labels.items() if lbl == 1])\r
\r
        def split_class_groups(cg):\r
            n = len(cg)\r
            if n <= 1:\r
                return set(cg), set(), set()\r
            nt = max(1, int(round(test_size * n))) if n >= 3 else 0\r
            nv = max(1, int(round(val_size * n))) if n >= 3 else (1 if n == 2 else 0)\r
            ntr = n - nt - nv\r
            if ntr < 1:\r
                ntr = 1\r
                if nv > 0:\r
                    nv -= 1\r
            return set(cg[nt + nv:]), set(cg[nt:nt + nv]), set(cg[:nt])\r
\r
        tr0, va0, te0 = split_class_groups(c0_groups)\r
        tr1, va1, te1 = split_class_groups(c1_groups)\r
\r
        train_groups = tr0 | tr1\r
        val_groups = va0 | va1\r
        test_groups = te0 | te1\r
\r
        if len(val_groups) == 0:\r
            val_groups = {list(train_groups)[0]}\r
            train_groups = train_groups - val_groups\r
        if len(test_groups) == 0:\r
            test_groups = {list(train_groups)[0]}\r
            train_groups = train_groups - test_groups\r
    else:\r
        shuffled_groups = rng.permutation(unique_groups)\r
        n_groups = len(shuffled_groups)\r
        n_test = max(1, int(round(test_size * n_groups)))\r
        n_val = max(1, int(round(val_size * n_groups)))\r
        n_train = n_groups - n_test - n_val\r
        if n_train < 1:\r
            n_train = 1\r
            if n_val > 1:\r
                n_val -= 1\r
            elif n_test > 1:\r
                n_test -= 1\r
\r
        test_groups = set(shuffled_groups[:n_test])\r
        val_groups = set(shuffled_groups[n_test : n_test + n_val])\r
        train_groups = set(shuffled_groups[n_test + n_val :])\r
\r
    # Assert zero overlap\r
    assert len(train_groups & val_groups) == 0, "Train-Val group leakage detected!"\r
    assert len(train_groups & test_groups) == 0, "Train-Test group leakage detected!"\r
    assert len(val_groups & test_groups) == 0, "Val-Test group leakage detected!"\r
\r
    train_idx = np.where(np.isin(groups, list(train_groups)))[0]\r
    val_idx = np.where(np.isin(groups, list(val_groups)))[0]\r
    test_idx = np.where(np.isin(groups, list(test_groups)))[0]\r
\r
    return train_idx, val_idx, test_idx\r
\r
\r
def train_and_evaluate_origin_framework(\r
    X: np.ndarray,\r
    y: np.ndarray,\r
    groups: np.ndarray,\r
    random_state: int = 42,\r
) -> Dict[str, Any]:\r
    \"\"\"\r
    Generic group-aware training framework with leakage prevention:\r
    1. Group-disjoint Train / Val / Test splitting.\r
    2. Training-only StandardScaler fitting.\r
    3. Multi-model comparative evaluation across LogisticRegression, LinearSVC, and RandomForest.\r
    4. Threshold calibration on Validation set only.\r
    5. Evaluation of calibrated threshold on held-out Test set.\r
    \"\"\"\r
    from sklearn.linear_model import LogisticRegression\r
    from sklearn.svm import LinearSVC\r
    from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score, f1_score, confusion_matrix\r
\r
    train_idx, val_idx, test_idx = build_group_disjoint_partitions(groups, y=y, random_state=random_state)\r
\r
    # Assert zero group leakage\r
    train_g = set(groups[train_idx])\r
    val_g = set(groups[val_idx])\r
    test_g = set(groups[test_idx])\r
    leakage_check = {\r
        "train_val_overlap": len(train_g & val_g),\r
        "train_test_overlap": len(train_g & test_g),\r
        "val_test_overlap": len(val_g & test_g),\r
        "zero_leakage_verified": (\r
            (len(train_g & val_g) == 0)\r
            and (len(train_g & test_g) == 0)\r
            and (len(val_g & test_g) == 0)\r
        ),\r
    }\r
\r
    X_train, y_train = X[train_idx], y[train_idx]\r
    X_val, y_val = X[val_idx], y[val_idx]\r
    X_test, y_test = X[test_idx], y[test_idx]\r
\r
    # Preprocessing: Fit scaler STRICTLY on training fold\r
    scaler = StandardScaler()\r
    X_train_scaled = scaler.fit_transform(X_train)\r
    X_val_scaled = scaler.transform(X_val)\r
    X_test_scaled = scaler.transform(X_test)\r
\r
    # Scaler isolation check: train mean is ~0, val mean is NOT forced to 0\r
    scaler_isolation = {\r
        "train_mean_norm": float(np.linalg.norm(np.mean(X_train_scaled, axis=0))),\r
        "val_mean_norm": float(np.linalg.norm(np.mean(X_val_scaled, axis=0))),\r
        "test_mean_norm": float(np.linalg.norm(np.mean(X_test_scaled, axis=0))),\r
        "scaler_strictly_fit_on_train": bool(np.allclose(np.mean(X_train_scaled, axis=0), 0, atol=1e-5)),\r
    }\r
\r
    candidate_clfs = {\r
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=random_state),\r
        "LinearSVC": LinearSVC(random_state=random_state, dual=False),\r
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=random_state),\r
    }\r
\r
    models_evaluation: Dict[str, Any] = {}\r
    for name, clf in candidate_clfs.items():\r
        clf.fit(X_train_scaled, y_train)\r
\r
        # Obtain continuous scores on val and test\r
        if hasattr(clf, "predict_proba"):\r
            scores_val = clf.predict_proba(X_val_scaled)[:, 1]\r
            scores_test = clf.predict_proba(X_test_scaled)[:, 1]\r
            threshold_candidates = np.linspace(0.1, 0.9, 17)\r
        elif hasattr(clf, "decision_function"):\r
            scores_val = clf.decision_function(X_val_scaled)\r
            scores_test = clf.decision_function(X_test_scaled)\r
            threshold_candidates = np.linspace(\r
                float(np.percentile(scores_val, 10)),\r
                float(np.percentile(scores_val, 90)),\r
                17,\r
            )\r
        else:\r
            scores_val = clf.predict(X_val_scaled)\r
            scores_test = clf.predict(X_test_scaled)\r
            threshold_candidates = np.array([0.5])\r
\r
        # Calibrate threshold on VALIDATION fold only\r
        best_t = 0.5 if hasattr(clf, "predict_proba") else 0.0\r
        best_val_bal_acc = -1.0\r
        for t in threshold_candidates:\r
            pred_val_t = (scores_val >= t).astype(int)\r
            b_acc = float(balanced_accuracy_score(y_val, pred_val_t))\r
            if b_acc > best_val_bal_acc:\r
                best_val_bal_acc = b_acc\r
                best_t = float(t)\r
\r
        # Apply calibrated threshold to held-out TEST set\r
        pred_val_opt = (scores_val >= best_t).astype(int)\r
        pred_test_opt = (scores_test >= best_t).astype(int)\r
\r
        # Compute metrics\r
        val_bal_acc = float(balanced_accuracy_score(y_val, pred_val_opt))\r
        val_rec = float(recall_score(y_val, pred_val_opt, zero_division=0))\r
        val_prec = float(precision_score(y_val, pred_val_opt, zero_division=0))\r
        val_f1 = float(f1_score(y_val, pred_val_opt, zero_division=0))\r
        val_cm = confusion_matrix(y_val, pred_val_opt, labels=[0, 1]).tolist()\r
\r
        test_bal_acc = float(balanced_accuracy_score(y_test, pred_test_opt))\r
        test_rec = float(recall_score(y_test, pred_test_opt, zero_division=0))\r
        test_prec = float(precision_score(y_test, pred_test_opt, zero_division=0))\r
        test_f1 = float(f1_score(y_test, pred_test_opt, zero_division=0))\r
        test_cm = confusion_matrix(y_test, pred_test_opt, labels=[0, 1]).tolist()\r
\r
        models_evaluation[name] = {\r
            "calibrated_threshold": round(best_t, 4),\r
            "validation": {\r
                "balanced_accuracy": round(val_bal_acc, 4),\r
                "precision": round(val_prec, 4),\r
                "recall": round(val_rec, 4),\r
                "f1": round(val_f1, 4),\r
                "confusion_matrix": val_cm,\r
            },\r
            "test_unseen_groups": {\r
                "balanced_accuracy": round(test_bal_acc, 4),\r
                "precision": round(test_prec, 4),\r
                "recall": round(test_rec, 4),\r
                "f1": round(test_f1, 4),\r
                "confusion_matrix": test_cm,\r
            },\r
        }\r
\r
    return {\r
        "partitions": {\r
            "n_train_samples": len(train_idx),\r
            "n_val_samples": len(val_idx),\r
            "n_test_samples": len(test_idx),\r
            "train_groups": sorted(list(train_g)),\r
            "val_groups": sorted(list(val_g)),\r
            "test_groups": sorted(list(test_g)),\r
        },\r
        "leakage_check": leakage_check,\r
        "scaler_isolation": scaler_isolation,\r
        "models_evaluation": models_evaluation,\r
    }\r
\r
\r
def run_origin_classifier_suite("""

assert target in raw, "Binary target not found in src/origin_classifier.py"
new_raw = raw.replace(target, insertion, 1)
p.write_bytes(new_raw)
print("Successfully cleanly patched src/origin_classifier.py in binary mode.")

