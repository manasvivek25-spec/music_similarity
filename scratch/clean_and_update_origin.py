from pathlib import Path

p = Path("src/origin_classifier.py")
raw = p.read_bytes()
# Clean all double-returns
text = raw.decode("utf-8").replace("\r\r\n", "\n").replace("\r\n", "\n")

# Now replace the function
idx1 = text.find("def build_group_disjoint_partitions(")
idx2 = text.find("def train_and_evaluate_origin_framework(")

new_build_func = """def build_group_disjoint_partitions(
    groups: np.ndarray,
    y: Optional[np.ndarray] = None,
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    \"\"\"
    Partition sample indices into Train, Validation, and Test subsets such that
    NO group (e.g. generator family, artist ID, recording session) appears in
    more than one partition. Guarantees zero group leakage. If labels `y` are
    provided, partitions are stratified across classes.

    Returns
    -------
    train_idx : np.ndarray
    val_idx : np.ndarray
    test_idx : np.ndarray
    \"\"\"
    unique_groups = np.unique(groups)
    if len(unique_groups) < 3:
        raise ValueError(
            f"At least 3 distinct groups required for train/val/test disjoint partitioning, found {len(unique_groups)}"
        )

    rng = np.random.RandomState(random_state)

    if y is not None:
        group_labels = {g: int(np.round(np.mean(y[groups == g]))) for g in unique_groups}
        c0_groups = rng.permutation([g for g, lbl in group_labels.items() if lbl == 0])
        c1_groups = rng.permutation([g for g, lbl in group_labels.items() if lbl == 1])

        def split_class_groups(cg):
            n = len(cg)
            if n <= 1:
                return set(cg), set(), set()
            nt = max(1, int(round(test_size * n))) if n >= 3 else 0
            nv = max(1, int(round(val_size * n))) if n >= 3 else (1 if n == 2 else 0)
            ntr = n - nt - nv
            if ntr < 1:
                ntr = 1
                if nv > 0:
                    nv -= 1
            return set(cg[nt + nv:]), set(cg[nt:nt + nv]), set(cg[:nt])

        tr0, va0, te0 = split_class_groups(c0_groups)
        tr1, va1, te1 = split_class_groups(c1_groups)

        train_groups = tr0 | tr1
        val_groups = va0 | va1
        test_groups = te0 | te1

        if len(val_groups) == 0:
            val_groups = {list(train_groups)[0]}
            train_groups = train_groups - val_groups
        if len(test_groups) == 0:
            test_groups = {list(train_groups)[0]}
            train_groups = train_groups - test_groups
    else:
        shuffled_groups = rng.permutation(unique_groups)
        n_groups = len(shuffled_groups)
        n_test = max(1, int(round(test_size * n_groups)))
        n_val = max(1, int(round(val_size * n_groups)))
        n_train = n_groups - n_test - n_val
        if n_train < 1:
            n_train = 1
            if n_val > 1:
                n_val -= 1
            elif n_test > 1:
                n_test -= 1

        test_groups = set(shuffled_groups[:n_test])
        val_groups = set(shuffled_groups[n_test : n_test + n_val])
        train_groups = set(shuffled_groups[n_test + n_val :])

    # Assert zero overlap
    assert len(train_groups & val_groups) == 0, "Train-Val group leakage detected!"
    assert len(train_groups & test_groups) == 0, "Train-Test group leakage detected!"
    assert len(val_groups & test_groups) == 0, "Val-Test group leakage detected!"

    train_idx = np.where(np.isin(groups, list(train_groups)))[0]
    val_idx = np.where(np.isin(groups, list(val_groups)))[0]
    test_idx = np.where(np.isin(groups, list(test_groups)))[0]

    return train_idx, val_idx, test_idx


"""

text = text[:idx1] + new_build_func + text[idx2:]
text = text.replace(
    "train_idx, val_idx, test_idx = build_group_disjoint_partitions(groups, random_state=random_state)",
    "train_idx, val_idx, test_idx = build_group_disjoint_partitions(groups, y=y, random_state=random_state)"
)

# Write using standard newline='\n'
with open(p, "w", encoding="utf-8", newline="\n") as f:
    f.write(text)

print("Successfully cleaned and updated src/origin_classifier.py")

