import re

with open("src/origin_classifier.py", "rb") as f:
    content = f.read().decode("utf-8")

# 1. Update imports
import_target = """from src.config import MODELS_DIR, SAMPLE_RATE"""
import_replacement = """import hashlib
import logging
from src.config import (
    MODELS_DIR, SAMPLE_RATE,
    LegacyArtifactError, LEGACY_ARTIFACT_HASHES,
    is_fresh_production_mode_active, is_legacy_mode_active, get_system_mode
)

logger = logging.getLogger(__name__)
TEST_FIXTURE_MODEL_FILE = "test_fixture_origin_classifier.joblib\""""

content = content.replace(import_target, import_replacement)

# 2. Add is_test_fixture_model and create_test_fixture_origin_classifier, and update load_origin_classifier and predict_origin
old_loaders = """def get_default_model_path() -> Path:
    \"\"\"Return the resolved path to the trained origin classifier artifact.\"\"\"
    return MODELS_DIR / MODEL_FILE_NAME


def load_origin_classifier(
    model_path: Optional[Union[str, Path]] = None
) -> Optional[Any]:
    \"\"\"
    Safely load the fitted classifier artifact from disk.

    Returns None if model file does not exist or fails to deserialize.
    \"\"\"
    path = Path(model_path) if model_path is not None else get_default_model_path()
    if not path.is_file():
        return None

    try:
        model = joblib.load(str(path))
        return model
    except Exception:
        return None


def predict_origin(
    audio_path: Union[str, Path],
    model_path: Optional[Union[str, Path]] = None
) -> OriginClassifierResult:
    \"\"\"
    Predict track origin (Human vs. AI) using the independent classifier.

    If model is missing or unconfigured, returns an honest 'Unconfigured' status.
    Never invents predictions or infers origin from DTW similarity scores.

    Parameters
    ----------
    audio_path : Union[str, Path]
        Path to target audio file.
    model_path : Optional[Union[str, Path]]
        Optional explicit model path override.

    Returns
    -------
    OriginClassifierResult
        Prediction details or explicit unconfigured notice.
    \"\"\"
    model = load_origin_classifier(model_path)
    if model is None:
        return OriginClassifierResult(
            predicted_origin="Unconfigured",
            confidence=None,
            probability_ai=None,
            model_status="Unconfigured / Missing Labeled Data",
            model_version=None,
        )

    try:
        features = extract_origin_features_from_audio(audio_path)
        features_2d = features.reshape(1, -1)

        pred_class = model.predict(features_2d)[0]
        label = "AI" if pred_class in (1, "AI", "ai") else "Human"

        prob_ai: Optional[float] = None
        confidence: Optional[float] = None

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(features_2d)[0]
            # Assume class 1 is AI if classes are [0, 1]
            classes = list(model.classes_)
            if 1 in classes:
                ai_idx = classes.index(1)
            elif "AI" in classes:
                ai_idx = classes.index("AI")
            else:
                ai_idx = len(classes) - 1

            prob_ai = float(probs[ai_idx])
            confidence = float(max(probs))

        return OriginClassifierResult(
            predicted_origin=label,
            confidence=round(confidence, 4) if confidence is not None else None,
            probability_ai=round(prob_ai, 4) if prob_ai is not None else None,
            model_status="Trained & Active",
            model_version=MODEL_VERSION,
        )
    except Exception:
        return OriginClassifierResult(
            predicted_origin="Unconfigured",
            confidence=None,
            probability_ai=None,
            model_status="Unconfigured / Missing Labeled Data",
            model_version=None,
        )"""

new_loaders = """def is_test_fixture_model(model_or_path: Any) -> bool:
    \"\"\"Return True if model object or path is an approved test-only fixture.\"\"\"
    if isinstance(model_or_path, (str, Path)):
        p = Path(model_or_path)
        if "test_fixture" in p.name.lower() or "fixture" in p.name.lower():
            return True
    if hasattr(model_or_path, "is_test_fixture") and getattr(model_or_path, "is_test_fixture"):
        return True
    if hasattr(model_or_path, "test_fixture_metadata"):
        return True
    return False


def create_test_fixture_origin_classifier(
    output_path: Optional[Union[str, Path]] = None
) -> Path:
    \"\"\"
    Synthesize and serialize a lightweight test-only origin classifier fixture.
    Clearly stamped with is_test_fixture=True and synthetic provenance metadata.
    Does NOT use any tracks or features from the retired 261-record corpus.
    \"\"\"
    out_p = Path(output_path) if output_path else MODELS_DIR / TEST_FIXTURE_MODEL_FILE
    out_p.parent.mkdir(parents=True, exist_ok=True)

    from sklearn.linear_model import LogisticRegression
    np.random.seed(42)
    X_synthetic = np.random.randn(20, 56).astype(np.float32)
    X_synthetic[:10] -= 1.0  # Class 0: Human
    X_synthetic[10:] += 1.0  # Class 1: AI
    y_synthetic = np.array([0]*10 + [1]*10, dtype=int)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(random_state=42, max_iter=200))
    ])
    pipe.fit(X_synthetic, y_synthetic)

    pipe.is_test_fixture = True
    pipe.test_fixture_metadata = {
        "is_test_fixture": True,
        "provenance": "synthetic_fixtures",
        "description": "Synthesized 56-dim acoustic fixture for browser E2E and regression testing",
        "created_at": "2026-11-01T20:00:00Z"
    }

    joblib.dump(pipe, str(out_p))
    logger.info(f"Created test fixture origin classifier at {out_p}")
    return out_p


def get_default_model_path() -> Path:
    \"\"\"Return the resolved path to the trained origin classifier artifact.\"\"\"
    return MODELS_DIR / MODEL_FILE_NAME


def load_origin_classifier(
    model_path: Optional[Union[str, Path]] = None,
    allow_legacy: bool = False,
    strict: bool = False,
) -> Optional[Any]:
    \"\"\"
    Safely load the fitted classifier artifact from disk.

    In fresh-production mode (default):
    - Strictly rejects legacy 261-record corpus checkpoints (models/origin_classifier.joblib).
    - If strict=True, raises LegacyArtifactError on legacy models.
    - If strict=False, logs warning and returns None (unconfigured).
    - Allows test fixtures (stamped with is_test_fixture or path matching test fixture).
    - Allows newly trained models with fresh production provenance.

    In legacy mode (MUSIC_SIMILARITY_MODE="legacy" or allow_legacy=True):
    - Loads historical checkpoint for benchmark evaluation.
    \"\"\"
    if model_path is not None:
        path = Path(model_path)
    else:
        env_fixture = os.getenv("MUSIC_ORIGIN_TEST_FIXTURE_PATH")
        env_model = os.getenv("MUSIC_ORIGIN_MODEL_PATH")
        if env_fixture and Path(env_fixture).is_file():
            path = Path(env_fixture)
        elif env_model and Path(env_model).is_file():
            path = Path(env_model)
        else:
            path = get_default_model_path()

    if not path.is_file():
        return None

    # Check if target is an approved test fixture
    if is_test_fixture_model(path):
        try:
            model = joblib.load(str(path))
            return model
        except Exception as exc:
            logger.warning(f"Failed to load test fixture model from {path}: {exc}")
            return None

    # Check for legacy 261-record checkpoint
    is_legacy_artifact = False
    try:
        content_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        if content_hash == LEGACY_ARTIFACT_HASHES.get("origin_classifier"):
            is_legacy_artifact = True
    except Exception:
        pass

    if path.resolve() == get_default_model_path().resolve():
        is_legacy_artifact = True

    if is_legacy_artifact:
        # Check if legacy loading is explicitly authorized
        if not allow_legacy and is_fresh_production_mode_active():
            msg = (
                f"Historical origin classifier checkpoint ({path.name}) from retired 261-record corpus "
                f"cannot be loaded in fresh-production mode (mode='{get_system_mode()}'). "
                "Explicit mode='legacy' or allow_legacy=True is strictly required."
            )
            if strict:
                raise LegacyArtifactError(msg)
            logger.warning(f"REJECTED LEGACY ARTIFACT: {msg}")
            return None

    try:
        model = joblib.load(str(path))
        if hasattr(model, "provenance_mode") and getattr(model, "provenance_mode") == "legacy":
            if not allow_legacy and is_fresh_production_mode_active():
                if strict:
                    raise LegacyArtifactError("Model object carries legacy 261 provenance in fresh-production mode.")
                return None
        return model
    except LegacyArtifactError:
        raise
    except Exception as exc:
        logger.warning(f"Failed to deserialize origin classifier from {path}: {exc}")
        return None


def predict_origin(
    audio_path: Union[str, Path],
    model_path: Optional[Union[str, Path]] = None,
    allow_legacy: bool = False,
) -> OriginClassifierResult:
    \"\"\"
    Predict track origin (Human vs. AI) using the independent classifier.

    If model is missing or unconfigured, returns an honest 'Unconfigured' status.
    Never invents predictions or infers origin from DTW similarity scores.

    Parameters
    ----------
    audio_path : Union[str, Path]
        Path to target audio file.
    model_path : Optional[Union[str, Path]]
        Optional explicit model path override.
    allow_legacy : bool
        If True, permits loading historical legacy 261-record checkpoint.

    Returns
    -------
    OriginClassifierResult
        Prediction details or explicit unconfigured notice.
    \"\"\"
    model = load_origin_classifier(model_path, allow_legacy=allow_legacy, strict=False)
    if model is None:
        return OriginClassifierResult(
            predicted_origin="Unconfigured",
            confidence=None,
            probability_ai=None,
            model_status="Unconfigured / Missing Labeled Data",
            model_version=None,
        )

    is_fixture = is_test_fixture_model(model) or (model_path and is_test_fixture_model(model_path))

    try:
        features = extract_origin_features_from_audio(audio_path)
        features_2d = features.reshape(1, -1)

        pred_class = model.predict(features_2d)[0]
        label = "AI" if pred_class in (1, "AI", "ai") else "Human"

        prob_ai: Optional[float] = None
        confidence: Optional[float] = None

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(features_2d)[0]
            classes = list(model.classes_)
            if 1 in classes:
                ai_idx = classes.index(1)
            elif "AI" in classes:
                ai_idx = classes.index("AI")
            else:
                ai_idx = len(classes) - 1

            prob_ai = float(probs[ai_idx])
            confidence = float(max(probs))

        if is_fixture:
            status_label = "Test Fixture Active"
            version_label = "test-fixture-1.0.0"
        elif is_fresh_production_mode_active():
            status_label = "Trained & Active"
            version_label = MODEL_VERSION
        else:
            status_label = "Historical Checkpoint Active (Legacy Mode)"
            version_label = f"legacy-{MODEL_VERSION}"

        return OriginClassifierResult(
            predicted_origin=label,
            confidence=round(confidence, 4) if confidence is not None else None,
            probability_ai=round(prob_ai, 4) if prob_ai is not None else None,
            model_status=status_label,
            model_version=version_label,
        )
    except Exception:
        return OriginClassifierResult(
            predicted_origin="Unconfigured",
            confidence=None,
            probability_ai=None,
            model_status="Unconfigured / Missing Labeled Data",
            model_version=None,
        )"""

# Handle CRLF vs LF
old_loaders_crlf = old_loaders.replace("\r\n", "\n").replace("\n", "\r\n")
old_loaders_lf = old_loaders.replace("\r\n", "\n")

if old_loaders_crlf in content:
    content = content.replace(old_loaders_crlf, new_loaders.replace("\n", "\r\n"))
    print("Replaced loaders CRLF")
elif old_loaders_lf in content:
    content = content.replace(old_loaders_lf, new_loaders)
    print("Replaced loaders LF")
else:
    print("Old loaders not found!")

with open("src/origin_classifier.py", "wb") as f:
    f.write(content.encode("utf-8"))
print("Done patching src/origin_classifier.py")
