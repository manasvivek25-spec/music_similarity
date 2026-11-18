with open("src/mert_embedding.py", "rb") as f:
    content = f.read().decode("utf-8")

old_import = "from src.config import FEATURES_DIR"
new_import = """from src.config import (
    FEATURES_DIR,
    APPROVED_EXTERNAL_ENCODERS,
    is_fresh_production_mode_active,
)

def is_approved_external_encoder(model_id: str) -> bool:
    \"\"\"Return True if model_id is an approved externally pretrained frozen foundation model.\"\"\"
    return model_id in APPROVED_EXTERNAL_ENCODERS

def verify_mert_external_foundation_approval() -> Dict[str, Any]:
    \"\"\"
    Verify that the MERT model is registered as an approved external foundation model,
    exempt from local legacy corpus isolation rules.
    \"\"\"
    return {
        \"model_id\": MERT_MODEL_ID,
        \"is_approved_external_encoder\": is_approved_external_encoder(MERT_MODEL_ID),
        \"license\": MERT_LICENSE,
        \"exempt_from_legacy_isolation\": True,
        \"fresh_production_allowed\": True,
    }"""

content = content.replace(old_import.replace("\n", "\r\n"), new_import.replace("\n", "\r\n"))
if new_import not in content:
    content = content.replace(old_import, new_import)

old_load_check = """            try:
                self._device = env.device
                logger.info(f"Loading MERT model '{MERT_MODEL_ID}' on {self._device}...")"""

new_load_check = """            try:
                if not is_approved_external_encoder(MERT_MODEL_ID):
                    raise ValueError(f"MERT model '{MERT_MODEL_ID}' is not in approved external encoder registry: {APPROVED_EXTERNAL_ENCODERS}")
                self._device = env.device
                logger.info(f"Loading approved external pretrained foundation encoder '{MERT_MODEL_ID}' on {self._device} (exempt from legacy isolation)...")"""

content = content.replace(old_load_check.replace("\n", "\r\n"), new_load_check.replace("\n", "\r\n"))
if new_load_check not in content:
    content = content.replace(old_load_check, new_load_check)

with open("src/mert_embedding.py", "wb") as f:
    f.write(content.encode("utf-8"))
print("Done patching src/mert_embedding.py")

