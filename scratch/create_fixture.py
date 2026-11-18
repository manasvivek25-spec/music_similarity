import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.origin_classifier import create_test_fixture_origin_classifier
p = create_test_fixture_origin_classifier()
print(f"Created: {p}, exists={p.exists()}")
