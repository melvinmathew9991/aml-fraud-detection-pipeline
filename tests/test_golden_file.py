"""
Golden-file regression test (ARCHITECTURE.md §7 / Sprint 3 DoD): reproduces
each of the 200 committed golden transactions' expected_score using ONLY the
committed serving bundle (model.txt + scaler.json) -- no scikit-learn, no
joblib model, no DuckDB. This is what proves the LightGBM native-format
round-trip and the pure-numpy StandardScaler reimplementation are each exact
to 1e-9, as standing regressions rather than the one-off check
ARCHITECTURE §12 originally ran.

Skipped (not failed) if the bundle or golden file haven't been generated yet
-- both are produced by scripts, not committed as an assumption of
pytest collection succeeding in every checkout state.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BUNDLE_DIR = PROJECT_ROOT / "model_bundle" / "v1"
GOLDEN_PATH = PROJECT_ROOT / "tests" / "golden" / "golden_transactions.csv"


def _require(path: Path):
    if not path.exists():
        pytest.skip(f"{path} not generated yet -- run export_bundle.py / generate_golden_file.py")


def test_golden_scores_reproduce_to_1e9():
    _require(BUNDLE_DIR / "model.txt")
    _require(BUNDLE_DIR / "scaler.json")
    _require(GOLDEN_PATH)

    lgb = pytest.importorskip("lightgbm")

    with open(BUNDLE_DIR / "scaler.json") as f:
        scaler = json.load(f)
    feature_names = scaler["feature_names"]
    mean = np.array(scaler["mean"])
    scale = np.array(scaler["scale"])

    golden = pd.read_csv(GOLDEN_PATH)
    raw = golden[feature_names].to_numpy(dtype="float64")
    scaled = (raw - mean) / scale

    booster = lgb.Booster(model_file=str(BUNDLE_DIR / "model.txt"))
    scores = booster.predict(scaled)

    expected = golden["expected_score"].to_numpy()
    max_abs_diff = np.max(np.abs(scores - expected))
    assert max_abs_diff < 1e-9, f"max abs diff {max_abs_diff:.3e} exceeds 1e-9"


def test_golden_file_has_both_classes():
    _require(GOLDEN_PATH)
    golden = pd.read_csv(GOLDEN_PATH)
    assert golden["is_fraud"].sum() > 0
    assert (golden["is_fraud"] == 0).sum() > 0


def test_bundle_meta_checksums_match_files():
    _require(BUNDLE_DIR / "bundle_meta.json")
    import hashlib

    with open(BUNDLE_DIR / "bundle_meta.json") as f:
        bundle_meta = json.load(f)

    for filename, expected_hash in bundle_meta["sha256"].items():
        path = BUNDLE_DIR / filename
        _require(path)
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        assert h.hexdigest() == expected_hash, f"{filename} checksum mismatch"
