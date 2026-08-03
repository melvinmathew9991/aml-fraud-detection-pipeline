"""
Skew test -- plumbing (ARCHITECTURE.md §7): feeds the exact golden-file
feature matrix through inference/score.py and asserts probabilities match
`expected_score` (computed at training time via the ORIGINAL joblib
sklearn/LightGBM model + sklearn StandardScaler -- see
generate_golden_file.py) to 1e-9.

This is deliberately the SERVING half of the training/serving skew problem
(ARCHITECTURE.md §2): it isolates model/scaler/threshold plumbing bugs --
a wrong scaler formula, a feature-order mismatch, a stale booster -- from
the STATE half, which test_skew_state.py isolates separately so a failure
identifies which one broke.

Skipped (not failed) if the bundle or golden file aren't generated yet.
"""

from pathlib import Path

import pandas as pd
import pytest

from inference.bundle import load_bundle
from inference.score import score_batch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BUNDLE_DIR = PROJECT_ROOT / "model_bundle" / "v1"
GOLDEN_PATH = PROJECT_ROOT / "tests" / "golden" / "golden_transactions.csv"


def _require(path: Path):
    if not path.exists():
        pytest.skip(f"{path} not generated yet -- run export_bundle.py / generate_golden_file.py")


def test_score_batch_matches_expected_score_to_1e9():
    _require(BUNDLE_DIR / "bundle_meta.json")
    _require(GOLDEN_PATH)

    bundle = load_bundle(BUNDLE_DIR)
    golden = pd.read_csv(GOLDEN_PATH)

    X = golden[bundle.feature_names].to_numpy(dtype="float64")
    expected = golden["expected_score"].to_numpy()

    results = score_batch(bundle, X.tolist())
    probabilities = [r.probability for r in results]

    max_abs_diff = max(abs(p - e) for p, e in zip(probabilities, expected, strict=True))
    assert max_abs_diff < 1e-9, f"max abs diff {max_abs_diff:.3e} exceeds 1e-9"


def test_score_one_matches_score_batch_for_same_row():
    _require(BUNDLE_DIR / "bundle_meta.json")
    _require(GOLDEN_PATH)

    from inference.score import score_one

    bundle = load_bundle(BUNDLE_DIR)
    golden = pd.read_csv(GOLDEN_PATH)
    X = golden[bundle.feature_names].to_numpy(dtype="float64")

    one = score_one(bundle, X[0].tolist())
    batch = score_batch(bundle, X[:1].tolist())[0]
    assert one.probability == pytest.approx(batch.probability, abs=1e-12)


def test_flagged_matches_threshold_comparison():
    _require(BUNDLE_DIR / "bundle_meta.json")
    _require(GOLDEN_PATH)

    bundle = load_bundle(BUNDLE_DIR)
    golden = pd.read_csv(GOLDEN_PATH)
    X = golden[bundle.feature_names].to_numpy(dtype="float64")

    results = score_batch(bundle, X.tolist())
    for r in results:
        assert r.flagged == (r.probability >= bundle.threshold.decision_threshold)
