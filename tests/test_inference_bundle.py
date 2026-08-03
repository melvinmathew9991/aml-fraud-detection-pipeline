"""
Unit tests for inference/bundle.py's integrity verification: a tampered,
partial, or missing bundle must fail to load rather than silently serve
wrong content (ARCHITECTURE.md §3's "refuses to serve a tampered or
partial bundle").
"""

import json
import shutil
from pathlib import Path

import pytest

from inference.bundle import (
    Bundle,
    BundleFileMissingError,
    BundleIntegrityError,
    load_bundle,
)

REAL_BUNDLE_DIR = Path(__file__).resolve().parents[1] / "model_bundle" / "v1"


def _require_real_bundle():
    if not (REAL_BUNDLE_DIR / "bundle_meta.json").exists():
        pytest.skip("model_bundle/v1 not present -- run export_bundle.py first.")


def test_load_bundle_succeeds_on_committed_bundle():
    _require_real_bundle()
    bundle = load_bundle(REAL_BUNDLE_DIR)
    assert isinstance(bundle, Bundle)
    assert bundle.bundle_version == "v1"
    # feature_version is the feature-SCHEMA version, not the feature count.
    # export_bundle.py wrote the count into both fields until 2026-08-03, and
    # the assertion that used to live here (len(feature_names) == feature_version)
    # passed only because of that bug -- it could never have caught it.
    # No training import here: this file runs in the serving-isolation job.
    assert len(bundle.feature_names) == bundle.bundle_meta["n_features"] == 18
    assert bundle.feature_version == 3


def test_missing_bundle_meta_raises(tmp_path):
    with pytest.raises(BundleFileMissingError):
        load_bundle(tmp_path)


def test_missing_file_listed_in_manifest_raises(tmp_path):
    _require_real_bundle()
    for name in ["bundle_meta.json", "scaler.json", "threshold.json"]:
        shutil.copy(REAL_BUNDLE_DIR / name, tmp_path / name)
    # model.txt and dest_state.parquet deliberately omitted, but bundle_meta
    # still lists them.
    with pytest.raises(BundleFileMissingError):
        load_bundle(tmp_path)


def test_tampered_file_raises_integrity_error(tmp_path):
    _require_real_bundle()
    for name in ["bundle_meta.json", "scaler.json", "threshold.json",
                 "model.txt", "dest_state.parquet"]:
        shutil.copy(REAL_BUNDLE_DIR / name, tmp_path / name)

    with open(tmp_path / "threshold.json") as f:
        payload = json.load(f)
    payload["decision_threshold"] = payload["decision_threshold"] * 2 + 1e-9
    with open(tmp_path / "threshold.json", "w") as f:
        json.dump(payload, f)

    with pytest.raises(BundleIntegrityError):
        load_bundle(tmp_path)


def test_corrupted_checksum_manifest_itself_raises(tmp_path):
    _require_real_bundle()
    for name in ["scaler.json", "threshold.json", "model.txt", "dest_state.parquet"]:
        shutil.copy(REAL_BUNDLE_DIR / name, tmp_path / name)

    with open(REAL_BUNDLE_DIR / "bundle_meta.json") as f:
        meta = json.load(f)
    meta["sha256"]["model.txt"] = "0" * 64
    with open(tmp_path / "bundle_meta.json", "w") as f:
        json.dump(meta, f)

    with pytest.raises(BundleIntegrityError):
        load_bundle(tmp_path)
