"""
bundle.py

Loads a versioned model_bundle/vN/ directory (ARCHITECTURE.md §3) and
verifies every file against bundle_meta.json's sha256 manifest before
anything is served from it. A tampered or partially-written bundle (e.g. an
interrupted `docker build` COPY, or a hand-edited threshold.json) must fail
loudly at load time -- "the API served a plausible-looking score against the
wrong model" is exactly the failure mode ARCHITECTURE.md §2 calls out for
the destination-state snapshot, and the same principle applies to every
other file in the bundle.

Deliberately does NOT depend on scikit-learn, duckdb, pandas, or joblib --
this module runs in the serving image (ARCHITECTURE.md §3's dependency
split), where none of those are installed.
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np


class BundleError(Exception):
    """Base class for bundle load failures."""


class BundleFileMissingError(BundleError):
    """A file bundle_meta.json's manifest requires is not on disk."""


class BundleIntegrityError(BundleError):
    """A file's sha256 does not match bundle_meta.json's manifest."""


@dataclass(frozen=True)
class Scaler:
    feature_names: list[str]
    mean: np.ndarray
    scale: np.ndarray

    def transform(self, X: np.ndarray) -> np.ndarray:
        """(x - mean) / scale -- the exact StandardScaler formula, reimplemented
        so the serving image never needs scikit-learn (ARCHITECTURE.md §3)."""
        return (X - self.mean) / self.scale


@dataclass(frozen=True)
class Threshold:
    decision_threshold: float
    reviews_per_day: float
    fold: int
    expected_precision: float
    expected_recall: float
    precision_ceiling: float


@dataclass(frozen=True)
class Bundle:
    bundle_dir: Path
    bundle_meta: dict
    booster: lgb.Booster
    scaler: Scaler
    threshold: Threshold
    dest_state_path: Path

    @property
    def bundle_version(self) -> str:
        return self.bundle_meta["bundle_version"]

    @property
    def feature_version(self) -> int:
        return self.bundle_meta["feature_version"]

    @property
    def feature_names(self) -> list[str]:
        return self.scaler.feature_names


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _verify_checksums(bundle_dir: Path, sha256_manifest: dict[str, str]) -> None:
    for filename, expected_hash in sha256_manifest.items():
        path = bundle_dir / filename
        if not path.exists():
            raise BundleFileMissingError(
                f"bundle_meta.json requires {filename!r} but {path} does not exist -- "
                "refusing to serve a partial bundle."
            )
        actual_hash = _sha256(path)
        if actual_hash != expected_hash:
            raise BundleIntegrityError(
                f"{filename} checksum mismatch: bundle_meta.json says {expected_hash}, "
                f"file on disk is {actual_hash} -- refusing to serve a tampered "
                "or corrupted bundle."
            )


def load_bundle(bundle_dir: Path) -> Bundle:
    """
    Loads and integrity-checks every file in `bundle_dir` against
    bundle_meta.json's sha256 manifest. Raises BundleError (or a subclass)
    on any failure -- callers (typically /ready) are expected to catch this
    and report an unready service rather than let it propagate as a 500 on
    the first scoring request.
    """
    meta_path = bundle_dir / "bundle_meta.json"
    if not meta_path.exists():
        raise BundleFileMissingError(f"{meta_path} does not exist.")
    with open(meta_path) as f:
        bundle_meta = json.load(f)

    _verify_checksums(bundle_dir, bundle_meta["sha256"])

    with open(bundle_dir / "scaler.json") as f:
        scaler_payload = json.load(f)
    scaler = Scaler(
        feature_names=scaler_payload["feature_names"],
        mean=np.array(scaler_payload["mean"], dtype="float64"),
        scale=np.array(scaler_payload["scale"], dtype="float64"),
    )

    with open(bundle_dir / "threshold.json") as f:
        threshold_payload = json.load(f)
    threshold = Threshold(
        decision_threshold=threshold_payload["decision_threshold"],
        reviews_per_day=threshold_payload["reviews_per_day"],
        fold=threshold_payload["fold"],
        expected_precision=threshold_payload["expected_precision"],
        expected_recall=threshold_payload["expected_recall"],
        precision_ceiling=threshold_payload["precision_ceiling"],
    )

    booster = lgb.Booster(model_file=str(bundle_dir / "model.txt"))

    return Bundle(
        bundle_dir=bundle_dir,
        bundle_meta=bundle_meta,
        booster=booster,
        scaler=scaler,
        threshold=threshold,
        dest_state_path=bundle_dir / "dest_state.parquet",
    )
