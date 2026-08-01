"""
export_bundle.py

Assembles the versioned, version-portable serving artifact from one
training run's output (ARCHITECTURE.md §3). Deliberately does NOT ship
`*.joblib`: pickled sklearn/LightGBM estimators are tied to the exact
library versions that wrote them, which turns every dependency bump into a
silent deserialization risk in production. Instead:

  model.txt        LightGBM's own native text format (booster_.save_model())
  scaler.json       {"feature_names", "mean", "scale"} -- StandardScaler is
                    just (x - mean) / scale, so shipping two arrays as JSON
                    and applying them in numpy removes scikit-learn (and its
                    scipy pull-in) from the serving image entirely
  threshold.json    the deployable decision rule -- score >= threshold means
                    "flag it" -- plus the operating point it was measured at
  dest_state.parquet  built separately by build_dest_state.py; this script
                    only reads it to include in the checksum manifest
  bundle_meta.json  bundle/feature version, git commit, run id, and a
                    per-file sha256 the API verifies at startup before
                    serving a tampered or partial bundle

Run `build_dest_state.py` before this script -- it will refuse to produce a
bundle_meta.json missing dest_state.parquet's checksum rather than silently
omitting it.
"""

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

import joblib

from config import PROJECT_ROOT

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("export_bundle")

BUNDLE_VERSION = "v1"


def _latest_run_dir(model_dir: Path) -> Path:
    runs = sorted((d for d in model_dir.iterdir() if d.is_dir() and (d / "metadata.json").exists()),
                  key=lambda d: d.name)
    if not runs:
        raise FileNotFoundError(f"No run directories with metadata.json found under {model_dir}")
    return runs[-1]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def export_scaler(scaler, feature_names: list[str], out_path: Path) -> None:
    scaler_feature_names = list(getattr(scaler, "feature_names_in_", feature_names))
    if scaler_feature_names != feature_names:
        raise ValueError(
            "Scaler's fitted feature order does not match metadata.json's feature_names -- "
            "refusing to export a bundle whose scaler and model could silently disagree on "
            f"column order. scaler={scaler_feature_names!r} metadata={feature_names!r}"
        )
    payload = {
        "feature_names": feature_names,
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)


def export_threshold(metadata: dict, out_path: Path) -> None:
    payload = {
        "decision_threshold": metadata["decision_threshold"],
        "reviews_per_day": metadata["review_capacity_per_day"],
        "fold": metadata["decision_threshold_fold"],
        "expected_precision": metadata["expected_precision_at_threshold"],
        "expected_recall": metadata["expected_recall_at_threshold"],
        "precision_ceiling": metadata["precision_ceiling_at_threshold"],
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)


def export_model(model, out_path: Path) -> None:
    booster = getattr(model, "booster_", None)
    if booster is None:
        raise TypeError(
            f"best_model_artifact's model ({type(model).__name__}) has no LightGBM booster_ -- "
            "the bundle format (ARCHITECTURE.md §3) only supports LightGBM's native text export. "
            "If a non-LightGBM model ever wins selection, this script needs a new branch, not a "
            "silent joblib fallback (that is exactly the version-portability problem the bundle "
            "format exists to avoid)."
        )
    booster.save_model(str(out_path))


def build_bundle(run_dir: Path, output_dir: Path, dest_state_path: Path) -> None:
    with open(run_dir / "metadata.json") as f:
        metadata = json.load(f)

    if not dest_state_path.exists():
        raise FileNotFoundError(
            f"{dest_state_path} does not exist -- run build_dest_state.py before export_bundle.py."
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    best_key = metadata["best_model_artifact"]
    model = joblib.load(run_dir / f"{best_key}.joblib")
    scaler = joblib.load(run_dir / "scaler.joblib")

    model_path = output_dir / "model.txt"
    scaler_path = output_dir / "scaler.json"
    threshold_path = output_dir / "threshold.json"
    meta_path = output_dir / "bundle_meta.json"

    export_model(model, model_path)
    export_scaler(scaler, metadata["feature_names"], scaler_path)
    export_threshold(metadata, threshold_path)
    logger.info("Wrote model.txt, scaler.json, threshold.json to %s", output_dir)

    sha256 = {
        p.name: _sha256(p) for p in (model_path, scaler_path, threshold_path, dest_state_path)
    }
    bundle_meta = {
        "bundle_version": BUNDLE_VERSION,
        "feature_version": len(metadata["feature_names"]),  # informational; see FEATURE_VERSION
        "n_features": len(metadata["feature_names"]),
        "git_commit": metadata["git_commit"],
        "run_id": metadata["run_id"],
        "model_name": metadata["best_model"],
        "model_artifact_key": best_key,
        "trained_at": metadata["run_id"],
        "sha256": sha256,
    }
    with open(meta_path, "w") as f:
        json.dump(bundle_meta, f, indent=2)
    logger.info("Wrote bundle_meta.json with sha256 for %d files", len(sha256))

    total_mb = sum(p.stat().st_size for p in (model_path, scaler_path, threshold_path,
                                               dest_state_path, meta_path)) / (1024 ** 2)
    logger.info("Bundle assembled at %s (run=%s, model=%s, %d features, ~%.2f MB total)",
                output_dir, metadata["run_id"], metadata["best_model"],
                len(metadata["feature_names"]), total_mb)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=None,
                         help="Run id under models/ to export (default: latest by name)")
    parser.add_argument("--output-dir", type=Path,
                         default=PROJECT_ROOT / "model_bundle" / BUNDLE_VERSION)
    parser.add_argument("--dest-state", type=Path,
                         default=PROJECT_ROOT / "model_bundle" / BUNDLE_VERSION / "dest_state.parquet")
    args = parser.parse_args()

    model_dir = PROJECT_ROOT / "models"
    run_dir = (model_dir / args.run_id) if args.run_id else _latest_run_dir(model_dir)
    if not run_dir.exists():
        logger.error("Run directory %s does not exist.", run_dir)
        sys.exit(1)

    build_bundle(run_dir, args.output_dir, args.dest_state)


if __name__ == "__main__":
    main()
