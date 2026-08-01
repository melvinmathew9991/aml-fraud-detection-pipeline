"""
generate_golden_file.py

Draws a fixed, committed sample of 200 real final-fold transactions plus the
best model's expected score for each, per ARCHITECTURE.md §7's golden-file
test: "any change to features, bundle, or model that moves a score fails CI
loudly." tests/test_golden_file.py is the CI side that checks against this.

Deliberately reimplements the minimal read path (cached `features` table +
the same fold split train_pipeline.py uses) rather than importing
train_pipeline.py itself, which has import-time side effects (it opens a
new log file and loads config at module scope) that don't belong in a
generation script run ad hoc from the command line.
"""

import argparse
import json
from pathlib import Path

import duckdb
import joblib
import numpy as np
import pandas as pd

from config import load_config, PROJECT_ROOT
from cv import time_based_folds
from features import FEATURE_COLUMNS


def _latest_run_dir(model_dir: Path) -> Path:
    runs = sorted((d for d in model_dir.iterdir() if d.is_dir() and (d / "metadata.json").exists()),
                  key=lambda d: d.name)
    if not runs:
        raise FileNotFoundError(f"No run directories with metadata.json found under {model_dir}")
    return runs[-1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=None, help="Run id under models/ (default: latest)")
    parser.add_argument("--n", type=int, default=200, help="Number of golden rows")
    parser.add_argument("--output", type=Path,
                         default=PROJECT_ROOT / "tests" / "golden" / "golden_transactions.csv")
    args = parser.parse_args()

    config = load_config()
    db_path = PROJECT_ROOT / config["data"]["processed_dir"] / "paysim.duckdb"

    con = duckdb.connect(str(db_path), read_only=True)
    try:
        feat_df = con.sql("SELECT * FROM features").df().fillna(0)
    finally:
        con.close()

    X = feat_df[FEATURE_COLUMNS]
    y = feat_df["isFraud"].to_numpy()
    step = feat_df["step"].to_numpy()

    folds = time_based_folds(step, config["cv"]["fold_boundaries"])
    test_mask = folds[-1][1]  # final fold, matching what train_pipeline.py's
                               # Sprint 2/3 analysis and the exported bundle
                               # were both derived from.
    X_test = X.loc[test_mask].reset_index(drop=True)
    y_test = y[test_mask]

    model_dir = PROJECT_ROOT / "models"
    run_dir = (model_dir / args.run_id) if args.run_id else _latest_run_dir(model_dir)
    with open(run_dir / "metadata.json") as f:
        metadata = json.load(f)

    model = joblib.load(run_dir / f"{metadata['best_model_artifact']}.joblib")
    scaler = joblib.load(run_dir / "scaler.joblib")

    # Stratified sample: half fraud, half non-fraud where available, so the
    # golden file exercises both classes rather than an almost-all-negative
    # random draw at this fraud rate.
    rng = np.random.default_rng(config["random_state"])
    fraud_idx = np.flatnonzero(y_test == 1)
    nonfraud_idx = np.flatnonzero(y_test == 0)
    n_fraud = min(args.n // 2, len(fraud_idx))
    n_nonfraud = args.n - n_fraud
    sample_idx = np.concatenate([
        rng.choice(fraud_idx, size=n_fraud, replace=False),
        rng.choice(nonfraud_idx, size=n_nonfraud, replace=False),
    ])
    rng.shuffle(sample_idx)

    X_sample = X_test.iloc[sample_idx].reset_index(drop=True)
    y_sample = y_test[sample_idx]

    X_scaled = scaler.transform(X_sample)
    expected_scores = model.predict_proba(X_scaled)[:, 1]

    out_df = X_sample.copy()
    out_df["is_fraud"] = y_sample
    out_df["expected_score"] = expected_scores
    out_df["source_run_id"] = metadata["run_id"]
    out_df["source_model_artifact"] = metadata["best_model_artifact"]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output, index=False)
    print(f"Wrote {len(out_df)} golden rows ({int(y_sample.sum())} fraud) to {args.output}")
    print(f"  Source run: {metadata['run_id']}, model: {metadata['best_model_artifact']}")


if __name__ == "__main__":
    main()
