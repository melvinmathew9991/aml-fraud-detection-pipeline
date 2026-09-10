"""
run_drift.py

The Sprint 8 drift job: computes PSI for every feature and for the model's own
score distribution, over successive time windows of the dataset, and writes the
three artifacts the dashboard's Drift page reads.

    python src/run_drift.py          (or: python tasks.py drift)

**Why the comparison windows are dataset time and not scored traffic.** An
earlier draft of the roadmap specified PSI against rolling live traffic. A
portfolio deployment receives tens of requests; PSI on that sample is noise
dressed as monitoring, and the job would have produced a chart that moved for
reasons having nothing to do with the data. PaySim spans 743 simulated hourly
steps over 6.36M rows, so the honest version of this analysis replays the
dataset's own time axis: reference = the final training fold, comparison =
each successive simulated day. ARCHITECTURE.md §10 records the decision; live
traffic is still surfaced on the Drift page, but as volume and score telemetry
that is labelled as not being a drift signal.

**Windows deliberately start at step 1, not after the reference cut.** The
first ~15 days sit *inside* the reference period, so their PSI is the
detector's noise floor measured on the data the reference was built from
rather than a number anyone has to take on trust. `in_reference_period` marks
them, and the dashboard shades them.

**Both scoring paths that matter are the deployed ones.** Rows are scored with
`model_bundle/v2` through `inference/`, the same code the API serves with, and
flagged against the bundle's own `decision_threshold` -- so `alert_rate` and
`precision_at_deployed_threshold` describe what production would actually have
produced in that window, not what a threshold re-fitted per window would.

Outputs (all under data/processed/, all committed -- the dashboard runs on
Streamlit Community Cloud with no dataset and no DuckDB, so a CSV is the only
interface that can reach it):

    drift_feature_psi.csv   one row per (window, feature)
    drift_score_psi.csv     one row per window: score PSI + operating point
    drift_reference.json    what the reference was, so a stale run is detectable
"""

import json
import logging
from datetime import UTC, datetime

import duckdb
import numpy as np
import pandas as pd

from config import PROJECT_ROOT, git_commit_hash, load_config
from cv import time_based_folds
from features import FEATURE_COLUMNS, FEATURE_VERSION
from inference.bundle import load_bundle
from monitoring.drift import (
    MIN_COMPARISON_ROWS,
    PSI_FEATURE_THRESHOLD,
    PSI_SCORE_THRESHOLD,
    compare,
    compare_score,
    prepare_reference,
)
from threshold import STEPS_PER_DAY, capacity_k, operating_point

CONFIG = load_config()

PROCESSED_DIR = PROJECT_ROOT / CONFIG["data"]["processed_dir"]
REPORTS_DIR = PROJECT_ROOT / "reports"
DB_PATH = PROCESSED_DIR / "paysim.duckdb"
BUNDLE_DIR = PROJECT_ROOT / "model_bundle" / "v2"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_PSI_CSV = PROCESSED_DIR / "drift_feature_psi.csv"
SCORE_PSI_CSV = PROCESSED_DIR / "drift_score_psi.csv"
REFERENCE_JSON = PROCESSED_DIR / "drift_reference.json"

# One simulated day per comparison window. `step` is one simulated hour, so
# this is the same unit threshold.py already uses to convert a window into
# review capacity -- monitoring and the operating point should not disagree
# about how long a day is.
WINDOW_STEPS = STEPS_PER_DAY

# Rows per scoring chunk. The full matrix is 6.36M x 18; `scaler.transform`
# promotes to float64, so scoring it in one call would allocate ~915MB for the
# scaled copy alone on a machine with 8GB total. At 500k rows the transient is
# ~72MB and the run still finishes in about a minute.
SCORE_CHUNK_ROWS = 500_000

RUN_ID = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(REPORTS_DIR / f"drift_{RUN_ID}.log"),
    ],
)
logger = logging.getLogger("run_drift")


def log_memory(stage: str) -> None:
    """Same RSS probe train_pipeline.py uses -- this job holds the full feature
    matrix in memory and the machine it was written on has 8GB."""
    try:
        import psutil
    except ImportError:
        return
    logger.info("  [memory] %-28s %7.0f MB RSS",
                stage, psutil.Process().memory_info().rss / (1024 ** 2))


def load_feature_frame() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Read the materialized `features` table produced by train_pipeline.py.

    This job deliberately does NOT build the table itself. Feature construction
    is a 6.36M-row window-function query owned by the training pipeline, and a
    monitoring job that could rebuild it would be a second place for
    FEATURE_VERSION to be interpreted -- exactly the duplication that
    ARCHITECTURE.md §2 spends its length avoiding. If the table is absent or
    stale, the fix is to run the pipeline, and this says so.
    """
    if not DB_PATH.exists():
        raise SystemExit(
            f"{DB_PATH} does not exist. The drift job reads the feature table the "
            "training pipeline materializes -- run `python tasks.py train` first."
        )

    con = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        duckdb_cfg = CONFIG.get("duckdb", {})
        con.execute(f"SET memory_limit='{duckdb_cfg.get('memory_limit', '2GB')}'")
        con.execute(f"SET threads={duckdb_cfg.get('threads', 2)}")

        table_exists = con.execute(
            "SELECT 1 FROM information_schema.tables WHERE table_name = 'features'"
        ).fetchone() is not None
        if not table_exists:
            raise SystemExit(
                f"No `features` table in {DB_PATH} -- run `python tasks.py train` first."
            )

        cached_version = con.execute(
            "SELECT feature_version FROM _cache_meta"
        ).fetchone()
        if cached_version and cached_version[0] != FEATURE_VERSION:
            raise SystemExit(
                f"The materialized features table was built at FEATURE_VERSION "
                f"{cached_version[0]}, but features.py is now at {FEATURE_VERSION}. "
                "Re-run the training pipeline before measuring drift against it."
            )

        columns = ", ".join(FEATURE_COLUMNS)
        frame = con.sql(f"SELECT {columns}, step, isFraud FROM features").df().fillna(0)
    finally:
        con.close()

    X = frame[FEATURE_COLUMNS].to_numpy(dtype="float32")
    step = frame["step"].to_numpy()
    is_fraud = frame["isFraud"].to_numpy()
    logger.info("Loaded %s rows x %d features (steps %d-%d)",
                f"{len(step):,}", X.shape[1], step.min(), step.max())
    return X, step, is_fraud


def score_all(bundle, X: np.ndarray) -> np.ndarray:
    """Score every row with the deployed bundle, in chunks (see SCORE_CHUNK_ROWS).

    Kept as float32 on the way out: 6.36M float64 scores is 51MB held for the
    whole run to buy precision that PSI's decile bins cannot use.
    """
    scores = np.empty(X.shape[0], dtype="float32")
    for start in range(0, X.shape[0], SCORE_CHUNK_ROWS):
        end = min(start + SCORE_CHUNK_ROWS, X.shape[0])
        chunk = np.asarray(X[start:end], dtype="float64")
        scores[start:end] = bundle.booster.predict(bundle.scaler.transform(chunk))
        logger.info("  scored %s / %s rows", f"{end:,}", f"{X.shape[0]:,}")
    return scores


def reference_mask(step: np.ndarray) -> np.ndarray:
    """The final training fold's rows.

    Derived by calling src/cv.py's own fold builder rather than re-deriving the
    0.8 quantile here. The reference for drift must be the exact population the
    shipped model was fitted on; a second hand-written copy of the fold
    boundary is a second thing to keep in sync, and this repo's recurring defect
    is precisely facts maintained in two places.
    """
    folds = time_based_folds(step, CONFIG["cv"]["fold_boundaries"])
    train_mask, _ = folds[-1]
    return train_mask


def window_bounds(step: np.ndarray) -> list[tuple[int, int]]:
    """Successive [start, end] step ranges of WINDOW_STEPS hours each, covering
    the full dataset. The last window is short wherever the data ends."""
    first, last = int(step.min()), int(step.max())
    return [(s, min(s + WINDOW_STEPS - 1, last))
            for s in range(first, last + 1, WINDOW_STEPS)]


def main() -> None:
    logger.info("=== Drift job %s (git %s) ===", RUN_ID, git_commit_hash())
    log_memory("start")

    bundle = load_bundle(BUNDLE_DIR)
    logger.info("Bundle %s (feature_version %d, model %s), decision_threshold %.6g",
                bundle.bundle_version, bundle.feature_version,
                bundle.bundle_meta["model_name"], bundle.threshold.decision_threshold)

    if bundle.feature_names != FEATURE_COLUMNS:
        raise SystemExit(
            "The bundle's feature order does not match features.py:FEATURE_COLUMNS. "
            "Scoring the feature table with this bundle would silently mismatch "
            "columns -- re-export the bundle."
        )

    X, step, is_fraud = load_feature_frame()
    log_memory("after feature load")

    scores = score_all(bundle, X)
    log_memory("after scoring")

    ref_mask = reference_mask(step)
    ref_steps = step[ref_mask]
    ref_cut = int(ref_steps.max())
    logger.info("Reference = final training fold: steps %d-%d, %s rows (%s fraud)",
                int(ref_steps.min()), ref_cut, f"{int(ref_mask.sum()):,}",
                f"{int(is_fraud[ref_mask].sum()):,}")

    # Prepared once per feature, then reused across every window -- see
    # monitoring/drift.ReferenceBins.
    prepared_features = {
        name: prepare_reference(X[ref_mask, j])
        for j, name in enumerate(FEATURE_COLUMNS)
    }
    prepared_scores = prepare_reference(scores[ref_mask])
    log_memory("after reference binning")

    threshold_value = bundle.threshold.decision_threshold
    reviews_per_day = CONFIG["review_capacity"]["reviews_per_day"]

    feature_rows: list[dict] = []
    score_rows: list[dict] = []

    for index, (start, end) in enumerate(window_bounds(step)):
        in_window = (step >= start) & (step <= end)
        n_rows = int(in_window.sum())
        window_days = (end - start + 1) / STEPS_PER_DAY

        n_breached = 0
        for j, name in enumerate(FEATURE_COLUMNS):
            result = compare(prepared_features[name], X[in_window, j],
                             threshold=PSI_FEATURE_THRESHOLD)
            n_breached += int(result.breached)
            feature_rows.append({
                "window": index,
                "step_start": start,
                "step_end": end,
                "feature": name,
                "psi": result.psi,
                "band": result.band,
                "binning": result.binning,
                "n_bins": result.n_bins,
                "n_rows": result.n_comparison,
                "breached": result.breached,
                "sufficient_sample": result.sufficient_sample,
            })

        # compare_score, not compare(..., threshold=PSI_SCORE_THRESHOLD): the
        # score band belongs to the detector, not to this call site. See
        # monitoring/drift.compare_score.
        score_result = compare_score(prepared_scores, scores[in_window])

        window_scores = scores[in_window]
        window_labels = is_fraud[in_window]
        flagged = window_scores >= threshold_value
        n_flagged = int(flagged.sum())
        n_fraud = int(window_labels.sum())
        true_positives = int(np.sum(flagged & (window_labels == 1)))

        # The retraining trigger's series (MONITORING.md §4). Precision at the
        # FIXED threshold is not comparable across windows here -- it swings
        # 0.06 to 0.96 purely with transaction volume, because a fixed score
        # cutoff does not fix a queue length. Precision at capacity holds the
        # queue at K = reviews_per_day x window_days instead, and daily fraud
        # count is stable across this dataset (216-320 per window), so its
        # ceiling is stable too and the series can be compared window to window.
        # This is the same distinction src/threshold.py exists to make.
        capacity = capacity_k(step[in_window], reviews_per_day, n_rows) if n_rows else 0
        at_capacity = (
            operating_point(window_labels, window_scores, capacity)
            if n_rows and capacity else None
        )
        # The bound is set by the queue actually worked, not by the requested K
        # -- score ties at the threshold can push n_flagged past it. Same
        # reasoning, and the same expression, as threshold.capacity_sweep.
        ceiling = (
            min(1.0, n_fraud / at_capacity["n_flagged"])
            if at_capacity and at_capacity["n_flagged"] else float("nan")
        )

        score_rows.append({
            "window": index,
            "step_start": start,
            "step_end": end,
            "window_days": window_days,
            # The control: windows inside the reference period measure the
            # detector's noise floor on the data the reference was built from.
            "in_reference_period": bool(end <= ref_cut),
            "n_rows": n_rows,
            "n_fraud": n_fraud,
            "fraud_rate": (n_fraud / n_rows) if n_rows else float("nan"),
            "score_psi": score_result.psi,
            "score_band": score_result.band,
            "score_breached": score_result.breached,
            "sufficient_sample": score_result.sufficient_sample,
            "features_breached": n_breached,
            "mean_score": float(window_scores.mean()) if n_rows else float("nan"),
            "p99_score": float(np.quantile(window_scores, 0.99)) if n_rows else float("nan"),
            # What the DEPLOYED threshold would have produced in this window.
            "n_flagged": n_flagged,
            "alert_rate": (n_flagged / n_rows) if n_rows else float("nan"),
            "alerts_per_day": (n_flagged / window_days) if window_days else float("nan"),
            # Capacity for the same window, from config -- so "the fixed
            # threshold produced more alerts than the team can work" is visible
            # as a number rather than left to the reader to divide.
            "capacity": capacity,
            "precision_at_deployed_threshold":
                (true_positives / n_flagged) if n_flagged else float("nan"),
            "recall_at_deployed_threshold":
                (true_positives / n_fraud) if n_fraud else float("nan"),
            "precision_at_capacity":
                at_capacity["precision"] if at_capacity else float("nan"),
            "recall_at_capacity":
                at_capacity["recall"] if at_capacity else float("nan"),
            "precision_ceiling_at_capacity": ceiling,
            # The actual health signal (MONITORING.md §4). Precision at capacity
            # is bounded above by fraud_count / queue_size, so a model that
            # ranks perfectly reports whatever that day's fraud count implies
            # and nothing about itself. Whether it is ON that bound is the
            # model-specific statement: falling off it means fraud has started
            # escaping the queue the team can actually work.
            "at_ceiling": bool(
                at_capacity is not None
                and np.isfinite(at_capacity["precision"])
                and np.isfinite(ceiling)
                and at_capacity["precision"] >= ceiling - 1e-9
            ),
        })

        logger.info(
            "window %2d  steps %3d-%3d  %9s rows  score PSI %7.4f (%-11s)  "
            "features breached %2d/%d%s",
            index, start, end, f"{n_rows:,}", score_result.psi, score_result.band,
            n_breached, len(FEATURE_COLUMNS),
            "" if score_result.sufficient_sample else "  [sample too small]",
        )

    feature_frame = pd.DataFrame(feature_rows)
    score_frame = pd.DataFrame(score_rows)
    feature_frame.to_csv(FEATURE_PSI_CSV, index=False)
    score_frame.to_csv(SCORE_PSI_CSV, index=False)

    reference_manifest = {
        "generated_at": datetime.now(UTC).isoformat(),
        "run_id": RUN_ID,
        "git_commit": git_commit_hash(),
        # The staleness contract. The scheduled check (.github/workflows/
        # monitoring.yml) compares these against what the live service reports
        # from /model-info: if the deployed model moves and this reference does
        # not, every PSI number below describes a model nobody is serving.
        "bundle_version": bundle.bundle_version,
        "feature_version": bundle.feature_version,
        "model_name": bundle.bundle_meta["model_name"],
        "decision_threshold": threshold_value,
        "features": FEATURE_COLUMNS,
        "reference": {
            "definition": "final training fold (src/cv.py, config cv.fold_boundaries)",
            "fold_boundaries": CONFIG["cv"]["fold_boundaries"],
            "step_min": int(ref_steps.min()),
            "step_max": ref_cut,
            "n_rows": int(ref_mask.sum()),
            "n_fraud": int(is_fraud[ref_mask].sum()),
        },
        "windows": {
            "unit": "simulated day",
            "steps_per_window": WINDOW_STEPS,
            "count": len(score_rows),
        },
        "thresholds": {
            "feature_psi": PSI_FEATURE_THRESHOLD,
            "score_psi": PSI_SCORE_THRESHOLD,
            "min_comparison_rows": MIN_COMPARISON_ROWS,
            "bands": {"stable": "< 0.10", "moderate": "0.10-0.25", "significant": ">= 0.25"},
        },
    }
    with open(REFERENCE_JSON, "w") as f:
        json.dump(reference_manifest, f, indent=2)
        f.write("\n")

    measured = score_frame[score_frame["sufficient_sample"]]
    logger.info("--- summary ---")
    logger.info("%d windows, %d with a usable sample (>= %s rows)",
                len(score_frame), len(measured), f"{MIN_COMPARISON_ROWS:,}")
    logger.info("score PSI breached in %d of those windows (threshold %.2f)",
                int(measured["score_breached"].sum()), PSI_SCORE_THRESHOLD)
    breaching_features = feature_frame[feature_frame["breached"]]["feature"]
    if len(breaching_features):
        logger.info("features breaching %.2f in at least one window: %s",
                    PSI_FEATURE_THRESHOLD,
                    ", ".join(f"{name} (x{count})" for name, count
                              in breaching_features.value_counts().items()))
    else:
        logger.info("no feature breached %.2f in any window", PSI_FEATURE_THRESHOLD)
    logger.info("wrote %s, %s, %s",
                FEATURE_PSI_CSV.name, SCORE_PSI_CSV.name, REFERENCE_JSON.name)
    log_memory("end")


if __name__ == "__main__":
    main()
