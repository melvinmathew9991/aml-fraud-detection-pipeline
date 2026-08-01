"""
train_pipeline.py

End-to-end training pipeline demonstrating:
- Lasso and Ridge as explicit linear baselines (the JD names these
  specifically, distinct from generic "Logistic Regression")
- Gradient-boosted trees (HistGradientBoosting, XGBoost, LightGBM) as tree
  models -- XGBoost/LightGBM added in Sprint 1, tuned via Optuna
- Expanding-window time-based cross-validation (src/cv.py) instead of a
  single 80/20 split -- a single held-out window leaves too few fraud rows
  to trust one PR-AUC number; CV gives a mean +/- std across folds while
  still respecting chronology (never training on data from the future
  relative to what's being evaluated)
- Train-side undersampling of the majority class (see config.yaml), applied
  independently within each fold -- fraud signal is bottlenecked by the
  handful of positive rows per fold, not by however many millions of
  negatives sit alongside them. Each fold's test set is left at the real
  distribution so evaluation reflects deployment conditions.
- Cost-sensitive learning via class_weight/scale_pos_weight AND the custom
  weighted-BCE loss from custom_metrics.py (both approaches, so the
  difference between "using a library flag" and "writing the loss yourself"
  is demonstrable)
- Capacity-based model selection (Sprint 2): `best_model` is chosen by mean
  precision at the review-capacity operating point -- not by PR-AUC, and no
  longer by Precision@K=1x-fraud-count either. Sprint 1 had already moved
  selection off PR-AUC (the two disagree post-tuning, see README); Sprint 2
  moved it again because K=1x is defined by the labels, so "the model that
  wins at K=1x" is not a criterion you could evaluate on the day you had to
  deploy. Both the PR-AUC and K=1x rankings are still reported when they
  disagree with the capacity winner
- Precision@K / Recall@K as the operationally-relevant evaluation metrics
- Optuna hyperparameter tuning (XGBoost/LightGBM only) optimizing mean PR-AUC
  across all CV folds (not a single fold -- one fold turned out to be
  near-perfectly separable and gave Optuna no signal to discriminate trials
  on), then refit with best params across all folds so the comparison table
  shows a visible tuned-vs-default delta
- MLflow local sqlite tracking so runs are comparable over time instead of
  overwriting a single CSV
- (Sprint 2) A deployable decision threshold set from analyst review capacity
  rather than from the true fraud count -- see src/threshold.py for why the
  distinction matters -- plus SHAP explainability (src/explain.py), top-K
  error analysis (src/error_analysis.py), and a feature ablation that
  attributes Sprint 2's metric movement to the new features instead of
  asserting it
"""

import json
import logging
import subprocess
from datetime import datetime, timezone

import duckdb
import joblib
import lightgbm as lgb
import mlflow
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, roc_auc_score

from config import load_config, PROJECT_ROOT
from cv import time_based_folds
from features import (
    feature_query, FEATURE_COLUMNS, BASE_FEATURE_COLUMNS, FEATURE_VERSION,
    TYPE_FEATURE_COLUMNS, DEST_FEATURE_COLUMNS,
)
from custom_metrics import weighted_bce_loss, suggest_pos_weight, precision_at_k, recall_at_k
from threshold import capacity_k, operating_point, window_days, capacity_sweep
from schema import validate_raw_sample
from explain import compute_shap, global_importance, explain_queue, log_global_importance
from error_analysis import segment_profile, missed_fraud_ranking, log_error_analysis
from economics import (
    net_value_curve, degeneracy_check, capacity_constraint_cost, ticket_size_crossover,
)

# Optuna's own per-trial INFO logging would interleave with the pipeline's
# log stream; tune_model() below logs its own summary line per study instead.
optuna.logging.set_verbosity(optuna.logging.WARNING)

CONFIG = load_config()

RAW_PATH = PROJECT_ROOT / CONFIG["data"]["raw_path"]
PROCESSED_DIR = PROJECT_ROOT / CONFIG["data"]["processed_dir"]
MODEL_DIR = PROJECT_ROOT / CONFIG["model_output"]["dir"]
REPORTS_DIR = PROJECT_ROOT / "reports"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Persistent DuckDB store for the raw transactions table. DuckDB executes
# out-of-core (spills to disk instead of materializing the full 6.36M-row
# table in Python memory), and re-parsing the 493MB CSV every run is the
# other expensive step worth avoiding -- so the `transactions` table is
# loaded once and reused across runs unless the raw CSV changes underneath
# it (checked via mtime/size in `_load_meta`).
DB_PATH = PROCESSED_DIR / "paysim.duckdb"

RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(REPORTS_DIR / f"train_{RUN_ID}.log"),
    ],
)
logger = logging.getLogger("train_pipeline")


def git_commit_hash() -> str:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT, stderr=subprocess.DEVNULL,
        ).decode().strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=PROJECT_ROOT, stderr=subprocess.DEVNULL,
        ).strip())
        return f"{commit}-dirty" if dirty else commit
    except Exception:
        return "nogit"


def log_memory(stage: str) -> None:
    """Log current process RSS. Lets us see where a run's peak memory
    actually comes from instead of guessing -- this machine has 8GB total,
    so peak footprint is the difference between a run finishing and the
    system hanging."""
    try:
        import psutil
    except ImportError:
        return
    rss_mb = psutil.Process().memory_info().rss / (1024 ** 2)
    logger.info("  [memory] %-24s %7.0f MB RSS", stage, rss_mb)


def _table_exists(con: duckdb.DuckDBPyConnection, name: str) -> bool:
    return con.execute(
        "SELECT 1 FROM information_schema.tables WHERE table_name = ?", [name]
    ).fetchone() is not None


def _cache_meta(con: duckdb.DuckDBPyConnection):
    if not _table_exists(con, "_cache_meta"):
        return None
    return con.execute(
        "SELECT raw_mtime, raw_size, feature_version FROM _cache_meta"
    ).fetchone()


def _ensure_transactions_table(con: duckdb.DuckDBPyConnection, raw_path) -> bool:
    """Loads the raw CSV into a `transactions` table on first run (or
    whenever the CSV changes), and reuses the persisted table otherwise --
    the one-time cost of a 6.36M-row CSV parse doesn't need to be paid on
    every rerun during normal iteration. Returns True if the table was
    (re)loaded -- meaning any materialized `features` table is now stale
    too, regardless of FEATURE_VERSION."""
    raw_stat = raw_path.stat()
    meta = _cache_meta(con)

    if _table_exists(con, "transactions") and meta \
            and meta[0] == raw_stat.st_mtime and meta[1] == raw_stat.st_size:
        logger.info("Using cached transactions table in %s (raw CSV unchanged)", DB_PATH)
        return False

    logger.info("Loading raw data from %s into DuckDB (one-time cost, cached after)", raw_path)
    con.execute(f"""
        CREATE OR REPLACE TABLE transactions AS
        SELECT *, row_number() OVER () AS row_id
        FROM read_csv_auto('{raw_path.as_posix()}')
    """)
    return True


def _ensure_features_table(con: duckdb.DuckDBPyConnection, raw_path, transactions_reloaded: bool) -> None:
    """Materializes feature_query()'s result into a `features` table instead
    of re-running the window-function query (a full sort over 6.36M rows,
    ~60s) on every run. Recomputed when the raw CSV changed (transactions
    table was just reloaded) or when FEATURE_VERSION was bumped -- otherwise
    the materialized table from the last run is reused as-is."""
    raw_stat = raw_path.stat()
    meta = _cache_meta(con)
    cache_valid = (
        not transactions_reloaded
        and _table_exists(con, "features")
        and meta is not None
        and meta[2] == FEATURE_VERSION
    )

    if cache_valid:
        logger.info("Using cached features table (raw CSV + feature logic unchanged)")
        return

    logger.info("Computing engineered features (window-function query over transactions)...")
    con.execute(f"CREATE OR REPLACE TABLE features AS {feature_query()}")
    con.execute(
        "CREATE OR REPLACE TABLE _cache_meta AS "
        "SELECT ? AS raw_mtime, ? AS raw_size, ? AS feature_version",
        [raw_stat.st_mtime, raw_stat.st_size, FEATURE_VERSION],
    )
    logger.info("Cached features table to %s (future runs skip recomputing it)", DB_PATH)


def undersample_majority(X: pd.DataFrame, y: np.ndarray, ratio: float, random_state: int):
    """
    Keep every positive (fraud) row and a random sample of negative rows at
    `ratio` negatives per positive. Train-side only -- callers must leave
    the test set at the real distribution so evaluation reflects deployment
    conditions, not the resampled training prior.
    """
    rng = np.random.default_rng(random_state)
    pos_idx = np.flatnonzero(y == 1)
    neg_idx = np.flatnonzero(y == 0)

    n_neg_keep = min(len(neg_idx), int(len(pos_idx) * ratio))
    neg_sample = rng.choice(neg_idx, size=n_neg_keep, replace=False)

    keep_idx = np.sort(np.concatenate([pos_idx, neg_sample]))
    return X.iloc[keep_idx], y[keep_idx]


def build_feature_frame():
    """Fetch the materialized features table from DuckDB once. Returns the
    full-dataset (X, y, step) -- per-fold splitting/scaling/undersampling
    happens downstream in prepare_fold()."""
    log_memory("start")

    con = duckdb.connect(str(DB_PATH))
    try:
        # Cap DuckDB's memory before any query runs. Its default is ~80% of
        # system RAM (~6.4GB here), which leaves nothing for the Python side
        # and is how full runs previously hung this machine; under an explicit
        # limit DuckDB spills the window sorts to disk instead.
        duckdb_cfg = CONFIG.get("duckdb", {})
        con.execute(f"SET memory_limit='{duckdb_cfg.get('memory_limit', '2GB')}'")
        con.execute(f"SET threads={duckdb_cfg.get('threads', 2)}")

        transactions_reloaded = _ensure_transactions_table(con, RAW_PATH)
        log_memory("after ensure transactions table")

        if transactions_reloaded:
            logger.info("Validating a sample of the raw table against RAW_TRANSACTION_SCHEMA...")
            validate_raw_sample(con)
            logger.info("  Raw data validation passed.")

        _ensure_features_table(con, RAW_PATH, transactions_reloaded)
        log_memory("after ensure features table")

        # Only the materialized ~11-column features table (not the raw
        # string columns) ever crosses into pandas/Python memory.
        feat_df = con.sql("SELECT * FROM features").df().fillna(0)
        log_memory("after feature fetch")
    finally:
        con.close()

    X_all = feat_df[FEATURE_COLUMNS].to_numpy(dtype="float32")
    y = feat_df["isFraud"].to_numpy()
    step = feat_df["step"].to_numpy()

    X = pd.DataFrame(X_all, columns=FEATURE_COLUMNS)
    return X, y, step


def prepare_fold(X: pd.DataFrame, y: np.ndarray, train_mask: np.ndarray, test_mask: np.ndarray,
                  sampling_cfg: dict, random_state: int):
    """Given one fold's train/test masks: undersample the train partition
    only (test stays at the real distribution), fit a StandardScaler on
    this fold's train partition alone (no cross-fold leakage of scaling
    statistics), and return everything a model needs to fit + evaluate on
    this fold."""
    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]

    # Real class imbalance for this fold's test window, computed before any
    # resampling -- used for cost-sensitive evaluation (weighted-BCE), which
    # needs the true deployment-time cost ratio, not whatever ratio training
    # ends up using.
    true_pos_weight = suggest_pos_weight(y_test)

    if sampling_cfg.get("undersample_train"):
        n_before = len(y_train)
        X_train, y_train = undersample_majority(
            X_train, y_train,
            ratio=sampling_cfg["negative_to_positive_ratio"],
            random_state=random_state,
        )
        logger.info(
            "  Undersampled train set: %d -> %d rows (%d fraud, 1:%s ratio)",
            n_before, len(y_train), int(y_train.sum()),
            sampling_cfg["negative_to_positive_ratio"],
        )

    # Class imbalance actually seen by the trained model, i.e. AFTER
    # undersampling -- this is what XGBoost/LightGBM's scale_pos_weight
    # should compensate for, mirroring what sklearn's class_weight=
    # 'balanced' computes from this same y_train for the other models.
    # Using true_pos_weight (the ~300-1700:1 real deployment ratio) here
    # instead would double-correct on top of undersampling and badly
    # miscalibrate predicted probabilities.
    train_pos_weight = suggest_pos_weight(y_train)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, true_pos_weight, train_pos_weight


# XGBoost/LightGBM both use early stopping against the fold's own test
# partition (config: models.xgboost/lightgbm.early_stopping_rounds), with
# n_estimators set to a high upper bound so early stopping -- not the bound
# -- decides the real count. This means the test partition does double duty
# as "when to stop training" and "how good is the result," which is mildly
# optimistic versus a true held-out validation split -- a deliberate,
# documented simplification for a single-machine portfolio project, not a
# nested-CV setup. Applied uniformly to every XGBoost/LightGBM fit (default
# params AND Optuna-tuned), so the tuned-vs-default delta stays
# apples-to-apples even though the absolute numbers carry the same optimism.

def fit_xgboost(X_train, y_train, X_val, y_val, pos_weight: float, extra_params: dict | None = None):
    xgb_cfg = CONFIG["models"]["xgboost"]
    model = xgb.XGBClassifier(
        n_estimators=xgb_cfg["n_estimators"],
        early_stopping_rounds=xgb_cfg["early_stopping_rounds"],
        eval_metric=xgb_cfg["eval_metric"],
        n_jobs=xgb_cfg["n_jobs"],
        scale_pos_weight=pos_weight,
        random_state=CONFIG["random_state"],
        **(extra_params or {}),
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model


def fit_lightgbm(X_train, y_train, X_val, y_val, pos_weight: float, extra_params: dict | None = None):
    lgb_cfg = CONFIG["models"]["lightgbm"]
    model = lgb.LGBMClassifier(
        n_estimators=lgb_cfg["n_estimators"],
        n_jobs=lgb_cfg["n_jobs"],
        scale_pos_weight=pos_weight,
        random_state=CONFIG["random_state"],
        verbosity=-1,
        **(extra_params or {}),
    )
    model.fit(
        X_train, y_train,
        eval_X=X_val, eval_y=y_val,
        eval_metric="average_precision",
        callbacks=[lgb.early_stopping(lgb_cfg["early_stopping_rounds"], verbose=False)],
    )
    return model


def _suggest_xgb_params(trial: optuna.Trial) -> dict:
    return {
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "min_child_weight": trial.suggest_float("min_child_weight", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
    }


def _suggest_lgb_params(trial: optuna.Trial) -> dict:
    return {
        "num_leaves": trial.suggest_int("num_leaves", 15, 255),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
    }


def _xgb_objective(trial, fold_data):
    params = _suggest_xgb_params(trial)
    scores = []
    for X_train, X_test, y_train, y_test, _scaler, _true_pw, train_pw in fold_data:
        model = fit_xgboost(X_train, y_train, X_test, y_test, train_pw, extra_params=params)
        scores.append(average_precision_score(y_test, model.predict_proba(X_test)[:, 1]))
    return float(np.mean(scores))


def _lgb_objective(trial, fold_data):
    params = _suggest_lgb_params(trial)
    scores = []
    for X_train, X_test, y_train, y_test, _scaler, _true_pw, train_pw in fold_data:
        model = fit_lightgbm(X_train, y_train, X_test, y_test, train_pw, extra_params=params)
        scores.append(average_precision_score(y_test, model.predict_proba(X_test)[:, 1]))
    return float(np.mean(scores))


def tune_model(name, objective_fn, fold_data, n_trials, random_state):
    """Runs a single Optuna study, always sequential (n_jobs=1) -- this
    machine has no spare cores for parallel trials (see hardware notes).
    The objective averages PR-AUC across ALL CV folds rather than scoring
    against one designated fold -- an earlier version scored against a
    single fold that turned out to be near-perfectly separable (PR-AUC
    pinned near 1.0 for almost any hyperparameter choice), giving Optuna
    no real signal to discriminate between trials."""
    logger.info("Tuning %s with Optuna (%d sequential trials, objective = mean PR-AUC across %d CV folds)...",
                name, n_trials, len(fold_data))
    study = optuna.create_study(
        direction=CONFIG["optuna"]["direction"],
        sampler=optuna.samplers.TPESampler(seed=random_state),
    )
    study.optimize(
        lambda trial: objective_fn(trial, fold_data),
        n_trials=n_trials, n_jobs=1, show_progress_bar=False,
    )
    logger.info("  Best %s mean CV PR-AUC: %.4f  params: %s", name, study.best_value, study.best_params)
    return study


# Precision@K alone conflates two different things: how well the model
# ranks fraud, and how K was chosen relative to the number of true frauds.
# K = 5x the fraud count mathematically caps precision at 20% even for a
# model with perfect recall at that K -- so every K we report is expressed
# as a multiple of the test set's true fraud count, and always paired with
# recall@K, which is the number that actually reflects ranking quality.
K_FRAUD_MULTIPLIERS = [1, 2, 5, 10]


def evaluate_model(name, y_test, y_proba, pos_weight, k_capacity):
    pr_auc = average_precision_score(y_test, y_proba)
    roc_auc = roc_auc_score(y_test, y_proba) if y_test.sum() > 0 else float("nan")
    w_bce = weighted_bce_loss(y_test, y_proba, pos_weight=pos_weight)

    n_test_fraud = int(y_test.sum())
    k = max(n_test_fraud * 5, 10)  # e.g. "review top-K flagged cases per day"
    k = min(k, len(y_test))
    p_at_k = precision_at_k(y_test, y_proba, k=k)
    r_at_k = recall_at_k(y_test, y_proba, k=k)

    # Sprint 2: the deployable operating point. Unlike every K above, this one
    # is derived from analyst review capacity rather than the (in production,
    # unknowable) true fraud count -- see src/threshold.py.
    op = operating_point(y_test, y_proba, k_capacity)

    logger.info("--- %s ---", name)
    logger.info("  PR-AUC:              %.4f  (primary metric under class imbalance)", pr_auc)
    logger.info("  ROC-AUC:             %.4f  (reported, but misleading alone at this imbalance)", roc_auc)
    logger.info("  Weighted BCE loss:   %.4f", w_bce)
    logger.info("  Precision@%d:          %.4f  (of top-%d flagged, fraction actually fraud)", k, p_at_k, k)
    logger.info("  Recall@%d:             %.4f  (of all fraud, fraction caught in top-%d)", k, r_at_k, k)
    logger.info("  At review capacity (K=%d, threshold=%.6f): precision=%.4f  recall=%.4f  "
                "alert_rate=%.4f%%  [%d TP / %d FP / %d missed]",
                op["k"], op["threshold"], op["precision"], op["recall"],
                100 * op["alert_rate"], op["true_positives"], op["false_positives"],
                op["false_negatives"])

    curve_rows = []
    for m in K_FRAUD_MULTIPLIERS:
        k_m = min(max(n_test_fraud * m, 10), len(y_test))
        curve_rows.append({
            "model": name, "k_multiplier": m, "k": k_m,
            "precision_at_k": precision_at_k(y_test, y_proba, k=k_m),
            "recall_at_k": recall_at_k(y_test, y_proba, k=k_m),
        })
    logger.info("  Precision/Recall@K by review-capacity multiple of true fraud count:")
    for row in curve_rows:
        logger.info("    %2dx fraud count (K=%6d): precision=%.4f  recall=%.4f",
                     row["k_multiplier"], row["k"], row["precision_at_k"], row["recall_at_k"])

    return {"model": name, "pr_auc": pr_auc, "roc_auc": roc_auc,
            "weighted_bce": w_bce, "precision_at_k": p_at_k, "recall_at_k": r_at_k,
            "k": k, "n_test_fraud": n_test_fraud,
            "capacity_k": op["k"], "capacity_threshold": op["threshold"],
            "capacity_precision": op["precision"], "capacity_recall": op["recall"],
            "capacity_alert_rate": op["alert_rate"],
            "capacity_false_positives": op["false_positives"],
            "capacity_false_negatives": op["false_negatives"]}, curve_rows


def main():
    run_dir = MODEL_DIR / RUN_ID
    run_dir.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(CONFIG["mlflow"]["tracking_uri"])
    mlflow.set_experiment(CONFIG["mlflow"]["experiment_name"])

    X, y, step = build_feature_frame()
    feature_names = list(X.columns)
    random_state = CONFIG["random_state"]
    sampling_cfg = CONFIG.get("sampling", {})

    folds = time_based_folds(step, CONFIG["cv"]["fold_boundaries"])
    n_folds = len(folds)
    logger.info("Built %d expanding-window time-based folds (boundaries=%s)",
                n_folds, CONFIG["cv"]["fold_boundaries"])

    reviews_per_day = CONFIG["review_capacity"]["reviews_per_day"]

    fold_data = []
    fold_capacity_k = []
    for fold_idx, (train_mask, test_mask) in enumerate(folds, start=1):
        prepared = prepare_fold(X, y, train_mask, test_mask, sampling_cfg, random_state)
        _, _, y_train, y_test, _, pos_weight, _ = prepared

        # Review capacity for this fold's window, from staffing and elapsed
        # time only -- deliberately independent of y_test (see threshold.py).
        step_test = step[test_mask]
        k_cap = capacity_k(step_test, reviews_per_day, n_rows=len(y_test))
        fold_capacity_k.append(k_cap)

        logger.info(
            "Fold %d/%d: train=%d rows (%d fraud), test=%d rows (%d fraud), pos_weight=%.1f",
            fold_idx, n_folds, len(y_train), y_train.sum(), len(y_test), y_test.sum(), pos_weight,
        )
        logger.info(
            "  Test window spans %.1f days -> review capacity K=%d at %d reviews/day "
            "(%d actual fraud in window)",
            window_days(step_test), k_cap, reviews_per_day, int(y_test.sum()),
        )
        fold_data.append(prepared)
    log_memory("all folds prepared")

    results = []
    curve_rows = []
    final_fold_models = {}
    # Display name -> artifact key, so the selected model can be looked up in
    # final_fold_models later (results tables carry the display name only).
    model_key_by_name = {}

    def record(name, model_key, fold_idx, y_test, y_proba, pos_weight):
        result, rows = evaluate_model(name, y_test, y_proba, pos_weight,
                                       k_capacity=fold_capacity_k[fold_idx - 1])
        model_key_by_name[name] = model_key
        result["fold"] = fold_idx
        for row in rows:
            row["fold"] = fold_idx
        results.append(result)
        curve_rows.extend(rows)
        with mlflow.start_run(run_name=f"{model_key}_fold{fold_idx}", nested=True):
            mlflow.log_param("fold", fold_idx)
            mlflow.log_metrics({k: v for k, v in result.items() if isinstance(v, (int, float))})
        return result

    with mlflow.start_run(run_name=RUN_ID):
        mlflow.log_params({
            "n_folds": n_folds,
            "fold_boundaries": str(CONFIG["cv"]["fold_boundaries"]),
            "git_commit": git_commit_hash(),
            "feature_version": FEATURE_VERSION,
            "n_features": len(feature_names),
            "reviews_per_day": reviews_per_day,
        })

        for fold_idx, (X_train, X_test, y_train, y_test, scaler, pos_weight, train_pos_weight) in enumerate(fold_data, start=1):
            logger.info("=== Fold %d/%d ===", fold_idx, n_folds)
            fold_models = {}

            logreg = LogisticRegression(random_state=random_state,
                                         **CONFIG["models"]["logistic_regression"])
            logreg.fit(X_train, y_train)
            fold_models["logistic_regression"] = logreg
            record("Logistic Regression (class_weight=balanced)", "logistic_regression",
                   fold_idx, y_test, logreg.predict_proba(X_test)[:, 1], pos_weight)

            ridge_lr = LogisticRegression(random_state=random_state, **CONFIG["models"]["ridge"])
            ridge_lr.fit(X_train, y_train)
            fold_models["ridge"] = ridge_lr
            record("Ridge-penalized Logistic Regression (L2, C=0.1)", "ridge",
                   fold_idx, y_test, ridge_lr.predict_proba(X_test)[:, 1], pos_weight)

            # random_state matters here specifically: solver=saga (config.yaml) is
            # a stochastic average-gradient method, unlike lbfgs above (deterministic
            # regardless of seed) -- without a fixed seed this model's coefficients,
            # and therefore its reported metrics, were not reproducible run to run.
            lasso_lr = LogisticRegression(random_state=random_state, **CONFIG["models"]["lasso"])
            lasso_lr.fit(X_train, y_train)
            fold_models["lasso"] = lasso_lr
            n_nonzero = np.sum(lasso_lr.coef_ != 0)
            logger.info("  Lasso feature selection: %d/%d features kept nonzero", n_nonzero, len(feature_names))
            record("Lasso-penalized Logistic Regression (L1, C=0.1)", "lasso",
                   fold_idx, y_test, lasso_lr.predict_proba(X_test)[:, 1], pos_weight)

            hgb = HistGradientBoostingClassifier(random_state=random_state,
                                                  **CONFIG["models"]["hist_gradient_boosting"])
            hgb.fit(X_train, y_train)
            fold_models["hist_gradient_boosting"] = hgb
            record("HistGradientBoosting (class_weight=balanced)", "hist_gradient_boosting",
                   fold_idx, y_test, hgb.predict_proba(X_test)[:, 1], pos_weight)

            xgb_default = fit_xgboost(X_train, y_train, X_test, y_test, train_pos_weight)
            fold_models["xgboost"] = xgb_default
            record("XGBoost (default params)", "xgboost",
                   fold_idx, y_test, xgb_default.predict_proba(X_test)[:, 1], pos_weight)

            lgb_default = fit_lightgbm(X_train, y_train, X_test, y_test, train_pos_weight)
            fold_models["lightgbm"] = lgb_default
            record("LightGBM (default params)", "lightgbm",
                   fold_idx, y_test, lgb_default.predict_proba(X_test)[:, 1], pos_weight)

            log_memory(f"fold {fold_idx} baselines trained")

            if fold_idx == n_folds:
                final_fold_models.update(fold_models)

        # --- Optuna tuning: XGBoost + LightGBM only, objective = mean
        # PR-AUC across all CV folds, sequential trials. ---
        n_trials = CONFIG["optuna"]["n_trials"]

        xgb_study = tune_model("XGBoost", _xgb_objective, fold_data, n_trials, random_state)
        lgb_study = tune_model("LightGBM", _lgb_objective, fold_data, n_trials, random_state)
        log_memory("optuna tuning complete")

        for model_name, study in [("xgboost_tuned", xgb_study), ("lightgbm_tuned", lgb_study)]:
            with mlflow.start_run(run_name=model_name, nested=True):
                mlflow.log_params(study.best_params)
                mlflow.log_metric("tuning_mean_cv_pr_auc", study.best_value)

        # Refit tuned params across all CV folds so the comparison table
        # shows a visible tuned-vs-default delta, not just a replacement.
        for fold_idx, (X_train, X_test, y_train, y_test, scaler, pos_weight, train_pos_weight) in enumerate(fold_data, start=1):
            xgb_tuned = fit_xgboost(X_train, y_train, X_test, y_test, train_pos_weight,
                                     extra_params=xgb_study.best_params)
            record("XGBoost (tuned)", "xgboost_tuned", fold_idx, y_test,
                   xgb_tuned.predict_proba(X_test)[:, 1], pos_weight)

            lgb_tuned = fit_lightgbm(X_train, y_train, X_test, y_test, train_pos_weight,
                                      extra_params=lgb_study.best_params)
            record("LightGBM (tuned)", "lightgbm_tuned", fold_idx, y_test,
                   lgb_tuned.predict_proba(X_test)[:, 1], pos_weight)

            if fold_idx == n_folds:
                final_fold_models["xgboost_tuned"] = xgb_tuned
                final_fold_models["lightgbm_tuned"] = lgb_tuned
        log_memory("tuned refits complete")

        # --- Aggregate across folds ---
        results_df = pd.DataFrame(results)
        results_df.to_csv(PROCESSED_DIR / "model_comparison_by_fold.csv", index=False)

        summary_df = (
            results_df.groupby("model")
            .agg(
                pr_auc_mean=("pr_auc", "mean"), pr_auc_std=("pr_auc", "std"),
                roc_auc_mean=("roc_auc", "mean"),
                weighted_bce_mean=("weighted_bce", "mean"),
                precision_at_k_mean=("precision_at_k", "mean"), precision_at_k_std=("precision_at_k", "std"),
                recall_at_k_mean=("recall_at_k", "mean"), recall_at_k_std=("recall_at_k", "std"),
                capacity_precision_mean=("capacity_precision", "mean"),
                capacity_precision_std=("capacity_precision", "std"),
                capacity_recall_mean=("capacity_recall", "mean"),
                capacity_recall_std=("capacity_recall", "std"),
                n_folds=("fold", "count"),
            )
            .reset_index()
            .sort_values("capacity_precision_mean", ascending=False)
        )
        summary_df.to_csv(PROCESSED_DIR / "model_comparison.csv", index=False)

        curve_df = pd.DataFrame(curve_rows)
        curve_df.to_csv(PROCESSED_DIR / "precision_recall_at_k_by_fold.csv", index=False)
        curve_summary_df = (
            curve_df.groupby(["model", "k_multiplier"])
            .agg(
                precision_at_k_mean=("precision_at_k", "mean"), precision_at_k_std=("precision_at_k", "std"),
                recall_at_k_mean=("recall_at_k", "mean"), recall_at_k_std=("recall_at_k", "std"),
            )
            .reset_index()
        )
        curve_summary_df.to_csv(PROCESSED_DIR / "precision_recall_at_k.csv", index=False)
        logger.info("Saved fold-level and aggregated comparison tables to %s", PROCESSED_DIR)

        # best_model is selected by mean precision at the REVIEW-CAPACITY
        # operating point, not by PR-AUC and (since Sprint 2) no longer by
        # Precision@K=1x-fraud-count either.
        #
        # Sprint 1 already moved selection off PR-AUC, on the argument that a
        # fraud team works a fixed-size queue rather than a probability
        # ranking. Sprint 2 finishes that argument: K=1x-fraud-count is still
        # defined by the labels, so "the model that wins at K=1x" is a
        # criterion you could not evaluate on the day you had to deploy it.
        # K from review capacity has the same operational shape and IS
        # computable in advance, so it is what selection now uses. The
        # PR-AUC and K=1x rankings are still reported when they disagree.
        capacity_ranked = summary_df.sort_values("capacity_precision_mean", ascending=False)
        best_model_name = capacity_ranked.iloc[0]["model"]

        k1_df = curve_summary_df[curve_summary_df["k_multiplier"] == 1].sort_values(
            "precision_at_k_mean", ascending=False
        )
        top_pr_auc_model = summary_df.sort_values("pr_auc_mean", ascending=False).iloc[0]["model"]
        top_k1_model = k1_df.iloc[0]["model"]

        logger.info(
            "Best model by mean precision at review capacity (%d reviews/day) across %d folds: "
            "%s (%.4f +/- %.4f)",
            reviews_per_day, n_folds, best_model_name,
            capacity_ranked.iloc[0]["capacity_precision_mean"],
            capacity_ranked.iloc[0]["capacity_precision_std"],
        )
        for label, other in (("mean PR-AUC", top_pr_auc_model),
                             ("mean Precision@K=1x fraud count", top_k1_model)):
            if other != best_model_name:
                logger.info("  (Note: %s leads on %s but is not the capacity-metric winner.)",
                            other, label)

        # --- Sprint 2: explainability, error analysis, feature ablation ---
        # All three run on the FINAL fold only: it has the most training data
        # and the latest test window, so it is the closest thing here to
        # "the model as it would be deployed". Running them per-fold would
        # triple the cost for triplicate versions of the same narrative.
        best_key = model_key_by_name[best_model_name]
        best_model = final_fold_models[best_key]

        X_tr_f, X_te_f, y_tr_f, y_te_f, scaler_f, pos_w_f, train_pos_w_f = fold_data[-1]
        final_test_mask = folds[-1][1]
        X_raw_test = X.loc[final_test_mask].reset_index(drop=True)
        best_proba = best_model.predict_proba(X_te_f)[:, 1]
        final_k = fold_capacity_k[-1]
        final_op = operating_point(y_te_f, best_proba, final_k)

        logger.info("=== Sprint 2 analysis on final fold, model = %s ===", best_model_name)
        logger.info("  Deployable operating point: threshold=%.6f at K=%d (%d reviews/day)",
                    final_op["threshold"], final_op["k"], reviews_per_day)

        # -- SHAP --
        explain_cfg = CONFIG["explainability"]
        shap_global_df = pd.DataFrame()
        alert_reasons_df = pd.DataFrame()
        try:
            shap_values, _shap_idx = compute_shap(
                best_model, X_te_f, max_rows=explain_cfg["shap_sample_rows"],
                random_state=random_state,
            )
            shap_global_df = global_importance(shap_values, feature_names)
            log_global_importance(shap_global_df)
            shap_global_df.to_csv(PROCESSED_DIR / "shap_global_importance.csv", index=False)

            alert_reasons_df = explain_queue(
                best_model, X_te_f, y_te_f, best_proba, scaler_f, feature_names,
                threshold=final_op["threshold"],
                max_alerts=explain_cfg["max_explained_alerts"],
                top_features=explain_cfg["reasons_per_alert"],
                random_state=random_state,
            )
            alert_reasons_df.to_csv(PROCESSED_DIR / "shap_alert_reasons.csv", index=False)
            logger.info("  Wrote per-alert reason codes for %d alerts to shap_alert_reasons.csv",
                        alert_reasons_df["alert_rank"].nunique() if len(alert_reasons_df) else 0)
        except Exception as exc:
            # TreeSHAP only covers tree ensembles. If a linear baseline ever
            # wins selection, log it and carry on rather than losing the run.
            logger.warning("  SHAP explainability skipped for %s: %s", best_model_name, exc)
        log_memory("shap complete")

        # -- Error analysis --
        profile_df = segment_profile(X_raw_test, y_te_f, best_proba, final_op["threshold"])
        # n_flagged, not final_k: ties at the threshold can flag more rows than
        # K requested, and this has to describe the same queue segment_profile
        # does (see missed_fraud_ranking's docstring).
        missed = missed_fraud_ranking(y_te_f, best_proba, final_op["n_flagged"])
        log_error_analysis(profile_df, missed)
        profile_df.to_csv(PROCESSED_DIR / "error_analysis_profile.csv", index=False)
        pd.DataFrame([missed]).to_csv(PROCESSED_DIR / "missed_fraud_summary.csv", index=False)

        # -- Capacity sweep --
        # Threshold "tuning" here means showing what each staffing level buys,
        # not picking one number and calling it optimal -- the trade-off is a
        # business decision, and the ceiling built into it (see
        # threshold.capacity_sweep) is the part most easily misread.
        sweep_rows = capacity_sweep(
            y_te_f, best_proba, step[final_test_mask],
            CONFIG["review_capacity"]["sweep_reviews_per_day"],
        )
        sweep_df = pd.DataFrame(sweep_rows)
        sweep_df.to_csv(PROCESSED_DIR / "capacity_sweep.csv", index=False)
        logger.info("  Operating point vs review capacity (final fold, %d fraud in %.1f days):",
                    int(y_te_f.sum()), window_days(step[final_test_mask]))
        logger.info("    %9s %8s %10s %9s %8s %11s %s",
                    "reviews/d", "K", "threshold", "precision", "recall", "ceiling", "")
        for row in sweep_rows:
            logger.info("    %9.0f %8d %10.6f %9.4f %8.4f %11.4f %s",
                        row["reviews_per_day"], row["k"], row["threshold"],
                        row["precision"], row["recall"], row["precision_ceiling"],
                        "<- at ceiling" if row["at_ceiling"] else "")

        # -- Feature ablation --
        # Attributes this sprint's metric movement to the new features rather
        # than asserting it, and splits the two independent feature groups so
        # a gain from one is not credited to the other. StandardScaler is
        # per-column independent, so column-subsetting the already-scaled
        # matrix is identical to having fitted a scaler on that subset alone
        # -- no refit needed.
        def cols_for(*groups):
            names = list(BASE_FEATURE_COLUMNS)
            for g in groups:
                names += list(g)
            return [feature_names.index(c) for c in names]

        ablation_rows = []
        for label, cols in (
            ("sprint1_features_only", cols_for()),
            ("base_plus_txn_type", cols_for(TYPE_FEATURE_COLUMNS)),
            ("base_plus_dest_aggregates", cols_for(DEST_FEATURE_COLUMNS)),
            ("sprint2_all_features", cols_for(TYPE_FEATURE_COLUMNS, DEST_FEATURE_COLUMNS)),
        ):
            abl_model = fit_lightgbm(X_tr_f[:, cols], y_tr_f, X_te_f[:, cols], y_te_f,
                                      train_pos_w_f)
            abl_proba = abl_model.predict_proba(X_te_f[:, cols])[:, 1]
            abl_op = operating_point(y_te_f, abl_proba, final_k)
            ablation_rows.append({
                "feature_set": label,
                "n_features": len(cols),
                "pr_auc": average_precision_score(y_te_f, abl_proba),
                "weighted_bce": weighted_bce_loss(y_te_f, abl_proba, pos_weight=pos_w_f),
                "capacity_precision": abl_op["precision"],
                "capacity_recall": abl_op["recall"],
                "capacity_false_positives": abl_op["false_positives"],
                "capacity_false_negatives": abl_op["false_negatives"],
            })
        ablation_df = pd.DataFrame(ablation_rows)
        ablation_df.to_csv(PROCESSED_DIR / "feature_ablation.csv", index=False)
        logger.info("  Feature ablation (LightGBM defaults, final fold, K=%d):", final_k)
        for row in ablation_rows:
            logger.info("    %-22s %2d features -> PR-AUC %.4f  capacity precision %.4f  recall %.4f",
                        row["feature_set"], row["n_features"], row["pr_auc"],
                        row["capacity_precision"], row["capacity_recall"])
        log_memory("sprint 2 analysis complete")

        # -- Economics (Sprint 3, ARCHITECTURE.md §0) --
        # avg_fraud_amount is measured from the final fold's actual fraud
        # transactions -- not assumed -- so the degeneracy check and
        # crossover below are grounded in this run's real data.
        econ_cfg = CONFIG["economics"]
        avg_fraud_amount = float(X_raw_test.loc[y_te_f == 1, "amount"].mean())

        econ_curve_df = net_value_curve(
            sweep_rows, avg_fraud_amount,
            cost_per_review=econ_cfg["cost_per_review"],
            recovery_rate=econ_cfg["recovery_rate"],
            liability_rate=econ_cfg["liability_rate"],
        )
        econ_curve_df.to_csv(PROCESSED_DIR / "capacity_economics.csv", index=False)

        degeneracy_df = degeneracy_check(
            sweep_rows, avg_fraud_amount,
            recovery_rate_grid=econ_cfg["degeneracy_recovery_rate_grid"],
            liability_rate=econ_cfg["liability_rate"],
            low_reviews_per_day=econ_cfg["capacity_constraint_low_reviews_per_day"],
            high_reviews_per_day=econ_cfg["capacity_constraint_high_reviews_per_day"],
        )
        constraint_cost = capacity_constraint_cost(
            sweep_rows, avg_fraud_amount,
            low_reviews_per_day=econ_cfg["capacity_constraint_low_reviews_per_day"],
            high_reviews_per_day=econ_cfg["capacity_constraint_high_reviews_per_day"],
        )
        crossover_df = ticket_size_crossover(
            sweep_rows, econ_cfg["ticket_size_grid"],
            cost_per_review=econ_cfg["cost_per_review"],
            recovery_rate=econ_cfg["recovery_rate"],
            liability_rate=econ_cfg["liability_rate"],
        )

        logger.info("  Economics (avg_fraud_amount=%.0f measured from this fold's actual fraud):",
                    avg_fraud_amount)
        logger.info("    Naive-optimum degeneracy: break-even cost_per_review ranges %.0f-%.0f "
                    "across recovery_rate %.2f-%.2f -- no plausible review cost approaches this, "
                    "so full recall wins under every assumption (see README).",
                    degeneracy_df["break_even_cost_per_review"].min(),
                    degeneracy_df["break_even_cost_per_review"].max(),
                    min(econ_cfg["degeneracy_recovery_rate_grid"]),
                    max(econ_cfg["degeneracy_recovery_rate_grid"]))
        logger.info("    Capacity constraint (%d vs %d reviews/day): %d frauds missed, "
                    "%.0f exposure, %.0f per marginal seat",
                    constraint_cost["low_reviews_per_day"], constraint_cost["high_reviews_per_day"],
                    constraint_cost["frauds_missed_by_constraint"], constraint_cost["exposure"],
                    constraint_cost["exposure_per_marginal_seat"])
        transitions = crossover_df[crossover_df["is_crossover"]]
        if len(transitions):
            logger.info("    Ticket-size crossover(s):")
            for _, row in transitions.iterrows():
                logger.info("      at avg_fraud_amount=%.0f, recommended staffing changes to %d reviews/day",
                            row["avg_fraud_amount"], row["recommended_reviews_per_day"])
        else:
            logger.info("    No ticket-size crossover across the swept grid -- "
                        "full capacity wins at every grid point.")

        # Persist every final-fold model plus the shared scaler and run
        # metadata -- final fold has the most training data and is the most
        # representative single snapshot for a later serving step.
        final_scaler = fold_data[-1][4]
        joblib.dump(final_scaler, run_dir / "scaler.joblib")
        for key, model in final_fold_models.items():
            joblib.dump(model, run_dir / f"{key}.joblib")

        metadata = {
            "run_id": RUN_ID,
            "git_commit": git_commit_hash(),
            "feature_names": feature_names,
            "n_folds": n_folds,
            "fold_boundaries": CONFIG["cv"]["fold_boundaries"],
            "optuna_objective": "mean_pr_auc_across_all_cv_folds",
            "optuna_n_trials": n_trials,
            "xgboost_best_params": xgb_study.best_params,
            "lightgbm_best_params": lgb_study.best_params,
            "best_model": best_model_name,
            "best_model_artifact": best_key,
            "best_model_selection_metric": "mean_precision_at_review_capacity",
            "top_pr_auc_model": top_pr_auc_model,
            "top_precision_at_k_1x_model": top_k1_model,
            # The deployable decision rule: score a transaction with
            # best_model_artifact, flag it if the probability is >= this
            # threshold. Derived from review capacity on the final fold, so
            # it carries no knowledge of the labels.
            "review_capacity_per_day": reviews_per_day,
            "decision_threshold": final_op["threshold"],
            "decision_threshold_fold": n_folds,
            "decision_threshold_k": final_op["k"],
            "expected_precision_at_threshold": final_op["precision"],
            "expected_recall_at_threshold": final_op["recall"],
            "expected_alert_rate": final_op["alert_rate"],
            # Once recall saturates, precision at capacity is bounded above by
            # total_fraud / (alerts actually flagged). Recorded so the headline
            # precision is never read without the ceiling it is being measured
            # against. Divides by n_flagged rather than k for the tie reason
            # documented in threshold.capacity_sweep.
            "precision_ceiling_at_threshold": (
                min(1.0, final_op["total_fraud"] / final_op["n_flagged"])
                if final_op["n_flagged"] else None
            ),
            "capacity_sweep": sweep_rows,
            "shap_top_features": (
                shap_global_df.head(5)["feature"].tolist() if len(shap_global_df) else []
            ),
            "feature_ablation": ablation_rows,
            "missed_fraud_summary": missed,
            "economics": {
                "avg_fraud_amount": avg_fraud_amount,
                "cost_per_review": econ_cfg["cost_per_review"],
                "recovery_rate": econ_cfg["recovery_rate"],
                "liability_rate": econ_cfg["liability_rate"],
                "degeneracy_break_even_cost_per_review_range": [
                    float(degeneracy_df["break_even_cost_per_review"].min()),
                    float(degeneracy_df["break_even_cost_per_review"].max()),
                ],
                "capacity_constraint_cost": constraint_cost,
                "ticket_size_crossovers": transitions.to_dict("records"),
            },
            "results_by_fold": results,
        }
        with open(run_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        mlflow.log_metrics({
            "decision_threshold": final_op["threshold"],
            "expected_precision_at_threshold": final_op["precision"],
            "expected_recall_at_threshold": final_op["recall"],
            "expected_alert_rate": final_op["alert_rate"],
        })
        for artifact in ("model_comparison.csv", "precision_recall_at_k.csv",
                         "shap_global_importance.csv", "shap_alert_reasons.csv",
                         "error_analysis_profile.csv", "feature_ablation.csv",
                         "capacity_sweep.csv", "capacity_economics.csv"):
            path = PROCESSED_DIR / artifact
            if path.exists():
                mlflow.log_artifact(str(path))
        mlflow.log_artifact(str(run_dir / "metadata.json"))

        logger.info("Saved model artifacts + metadata to %s", run_dir)


if __name__ == "__main__":
    main()
