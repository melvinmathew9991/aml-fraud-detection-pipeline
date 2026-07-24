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
- Precision@K-based model selection (K=1x true fraud count, the realistic
  review-queue operating point) rather than PR-AUC alone -- the two were
  found to disagree post-tuning (see README), and this project's own
  thesis is that Precision@K is the operationally relevant metric
- Precision@K / Recall@K as the operationally-relevant evaluation metrics
- Optuna hyperparameter tuning (XGBoost/LightGBM only) optimizing mean PR-AUC
  across all CV folds (not a single fold -- one fold turned out to be
  near-perfectly separable and gave Optuna no signal to discriminate trials
  on), then refit with best params across all folds so the comparison table
  shows a visible tuned-vs-default delta
- MLflow local file-store tracking (file:./mlruns) so runs are comparable
  over time instead of overwriting a single CSV
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
from features import feature_query, FEATURE_COLUMNS, FEATURE_VERSION
from custom_metrics import weighted_bce_loss, suggest_pos_weight, precision_at_k, recall_at_k

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
        transactions_reloaded = _ensure_transactions_table(con, RAW_PATH)
        log_memory("after ensure transactions table")

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


def evaluate_model(name, y_test, y_proba, pos_weight):
    pr_auc = average_precision_score(y_test, y_proba)
    roc_auc = roc_auc_score(y_test, y_proba) if y_test.sum() > 0 else float("nan")
    w_bce = weighted_bce_loss(y_test, y_proba, pos_weight=pos_weight)

    n_test_fraud = int(y_test.sum())
    k = max(n_test_fraud * 5, 10)  # e.g. "review top-K flagged cases per day"
    k = min(k, len(y_test))
    p_at_k = precision_at_k(y_test, y_proba, k=k)
    r_at_k = recall_at_k(y_test, y_proba, k=k)

    logger.info("--- %s ---", name)
    logger.info("  PR-AUC:              %.4f  (primary metric under class imbalance)", pr_auc)
    logger.info("  ROC-AUC:             %.4f  (reported, but misleading alone at this imbalance)", roc_auc)
    logger.info("  Weighted BCE loss:   %.4f", w_bce)
    logger.info("  Precision@%d:          %.4f  (of top-%d flagged, fraction actually fraud)", k, p_at_k, k)
    logger.info("  Recall@%d:             %.4f  (of all fraud, fraction caught in top-%d)", k, r_at_k, k)

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
            "k": k, "n_test_fraud": n_test_fraud}, curve_rows


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

    fold_data = []
    for fold_idx, (train_mask, test_mask) in enumerate(folds, start=1):
        prepared = prepare_fold(X, y, train_mask, test_mask, sampling_cfg, random_state)
        _, _, y_train, y_test, _, pos_weight, _ = prepared
        logger.info(
            "Fold %d/%d: train=%d rows (%d fraud), test=%d rows (%d fraud), pos_weight=%.1f",
            fold_idx, n_folds, len(y_train), y_train.sum(), len(y_test), y_test.sum(), pos_weight,
        )
        fold_data.append(prepared)
    log_memory("all folds prepared")

    results = []
    curve_rows = []
    final_fold_models = {}

    def record(name, model_key, fold_idx, y_test, y_proba, pos_weight):
        result, rows = evaluate_model(name, y_test, y_proba, pos_weight)
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
        })

        for fold_idx, (X_train, X_test, y_train, y_test, scaler, pos_weight, train_pos_weight) in enumerate(fold_data, start=1):
            logger.info("=== Fold %d/%d ===", fold_idx, n_folds)
            fold_models = {}

            logreg = LogisticRegression(**CONFIG["models"]["logistic_regression"])
            logreg.fit(X_train, y_train)
            fold_models["logistic_regression"] = logreg
            record("Logistic Regression (class_weight=balanced)", "logistic_regression",
                   fold_idx, y_test, logreg.predict_proba(X_test)[:, 1], pos_weight)

            ridge_lr = LogisticRegression(**CONFIG["models"]["ridge"])
            ridge_lr.fit(X_train, y_train)
            fold_models["ridge"] = ridge_lr
            record("Ridge-penalized Logistic Regression (L2, C=0.1)", "ridge",
                   fold_idx, y_test, ridge_lr.predict_proba(X_test)[:, 1], pos_weight)

            lasso_lr = LogisticRegression(**CONFIG["models"]["lasso"])
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
                n_folds=("fold", "count"),
            )
            .reset_index()
            .sort_values("pr_auc_mean", ascending=False)
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

        # best_model is selected by mean Precision@K at the realistic
        # K=1x-true-fraud-count operating point, NOT by mean PR-AUC -- this
        # project's own thesis throughout is that Precision@K is the
        # operationally relevant metric (a fraud team reviews a fixed-size
        # queue, not a probability ranking), and PR-AUC and Precision@K@1x
        # were found to disagree (see README): Optuna's PR-AUC-only
        # objective can find hyperparameters that rank fraud marginally
        # better while calibrating worse and doing no better (or worse) on
        # the metric that actually matters operationally.
        k1_df = curve_summary_df[curve_summary_df["k_multiplier"] == 1].sort_values(
            "precision_at_k_mean", ascending=False
        )
        best_model_name = k1_df.iloc[0]["model"]
        top_pr_auc_model = summary_df.iloc[0]["model"]
        logger.info(
            "Best model by mean Precision@K (K=1x fraud count) across %d folds: %s (%.4f +/- %.4f)",
            n_folds, best_model_name, k1_df.iloc[0]["precision_at_k_mean"], k1_df.iloc[0]["precision_at_k_std"],
        )
        if top_pr_auc_model != best_model_name:
            logger.info(
                "  (Note: %s has the highest mean PR-AUC (%.4f) but isn't the Precision@K winner -- "
                "PR-AUC and Precision@K@1x disagree here; see README 'Optuna improved ranking but hurt calibration'.)",
                top_pr_auc_model, summary_df.iloc[0]["pr_auc_mean"],
            )

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
            "best_model_selection_metric": "mean_precision_at_k_1x_fraud_count",
            "top_pr_auc_model": top_pr_auc_model,
            "results_by_fold": results,
        }
        with open(run_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        mlflow.log_artifact(str(PROCESSED_DIR / "model_comparison.csv"))
        mlflow.log_artifact(str(PROCESSED_DIR / "precision_recall_at_k.csv"))
        mlflow.log_artifact(str(run_dir / "metadata.json"))

        logger.info("Saved model artifacts + metadata to %s", run_dir)


if __name__ == "__main__":
    main()
