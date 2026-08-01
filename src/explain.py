"""
explain.py

SHAP explainability for the selected fraud model.

Two distinct audiences, two outputs, because they are not the same artifact:

  * Global (`global_importance`) -- "which features does this model actually
    rely on?" Model-governance and feature-engineering question. Answered by
    mean |SHAP| over a sample of the evaluation window.

  * Local (`explain_queue`) -- "why is THIS transaction in my review queue?"
    The analyst's question. An alert that says only "score 0.97" is not
    actionable; one that says "score 0.97, driven by amount_to_balance_ratio
    = 1.00 and dest_prior_txn_count = 0" tells the reviewer where to look,
    and is the kind of per-decision reason code an AML audit expects to see.

Both run on a SAMPLE, never the full 6.36M rows: exact TreeSHAP is O(rows)
with a large constant, and this machine has 8GB total (see hardware notes).
Sample sizes are config-driven (`explainability.*` in config.yaml).

A note on scaling: the pipeline feeds models StandardScaler-scaled features,
so SHAP attributions are computed in scaled space. Attribution magnitudes are
per-feature and unaffected by that, but the feature VALUES a human reads must
be un-scaled to mean anything -- so every local explanation reports the
inverse-transformed raw value alongside the contribution.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger("train_pipeline.explain")


def _binary_shap_values(explanation) -> np.ndarray:
    """
    Normalize SHAP output to a 2-D (n_rows, n_features) array of
    positive-class attributions.

    Different tree libraries disagree on shape for binary classification:
    XGBoost/LightGBM typically return (n, features) for the positive class,
    while sklearn's HistGradientBoosting can return (n, features, 2) with one
    slice per class. Rather than special-casing model types, normalize on
    shape and take the positive-class slice when a class axis is present.
    """
    values = explanation.values if hasattr(explanation, "values") else explanation
    values = np.asarray(values)

    if values.ndim == 3:
        # (n, features, classes) -> positive class
        return values[..., -1]
    return values


def compute_shap(model, X: np.ndarray, max_rows: int, random_state: int):
    """
    Run TreeSHAP over at most `max_rows` rows sampled from X.

    Returns (shap_values, sample_index) so callers can line attributions back
    up with the rows they came from. `check_additivity=False`: the pipeline
    trains on float32 features, and TreeSHAP's additivity assertion compares
    summed attributions against the model's float64 margin, which trips on
    float32 rounding for deep ensembles even when nothing is wrong.
    """
    import shap  # imported lazily -- pulls in numba, ~2s

    rng = np.random.default_rng(random_state)
    n = len(X)
    if n > max_rows:
        idx = np.sort(rng.choice(n, size=max_rows, replace=False))
    else:
        idx = np.arange(n)

    explainer = shap.TreeExplainer(model)
    values = _binary_shap_values(explainer(X[idx], check_additivity=False))
    return values, idx


def global_importance(shap_values: np.ndarray, feature_names: list[str]) -> pd.DataFrame:
    """
    Mean |SHAP| per feature -- the standard global-importance summary.

    Preferred over a tree's built-in `feature_importances_` (split counts or
    gain) because those are computed on the training data and are known to
    inflate high-cardinality features; SHAP is measured on held-out rows and
    is in units of the model's output margin.

    `mean_signed_shap` is reported alongside because sign matters
    operationally: a feature can be important overall while pushing risk DOWN
    on average, which mean |SHAP| alone would hide.
    """
    mean_abs = np.abs(shap_values).mean(axis=0)
    mean_signed = shap_values.mean(axis=0)

    df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs,
        "mean_signed_shap": mean_signed,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    total = df["mean_abs_shap"].sum()
    df["share_of_total_abs_shap"] = df["mean_abs_shap"] / total if total > 0 else np.nan
    df["rank"] = np.arange(1, len(df) + 1)
    return df


def explain_queue(model, X_scaled: np.ndarray, y_true: np.ndarray, y_proba: np.ndarray,
                  scaler, feature_names: list[str], threshold: float,
                  max_alerts: int, top_features: int, random_state: int) -> pd.DataFrame:
    """
    Per-alert reason codes for the highest-scoring transactions in the review
    queue: for each alert, the `top_features` features contributing most to
    ITS score, with each feature's raw (un-scaled) value.

    Explains the top `max_alerts` by score rather than a random sample of the
    queue -- these are the cases an analyst opens first, so they are the ones
    whose explanations have to hold up.
    """
    flagged_idx = np.flatnonzero(y_proba >= threshold)
    # Highest-scoring alerts first, capped.
    order = flagged_idx[np.argsort(y_proba[flagged_idx])[::-1]][:max_alerts]
    if len(order) == 0:
        return pd.DataFrame()

    shap_values, _ = compute_shap(model, X_scaled[order], max_rows=len(order),
                                  random_state=random_state)
    raw_values = scaler.inverse_transform(X_scaled[order])

    rows = []
    for i, row_idx in enumerate(order):
        contributions = shap_values[i]
        # Rank this alert's features by absolute contribution to ITS score.
        top = np.argsort(np.abs(contributions))[::-1][:top_features]
        for rank, feat_idx in enumerate(top, start=1):
            rows.append({
                "alert_rank": i + 1,
                "row_index": int(row_idx),
                "fraud_score": float(y_proba[row_idx]),
                "is_fraud": int(y_true[row_idx]),
                "reason_rank": rank,
                "feature": feature_names[feat_idx],
                "feature_value": float(raw_values[i, feat_idx]),
                "shap_contribution": float(contributions[feat_idx]),
            })
    return pd.DataFrame(rows)


def log_global_importance(df: pd.DataFrame, top_n: int = 10) -> None:
    logger.info("  SHAP global importance (mean |SHAP| over sampled test rows):")
    for _, row in df.head(top_n).iterrows():
        direction = "raises" if row["mean_signed_shap"] > 0 else "lowers"
        logger.info(
            "    %2d. %-32s %.5f  (%.1f%% of total; on average %s risk)",
            int(row["rank"]), row["feature"], row["mean_abs_shap"],
            100 * row["share_of_total_abs_shap"], direction,
        )
    dead = df[df["mean_abs_shap"] <= 1e-9]
    if len(dead):
        logger.info("    %d feature(s) with effectively zero attribution: %s",
                    len(dead), ", ".join(dead["feature"].tolist()))
