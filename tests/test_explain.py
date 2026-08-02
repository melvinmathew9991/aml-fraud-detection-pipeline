"""
Unit tests for explain.py against hand-computed values -- flagged as a
coverage gap in the Sprint 4 audit. Covers the two branches that were
previously untested: _binary_shap_values' shape-normalization (the 3-D
HistGradientBoosting case never exercised by any test) and explain_queue's
un-scaling of raw feature values, which its own docstring claims but which
nothing pinned as a regression.

explain_queue's dependency on `compute_shap` (which lazily imports `shap`
and calls a real TreeExplainer) is monkeypatched out here -- these tests
are about explain_queue's OWN post-processing logic (alert selection,
top-feature ranking, un-scaling), not about SHAP's correctness, which
train_pipeline's Sprint-2 feasibility validation and the offline SHAP run
already cover.
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import explain
from explain import _binary_shap_values, explain_queue, global_importance


# ------------------------------------------------------------ _binary_shap_values

def test_binary_shap_values_2d_passthrough():
    values = np.array([[0.1, 0.2], [0.3, 0.4]])
    result = _binary_shap_values(values)
    np.testing.assert_array_equal(result, values)


def test_binary_shap_values_3d_takes_positive_class_slice():
    # (n=2, features=2, classes=2) -- HistGradientBoosting-style output.
    values = np.array([
        [[0.9, 0.1], [0.8, 0.2]],   # row 0: class0 vs class1 per feature
        [[0.7, 0.3], [0.6, 0.4]],   # row 1
    ])
    result = _binary_shap_values(values)
    expected = np.array([[0.1, 0.2], [0.3, 0.4]])  # last-class ([..., -1]) slice
    np.testing.assert_array_equal(result, expected)


def test_binary_shap_values_unwraps_dot_values_attribute():
    # shap's Explanation objects expose .values rather than being arrays.
    fake_explanation = SimpleNamespace(values=np.array([[1.0, 2.0]]))
    result = _binary_shap_values(fake_explanation)
    np.testing.assert_array_equal(result, np.array([[1.0, 2.0]]))


# ------------------------------------------------------------- global_importance

def test_global_importance_hand_computed():
    # 3 rows, 2 features.
    shap_values = np.array([
        [1.0, -2.0],
        [-1.0, -2.0],
        [3.0, 0.0],
    ])
    df = global_importance(shap_values, ["a", "b"])
    row = df.set_index("feature")

    # mean|SHAP|: a = (1+1+3)/3 = 1.667, b = (2+2+0)/3 = 1.333
    assert row.loc["a", "mean_abs_shap"] == pytest.approx(5 / 3)
    assert row.loc["b", "mean_abs_shap"] == pytest.approx(4 / 3)
    # mean signed: a = (1-1+3)/3 = 1.0, b = (-2-2+0)/3 = -1.333
    assert row.loc["a", "mean_signed_shap"] == pytest.approx(1.0)
    assert row.loc["b", "mean_signed_shap"] == pytest.approx(-4 / 3)

    total = 5 / 3 + 4 / 3
    assert row.loc["a", "share_of_total_abs_shap"] == pytest.approx((5 / 3) / total)


def test_global_importance_sorted_by_mean_abs_shap_descending_with_rank():
    shap_values = np.array([[0.1, 0.9], [0.1, 0.8]])
    df = global_importance(shap_values, ["low", "high"])
    assert df.iloc[0]["feature"] == "high"
    assert df.iloc[0]["rank"] == 1
    assert df.iloc[1]["feature"] == "low"
    assert df.iloc[1]["rank"] == 2


def test_global_importance_zero_total_does_not_divide_by_zero():
    shap_values = np.zeros((2, 2))
    df = global_importance(shap_values, ["a", "b"])
    assert df["share_of_total_abs_shap"].isna().all()


# --------------------------------------------------------------- explain_queue

class _IdentityScaler:
    """Stand-in for sklearn's StandardScaler with mean=0/scale=1, so raw ==
    scaled and un-scaling is trivially checkable."""
    def inverse_transform(self, X):
        return X * 2.0 + 10.0  # arbitrary invertible transform to prove it's applied


def test_explain_queue_returns_empty_frame_when_nothing_flagged():
    X_scaled = np.zeros((5, 2))
    y_true = np.zeros(5, dtype=int)
    y_proba = np.full(5, 0.1)
    result = explain_queue(
        model=None, X_scaled=X_scaled, y_true=y_true, y_proba=y_proba,
        scaler=_IdentityScaler(), feature_names=["a", "b"], threshold=0.5,
        max_alerts=10, top_features=2, random_state=0,
    )
    assert result.empty


def test_explain_queue_orders_alerts_by_score_descending_and_unscales_values(monkeypatch):
    # 4 rows; rows 1 and 3 flagged (proba >= 0.5), row 3's score is higher.
    X_scaled = np.array([[0.0, 0.0], [1.0, 2.0], [0.0, 0.0], [3.0, 4.0]])
    y_true = np.array([0, 1, 0, 1])
    y_proba = np.array([0.1, 0.6, 0.2, 0.9])

    # Fake compute_shap: return a fixed contribution matrix keyed to the
    # ORDER explain_queue passes in (already alert-sorted), not to X_scaled's
    # original row order -- exercises that explain_queue reindexes correctly.
    def fake_compute_shap(model, X, max_rows, random_state):
        # X here is X_scaled[order] -- 2 rows (row3 first, row1 second).
        return np.array([[5.0, 1.0], [2.0, 8.0]]), np.arange(len(X))

    monkeypatch.setattr(explain, "compute_shap", fake_compute_shap)

    result = explain_queue(
        model=None, X_scaled=X_scaled, y_true=y_true, y_proba=y_proba,
        scaler=_IdentityScaler(), feature_names=["a", "b"], threshold=0.5,
        max_alerts=10, top_features=1, random_state=0,
    )

    # Row 3 (score 0.9) must be alert_rank 1; row 1 (score 0.6) alert_rank 2.
    assert result.loc[result["alert_rank"] == 1, "row_index"].iloc[0] == 3
    assert result.loc[result["alert_rank"] == 2, "row_index"].iloc[0] == 1

    # top_features=1 -> only the larger-|contribution| feature per alert.
    # Alert 1 (row 3): contributions [5.0, 1.0] -> feature "a" wins.
    alert1 = result[result["alert_rank"] == 1].iloc[0]
    assert alert1["feature"] == "a"
    assert alert1["shap_contribution"] == pytest.approx(5.0)
    # raw value un-scaled via _IdentityScaler: X_scaled[3] = [3.0, 4.0] -> a=3.0*2+10=16.0
    assert alert1["feature_value"] == pytest.approx(16.0)

    # Alert 2 (row 1): contributions [2.0, 8.0] -> feature "b" wins.
    alert2 = result[result["alert_rank"] == 2].iloc[0]
    assert alert2["feature"] == "b"
    # X_scaled[1] = [1.0, 2.0] -> b = 2.0*2+10 = 14.0
    assert alert2["feature_value"] == pytest.approx(14.0)


def test_explain_queue_respects_max_alerts_cap(monkeypatch):
    X_scaled = np.zeros((5, 2))
    y_true = np.array([1, 1, 1, 1, 1])
    y_proba = np.array([0.9, 0.8, 0.7, 0.6, 0.5])

    def fake_compute_shap(model, X, max_rows, random_state):
        return np.ones((len(X), 2)), np.arange(len(X))

    monkeypatch.setattr(explain, "compute_shap", fake_compute_shap)

    result = explain_queue(
        model=None, X_scaled=X_scaled, y_true=y_true, y_proba=y_proba,
        scaler=_IdentityScaler(), feature_names=["a", "b"], threshold=0.5,
        max_alerts=2, top_features=2, random_state=0,
    )
    assert result["alert_rank"].nunique() == 2
