"""
Unit tests for threshold.py: window_days endpoint inclusivity, capacity_k
clamping, and operating_point's confusion-matrix arithmetic -- including a
regression test for the n_flagged > k tie path (Sprint 2 audit fix,
previously exercised only by chance, per ROADMAP.md's test-plan note).
"""

import math

import numpy as np
import pytest

from threshold import window_days, capacity_k, threshold_at_k, operating_point, capacity_sweep


def test_window_days_is_endpoint_inclusive():
    # steps 100..123 is 24 distinct simulated hours -> exactly 1.0 days,
    # not 23/24 -- this is the "+1" in window_days's span_steps.
    step_test = np.arange(100, 124)
    assert window_days(step_test) == pytest.approx(1.0)


def test_window_days_empty_array_is_zero():
    assert window_days(np.array([])) == 0.0


def test_window_days_single_step_is_one_hour():
    step_test = np.array([50, 50, 50])
    assert window_days(step_test) == pytest.approx(1 / 24)


def test_capacity_k_basic_arithmetic():
    step_test = np.arange(0, 24)  # 1.0 days
    k = capacity_k(step_test, reviews_per_day=60, n_rows=1000)
    assert k == 60


def test_capacity_k_clamps_to_at_least_one():
    step_test = np.arange(0, 24)
    k = capacity_k(step_test, reviews_per_day=0, n_rows=1000)
    assert k == 1


def test_capacity_k_clamps_to_n_rows():
    step_test = np.arange(0, 24 * 100)  # 100 days
    k = capacity_k(step_test, reviews_per_day=1000, n_rows=50)
    assert k == 50


def test_capacity_k_ignores_labels_by_construction():
    # capacity_k's signature takes no y_true at all -- this test exists to
    # pin that contract so a future refactor can't quietly reintroduce it.
    import inspect
    params = inspect.signature(capacity_k).parameters
    assert "y_true" not in params


def test_threshold_at_k_picks_kth_largest():
    y_pred_proba = np.array([0.1, 0.5, 0.3, 0.9, 0.7])
    # sorted desc: 0.9, 0.7, 0.5, 0.3, 0.1 -> k=2 -> 0.7
    assert threshold_at_k(y_pred_proba, k=2) == pytest.approx(0.7)


def test_threshold_at_k_rejects_out_of_range_k():
    y_pred_proba = np.array([0.1, 0.5, 0.3])
    with pytest.raises(ValueError):
        threshold_at_k(y_pred_proba, k=0)
    with pytest.raises(ValueError):
        threshold_at_k(y_pred_proba, k=4)


def test_operating_point_basic_confusion_matrix():
    y_true = np.array([1, 0, 1, 0, 0])
    y_pred_proba = np.array([0.9, 0.2, 0.8, 0.1, 0.05])
    op = operating_point(y_true, y_pred_proba, k=2)

    assert op["n_flagged"] == 2
    assert op["true_positives"] == 2
    assert op["false_positives"] == 0
    assert op["false_negatives"] == 0
    assert op["precision"] == pytest.approx(1.0)
    assert op["recall"] == pytest.approx(1.0)


def test_operating_point_tie_path_n_flagged_exceeds_k():
    """
    Regression test for the Sprint 2 audit finding: tree ensembles routinely
    produce exactly-equal scores, so the requested k and the actually-flagged
    count diverge whenever the k-th and (k+1)-th scores tie. operating_point
    must flag every row >= threshold_at_k(k), not just the first k by index,
    and precision/recall must be computed over that larger flagged set.
    """
    # Three rows tie at 0.9; k=2 asks for the top 2, but the tie forces all
    # three above the threshold into the queue.
    y_true = np.array([1, 0, 1, 0, 0])
    y_pred_proba = np.array([0.9, 0.9, 0.9, 0.5, 0.1])

    threshold = threshold_at_k(y_pred_proba, k=2)
    assert threshold == pytest.approx(0.9)

    op = operating_point(y_true, y_pred_proba, k=2)
    assert op["k"] == 2
    assert op["n_flagged"] == 3          # more rows flagged than k requested
    assert op["true_positives"] == 2     # rows 0 and 2
    assert op["false_positives"] == 1    # row 1
    assert op["precision"] == pytest.approx(2 / 3)
    assert op["recall"] == pytest.approx(1.0)


def test_operating_point_no_fraud_in_window():
    y_true = np.array([0, 0, 0, 0])
    y_pred_proba = np.array([0.1, 0.9, 0.4, 0.2])
    op = operating_point(y_true, y_pred_proba, k=1)
    assert op["total_fraud"] == 0
    assert math.isnan(op["recall"])


def test_capacity_sweep_ceiling_uses_n_flagged_not_k():
    """
    precision_ceiling must divide by n_flagged (the queue actually worked),
    not k (the requested size) -- otherwise a perfect ranker sitting exactly
    at the tie-inflated queue size would read as *below* its own ceiling.
    """
    y_true = np.array([1, 0, 1, 0, 0])
    y_pred_proba = np.array([0.9, 0.9, 0.9, 0.5, 0.1])
    step_test = np.arange(0, 24)  # 1 day

    rows = capacity_sweep(y_true, y_pred_proba, step_test, reviews_per_day_grid=[2])
    row = rows[0]

    assert row["n_flagged"] == 3
    assert row["precision_ceiling"] == pytest.approx(min(1.0, 2 / 3))
    assert row["at_ceiling"] is True


def test_capacity_sweep_reports_one_row_per_grid_point():
    y_true = np.array([0] * 90 + [1] * 10)
    rng = np.random.default_rng(0)
    y_pred_proba = rng.uniform(0, 1, 100)
    step_test = np.arange(0, 24 * 10)  # 10 days

    grid = [5, 10, 20]
    rows = capacity_sweep(y_true, y_pred_proba, step_test, reviews_per_day_grid=grid)
    assert [r["reviews_per_day"] for r in rows] == grid
