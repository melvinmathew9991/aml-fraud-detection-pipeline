"""
Unit tests for custom_metrics.py: weighted BCE, pos-weight suggestion, and
Precision@K/Recall@K, each checked against a hand-computed reference rather
than re-deriving the same formula the source uses.
"""

import math

import numpy as np
import pytest

from custom_metrics import (
    precision_at_k,
    recall_at_k,
    suggest_pos_weight,
    weighted_bce_loss,
)


def test_suggest_pos_weight_basic():
    y = np.array([0, 0, 0, 1])
    assert suggest_pos_weight(y) == pytest.approx(3.0)


def test_suggest_pos_weight_no_positives_does_not_divide_by_zero():
    y = np.array([0, 0, 0, 0])
    # max(n_pos, 1) guards this -- weight degenerates to n_neg, not inf/nan.
    assert suggest_pos_weight(y) == pytest.approx(4.0)


def test_weighted_bce_loss_matches_hand_computed_formula():
    y_true = np.array([0, 1])
    y_pred = np.array([0.5, 0.5])
    pos_weight = 3.0

    # Reference computed independently of the source's implementation.
    expected = -np.mean([
        math.log(1 - 0.5),          # y=0 term
        pos_weight * math.log(0.5),  # y=1 term
    ])
    assert weighted_bce_loss(y_true, y_pred, pos_weight) == pytest.approx(expected)


def test_weighted_bce_loss_unweighted_reduces_to_plain_bce():
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0.2, 0.8, 0.6, 0.3])
    expected = -np.mean([
        math.log(1 - 0.2),
        math.log(0.8),
        math.log(0.6),
        math.log(1 - 0.3),
    ])
    assert weighted_bce_loss(y_true, y_pred, pos_weight=1.0) == pytest.approx(expected)


def test_weighted_bce_loss_handles_float32_without_overflow():
    # float32(1 - 1e-12) rounds to exactly 1.0, which would make log(1-p) =
    # -inf if the eps-clip were applied before promoting to float64.
    y_true = np.array([0, 1], dtype=np.float32)
    y_pred = np.array([1.0, 0.0], dtype=np.float32)
    loss = weighted_bce_loss(y_true, y_pred, pos_weight=5.0)
    assert math.isfinite(loss)


def test_precision_at_k_hand_computed():
    y_true = np.array([0, 0, 1, 0, 1])
    y_pred = np.array([0.1, 0.4, 0.9, 0.2, 0.8])

    # top-2 by score: index 2 (0.9, fraud), index 4 (0.8, fraud)
    assert precision_at_k(y_true, y_pred, k=2) == pytest.approx(1.0)
    # top-3: adds index 1 (0.4, not fraud) -> 2/3
    assert precision_at_k(y_true, y_pred, k=3) == pytest.approx(2 / 3)


def test_recall_at_k_hand_computed():
    y_true = np.array([0, 0, 1, 0, 1])
    y_pred = np.array([0.1, 0.4, 0.9, 0.2, 0.8])

    # both frauds are in the top 2 -> full recall already
    assert recall_at_k(y_true, y_pred, k=2) == pytest.approx(1.0)
    assert recall_at_k(y_true, y_pred, k=1) == pytest.approx(0.5)


def test_recall_at_k_no_positives_returns_nan():
    y_true = np.array([0, 0, 0])
    y_pred = np.array([0.1, 0.5, 0.9])
    assert math.isnan(recall_at_k(y_true, y_pred, k=2))


@pytest.mark.parametrize("fn", [precision_at_k, recall_at_k])
def test_at_k_rejects_out_of_range_k(fn):
    y_true = np.array([0, 1, 0])
    y_pred = np.array([0.1, 0.9, 0.2])
    with pytest.raises(ValueError):
        fn(y_true, y_pred, k=0)
    with pytest.raises(ValueError):
        fn(y_true, y_pred, k=4)
