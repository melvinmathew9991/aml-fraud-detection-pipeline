"""
Unit tests for cv.py: fold count, no train/test overlap, and chronological
ordering (a fold's test window is strictly after that fold's train cutoff --
the leakage-safety property expanding-window CV exists to guarantee).
"""

import numpy as np
import pytest

from cv import time_based_folds


def test_boundaries_too_short_raises():
    with pytest.raises(ValueError):
        time_based_folds(np.arange(100), boundaries=[1.0])


def test_produces_len_boundaries_minus_one_folds():
    step = np.arange(1000)
    folds = time_based_folds(step, boundaries=[0.4, 0.6, 0.8, 1.0])
    assert len(folds) == 3


def test_train_and_test_masks_never_overlap():
    step = np.arange(1000)
    folds = time_based_folds(step, boundaries=[0.4, 0.6, 0.8, 1.0])
    for train_mask, test_mask in folds:
        assert not np.any(train_mask & test_mask)


def test_test_window_is_strictly_after_train_cutoff():
    """No leakage: every row used for training in fold i has step <= that
    fold's cutoff quantile, and every row tested in fold i has step strictly
    greater than it -- the property that makes this "expanding-window" CV
    rather than a random shuffle split."""
    rng = np.random.default_rng(0)
    step = rng.integers(0, 1000, size=5000)
    boundaries = [0.4, 0.6, 0.8, 1.0]
    cuts = [np.quantile(step, b) for b in boundaries]
    folds = time_based_folds(step, boundaries)
    for i, (train_mask, test_mask) in enumerate(folds):
        assert np.all(step[train_mask] <= cuts[i])
        assert np.all(step[test_mask] > cuts[i])


def test_folds_are_expanding_windows():
    """Later folds train on strictly more (or equal) data than earlier ones,
    since each fold's train cutoff is a later quantile than the last."""
    step = np.arange(1000)
    folds = time_based_folds(step, boundaries=[0.4, 0.6, 0.8, 1.0])
    train_counts = [int(train_mask.sum()) for train_mask, _ in folds]
    assert train_counts == sorted(train_counts)


def test_test_windows_are_sequential_and_non_overlapping_across_folds():
    step = np.arange(1000)
    folds = time_based_folds(step, boundaries=[0.4, 0.6, 0.8, 1.0])
    test_masks = [test_mask for _, test_mask in folds]
    for i in range(len(test_masks) - 1):
        assert not np.any(test_masks[i] & test_masks[i + 1])
        # fold i's test rows are all chronologically before fold i+1's
        rows_i = step[test_masks[i]]
        rows_next = step[test_masks[i + 1]]
        if len(rows_i) and len(rows_next):
            assert rows_i.max() <= rows_next.min()


def test_final_fold_boundary_matches_single_split_holdout():
    """With boundaries ending in 1.0, the last fold's test window is
    everything above the second-to-last cut -- equivalent to what an old
    single `time_cutoff_quantile` split would have produced."""
    step = np.arange(1000)
    boundaries = [0.8, 1.0]
    folds = time_based_folds(step, boundaries)
    assert len(folds) == 1
    train_mask, test_mask = folds[0]
    cut = np.quantile(step, 0.8)
    assert np.array_equal(train_mask, step <= cut)
    assert np.array_equal(test_mask, step > cut)
