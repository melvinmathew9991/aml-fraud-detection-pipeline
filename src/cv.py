"""
cv.py

Time-based cross-validation folds, used instead of a single 80/20 split.
A single held-out window (the pre-Sprint-1 approach) leaves only ~4,250
fraud rows in the test set -- noisy enough that one unlucky window can
swing PR-AUC/Precision@K without the model actually changing. Expanding
the same idea to multiple sequential windows gives a mean +/- std instead
of a single point estimate, without ever training on data from the future
relative to what's being evaluated (the leakage-safety property a random
shuffle split would break).
"""

import numpy as np


def time_based_folds(step: np.ndarray, boundaries: list[float]) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Expanding-window time-based folds over `step`.

    `boundaries` is a list of quantiles (e.g. [0.4, 0.6, 0.8, 1.0]) marking
    the end of each sequential block. Fold i trains on everything up to
    block i's cut and tests on block i+1 alone -- so len(boundaries) - 1
    folds are produced, each with strictly more training data and a later,
    non-overlapping test window than the last. With boundaries ending in
    1.0, the final fold's test window is `step` above the second-to-last
    cut, matching what a single `time_cutoff_quantile` split would produce.
    """
    if len(boundaries) < 2:
        raise ValueError("boundaries must have at least 2 entries to produce a fold")

    cuts = [np.quantile(step, b) for b in boundaries]
    folds = []
    for i in range(len(cuts) - 1):
        train_mask = step <= cuts[i]
        test_mask = (step > cuts[i]) & (step <= cuts[i + 1])
        folds.append((train_mask, test_mask))
    return folds
