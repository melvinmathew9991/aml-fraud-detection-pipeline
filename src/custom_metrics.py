"""
custom_metrics.py

Two things this module exists to prove, honestly:

1. A custom weighted binary cross-entropy loss (the JD's exact phrase:
   "custom loss functions (weighted BCE, cost-sensitive)") -- not just
   sklearn's built-in class_weight='balanced', but a hand-written loss
   with an explicit, tunable positive-class weight.

2. Precision@K -- the retrieval/ranking metric that MEDBOT's own project
   audit explicitly flagged as missing ("There is no retrieval-quality
   metric (precision@k, recall, MRR)"). Implementing it here, in a new
   project, is a direct, honest close of a gap you found in your own
   prior work -- not a metric borrowed from nowhere.
"""

import numpy as np


def weighted_bce_loss(y_true: np.ndarray, y_pred_proba: np.ndarray, pos_weight: float) -> float:
    """
    Weighted binary cross-entropy: weights the loss contribution of the
    positive (fraud) class by `pos_weight`, so misclassifying a rare fraud
    case is penalized `pos_weight` times more than misclassifying a
    legitimate transaction.

    loss = -mean[ pos_weight * y * log(p) + (1 - y) * log(1 - p) ]

    This is the manual, from-scratch version of what XGBoost's
    scale_pos_weight or a custom Keras/PyTorch loss function would do --
    written explicitly here so the mechanism is demonstrable, not just
    invoked as a library flag.
    """
    eps = 1e-12  # avoid log(0)
    # Force float64 before clipping: in float32, 1 - 1e-12 rounds to exactly
    # 1.0 (below float32's precision near 1), so a float32 y_pred_proba
    # (as produced when the caller trains on float32 features) would defeat
    # the clip and yield log(0) = -inf.
    p = np.clip(y_pred_proba.astype(np.float64), eps, 1 - eps)
    y = y_true.astype(float)

    per_sample_loss = -(pos_weight * y * np.log(p) + (1 - y) * np.log(1 - p))
    return float(np.mean(per_sample_loss))


def suggest_pos_weight(y_true: np.ndarray) -> float:
    """
    Standard cost-sensitive heuristic: weight = (# negative) / (# positive).
    With ~0.026% fraud in the sample data, this comes out to roughly 3800:1 --
    demonstrating why naive accuracy is meaningless here and PR-AUC/weighted
    loss are the only metrics that mean anything on this data.
    """
    n_pos = y_true.sum()
    n_neg = len(y_true) - n_pos
    return float(n_neg / max(n_pos, 1))


def precision_at_k(y_true: np.ndarray, y_pred_proba: np.ndarray, k: int) -> float:
    """
    Precision@K: of the top-K highest-risk transactions by predicted fraud
    probability, what fraction are actually fraud?

    This is the operationally relevant metric for a fraud-review team with
    a fixed daily investigation capacity of K cases -- more directly useful
    than ROC-AUC or even PR-AUC for that specific business question, which
    is exactly the "business-aligned metrics" framing the JD asks for.
    """
    if k <= 0 or k > len(y_true):
        raise ValueError(f"k must be between 1 and {len(y_true)}, got {k}")

    top_k_idx = np.argsort(y_pred_proba)[::-1][:k]
    return float(y_true[top_k_idx].sum() / k)


def recall_at_k(y_true: np.ndarray, y_pred_proba: np.ndarray, k: int) -> float:
    """
    Recall@K: of all actual fraud cases, what fraction landed in the top-K
    highest-risk transactions?

    Precision@K on its own is easy to misread as a model quality score, but
    it is a function of *both* the model and how K was chosen relative to
    the true positive count -- e.g. picking K = 5x the number of frauds
    mathematically caps precision at 20% even if the model ranks every
    single fraud first (recall@K = 100%). Reporting recall@K alongside it
    is what tells you which of those two you're actually looking at.
    """
    if k <= 0 or k > len(y_true):
        raise ValueError(f"k must be between 1 and {len(y_true)}, got {k}")

    n_pos = y_true.sum()
    if n_pos == 0:
        return float("nan")

    top_k_idx = np.argsort(y_pred_proba)[::-1][:k]
    return float(y_true[top_k_idx].sum() / n_pos)


if __name__ == "__main__":
    # Quick self-test with synthetic data
    rng = np.random.default_rng(42)
    y_true = np.array([0] * 995 + [1] * 5)
    y_pred = rng.uniform(0, 0.3, 1000)
    y_pred[-5:] = rng.uniform(0.7, 0.99, 5)  # model correctly scores frauds higher

    w = suggest_pos_weight(y_true)
    print(f"Suggested pos_weight for {y_true.sum()}/{len(y_true)} positive rate: {w:.1f}")
    print(f"Weighted BCE loss: {weighted_bce_loss(y_true, y_pred, pos_weight=w):.4f}")
    print(f"Unweighted (pos_weight=1) BCE loss: {weighted_bce_loss(y_true, y_pred, pos_weight=1.0):.4f}")
    print(f"Precision@10: {precision_at_k(y_true, y_pred, k=10):.3f}")
    print(f"Precision@50: {precision_at_k(y_true, y_pred, k=50):.3f}")
    print(f"Recall@10:    {recall_at_k(y_true, y_pred, k=10):.3f}")
    print(f"Recall@50:    {recall_at_k(y_true, y_pred, k=50):.3f}")
