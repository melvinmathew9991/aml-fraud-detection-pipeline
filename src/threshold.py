"""
threshold.py

Turns a ranking into an actual operating decision.

Every Precision@K number this project reported before Sprint 2 used
K = <multiple> x the test window's TRUE FRAUD COUNT. That is fine for
comparing models offline, but it is not a rule you can deploy: in production
nobody knows the true fraud count for today's transactions -- that number is
only available in hindsight, which is precisely what a fraud model exists to
predict. Tuning an operating point against it is a subtle form of using the
labels at decision time.

What IS knowable in advance is how many alerts a review team can actually
work through in a day. So Sprint 2 sets K from a configured review capacity
(`review_capacity.reviews_per_day` in config.yaml) scaled by the length of
the evaluation window, and derives the score threshold from that K. The
resulting threshold is a deployable artifact: a number you can ship with the
model and apply to a single incoming transaction, with no knowledge of the
labels.

The label-derived multiplier curve is still computed alongside it (see
train_pipeline.K_FRAUD_MULTIPLIERS) because it remains the cleaner
model-vs-model comparison. The two answer different questions and Sprint 2
reports both rather than replacing one with the other.
"""

import numpy as np

# PaySim's `step` column is one simulated hour.
STEPS_PER_DAY = 24


def window_days(step_test: np.ndarray) -> float:
    """Length of an evaluation window in days, from its `step` range.

    Inclusive of both endpoints: a window covering steps 100..123 is 24
    distinct hours, i.e. exactly one day, not 23/24 of one.
    """
    if len(step_test) == 0:
        return 0.0
    span_steps = int(step_test.max() - step_test.min()) + 1
    return span_steps / STEPS_PER_DAY


def capacity_k(step_test: np.ndarray, reviews_per_day: float, n_rows: int) -> int:
    """
    K = (analyst reviews available per day) x (days in the evaluation window),
    clamped to [1, n_rows].

    Note this deliberately does NOT look at y_true. That is the whole point:
    K here is a function of staffing and elapsed time only, both of which a
    deployed system knows without seeing a single label.
    """
    k = int(round(reviews_per_day * window_days(step_test)))
    return int(np.clip(k, 1, n_rows))


def threshold_at_k(y_pred_proba: np.ndarray, k: int) -> float:
    """
    The score cutoff that admits the top-K transactions: the K-th highest
    predicted probability. Scoring a new transaction at or above this value
    means "put it in the review queue".
    """
    if k <= 0 or k > len(y_pred_proba):
        raise ValueError(f"k must be between 1 and {len(y_pred_proba)}, got {k}")
    # -k'th order statistic == k'th largest.
    return float(np.partition(y_pred_proba, -k)[-k])


def operating_point(y_true: np.ndarray, y_pred_proba: np.ndarray, k: int) -> dict:
    """
    Full confusion-matrix view of the decision "review everything scoring at
    or above threshold_at_k(k)".

    `n_flagged` can exceed `k` when scores tie at the threshold -- tree
    ensembles produce plenty of exactly-equal probabilities, so this is
    common rather than a corner case. Precision/recall are reported over the
    actual flagged set (>= threshold), not over a truncated top-K slice, so
    they describe what the review team would really see. `n_flagged` is
    returned alongside `k` precisely so that gap stays visible instead of
    being silently rounded away.
    """
    threshold = threshold_at_k(y_pred_proba, k)
    flagged = y_pred_proba >= threshold

    n_flagged = int(flagged.sum())
    tp = int(np.sum(flagged & (y_true == 1)))
    fp = n_flagged - tp
    total_fraud = int(y_true.sum())
    fn = total_fraud - tp

    return {
        "k": int(k),
        "threshold": threshold,
        "n_flagged": n_flagged,
        "n_rows": int(len(y_true)),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "total_fraud": total_fraud,
        "precision": float(tp / n_flagged) if n_flagged else float("nan"),
        "recall": float(tp / total_fraud) if total_fraud else float("nan"),
        "alert_rate": float(n_flagged / len(y_true)) if len(y_true) else float("nan"),
    }


def capacity_sweep(y_true: np.ndarray, y_pred_proba: np.ndarray, step_test: np.ndarray,
                   reviews_per_day_grid: list[float]) -> list[dict]:
    """
    The operating point as a function of staffing -- i.e. the actual output a
    capacity decision needs.

    A single precision number at a single K invites the wrong reading, because
    precision at any K where recall is already saturated is bounded above by
    (total fraud / K): once the model ranks every fraud inside the queue,
    every additional review slot can only add a false positive. Sweeping the
    capacity makes that ceiling visible instead of leaving it as a trap, and
    turns "what precision does the model get?" into the question a fraud lead
    actually asks: how many reviewers do we need, and what do we give up at
    each level?

    `precision_ceiling` is that bound; a model sitting on it is ranking
    perfectly at that capacity and cannot be improved except by re-sizing the
    queue.
    """
    days = window_days(step_test)
    total_fraud = int(y_true.sum())
    rows = []
    for rpd in reviews_per_day_grid:
        k = capacity_k(step_test, rpd, n_rows=len(y_true))
        op = operating_point(y_true, y_pred_proba, k)
        op["reviews_per_day"] = rpd
        op["window_days"] = days
        # The bound is set by the queue actually worked (`n_flagged`), not by
        # the requested `k`. Ties at the threshold can push n_flagged past k,
        # and precision is measured over n_flagged -- so dividing by k here
        # would report a ceiling the model provably cannot reach and make
        # `at_ceiling` read False for a perfect ranker.
        n_flagged = op["n_flagged"]
        op["precision_ceiling"] = (
            min(1.0, total_fraud / n_flagged) if n_flagged else float("nan")
        )
        op["at_ceiling"] = bool(
            np.isfinite(op["precision"]) and op["precision"] >= op["precision_ceiling"] - 1e-9
        )
        rows.append(op)
    return rows


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    y = np.array([0] * 9900 + [1] * 100)
    p = rng.uniform(0, 0.4, 10000)
    p[-100:] = rng.uniform(0.5, 0.99, 100)
    step = np.repeat(np.arange(100, 100 + 48), 10000 // 48 + 1)[:10000]

    print(f"window_days: {window_days(step):.2f}")
    k = capacity_k(step, reviews_per_day=60, n_rows=len(y))
    print(f"capacity_k @60/day: {k}")
    for key, val in operating_point(y, p, k).items():
        print(f"  {key:18s} {val}")
