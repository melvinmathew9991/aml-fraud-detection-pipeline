"""
inference/score.py

Feature vector -> probability -> flag/no-flag against the bundled decision
threshold, plus live per-request reason codes (ARCHITECTURE.md §4/§5).

Reason codes use LightGBM's native TreeSHAP (`booster.predict(...,
pred_contrib=True)`) rather than the `shap` package: exact per-feature
contributions, computed with only LightGBM itself, no `numba`/`llvmlite` in
the serving image. `pred_contrib` returns one row of (n_features + 1)
values per input row -- the last column is the expected-value base offset,
and the first n_features sum with it to the raw margin (see
ARCHITECTURE.md §12, verified to 8e-14 against the training-time SHAP path).
"""

from dataclasses import dataclass

import numpy as np

from .bundle import Bundle

DEFAULT_TOP_N_REASONS = 4


@dataclass(frozen=True)
class Reason:
    feature: str
    contribution: float
    value: float


@dataclass(frozen=True)
class ScoreResult:
    probability: float
    flagged: bool
    decision_threshold: float
    reasons: list[Reason]


def _top_reasons(feature_names: list[str], feature_row: np.ndarray,
                  contrib_row: np.ndarray, top_n: int) -> list[Reason]:
    order = np.argsort(-np.abs(contrib_row))[:top_n]
    return [
        Reason(
            feature=feature_names[i],
            contribution=float(contrib_row[i]),
            value=float(feature_row[i]),
        )
        for i in order
    ]


def score_one(bundle: Bundle, feature_vector: list[float],
              top_n_reasons: int = DEFAULT_TOP_N_REASONS) -> ScoreResult:
    """Scores a single transaction's already-computed feature vector."""
    X = np.asarray(feature_vector, dtype="float64").reshape(1, -1)
    scaled = bundle.scaler.transform(X)

    probability = float(bundle.booster.predict(scaled)[0])
    contrib = bundle.booster.predict(scaled, pred_contrib=True)[0]
    feature_contrib = contrib[:-1]  # drop the trailing base-value column

    return ScoreResult(
        probability=probability,
        flagged=probability >= bundle.threshold.decision_threshold,
        decision_threshold=bundle.threshold.decision_threshold,
        reasons=_top_reasons(bundle.feature_names, X[0], feature_contrib, top_n_reasons),
    )


def score_batch(bundle: Bundle, feature_matrix: list[list[float]],
                 top_n_reasons: int = DEFAULT_TOP_N_REASONS) -> list[ScoreResult]:
    """
    Scores many transactions in one booster.predict() call rather than
    looping score_one() per row -- the whole point of a batch endpoint is
    to amortize that call, not to hide N separate model invocations behind
    one HTTP request.
    """
    X = np.asarray(feature_matrix, dtype="float64")
    if X.ndim == 1:
        X = X.reshape(1, -1)
    scaled = bundle.scaler.transform(X)

    probabilities = bundle.booster.predict(scaled)
    # lightgbm's type stubs declare predict() -> list, not ndarray, so mypy
    # rejects the 2D slice below without this -- predict() actually returns
    # an ndarray for 2D input at runtime (asarray is a zero-copy no-op then).
    contribs = np.asarray(bundle.booster.predict(scaled, pred_contrib=True))
    feature_contribs = contribs[:, :-1]

    results = []
    for i in range(X.shape[0]):
        probability = float(probabilities[i])
        results.append(ScoreResult(
            probability=probability,
            flagged=probability >= bundle.threshold.decision_threshold,
            decision_threshold=bundle.threshold.decision_threshold,
            reasons=_top_reasons(bundle.feature_names, X[i], feature_contribs[i], top_n_reasons),
        ))
    return results
