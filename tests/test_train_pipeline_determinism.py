"""
Regression test for a real reproducibility bug found during Sprint 3
verification: config.yaml documented that `random_state` is "injected
separately... so it stays consistent across all models," but
train_pipeline.py never actually passed it to the three LogisticRegression
constructions (plain, ridge, lasso). Lasso's config sets solver=saga
(a stochastic method), so without a fixed seed its coefficients -- and every
metric derived from them -- were not reproducible run to run.

Checked by inspecting train_pipeline.py's source rather than importing it:
the module has import-time side effects (opens a log file, loads config at
module scope) that don't belong in a unit test, and the invariant here is
about what argument each call site passes, not about behavior that needs a
live model fit to observe.
"""

import re
from pathlib import Path

SRC = (Path(__file__).resolve().parents[1] / "src" / "train_pipeline.py").read_text()


def test_every_logisticregression_call_passes_random_state():
    calls = re.findall(r"LogisticRegression\([^)]*\)", SRC, flags=re.DOTALL)
    assert len(calls) == 3, f"expected 3 LogisticRegression(...) call sites, found {len(calls)}"
    for call in calls:
        assert "random_state" in call, (
            f"LogisticRegression call site missing random_state, so its output "
            f"(especially Lasso's solver=saga fit) is not reproducible: {call!r}"
        )


def test_saga_solver_is_actually_reproducible_given_a_fixed_seed():
    """Confirms the fix works, not just that the argument is present."""
    from sklearn.linear_model import LogisticRegression
    import numpy as np

    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 5))
    y = (X[:, 0] + rng.normal(scale=0.1, size=200) > 0).astype(int)

    kwargs = dict(solver="saga", l1_ratio=1, C=0.1, max_iter=1000, random_state=42)
    model_a = LogisticRegression(**kwargs).fit(X, y)
    model_b = LogisticRegression(**kwargs).fit(X, y)

    assert np.array_equal(model_a.coef_, model_b.coef_)
