"""
Tests dashboard/common.py's data-loading and net-value computation --
the logic behind the capacity & economics explorer (dashboard/pages/
3_Capacity_Economics_Explorer.py), pulled into a plain importable module
specifically so it's testable without Streamlit's AppTest machinery.

Sprint 5's DoD (ROADMAP.md): "capacity explorer reproduces
capacity_sweep.csv exactly at 500/day" -- these tests are what pins that.
"""

import pytest

from common import (
    AVG_FRAUD_AMOUNT,
    SENSITIVITY_RECOVERY_RATE_GRID,
    capacity_sweep_rows,
    compute_net_value_curve,
)


def test_capacity_sweep_rows_matches_committed_csv_at_500_per_day():
    rows = capacity_sweep_rows()
    row_500 = next(r for r in rows if r["reviews_per_day"] == 500)
    # These are the exact published numbers (README.md "Results" table).
    assert row_500["k"] == 8083
    assert row_500["true_positives"] == 4250
    assert row_500["false_positives"] == 3835
    assert row_500["false_negatives"] == 0
    assert row_500["precision"] == pytest.approx(0.5257, abs=1e-4)
    assert row_500["recall"] == pytest.approx(1.0)
    assert row_500["precision_ceiling"] == pytest.approx(row_500["precision"])


def test_capacity_sweep_rows_covers_the_documented_grid():
    rows = capacity_sweep_rows()
    assert sorted(r["reviews_per_day"] for r in rows) == [100, 250, 500, 1000, 2000, 5000]


def test_net_value_curve_matches_committed_capacity_economics_csv_at_500(tmp_path):
    # capacity_economics.csv was generated with config.yaml's economics
    # defaults (cost_per_review=200, recovery_rate=0.5, liability_rate=1.0)
    # against this same avg_fraud_amount -- reproducing its 500/day row
    # exactly is the DoD requirement.
    curve = compute_net_value_curve(cost_per_review=200, recovery_rate=0.5, liability_rate=1.0)
    row_500 = curve[curve["reviews_per_day"] == 500].iloc[0]
    assert row_500["net_value"] == pytest.approx(3.339824e9, rel=1e-5)


def test_net_value_curve_hand_computed_at_one_point():
    # k=1617 row (100/day): tp=1617, n_flagged=1617, fn=2633.
    curve = compute_net_value_curve(cost_per_review=200, recovery_rate=0.5, liability_rate=1.0)
    row_100 = curve[curve["reviews_per_day"] == 100].iloc[0]
    expected = (1617 * AVG_FRAUD_AMOUNT * 0.5) - (1617 * 200) - (2633 * AVG_FRAUD_AMOUNT * 1.0)
    assert row_100["net_value"] == pytest.approx(expected)


def test_optimum_always_reaches_recall_saturation_at_realistic_cost():
    # The page's own degeneracy check, reproduced here as a regression: at
    # a realistic review cost (well under the 84,942-161,795 break-even
    # documented in README.md), the optimum should sit at the
    # recall-saturation point (500/day) across the whole sensitivity band.
    rows = capacity_sweep_rows()
    saturation_rpd = min(r["reviews_per_day"] for r in rows if r["recall"] >= 0.9999)
    assert saturation_rpd == 500

    for recovery_rate in SENSITIVITY_RECOVERY_RATE_GRID:
        curve = compute_net_value_curve(cost_per_review=200, recovery_rate=recovery_rate,
                                        liability_rate=1.0)
        optimum_rpd = curve.loc[curve["net_value"].idxmax(), "reviews_per_day"]
        assert optimum_rpd >= saturation_rpd


def test_optimum_drops_below_saturation_past_the_documented_break_even():
    # Above the break-even range, a low recovery rate should make the
    # optimum retreat from full-recall staffing -- the "genuine tradeoff"
    # branch the page shows must be reachable, not dead code.
    curve = compute_net_value_curve(cost_per_review=200_000, recovery_rate=0.05,
                                    liability_rate=1.0)
    optimum_rpd = curve.loc[curve["net_value"].idxmax(), "reviews_per_day"]
    assert optimum_rpd < 500
