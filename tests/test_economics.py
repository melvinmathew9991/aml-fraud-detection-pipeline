"""
Unit tests for economics.py -- pure functions of counts and rates, no model
involved, matching ARCHITECTURE.md §4's design intent. The degeneracy_check
and capacity_constraint_cost expected values are cross-checked against the
figures already published in README.md/ARCHITECTURE.md (85,000-161,875
break-even cost, ~327M exposure at 250 vs 500 reviews/day), so a future
retrain that silently changes the capacity sweep would be caught here too.
"""

import pytest

from economics import (
    net_value, net_value_curve, degeneracy_check,
    capacity_constraint_cost, ticket_size_crossover,
)

# Shaped like the published final-fold capacity sweep (README "Setting the
# operating point from capacity, not from the labels").
SWEEP_ROWS = [
    {"reviews_per_day": 100, "k": 1617, "n_flagged": 1619,
     "true_positives": 1619, "false_positives": 0, "false_negatives": 2631},
    {"reviews_per_day": 250, "k": 4042, "n_flagged": 4042,
     "true_positives": 4042, "false_positives": 0, "false_negatives": 208},
    {"reviews_per_day": 500, "k": 8083, "n_flagged": 8083,
     "true_positives": 4250, "false_positives": 3833, "false_negatives": 0},
]
AVG_FRAUD_AMOUNT = 1_572_443


def test_net_value_hand_computed():
    # 10 caught * 100 * 0.5 - 20 reviewed * 5 - 2 missed * 100 * 1.0
    val = net_value(true_positives=10, n_flagged=20, false_negatives=2,
                     avg_fraud_amount=100, cost_per_review=5,
                     recovery_rate=0.5, liability_rate=1.0)
    assert val == pytest.approx(10 * 100 * 0.5 - 20 * 5 - 2 * 100 * 1.0)


def test_net_value_curve_matches_net_value_per_row():
    df = net_value_curve(SWEEP_ROWS, AVG_FRAUD_AMOUNT, cost_per_review=200,
                          recovery_rate=0.5, liability_rate=1.0)
    for row in SWEEP_ROWS:
        expected = net_value(row["true_positives"], row["n_flagged"], row["false_negatives"],
                              AVG_FRAUD_AMOUNT, 200, 0.5, 1.0)
        actual = df.loc[df["reviews_per_day"] == row["reviews_per_day"], "net_value"].iloc[0]
        assert actual == pytest.approx(expected)


def test_degeneracy_check_matches_published_break_even_range():
    df = degeneracy_check(SWEEP_ROWS, AVG_FRAUD_AMOUNT,
                           recovery_rate_grid=[0.05, 1.0],
                           low_reviews_per_day=250, high_reviews_per_day=500)
    low = df.loc[df["recovery_rate"] == 0.05, "break_even_cost_per_review"].iloc[0]
    high = df.loc[df["recovery_rate"] == 1.0, "break_even_cost_per_review"].iloc[0]
    # README/ARCHITECTURE state this range as "85,000 to 161,875".
    assert low == pytest.approx(84_984.3, abs=1)
    assert high == pytest.approx(161_874.9, abs=1)


def test_degeneracy_check_break_even_scales_with_recovery_rate():
    df = degeneracy_check(SWEEP_ROWS, AVG_FRAUD_AMOUNT,
                           recovery_rate_grid=[0.1, 0.5, 1.0])
    # Higher recovery_rate makes staffing up more valuable, so the
    # break-even review cost (the threshold at which it stops paying off)
    # should be strictly increasing.
    values = df.sort_values("recovery_rate")["break_even_cost_per_review"].tolist()
    assert values == sorted(values)


def test_capacity_constraint_cost_matches_published_exposure():
    result = capacity_constraint_cost(SWEEP_ROWS, AVG_FRAUD_AMOUNT,
                                       low_reviews_per_day=250, high_reviews_per_day=500)
    assert result["frauds_missed_by_constraint"] == 208
    # ARCHITECTURE §0: "you miss 208 frauds -- about 327M in exposure"
    assert result["exposure"] == pytest.approx(327_068_144, abs=1)
    assert result["exposure_per_marginal_seat"] == pytest.approx(327_068_144 / 250, abs=1)


def test_ticket_size_crossover_favors_full_capacity_at_high_ticket_size():
    df = ticket_size_crossover(SWEEP_ROWS, avg_fraud_amount_grid=[AVG_FRAUD_AMOUNT],
                                cost_per_review=200, recovery_rate=0.5, liability_rate=1.0)
    assert df.iloc[0]["recommended_reviews_per_day"] == 500


def test_ticket_size_crossover_favors_low_capacity_at_low_ticket_size():
    # At a UPI-scale ticket size, the fixed review cost dominates the value
    # of catching one more fraud, so the lowest capacity in the grid wins.
    df = ticket_size_crossover(SWEEP_ROWS, avg_fraud_amount_grid=[50],
                                cost_per_review=200, recovery_rate=0.5, liability_rate=1.0)
    assert df.iloc[0]["recommended_reviews_per_day"] == 100


def test_ticket_size_crossover_marks_transitions():
    grid = [50, 1_000, 10_000, AVG_FRAUD_AMOUNT]
    df = ticket_size_crossover(SWEEP_ROWS, avg_fraud_amount_grid=grid,
                                cost_per_review=200, recovery_rate=0.5, liability_rate=1.0)
    assert df.iloc[0]["is_crossover"] == False
    # recommendation must be non-decreasing as ticket size grows, holding
    # the other two rates fixed
    recs = df["recommended_reviews_per_day"].tolist()
    assert recs == sorted(recs)
    # at least one crossover happens somewhere across this wide a grid
    assert df["is_crossover"].iloc[1:].any()
