"""
economics.py

Turns the Sprint 2 capacity sweep (an exchange rate: "~18 false positives
per marginal fraud caught") into an actual business decision, per
ARCHITECTURE.md §0.

    net_value(K) = frauds_caught(K)  * avg_fraud_amount * recovery_rate
                 - alerts_reviewed(K) * cost_per_review
                 - frauds_missed(K)  * avg_fraud_amount * liability_rate

This is a pure function of the counts already in a capacity_sweep row
(true_positives, n_flagged, false_negatives) plus three business rates --
no model involved, which is what makes it unit-testable in isolation and
reusable against any model's capacity sweep (see ARCHITECTURE §4).

THE NAIVE VERSION IS DEGENERATE. Tested against the real final-fold data:
average fraud amount ~1.57M against a review that plausibly costs tens to
low-hundreds of currency units means the break-even review cost for staffing
up from 250/day to 500/day is 85,000-162,000/alert across recovery rates
0.05-1.00 (see degeneracy_check). No real review costs that, so full recall
wins under every plausible assumption and "the optimum" never moves --
shipping that as the headline finding would be reporting arithmetic as
insight. Two non-degenerate questions instead:

  1. capacity_constraint_cost -- what does under-staffing cost, in exposure
     and per analyst seat? (Sensitivity, not an optimum.)
  2. ticket_size_crossover -- at what average fraud amount does the
     recommended staffing level stop being "as much as possible"? PaySim's
     1.57M average is nowhere near this crossover; UPI-scale fraud (hundreds
     to low thousands per transaction) may be.
"""

import numpy as np
import pandas as pd


def net_value(true_positives: int, n_flagged: int, false_negatives: int,
              avg_fraud_amount: float, cost_per_review: float,
              recovery_rate: float, liability_rate: float) -> float:
    """
    Net value of operating at one capacity/threshold point.

    `true_positives` and `false_negatives` are read from an
    `threshold.operating_point` (or `capacity_sweep`) row -- frauds caught
    and frauds missed at that point -- and `n_flagged` is that row's
    `n_flagged` (alerts actually reviewed), not the requested `k`, for the
    same tie-handling reason `capacity_sweep`'s own ceiling calculation uses
    `n_flagged` rather than `k`.
    """
    return (
        true_positives * avg_fraud_amount * recovery_rate
        - n_flagged * cost_per_review
        - false_negatives * avg_fraud_amount * liability_rate
    )


def net_value_curve(sweep_rows: list[dict], avg_fraud_amount: float,
                     cost_per_review: float, recovery_rate: float,
                     liability_rate: float) -> pd.DataFrame:
    """
    Adds `net_value` to every row of a capacity sweep (as produced by
    `threshold.capacity_sweep`), sorted by reviews_per_day. This is the
    curve the dashboard's capacity explorer plots.
    """
    rows = []
    for row in sweep_rows:
        rows.append({
            "reviews_per_day": row["reviews_per_day"],
            "k": row["k"],
            "n_flagged": row["n_flagged"],
            "true_positives": row["true_positives"],
            "false_positives": row["false_positives"],
            "false_negatives": row["false_negatives"],
            "precision": row.get("precision", float("nan")),
            "recall": row.get("recall", float("nan")),
            "net_value": net_value(
                row["true_positives"], row["n_flagged"], row["false_negatives"],
                avg_fraud_amount, cost_per_review, recovery_rate, liability_rate,
            ),
        })
    return pd.DataFrame(rows).sort_values("reviews_per_day").reset_index(drop=True)


def degeneracy_check(sweep_rows: list[dict], avg_fraud_amount: float,
                      recovery_rate_grid: list[float], liability_rate: float = 1.0,
                      low_reviews_per_day: float = 250, high_reviews_per_day: float = 500,
                      ) -> pd.DataFrame:
    """
    The break-even `cost_per_review` above which staffing UP from
    `low_reviews_per_day` to `high_reviews_per_day` stops being worth it --
    i.e. the cost at which net_value(high) == net_value(low).

    Solving net_value(high) > net_value(low) for cost_per_review:

        marginal_frauds * avg_fraud_amount * recovery_rate
      + marginal_frauds_avoided_as_missed * avg_fraud_amount * liability_rate
      > marginal_reviews * cost_per_review

    where marginal_frauds = tp(high) - tp(low) and, since raising capacity
    can only ever reduce (never increase) false negatives, the missed-fraud
    term collapses to the same marginal_frauds count -- catching a fraud and
    no-longer-missing it are the same event. `liability_rate` defaults to
    1.0 (a missed fraud is treated as a full loss) while `recovery_rate` is
    the swept parameter, matching the reading that recovery on a CAUGHT
    fraud is uncertain (money already moved) while a MISSED fraud's cost to
    the business is not in question.

    Returns one row per recovery_rate: the break-even cost_per_review, and
    whether that break-even is "high" in the sense of exceeding a token
    plausible review cost (never true for PaySim's amounts -- that is
    exactly the degenerate finding this function exists to make explicit
    rather than assert).
    """
    low = next(r for r in sweep_rows if r["reviews_per_day"] == low_reviews_per_day)
    high = next(r for r in sweep_rows if r["reviews_per_day"] == high_reviews_per_day)

    marginal_frauds = high["true_positives"] - low["true_positives"]
    marginal_reviews = high["n_flagged"] - low["n_flagged"]

    rows = []
    for recovery_rate in recovery_rate_grid:
        break_even_cost = (
            marginal_frauds * avg_fraud_amount * (recovery_rate + liability_rate)
            / marginal_reviews
        )
        rows.append({
            "recovery_rate": recovery_rate,
            "liability_rate": liability_rate,
            "marginal_frauds": marginal_frauds,
            "marginal_reviews": marginal_reviews,
            "break_even_cost_per_review": break_even_cost,
        })
    return pd.DataFrame(rows)


def capacity_constraint_cost(sweep_rows: list[dict], avg_fraud_amount: float,
                              low_reviews_per_day: float, high_reviews_per_day: float,
                              ) -> dict:
    """
    Question 1 (ARCHITECTURE §0): what does staffing only
    `low_reviews_per_day` instead of `high_reviews_per_day` cost?

    Linear and honest: frauds missed by the constraint, their dollar
    exposure, and the exposure expressed per additional analyst-seat-day so
    a manager can put a number on one more hire. Deliberately not an
    "optimum" -- this is a sensitivity, not a claim that `high_reviews_per_day`
    is the right number to staff.
    """
    low = next(r for r in sweep_rows if r["reviews_per_day"] == low_reviews_per_day)
    high = next(r for r in sweep_rows if r["reviews_per_day"] == high_reviews_per_day)

    frauds_missed_by_constraint = high["true_positives"] - low["true_positives"]
    exposure = frauds_missed_by_constraint * avg_fraud_amount
    marginal_seats = high_reviews_per_day - low_reviews_per_day

    return {
        "low_reviews_per_day": low_reviews_per_day,
        "high_reviews_per_day": high_reviews_per_day,
        "frauds_missed_by_constraint": frauds_missed_by_constraint,
        "avg_fraud_amount": avg_fraud_amount,
        "exposure": exposure,
        "marginal_seats": marginal_seats,
        "exposure_per_marginal_seat": exposure / marginal_seats if marginal_seats else float("nan"),
    }


def ticket_size_crossover(sweep_rows: list[dict], avg_fraud_amount_grid: list[float],
                           cost_per_review: float, recovery_rate: float,
                           liability_rate: float) -> pd.DataFrame:
    """
    Question 2 (ARCHITECTURE §0): sweeping `avg_fraud_amount` (not this
    fold's actual value -- a hypothetical range spanning UPI-scale tickets up
    to PaySim's own average) to find where the recommended staffing level
    stops being "as much capacity as available" and an under-staffed,
    higher-threshold policy becomes the higher-net-value choice.

    For each grid point, evaluates net_value at every capacity level already
    in `sweep_rows` and reports the argmax -- the capacity a decision-maker
    would pick at that ticket size, holding the other two business rates
    fixed. A `recommended_reviews_per_day` that decreases as `avg_fraud_amount`
    shrinks is the crossover in the shape the ARCHITECTURE doc describes it.
    """
    rows = []
    for avg_fraud_amount in avg_fraud_amount_grid:
        curve = net_value_curve(sweep_rows, avg_fraud_amount, cost_per_review,
                                 recovery_rate, liability_rate)
        best = curve.loc[curve["net_value"].idxmax()]
        rows.append({
            "avg_fraud_amount": avg_fraud_amount,
            "recommended_reviews_per_day": int(best["reviews_per_day"]),
            "net_value_at_recommendation": float(best["net_value"]),
        })
    df = pd.DataFrame(rows).sort_values("avg_fraud_amount").reset_index(drop=True)
    df["is_crossover"] = df["recommended_reviews_per_day"].ne(
        df["recommended_reviews_per_day"].shift()
    )
    df.loc[0, "is_crossover"] = False  # the first row has nothing to cross over from
    return df


if __name__ == "__main__":
    # Self-test with a synthetic sweep shaped like the real final-fold one:
    # recall saturates well before the top of the capacity grid.
    sweep_rows = [
        {"reviews_per_day": 100, "k": 1617, "n_flagged": 1619,
         "true_positives": 1619, "false_positives": 0, "false_negatives": 2631},
        {"reviews_per_day": 250, "k": 4042, "n_flagged": 4042,
         "true_positives": 4042, "false_positives": 0, "false_negatives": 208},
        {"reviews_per_day": 500, "k": 8083, "n_flagged": 8083,
         "true_positives": 4250, "false_positives": 3833, "false_negatives": 0},
    ]
    avg_fraud_amount = 1_572_443

    print("Degeneracy check (naive optimum):")
    print(degeneracy_check(sweep_rows, avg_fraud_amount,
                            recovery_rate_grid=[0.05, 0.5, 1.0]).to_string(index=False))

    print("\nCapacity constraint cost (250 vs 500/day):")
    for k, v in capacity_constraint_cost(sweep_rows, avg_fraud_amount, 250, 500).items():
        print(f"  {k}: {v}")

    print("\nTicket-size crossover:")
    grid = [100, 1_000, 10_000, 100_000, 1_000_000, 2_000_000]
    print(ticket_size_crossover(sweep_rows, grid, cost_per_review=200,
                                 recovery_rate=0.5, liability_rate=1.0).to_string(index=False))
