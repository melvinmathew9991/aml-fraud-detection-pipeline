"""
Tests build_dest_state's aggregation logic against a small hand-built table
-- merchant exclusion, whole-history (not "prior-row") counting, and the
trailing-24h velocity slice ending at the snapshot step.
"""

import duckdb
import pandas as pd
import pytest

from build_dest_state import build_dest_state

ROWS = [
    # step, amount, nameDest
    (0,  100.0, "C1"),
    (0,  50.0,  "C1"),
    (10, 200.0, "C1"),
    (30, 400.0, "C1"),
    (5,  75.0,  "C2"),
    (5,  60.0,  "M999999999"),  # merchant -- must not appear in the output
]


@pytest.fixture
def con():
    con = duckdb.connect(":memory:")
    df = pd.DataFrame(ROWS, columns=["step", "amount", "nameDest"])
    con.register("df", df)
    con.execute("CREATE TABLE transactions AS SELECT * FROM df")
    yield con
    con.close()


def test_merchants_excluded(con):
    df, _ = build_dest_state(con)
    assert "M999999999" not in df["name_dest"].values
    assert set(df["name_dest"]) == {"C1", "C2"}


def test_snapshot_step_is_max_step(con):
    _, snapshot_step = build_dest_state(con)
    assert snapshot_step == 30


def test_whole_history_counted_not_just_prior_rows(con):
    # Unlike the training-time window function, the snapshot has no "current
    # row" of its own -- every one of C1's 4 rows counts.
    df, _ = build_dest_state(con)
    c1 = df.set_index("name_dest").loc["C1"]
    assert c1["prior_txn_count"] == 4
    assert c1["prior_avg_amount"] == pytest.approx((100 + 50 + 200 + 400) / 4)


def test_velocity_window_ends_at_snapshot_step(con):
    # snapshot_step=30, 24h window -> step > 6. Only row3 (step=10) and row4
    # (step=30) qualify; rows 1/2 (step=0) are outside it.
    df, _ = build_dest_state(con)
    c1 = df.set_index("name_dest").loc["C1"]
    assert c1["txn_count_24h"] == 2
    assert c1["amount_sum_24h"] == pytest.approx(200 + 400)


def test_single_transaction_destination(con):
    df, _ = build_dest_state(con)
    c2 = df.set_index("name_dest").loc["C2"]
    assert c2["prior_txn_count"] == 1
    assert c2["prior_avg_amount"] == pytest.approx(75.0)
