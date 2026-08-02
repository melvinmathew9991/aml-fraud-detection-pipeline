"""
Tests feature_query()'s SQL directly against a small, hand-built DuckDB
table -- not the 6.36M-row production data -- so every case here has an
independently hand-computed expected value.

Covers:
  * first-transaction cases (COUNT/AVG over an empty window -> 0, per the
    COALESCE convention)
  * div-by-zero guards (amount_to_balance_ratio, dest_amount_to_prior_avg_ratio)
  * leakage safety: dest_prior_txn_count only sees rows strictly before the
    current one in (step, row_id) order
  * the RANGE-vs-ROWS distinction for velocity: dest_txn_count_24h excludes
    same-step ("same simulated hour") peers entirely, even though the
    ROWS-framed prior-count aggregate includes them (ties broken by row_id)
  * merchant detection and transaction-type indicators
"""

import duckdb
import pandas as pd
import pytest

from features import FEATURE_COLUMNS, feature_query

# Six hand-built rows, row_id assigned by insertion order (mirrors what
# _ensure_transactions_table's row_number() OVER () does on the real CSV).
#
#   row 1: step=0,  C1, amount=100  (first txn to C1)
#   row 2: step=0,  C1, amount=50   (same step as row1 -- tie case)
#   row 3: step=10, C1, amount=200  (10h after row1/2 -- within 24h velocity)
#   row 4: step=30, C1, amount=400  (30h after row1/2 -- outside their 24h
#                                     window, but within 24h of row3)
#   row 5: step=5,  C2, amount=75   (first txn to C2; oldbalanceOrg=0 ->
#                                     div-by-zero guard on amount_to_balance_ratio)
#   row 6: step=5,  M-prefixed dest, amount=60  (merchant destination)
ROWS = [
    # step, type,       amount, nameOrig, oldOrig, newOrig, nameDest,     oldDest, newDest, isFraud
    (0,  "TRANSFER", 100.0, "CORIG1", 1000.0, 900.0, "C1",         0.0, 100.0, 0),
    (0,  "TRANSFER", 50.0,  "CORIG2", 500.0,  450.0, "C1",         0.0, 50.0,  0),
    (10, "CASH_OUT", 200.0, "CORIG3", 300.0,  100.0, "C1",         0.0, 200.0, 0),
    (30, "CASH_OUT", 400.0, "CORIG4", 0.0,    0.0,   "C1",         0.0, 400.0, 0),
    (5,  "PAYMENT",  75.0,  "CORIG5", 0.0,    0.0,   "C2",         0.0, 75.0,  0),
    (5,  "PAYMENT",  60.0,  "CORIG6", 200.0,  140.0, "M999999999", 0.0, 60.0,  0),
]


@pytest.fixture
def con():
    con = duckdb.connect(":memory:")
    df = pd.DataFrame(ROWS, columns=[
        "step", "type", "amount", "nameOrig", "oldbalanceOrg", "newbalanceOrig",
        "nameDest", "oldbalanceDest", "newbalanceDest", "isFraud",
    ])
    df.insert(len(df.columns), "row_id", range(1, len(df) + 1))
    con.register("df", df)
    con.execute("CREATE TABLE transactions AS SELECT * FROM df")
    yield con
    con.close()


@pytest.fixture
def feat(con):
    result = con.sql(feature_query()).df()
    # Recover which hand-built row each output row came from by amount,
    # which is unique per row in this fixture.
    return result.set_index(result["amount"])


def test_feature_query_returns_expected_columns(feat):
    for col in FEATURE_COLUMNS:
        assert col in feat.columns


def test_first_transaction_defaults_to_zero_not_null(feat):
    row1 = feat.loc[100.0]
    assert row1["dest_prior_txn_count"] == 0
    assert row1["dest_prior_avg_amount"] == 0
    assert row1["dest_txn_count_24h"] == 0
    assert row1["dest_amount_sum_24h"] == 0


def test_div_by_zero_guard_on_dest_prior_avg_ratio(feat):
    # row1: no prior history to C1 -> ratio degrades to amount / (0 + 1) = amount
    row1 = feat.loc[100.0]
    assert row1["dest_amount_to_prior_avg_ratio"] == pytest.approx(100.0)


def test_div_by_zero_guard_on_orig_balance_ratio(feat):
    # row5 (C2): oldbalanceOrg = 0 -> amount_to_balance_ratio = amount / (0+1)
    row5 = feat.loc[75.0]
    assert row5["amount_to_balance_ratio"] == pytest.approx(75.0)


def test_prior_txn_count_counts_same_step_peer_via_row_id_tiebreak(feat):
    # row2 is the same step (0) as row1 but a later row_id -> the ROWS frame
    # (ordered by step, row_id) counts row1 as prior for row2.
    row2 = feat.loc[50.0]
    assert row2["dest_prior_txn_count"] == 1
    assert row2["dest_prior_avg_amount"] == pytest.approx(100.0)


def test_velocity_range_frame_excludes_same_step_peers(feat):
    # row2 shares row1's step (0) exactly -- under the RANGE frame's peer-
    # group semantics, "1 PRECEDING" excludes the ENTIRE current step's
    # peers, unlike the ROWS-framed prior-count above. So row2's velocity
    # sees nothing, even though its prior-count aggregate saw row1.
    row2 = feat.loc[50.0]
    assert row2["dest_txn_count_24h"] == 0
    assert row2["dest_amount_sum_24h"] == 0


def test_velocity_window_includes_prior_rows_within_24_simulated_hours(feat):
    # row3 (step=10): rows 1 and 2 (step=0) are both within the trailing 24h.
    row3 = feat.loc[200.0]
    assert row3["dest_prior_txn_count"] == 2
    assert row3["dest_prior_avg_amount"] == pytest.approx(75.0)  # (100+50)/2
    assert row3["dest_txn_count_24h"] == 2
    assert row3["dest_amount_sum_24h"] == pytest.approx(150.0)


def test_velocity_window_excludes_rows_older_than_24_simulated_hours(feat):
    # row4 (step=30): rows 1/2 (step=0) are 30h back -- outside the 24h
    # velocity window -- but row3 (step=10) is only 20h back -- inside it.
    # The plain prior-count aggregate (unbounded lookback) still sees all
    # three, which is exactly the leakage-safe-but-unbounded vs.
    # bounded-velocity distinction the two feature groups exist to capture.
    row4 = feat.loc[400.0]
    assert row4["dest_prior_txn_count"] == 3
    assert row4["dest_txn_count_24h"] == 1
    assert row4["dest_amount_sum_24h"] == pytest.approx(200.0)


def test_merchant_destination_flagged(feat):
    row6 = feat.loc[60.0]
    assert row6["dest_is_merchant"] == 1
    row1 = feat.loc[100.0]
    assert row1["dest_is_merchant"] == 0


def test_transaction_type_indicators(feat):
    row1 = feat.loc[100.0]  # TRANSFER
    assert row1["is_transfer"] == 1
    assert row1["is_cash_out"] == 0
    assert row1["is_cash_in"] == 0
    assert row1["is_debit"] == 0

    row3 = feat.loc[200.0]  # CASH_OUT
    assert row3["is_transfer"] == 0
    assert row3["is_cash_out"] == 1


def test_hour_of_day_and_is_night(feat):
    row1 = feat.loc[100.0]  # step 0
    assert row1["hour_of_day"] == 0
    assert row1["is_night"] == 1

    row3 = feat.loc[200.0]  # step 10
    assert row3["hour_of_day"] == 10
    assert row3["is_night"] == 0


def test_orig_balance_mismatch_flags_discrepancy(feat):
    # row1: oldbalanceOrg - newbalanceOrig = 100, amount = 100 -> no mismatch
    row1 = feat.loc[100.0]
    assert row1["orig_balance_mismatch"] == 0


def test_dead_features_are_gone(feat):
    assert "orig_prior_txn_count" not in feat.columns
    assert "orig_prior_avg_amount" not in feat.columns
