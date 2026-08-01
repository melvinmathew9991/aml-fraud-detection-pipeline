"""
features.py

Feature engineering matching the JD's exact framing: "temporal, behavioral,
aggregated features" on "structured, semi-structured, and unstructured data."

Computed as a single SQL query over DuckDB rather than pandas: DuckDB
executes out-of-core (spills to disk instead of loading the full 6.36M-row
table into Python memory), and the account-level aggregates below are exactly
SQL window functions -- no need to hand-roll them with a full-frame sort +
groupby cumsum/cumcount the way the old pandas version did.

All features are computed leakage-safely: aggregates are built only from
information available strictly BEFORE the transaction in question (or from
static account-level history), never from the transaction's own outcome.

Sprint 2 added transaction-type indicators plus destination-side velocity and
graph (fan-in) aggregates. Which side those aggregates could live on was
decided by profiling the data, not by assumption -- see ACCOUNT STRUCTURE.
"""

# ---------------------------------------------------------------------------
# ACCOUNT STRUCTURE (profiled 2026-07-28 against the full 6.36M-row table)
#
#   rows 6,362,620 | distinct nameOrig 6,353,307 | distinct nameDest 2,722,362
#
# nameOrig is effectively a per-transaction identifier: 6,344,009 of the
# 6,353,307 origin accounts appear exactly ONCE, and only 0.15% of rows have
# any prior transaction from the same origin at all. Three consequences that
# shaped the Sprint 2 feature set:
#
#   1. Origin-side velocity ("transactions/hour per account", as the roadmap
#      originally specified) is not computable on this dataset. It would be
#      zero for 99.85% of rows. Velocity is therefore built on the
#      DESTINATION side, which is where PaySim's repeat structure actually
#      lives -- and which is also the more AML-relevant direction anyway: the
#      pattern worth detecting is a mule account receiving rapid inflows.
#
#   2. Fan-in as "distinct prior senders to this destination" is redundant
#      here. Because origins are essentially never reused, every
#      (nameOrig, nameDest) pair in the dataset is unique, so distinct-sender
#      count is exactly equal to prior-transaction count (verified: 0 rows
#      differ). Only the count is kept; shipping both would be two names for
#      one column. Fan-OUT per origin is degenerate for the same reason.
#
#   3. The pre-existing orig_prior_txn_count / orig_prior_avg_amount features
#      are near-dead (nonzero on 0.15% of rows). They are deliberately kept
#      rather than removed, so the Sprint 2 feature set is a superset of the
#      Sprint 1 one and the comparison stays apples-to-apples -- and SHAP is
#      expected to rank them near-worthless, which is a useful demonstration
#      that the explainability layer catches dead weight.
#
# Destination-side repeat structure is real but confined to non-merchant
# accounts: the 2,150,401 'M'-prefixed merchant destinations average 1.0005
# transactions each, while the 571,961 'C'-prefixed customer destinations
# average 7.36 (max 113). 57.2% of all rows have at least one prior
# transaction to the same destination.
# ---------------------------------------------------------------------------

# Bump whenever feature_query()'s SQL changes in a way that changes output
# values -- train_pipeline.py materializes the query result into a `features`
# table and reuses it across runs, keyed on this plus the raw CSV's
# mtime/size, so a stale materialized table can't silently serve features
# from an older version of this file.
# v2 (Sprint 2): + transaction-type indicators, destination velocity/fan-in.
FEATURE_VERSION = 2

# Velocity lookback in `step` units. PaySim's `step` is one simulated hour,
# so this is a 24-hour trailing window.
VELOCITY_WINDOW_HOURS = 24

# Features carried over from Sprint 1, unchanged. Kept as a separate list so
# the Sprint 2 ablation (the `cols_for`/`ablation_rows` block in
# train_pipeline.main) can train on exactly the old feature set without
# hardcoding a second copy of the names.
BASE_FEATURE_COLUMNS = [
    "amount", "hour_of_day", "is_night",
    "orig_balance_delta", "dest_balance_delta", "orig_balance_mismatch",
    "orig_emptied", "amount_to_balance_ratio", "dest_is_merchant",
    "orig_prior_txn_count", "orig_prior_avg_amount",
]

# Sprint 2 additions, split into the two independent groups they form so the
# ablation can attribute movement to each separately rather than crediting
# "Sprint 2" as a block.
# Transaction type. Fraud occurs in only 2 of the 5 types (TRANSFER 0.77%,
# CASH_OUT 0.18%; CASH_IN/PAYMENT/DEBIT are 0.00% across 3.59M rows), yet type
# was not a feature at all before Sprint 2 -- the model was only seeing it
# indirectly through dest_is_merchant. PAYMENT is the reference level (all four
# indicators zero) to avoid the dummy trap for the linear baselines.
TYPE_FEATURE_COLUMNS = [
    "is_transfer", "is_cash_out", "is_cash_in", "is_debit",
]

# Destination graph/history aggregates (strictly prior transactions), plus
# destination velocity over a trailing VELOCITY_WINDOW_HOURS window.
DEST_FEATURE_COLUMNS = [
    "dest_prior_txn_count", "dest_prior_avg_amount",
    "dest_amount_to_prior_avg_ratio",
    "dest_txn_count_24h", "dest_amount_sum_24h",
]

# Derived, never hand-maintained. This was previously a second hand-written
# copy of the two lists above, which meant editing one group without editing
# the copy would silently desynchronize the trained feature set from the
# ablation's per-group arms -- with no error raised, since both lists were
# independently valid.
SPRINT2_FEATURE_COLUMNS = TYPE_FEATURE_COLUMNS + DEST_FEATURE_COLUMNS

FEATURE_COLUMNS = BASE_FEATURE_COLUMNS + SPRINT2_FEATURE_COLUMNS


def feature_query(table: str = "transactions") -> str:
    """
    Builds FEATURE_COLUMNS plus step/isFraud (needed by the caller for the
    time-based split and labels) from the raw `table`.

    Leakage safety, per window:

    `w_orig` / `w_dest` (ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
        Each account's OWN PRIOR HISTORY ONLY -- the frame explicitly excludes
        the current row, the SQL equivalent of the leakage-safety pattern
        already used in the Clinical EMR and Nectar projects. Ties on `step`
        (multiple transactions in the same simulated hour) are broken by
        `row_id`, assigned once when the CSV is loaded into DuckDB, so
        ordering is deterministic across runs. COUNT/AVG over an empty window
        (an account's first transaction) return 0/NULL respectively; COALESCE
        handles the NULL case the same way the old pandas `.fillna(0)` did.

    `w_dest_velocity` (RANGE BETWEEN 24 PRECEDING AND 1 PRECEDING)
        A RANGE frame over `step` rather than a ROWS frame, so "last 24 hours"
        means 24 simulated hours of wall-clock, not the previous 24 rows. It
        ends at `1 PRECEDING`, which under RANGE semantics excludes every row
        sharing the current row's `step` -- so same-hour peer transactions are
        excluded too, not just the current row. That is deliberately
        conservative: intra-hour ordering in PaySim is an artifact of row
        order, so treating concurrent transactions as "not yet observed" is
        the defensible reading.

    All of these are also strictly past-only with respect to the CV folds:
    an aggregate for a row in a later fold's test window may draw on rows from
    an earlier training window, which is exactly what a production feature
    store would serve at scoring time, but never the reverse.
    """
    return f"""
        SELECT
            step,
            amount,
            CAST(step % 24 AS INTEGER) AS hour_of_day,
            CAST(step % 24 BETWEEN 0 AND 5 AS TINYINT) AS is_night,
            (oldbalanceOrg - newbalanceOrig) AS orig_balance_delta,
            (newbalanceDest - oldbalanceDest) AS dest_balance_delta,
            CAST(ABS((oldbalanceOrg - newbalanceOrig) - amount) > 0.01 AS TINYINT)
                AS orig_balance_mismatch,
            CAST(oldbalanceOrg > 0 AND newbalanceOrig = 0 AS TINYINT) AS orig_emptied,
            (amount / (oldbalanceOrg + 1)) AS amount_to_balance_ratio,
            CAST(nameDest LIKE 'M%' AS TINYINT) AS dest_is_merchant,
            COUNT(*) OVER w_orig AS orig_prior_txn_count,
            COALESCE(AVG(amount) OVER w_orig, 0) AS orig_prior_avg_amount,

            CAST(type = 'TRANSFER' AS TINYINT) AS is_transfer,
            CAST(type = 'CASH_OUT' AS TINYINT) AS is_cash_out,
            CAST(type = 'CASH_IN'  AS TINYINT) AS is_cash_in,
            CAST(type = 'DEBIT'    AS TINYINT) AS is_debit,

            COUNT(*) OVER w_dest AS dest_prior_txn_count,
            COALESCE(AVG(amount) OVER w_dest, 0) AS dest_prior_avg_amount,
            -- How unusual is this amount for this destination? The +1 guards
            -- the no-prior-history case (ratio degrades to `amount`), the
            -- same convention amount_to_balance_ratio above already uses.
            (amount / (COALESCE(AVG(amount) OVER w_dest, 0) + 1))
                AS dest_amount_to_prior_avg_ratio,

            COUNT(*) OVER w_dest_velocity AS dest_txn_count_24h,
            COALESCE(SUM(amount) OVER w_dest_velocity, 0) AS dest_amount_sum_24h,

            isFraud
        FROM {table}
        WINDOW
            w_orig AS (
                PARTITION BY nameOrig ORDER BY step, row_id
                ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
            ),
            w_dest AS (
                PARTITION BY nameDest ORDER BY step, row_id
                ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
            ),
            w_dest_velocity AS (
                PARTITION BY nameDest ORDER BY step
                RANGE BETWEEN {VELOCITY_WINDOW_HOURS} PRECEDING AND 1 PRECEDING
            )
    """
