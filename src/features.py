"""
features.py

Feature engineering matching the JD's exact framing: "temporal, behavioral,
aggregated features" on "structured, semi-structured, and unstructured data."

Computed as a single SQL query over DuckDB rather than pandas: DuckDB
executes out-of-core (spills to disk instead of loading the full 6.36M-row
table into Python memory), and the account-level aggregate below is exactly
a SQL window function -- no need to hand-roll it with a full-frame sort +
groupby cumsum/cumcount the way the old pandas version did.

All features are computed leakage-safely: aggregates are built only from
information available strictly BEFORE the transaction in question (or from
static account-level history), never from the transaction's own outcome.
"""

# Bump whenever feature_query()'s SQL changes in a way that changes output
# values -- train_pipeline.py materializes the query result into a `features`
# table and reuses it across runs, keyed on this plus the raw CSV's
# mtime/size, so a stale materialized table can't silently serve features
# from an older version of this file.
FEATURE_VERSION = 1

FEATURE_COLUMNS = [
    "amount", "hour_of_day", "is_night",
    "orig_balance_delta", "dest_balance_delta", "orig_balance_mismatch",
    "orig_emptied", "amount_to_balance_ratio", "dest_is_merchant",
    "orig_prior_txn_count", "orig_prior_avg_amount",
]


def feature_query(table: str = "transactions") -> str:
    """
    Builds FEATURE_COLUMNS plus step/isFraud (needed by the caller for the
    time-based split and labels) from the raw `table`.

    orig_prior_txn_count / orig_prior_avg_amount: each account's OWN PRIOR
    HISTORY ONLY, via a window frame that explicitly excludes the current
    row (ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) -- the SQL
    equivalent of the leakage-safety pattern already used in the Clinical
    EMR and Nectar projects. Ties on `step` (multiple transactions in the
    same simulated hour) are broken by `row_id`, assigned once when the CSV
    is loaded into DuckDB, so ordering is deterministic across runs.
    COUNT/AVG over an empty window (an account's first transaction) return
    0/NULL respectively; COALESCE handles the NULL case the same way the
    old pandas `.fillna(0)` did.
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
            COUNT(*) OVER w AS orig_prior_txn_count,
            COALESCE(AVG(amount) OVER w, 0) AS orig_prior_avg_amount,
            isFraud
        FROM {table}
        WINDOW w AS (
            PARTITION BY nameOrig ORDER BY step, row_id
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        )
    """
