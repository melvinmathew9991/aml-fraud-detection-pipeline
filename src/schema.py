"""
schema.py

A pandera schema for the raw PaySim table, validated at ingest time so a
malformed or unexpectedly-shaped CSV fails loudly before 28 minutes of
training rather than producing silently wrong features.

Deliberately validated against a SAMPLE, not the full 6.36M-row table:
pandera validates pandas DataFrames, and materializing all 6.36M rows into
pandas just to check dtypes/ranges is exactly the full-frame-in-memory cost
`features.py`'s DuckDB rewrite exists to avoid (see its docstring and
ROADMAP's data-layer-hardening entry). A schema violation that exists
anywhere in the table has a high chance of appearing in a few hundred
thousand sampled rows too, and this is the same sampling tradeoff
`explainability.shap_sample_rows` already makes for the same reason -- traded
off explicitly here rather than silently.
"""

import duckdb
import pandera.pandas as pa
from pandera.pandas import Column, Check

VALID_TYPES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]

RAW_TRANSACTION_SCHEMA = pa.DataFrameSchema(
    {
        "step": Column(int, Check.ge(0), nullable=False),
        "type": Column(str, Check.isin(VALID_TYPES), nullable=False),
        "amount": Column(float, Check.ge(0), nullable=False),
        "nameOrig": Column(str, nullable=False),
        "oldbalanceOrg": Column(float, Check.ge(0), nullable=False),
        "newbalanceOrig": Column(float, Check.ge(0), nullable=False),
        "nameDest": Column(str, nullable=False),
        "oldbalanceDest": Column(float, Check.ge(0), nullable=False),
        "newbalanceDest": Column(float, Check.ge(0), nullable=False),
        "isFraud": Column(int, Check.isin([0, 1]), nullable=False),
        "isFlaggedFraud": Column(int, Check.isin([0, 1]), nullable=False),
    },
    strict=False,  # the table also carries `row_id`, added by the loader
    coerce=False,  # a wrong dtype should fail the check, not be silently cast
)


def validate_raw_sample(con: duckdb.DuckDBPyConnection, table: str = "transactions",
                         n: int = 50_000) -> None:
    """
    Validates `n` rows of `table` against RAW_TRANSACTION_SCHEMA.

    Uses DuckDB's reservoir sample (`USING SAMPLE`), not `ORDER BY random()`
    -- the latter forces a full sort over every row, which is the exact cost
    this function exists to avoid paying.

    Raises pandera.errors.SchemaErrors (via lazy=True) listing every failing
    row/column at once, rather than stopping at the first violation.
    """
    sample_df = con.sql(
        f"SELECT * FROM {table} USING SAMPLE {n} ROWS (reservoir)"
    ).df()
    RAW_TRANSACTION_SCHEMA.validate(sample_df, lazy=True)
