"""
Tests RAW_TRANSACTION_SCHEMA / validate_raw_sample against small hand-built
DuckDB tables -- a valid one, and one seeded with the exact defect classes
the schema exists to catch (bad enum value, out-of-range amount, null key).
"""

import duckdb
import pandas as pd
import pandera
import pytest

from schema import validate_raw_sample

VALID_ROW = dict(
    step=1, type="TRANSFER", amount=100.0, nameOrig="C1",
    oldbalanceOrg=1000.0, newbalanceOrig=900.0, nameDest="C2",
    oldbalanceDest=0.0, newbalanceDest=100.0, isFraud=0, isFlaggedFraud=0,
)


def _make_table(con, rows):
    df = pd.DataFrame(rows)
    df.insert(len(df.columns), "row_id", range(1, len(df) + 1))
    con.register("df", df)
    con.execute("CREATE TABLE transactions AS SELECT * FROM df")


def test_valid_rows_pass():
    con = duckdb.connect(":memory:")
    _make_table(con, [VALID_ROW, {**VALID_ROW, "amount": 50.0}])
    validate_raw_sample(con, n=10)  # must not raise
    con.close()


def test_invalid_type_enum_rejected():
    con = duckdb.connect(":memory:")
    _make_table(con, [{**VALID_ROW, "type": "NOT_A_REAL_TYPE"}])
    with pytest.raises(pandera.errors.SchemaErrors):
        validate_raw_sample(con, n=10)
    con.close()


def test_negative_amount_rejected():
    con = duckdb.connect(":memory:")
    _make_table(con, [{**VALID_ROW, "amount": -5.0}])
    with pytest.raises(pandera.errors.SchemaErrors):
        validate_raw_sample(con, n=10)
    con.close()


def test_isfraud_outside_zero_one_rejected():
    con = duckdb.connect(":memory:")
    _make_table(con, [{**VALID_ROW, "isFraud": 2}])
    with pytest.raises(pandera.errors.SchemaErrors):
        validate_raw_sample(con, n=10)
    con.close()
