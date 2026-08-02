"""
Skew test -- state (ARCHITECTURE.md §7): for a sample of real destinations,
asserts inference/state.py's lookup against the COMMITTED dest_state.parquet
matches a query computed directly and independently against the live
DuckDB table, at that table's current max(step).

This isolates the STATE half of the training/serving skew problem
(ARCHITECTURE.md §2) from the PLUMBING half test_skew_plumbing.py checks --
a failure here means the hashed-array load/lookup path in state.py (or a
stale committed snapshot) disagrees with what the source table actually
contains, not a model/scaler bug.

Coupled to the local DuckDB store matching the table the committed
model_bundle/v1/dest_state.parquet was built from -- if the store has been
regenerated or grown since, this test's independent recomputation will
(correctly) stop agreeing with the frozen snapshot and should be re-run
after `build_dest_state.py` + `export_bundle.py` regenerate the bundle.
Skipped (not failed) if either the local DuckDB store or the bundle is
absent, matching test_golden_file.py's pattern for optional local state.
"""

from pathlib import Path

import duckdb
import pytest

from config import PROJECT_ROOT, load_config
from inference.state import load_dest_state

BUNDLE_DIR = PROJECT_ROOT / "model_bundle" / "v1"
VELOCITY_WINDOW_HOURS = 24
SAMPLE_SIZE = 30


def _require_bundle():
    if not (BUNDLE_DIR / "dest_state.parquet").exists():
        pytest.skip("model_bundle/v1/dest_state.parquet not generated yet.")


def _require_duckdb_store():
    config = load_config()
    db_path = PROJECT_ROOT / config["data"]["processed_dir"] / "paysim.duckdb"
    if not db_path.exists():
        pytest.skip(f"No local DuckDB store at {db_path} -- run train_pipeline.py first.")
    return db_path


def test_state_lookup_matches_independent_duckdb_query():
    _require_bundle()
    db_path = _require_duckdb_store()

    ds = load_dest_state(BUNDLE_DIR / "dest_state.parquet")

    con = duckdb.connect(str(db_path), read_only=True)
    try:
        snapshot_step = con.execute("SELECT MAX(step) FROM transactions").fetchone()[0]

        sample = con.sql(f"""
            SELECT DISTINCT nameDest FROM transactions
            WHERE nameDest NOT LIKE 'M%'
            USING SAMPLE {SAMPLE_SIZE} ROWS (reservoir)
        """).df()["nameDest"].tolist()

        assert len(sample) > 0, "sample query returned no destinations"

        for name_dest in sample:
            independent = con.execute(f"""
                SELECT
                    COUNT(*) AS prior_txn_count,
                    AVG(amount) AS prior_avg_amount,
                    COUNT(*) FILTER (WHERE step > ? - {VELOCITY_WINDOW_HOURS}) AS txn_count_24h,
                    COALESCE(SUM(amount) FILTER (WHERE step > ? - {VELOCITY_WINDOW_HOURS}), 0)
                        AS amount_sum_24h
                FROM transactions
                WHERE nameDest = ?
            """, [snapshot_step, snapshot_step, name_dest]).fetchone()

            expected = {
                "prior_txn_count": int(independent[0]),
                "prior_avg_amount": float(independent[1]),
                "txn_count_24h": int(independent[2]),
                "amount_sum_24h": float(independent[3]),
            }
            actual, hit = ds.lookup(name_dest)
            assert hit is True, f"{name_dest} present in source table but missing from snapshot"

            assert actual["prior_txn_count"] == expected["prior_txn_count"], name_dest
            assert actual["prior_avg_amount"] == pytest.approx(
                expected["prior_avg_amount"], rel=1e-5), name_dest
            assert actual["txn_count_24h"] == expected["txn_count_24h"], name_dest
            assert actual["amount_sum_24h"] == pytest.approx(
                expected["amount_sum_24h"], rel=1e-5), name_dest
    finally:
        con.close()


def test_state_lookup_merchant_destinations_are_cold_start():
    _require_bundle()
    db_path = _require_duckdb_store()

    ds = load_dest_state(BUNDLE_DIR / "dest_state.parquet")

    con = duckdb.connect(str(db_path), read_only=True)
    try:
        merchants = con.sql("""
            SELECT DISTINCT nameDest FROM transactions
            WHERE nameDest LIKE 'M%'
            USING SAMPLE 10 ROWS (reservoir)
        """).df()["nameDest"].tolist()
    finally:
        con.close()

    assert len(merchants) > 0
    for name_dest in merchants:
        _, hit = ds.lookup(name_dest)
        assert hit is False, f"merchant {name_dest} unexpectedly present in dest_state snapshot"
