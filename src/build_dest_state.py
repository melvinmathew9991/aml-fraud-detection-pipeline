"""
build_dest_state.py

Emits the per-destination state snapshot that resolves the training/serving
skew problem described in ARCHITECTURE.md §2: five of the model's features
are stateful aggregates over destination account history, computed at
training time as SQL window functions over the full raw table. A single
incoming transaction at serving time has no window to look back through, so
this snapshot is what the API consults instead.

Frozen at the end of the dataset (the max `step` seen), which is the
snapshot's documented limitation (ARCHITECTURE §2.1): it is a point-in-time
approximation, not an online feature store. Every non-merchant destination's
*entire* history counts as "prior" here -- unlike the training-time window
functions, which explicitly exclude the current row -- because a snapshot
taken after the last training row has no "current row" of its own; every
row that happened, happened before whatever transaction scores against this
snapshot next. That is also why cold start (an unknown destination) and a
destination's real first transaction produce the same zeros: both are "no
history yet," computed the same way.

Merchant destinations ('M%') are NOT stored -- they average ~1.0005
transactions each, so their prior history is ~always empty, and dropping
them removes 79% of rows for negligible fidelity loss (ARCHITECTURE §2).
They resolve to the cold-start default at serving time instead.
"""

import argparse
import logging
import sys
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from config import PROJECT_ROOT, load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("build_dest_state")

VELOCITY_WINDOW_HOURS = 24  # matches features.VELOCITY_WINDOW_HOURS
MAX_BUNDLE_SIZE_MB = 20  # Sprint 3 DoD ceiling

# Embedded as parquet file-level metadata rather than a separate sidecar
# file, so the snapshot's own artifact is the single source of truth for
# when it was taken -- ARCHITECTURE.md §5 requires /model-info to report
# this ("snapshot as-of step"). inference/state.py reads the same key; kept
# in sync by convention (a hardcoded string in both, not a shared import --
# state.py cannot depend on this training-only module, which pulls in
# duckdb).
SNAPSHOT_STEP_METADATA_KEY = b"snapshot_step"


def build_dest_state(con: duckdb.DuckDBPyConnection, table: str = "transactions"):
    """
    Returns (dataframe, snapshot_step). One row per non-merchant destination
    that has ever appeared in `table`, aggregated over its ENTIRE history
    (not "prior to some row" -- there is no current row for a snapshot taken
    after the dataset ends) plus a trailing-24h velocity slice ending at the
    snapshot step.
    """
    snapshot_step = con.execute(f"SELECT MAX(step) FROM {table}").fetchone()[0]

    df = con.sql(f"""
        SELECT
            nameDest AS name_dest,
            COUNT(*) AS prior_txn_count,
            AVG(amount) AS prior_avg_amount,
            COUNT(*) FILTER (WHERE step > {snapshot_step} - {VELOCITY_WINDOW_HOURS})
                AS txn_count_24h,
            COALESCE(SUM(amount) FILTER (
                WHERE step > {snapshot_step} - {VELOCITY_WINDOW_HOURS}
            ), 0) AS amount_sum_24h
        FROM {table}
        WHERE nameDest NOT LIKE 'M%'
        GROUP BY nameDest
    """).df()

    df["prior_txn_count"] = df["prior_txn_count"].astype("int32")
    df["prior_avg_amount"] = df["prior_avg_amount"].astype("float64")
    df["txn_count_24h"] = df["txn_count_24h"].astype("int32")
    df["amount_sum_24h"] = df["amount_sum_24h"].astype("float64")

    return df, snapshot_step


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=PROJECT_ROOT / "model_bundle" / "v1" / "dest_state.parquet",
        help="Output parquet path (default: model_bundle/v1/dest_state.parquet)",
    )
    args = parser.parse_args()

    config = load_config()
    db_path = PROJECT_ROOT / config["data"]["processed_dir"] / "paysim.duckdb"
    if not db_path.exists():
        logger.error("No DuckDB store at %s -- run train_pipeline.py first.", db_path)
        sys.exit(1)

    con = duckdb.connect(str(db_path), read_only=True)
    try:
        df, snapshot_step = build_dest_state(con)
    finally:
        con.close()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df, preserve_index=False)
    table = table.replace_schema_metadata({
        **(table.schema.metadata or {}),
        SNAPSHOT_STEP_METADATA_KEY: str(snapshot_step).encode("utf-8"),
    })
    pq.write_table(table, args.output, compression="zstd")

    size_mb = args.output.stat().st_size / (1024 ** 2)
    logger.info("Wrote %d destination rows (snapshot as of step %d) to %s (%.2f MB)",
                len(df), snapshot_step, args.output, size_mb)
    logger.info("  prior_txn_count: mean %.2f, max %d (every stored row has history by "
                "construction -- destinations with none are the cold-start case, resolved "
                "at serving time by NOT appearing here at all)",
                df["prior_txn_count"].mean(), int(df["prior_txn_count"].max()))

    if size_mb > MAX_BUNDLE_SIZE_MB:
        logger.error("dest_state.parquet is %.2f MB, exceeding the %d MB Sprint 3 budget.",
                     size_mb, MAX_BUNDLE_SIZE_MB)
        sys.exit(1)


if __name__ == "__main__":
    main()
