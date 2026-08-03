"""
generate_dashboard_sample.py

Draws the bundled 50,000-row sample the Streamlit dashboard's Batch upload
page (Sprint 5, ARCHITECTURE.md §6) scores when the user hasn't uploaded a
CSV -- so that page works with zero upload, the same reasoning
generate_golden_file.py already applies to the golden-file test.

Stratification: every fraud transaction in the dataset (8,213 rows) plus a
random sample of legitimate ones filling the remainder to 50,000 -- not a
fixed ratio picked to look nice. That keeps the design simple to state and
honest (no fraud is invented or duplicated) while giving the demo enough
positive cases to be worth clicking through: a purely random 50k draw at
PaySim's real ~0.13% fraud rate would carry only ~65 fraud rows, most of
which wouldn't stand out in a UI table.

Columns match schema.py's RAW_TRANSACTION_SCHEMA exactly -- this is what a
real analyst's CSV upload looks like, raw and unscored. isFraud is included
so the dashboard can show ground truth for demo purposes (a real production
upload would not have it; the API's /score/batch request schema doesn't
accept or require it -- the dashboard drops it before calling the API).

Deliberately reimplements the minimal read path against the raw
`transactions` table rather than importing train_pipeline.py, for the same
reason generate_golden_file.py does: no import-time side effects.
"""

import argparse
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from config import PROJECT_ROOT, load_config

RAW_COLUMNS = [
    "step", "type", "amount", "nameOrig", "oldbalanceOrg", "newbalanceOrig",
    "nameDest", "oldbalanceDest", "newbalanceDest", "isFraud", "isFlaggedFraud",
]
SAMPLE_SIZE = 50_000


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                         default=PROJECT_ROOT / "dashboard" / "sample_transactions.csv")
    args = parser.parse_args()

    config = load_config()
    db_path = PROJECT_ROOT / config["data"]["processed_dir"] / "paysim.duckdb"
    if not db_path.exists():
        raise FileNotFoundError(f"No DuckDB store at {db_path} -- run train_pipeline.py first.")

    con = duckdb.connect(str(db_path), read_only=True)
    try:
        fraud = con.sql(f"""
            SELECT {', '.join(RAW_COLUMNS)} FROM transactions WHERE isFraud = 1
        """).df()

        n_nonfraud = SAMPLE_SIZE - len(fraud)
        if n_nonfraud < 0:
            raise ValueError(
                f"{len(fraud)} fraud rows exceed the {SAMPLE_SIZE}-row sample budget -- "
                "raise SAMPLE_SIZE or cap the fraud draw."
            )
        # ORDER BY random() + LIMIT rather than USING SAMPLE ... (reservoir,
        # seed): DuckDB's reservoir sampler merges per-thread reservoirs and
        # can silently return fewer rows than requested on a large parallel
        # scan (observed: 41,743 of 41,787 requested) -- fine for a training
        # feature-set sample where the exact count doesn't matter, not fine
        # here where the script's own docstring promises 50,000 rows.
        # setseed() makes the sort's random() draw reproducible.
        con.execute(f"SELECT setseed({(config['random_state'] % 1000) / 1000})")
        nonfraud = con.sql(f"""
            SELECT {', '.join(RAW_COLUMNS)} FROM transactions WHERE isFraud = 0
            ORDER BY random() LIMIT {n_nonfraud}
        """).df()
    finally:
        con.close()

    rng = np.random.default_rng(config["random_state"])
    sample = pd.concat([fraud, nonfraud], ignore_index=True)
    sample = sample.iloc[rng.permutation(len(sample))].reset_index(drop=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(args.output, index=False)

    print(f"Wrote {len(sample)} rows ({int(sample['isFraud'].sum())} fraud, "
          f"{sample['isFraud'].mean() * 100:.2f}% fraud rate) to {args.output}")
    print(f"  file size: {args.output.stat().st_size / (1024 ** 2):.2f} MB")


if __name__ == "__main__":
    main()
