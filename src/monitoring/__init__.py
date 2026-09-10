"""
monitoring/

Sprint 8. Population Stability Index over the dataset's own time axis
(ARCHITECTURE.md §10).

This package holds the detector only -- `drift.py`, pure numpy, no
pandas/duckdb/lightgbm. The job that feeds it lives at `src/run_drift.py`,
following the same split the rest of `src/` already uses: packages are
libraries, top-level scripts are runnable jobs (`export_bundle.py`,
`build_dest_state.py`).

Keeping the detector free of the data layer is what makes the injected-shift
tests possible: `tests/test_drift.py` constructs distributions directly and
asserts the detector fires, rather than needing a 6.36M-row DuckDB store to
exercise a PSI calculation. A detector that could only be tested by running the
whole job is a detector nobody tests.
"""
