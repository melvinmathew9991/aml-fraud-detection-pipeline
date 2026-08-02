"""
Unit tests for inference/state.py: the hashed-array lookup, cold-start
behavior, and the collision guard ARCHITECTURE.md §2 requires ("checked at
build time and the build fails rather than serving a silently wrong row").
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from inference.state import COLD_START, DestState, DestStateCollisionError, load_dest_state


def _write_parquet(tmp_path: Path, rows: list[dict]) -> Path:
    path = tmp_path / "dest_state.parquet"
    pd.DataFrame(rows).to_parquet(path, engine="pyarrow", index=False)
    return path


def test_load_and_lookup_known_destination(tmp_path):
    path = _write_parquet(tmp_path, [
        {"name_dest": "C1", "prior_txn_count": 5, "prior_avg_amount": 200.0,
         "txn_count_24h": 2, "amount_sum_24h": 150.0},
        {"name_dest": "C2", "prior_txn_count": 0, "prior_avg_amount": 0.0,
         "txn_count_24h": 0, "amount_sum_24h": 0.0},
    ])
    ds = load_dest_state(path)

    values, hit = ds.lookup("C1")
    assert hit is True
    assert values == {"prior_txn_count": 5, "prior_avg_amount": 200.0,
                       "txn_count_24h": 2, "amount_sum_24h": 150.0}


def test_lookup_unknown_destination_returns_cold_start(tmp_path):
    path = _write_parquet(tmp_path, [
        {"name_dest": "C1", "prior_txn_count": 5, "prior_avg_amount": 200.0,
         "txn_count_24h": 2, "amount_sum_24h": 150.0},
    ])
    ds = load_dest_state(path)
    values, hit = ds.lookup("C_NEVER_SEEN")
    assert hit is False
    assert values == COLD_START


def test_lookup_on_empty_snapshot_is_cold_start():
    ds = DestState(
        keys=np.array([], dtype="uint64"), count=np.array([], dtype="int32"),
        avg=np.array([], dtype="float32"), c24=np.array([], dtype="int32"),
        s24=np.array([], dtype="float32"),
    )
    values, hit = ds.lookup("anything")
    assert hit is False
    assert values == COLD_START


def test_lookup_many_matches_lookup_one_at_a_time(tmp_path):
    rows = [
        {"name_dest": f"C{i}", "prior_txn_count": i, "prior_avg_amount": float(i) * 10,
         "txn_count_24h": i % 3, "amount_sum_24h": float(i) * 2}
        for i in range(50)
    ]
    path = _write_parquet(tmp_path, rows)
    ds = load_dest_state(path)

    names = [f"C{i}" for i in range(0, 50, 7)] + ["C_UNKNOWN"]
    batch_results, batch_hits = ds.lookup_many(names)
    for i, name in enumerate(names):
        single_values, single_hit = ds.lookup(name)
        assert batch_results[i] == single_values
        assert bool(batch_hits[i]) == single_hit


def test_hash_collision_raises_and_refuses_to_load(tmp_path, monkeypatch):
    path = _write_parquet(tmp_path, [
        {"name_dest": "C1", "prior_txn_count": 1, "prior_avg_amount": 1.0,
         "txn_count_24h": 1, "amount_sum_24h": 1.0},
        {"name_dest": "C2", "prior_txn_count": 2, "prior_avg_amount": 2.0,
         "txn_count_24h": 2, "amount_sum_24h": 2.0},
    ])

    import inference.state as state_module

    def fake_hash_dest(name_dest):
        return np.uint64(42)  # force a collision between C1 and C2

    monkeypatch.setattr(state_module, "_hash_dest", fake_hash_dest)
    with pytest.raises(DestStateCollisionError):
        load_dest_state(path)
