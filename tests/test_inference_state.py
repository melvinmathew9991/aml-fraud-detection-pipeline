"""
Unit tests for inference/state.py: the hashed-array lookup, cold-start
behavior, and the collision guard ARCHITECTURE.md §2 requires ("checked at
build time and the build fails rather than serving a silently wrong row").
"""

from pathlib import Path

import numpy as np
import pytest

from inference.state import (
    COLD_START,
    DestState,
    DestStateCollisionError,
    hash_many,
    load_dest_state,
)


def _write_state(tmp_path: Path, rows: list[dict], snapshot_step: int | None = None,
                 sort: bool = True) -> Path:
    """Write a dest_state.npz the way build_dest_state.py does.

    The v2 bundle ships precomputed, pre-sorted hash keys instead of account
    names, so these fixtures hash and sort here -- the same work the build does
    once, which the serving path no longer repeats on every cold start.

    `sort=False` writes them in the given order, which is how the "unsorted keys
    are refused" test produces a bundle the loader must reject.
    """
    path = tmp_path / "dest_state.npz"
    keys = hash_many([r["name_dest"] for r in rows])
    order = np.argsort(keys, kind="stable") if sort else np.arange(len(keys))
    np.savez_compressed(
        path,
        keys=keys[order],
        count=np.array([r["prior_txn_count"] for r in rows], dtype="int32")[order],
        avg=np.array([r["prior_avg_amount"] for r in rows], dtype="float32")[order],
        c24=np.array([r["txn_count_24h"] for r in rows], dtype="int32")[order],
        s24=np.array([r["amount_sum_24h"] for r in rows], dtype="float32")[order],
        snapshot_step=(np.array([], dtype="int64") if snapshot_step is None
                       else np.array([snapshot_step], dtype="int64")),
    )
    return path


def test_load_and_lookup_known_destination(tmp_path):
    path = _write_state(tmp_path, [
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


def test_snapshot_step_round_trips_through_the_npz(tmp_path):
    path = _write_state(tmp_path, [
        {"name_dest": "C1", "prior_txn_count": 5, "prior_avg_amount": 200.0,
         "txn_count_24h": 2, "amount_sum_24h": 150.0},
    ], snapshot_step=743)
    ds = load_dest_state(path)
    assert ds.snapshot_step == 743


def test_snapshot_step_none_when_metadata_absent(tmp_path):
    # Older-format parquet with no embedded metadata (pre-Sprint-5) must not
    # crash the loader -- just report snapshot_step as unknown.
    path = _write_state(tmp_path, [
        {"name_dest": "C1", "prior_txn_count": 5, "prior_avg_amount": 200.0,
         "txn_count_24h": 2, "amount_sum_24h": 150.0},
    ])
    ds = load_dest_state(path)
    assert ds.snapshot_step is None


def test_lookup_unknown_destination_returns_cold_start(tmp_path):
    path = _write_state(tmp_path, [
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
    path = _write_state(tmp_path, rows)
    ds = load_dest_state(path)

    names = [f"C{i}" for i in range(0, 50, 7)] + ["C_UNKNOWN"]
    batch_results, batch_hits = ds.lookup_many(names)
    for i, name in enumerate(names):
        single_values, single_hit = ds.lookup(name)
        assert batch_results[i] == single_values
        assert bool(batch_hits[i]) == single_hit


def test_duplicate_keys_are_refused(tmp_path, monkeypatch):
    """A snapshot with two identical keys is refused rather than served.

    The v2 bundle precomputes the hashes at build time, so the collision is now
    caught where it can be prevented -- `build_dest_state.py` refuses to WRITE
    such a bundle. This is the second line of defence: the loader still verifies
    that the keys it was handed are strictly increasing, because a duplicate
    makes `searchsorted` return one destination's history under another's name,
    and that must never be served whatever produced the file.
    """
    import inference.state as state_module
    monkeypatch.setattr(state_module, "_hash_dest", lambda name_dest: np.uint64(42))

    path = _write_state(tmp_path, [
        {"name_dest": "C1", "prior_txn_count": 1, "prior_avg_amount": 1.0,
         "txn_count_24h": 1, "amount_sum_24h": 1.0},
        {"name_dest": "C2", "prior_txn_count": 2, "prior_avg_amount": 2.0,
         "txn_count_24h": 2, "amount_sum_24h": 2.0},
    ])

    with pytest.raises(DestStateCollisionError, match="strictly increasing"):
        load_dest_state(path)


def test_unsorted_keys_are_refused(tmp_path):
    """The lookup is a `searchsorted`, which silently returns wrong answers on
    unsorted keys rather than failing. Cheap O(n) check, no re-sort."""
    # Written directly with descending keys rather than by hashing two ids and
    # hoping they land out of order -- a test that skips when the hash happens
    # to cooperate is a test that does not run.
    path = tmp_path / "dest_state.npz"
    np.savez_compressed(
        path,
        keys=np.array([99, 1], dtype="uint64"),
        count=np.array([2, 1], dtype="int32"),
        avg=np.array([2.0, 1.0], dtype="float32"),
        c24=np.array([2, 1], dtype="int32"),
        s24=np.array([2.0, 1.0], dtype="float32"),
        snapshot_step=np.array([743], dtype="int64"),
    )

    with pytest.raises(DestStateCollisionError, match="strictly increasing"):
        load_dest_state(path)
