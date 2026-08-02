"""
state.py

Destination state lookup -- resolves the training/serving skew problem
described in ARCHITECTURE.md §2. Five of the model's features are stateful
aggregates over destination account history, computed at training time as
SQL window functions over the full raw table. A single incoming transaction
at serving time has no window to look back through, so this module answers
"what does this destination's history look like as of the bundled
snapshot?" from the committed `dest_state.parquet`.

Per ARCHITECTURE.md §2's "In-memory representation" section: the parquet is
NOT loaded into a dict or a DataFrame (571,961 Python dict entries would
cost >100MB of object overhead, a real OOM risk on a 512MiB instance).
Instead it is loaded once as five parallel numpy arrays, sorted by a 64-bit
hash of `name_dest`, and looked up with `np.searchsorted` -- O(log n), no
per-row Python objects, ~13.7MB resident for 571,961 destinations.

Cold-start policy (unknown destination, including every merchant `M%`
account -- merchants are not stored in the snapshot at all, see
build_dest_state.py): prior_txn_count=0, prior_avg_amount=0,
txn_count_24h=0, amount_sum_24h=0. This is not a fallback hack -- it is
exactly what the training SQL produces for an account's first transaction
(COUNT over an empty window is 0, COALESCE(AVG(...), 0), and the `+1` ratio
guard in features.py makes dest_amount_to_prior_avg_ratio degrade to
`amount`). Training and serving agree by construction.
"""

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

# Cold-start / unknown-destination values, shared by features.py.
COLD_START = {
    "prior_txn_count": 0,
    "prior_avg_amount": 0.0,
    "txn_count_24h": 0,
    "amount_sum_24h": 0.0,
}


class DestStateCollisionError(Exception):
    """Two distinct name_dest values hashed to the same 64-bit key.

    Per ARCHITECTURE.md §2: expected count at 2^64 over ~572k keys is
    effectively zero, but this is checked rather than assumed -- the build
    (here, the load) fails rather than silently serving one destination's
    state under another's name.
    """


def _hash_dest(name_dest: str) -> np.uint64:
    """64-bit blake2b hash of a destination account id, as an unsigned int."""
    digest = hashlib.blake2b(name_dest.encode("utf-8"), digest_size=8).digest()
    return np.uint64(int.from_bytes(digest, byteorder="big"))


def _hash_many(names) -> np.ndarray:
    return np.array([_hash_dest(n) for n in names], dtype="uint64")


@dataclass(frozen=True)
class DestState:
    keys: np.ndarray          # uint64[n], sorted
    count: np.ndarray         # int32[n]
    avg: np.ndarray           # float32[n]
    c24: np.ndarray           # int32[n]
    s24: np.ndarray           # float32[n]
    snapshot_step: int | None = None

    def lookup(self, name_dest: str) -> tuple[dict, bool]:
        """Returns (state_dict, state_hit) for one destination.

        state_hit is False for both a genuinely unknown destination AND a
        merchant account (never stored) -- callers that need to distinguish
        "known to be a merchant" from "unknown" should consult
        dest_is_merchant separately (it's computed from the name prefix
        alone in features.py, not from this lookup).
        """
        if len(self.keys) == 0:
            return dict(COLD_START), False
        key = _hash_dest(name_dest)
        idx = int(np.searchsorted(self.keys, key))
        if idx < len(self.keys) and self.keys[idx] == key:
            return {
                "prior_txn_count": int(self.count[idx]),
                "prior_avg_amount": float(self.avg[idx]),
                "txn_count_24h": int(self.c24[idx]),
                "amount_sum_24h": float(self.s24[idx]),
            }, True
        return dict(COLD_START), False

    def lookup_many(self, names_dest) -> tuple[list[dict], np.ndarray]:
        """Vectorized form of lookup(), for /score/batch."""
        n = len(names_dest)
        if len(self.keys) == 0 or n == 0:
            return [dict(COLD_START) for _ in range(n)], np.zeros(n, dtype=bool)

        query_keys = _hash_many(names_dest)
        idx = np.searchsorted(self.keys, query_keys)
        idx_clipped = np.clip(idx, 0, len(self.keys) - 1)
        hits = self.keys[idx_clipped] == query_keys

        results = []
        for i in range(n):
            if hits[i]:
                j = int(idx_clipped[i])
                results.append({
                    "prior_txn_count": int(self.count[j]),
                    "prior_avg_amount": float(self.avg[j]),
                    "txn_count_24h": int(self.c24[j]),
                    "amount_sum_24h": float(self.s24[j]),
                })
            else:
                results.append(dict(COLD_START))
        return results, hits


def load_dest_state(parquet_path: Path) -> DestState:
    table = pq.read_table(
        parquet_path,
        columns=["name_dest", "prior_txn_count", "prior_avg_amount",
                 "txn_count_24h", "amount_sum_24h"],
    )
    names = table.column("name_dest").to_pylist()
    count = table.column("prior_txn_count").to_numpy(zero_copy_only=False).astype("int32")
    avg = table.column("prior_avg_amount").to_numpy(zero_copy_only=False).astype("float32")
    c24 = table.column("txn_count_24h").to_numpy(zero_copy_only=False).astype("int32")
    s24 = table.column("amount_sum_24h").to_numpy(zero_copy_only=False).astype("float32")

    keys = _hash_many(names)
    order = np.argsort(keys, kind="stable")
    keys = keys[order]

    dup = keys[:-1] == keys[1:]
    if dup.any():
        collided_at = int(np.flatnonzero(dup)[0])
        raise DestStateCollisionError(
            f"64-bit hash collision at sorted index {collided_at} while loading "
            f"{parquet_path} -- two distinct name_dest values hash to the same "
            "key. Refusing to load a state snapshot that could serve the wrong "
            "destination's history."
        )

    return DestState(
        keys=keys,
        count=count[order],
        avg=avg[order],
        c24=c24[order],
        s24=s24[order],
    )
