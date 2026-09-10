"""
state.py

Destination state lookup -- resolves the training/serving skew problem
described in ARCHITECTURE.md §2. Five of the model's features are stateful
aggregates over destination account history, computed at training time as
SQL window functions over the full raw table. A single incoming transaction
at serving time has no window to look back through, so this module answers
"what does this destination's history look like as of the bundled
snapshot?" from the committed `dest_state.npz`.

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


def hash_many(names) -> np.ndarray:
    """Hash a sequence of destination ids.

    Public because `build_dest_state.py` imports it. The snapshot now ships
    precomputed keys rather than account names, so the build and the lookup must
    agree on the hash exactly -- sharing the function is what guarantees that,
    instead of two implementations that merely look alike.
    """
    return np.array([_hash_dest(n) for n in names], dtype="uint64")


_hash_many = hash_many  # retained: existing tests import the private name


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


SNAPSHOT_STEP_METADATA_KEY = b"snapshot_step"  # written by build_dest_state.py


def load_dest_state(state_path: Path) -> DestState:
    """Load the snapshot from the bundle's `dest_state.npz`.

    The arrays are stored ready to use: keys already hashed, already sorted, and
    the account names not stored at all -- `DestState` never keeps them, so
    shipping 571,961 strings only to hash and discard them was work the build
    could do once instead of every cold start.

    Measured on the previous parquet format, that work was **2,268 ms** of the
    startup path (627 ms parquet parse, 556 ms materialising the strings,
    1,016 ms hashing them, 68 ms sorting). Reading the arrays back is ~100 ms,
    and dropping the format also dropped `pyarrow` -- 84.3 MB of the 133.6 MB
    serving dependency footprint, for this one call. See ARCHITECTURE.md §3.

    The collision check moved to build time with the hashing, which is where it
    always belonged: a bundle that could serve one destination's history under
    another's name should never be written, not merely refused on load.
    """
    with np.load(state_path) as payload:
        keys = payload["keys"].astype("uint64", copy=False)
        count = payload["count"].astype("int32", copy=False)
        avg = payload["avg"].astype("float32", copy=False)
        c24 = payload["c24"].astype("int32", copy=False)
        s24 = payload["s24"].astype("float32", copy=False)
        stored_step = payload["snapshot_step"]
        snapshot_step = None if stored_step.size == 0 else int(stored_step.reshape(-1)[0])

    # Cheap invariant, not a re-derivation: the lookup is a searchsorted over
    # `keys` and silently returns wrong answers if the build ever emits them
    # unsorted. O(n) to check against O(n log n) to redo.
    if keys.size > 1 and not np.all(keys[:-1] < keys[1:]):
        raise DestStateCollisionError(
            f"{state_path} -- keys are not strictly increasing, so they are "
            "either unsorted or contain a duplicate. Refusing to load a state "
            "snapshot whose lookups would be undefined."
        )

    return DestState(
        keys=keys,
        count=count,
        avg=avg,
        c24=c24,
        s24=s24,
        snapshot_step=snapshot_step,
    )
