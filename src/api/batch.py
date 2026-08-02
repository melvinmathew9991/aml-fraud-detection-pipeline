"""
api/batch.py

Within-batch destination-state accumulation for POST /score/batch
(ARCHITECTURE.md §4/§5): "aggregates accumulate within the batch" is the
project's honest, cheap demonstration of incremental feature state, and the
reason the separate streaming sprint was cut (ARCHITECTURE.md §16).

Simplifying assumption, stated plainly: every transaction in a batch is
treated as arriving "now" -- i.e. after the bundled snapshot's frozen step,
and close enough together that any batch-internal repeat to the same
destination is, by construction, within the 24h velocity window as well as
the unbounded prior-history window. That collapses what would otherwise be
two differently-windowed accumulators (ARCHITECTURE.md §2's `w_dest` vs
`w_dest_velocity`) into one: a batch row occurring after other rows to the
same destination in the same batch sees them counted in BOTH
dest_prior_txn_count and dest_txn_count_24h. This is exact for the common
case (a batch scored in one HTTP request over a short wall-clock span) and
is the same reading `state.py`'s cold-start policy already uses: no
attempt to reconstruct real elapsed time between requests, because a
serving snapshot has none to offer.

Rows are processed in the order submitted -- that order is the caller's
claim about arrival order, and this accumulator does not second-guess it
(e.g. by re-sorting on `step`).
"""

from dataclasses import dataclass, field

from inference.state import DestState


@dataclass
class _Accumulated:
    count: int = 0
    amount_sum: float = 0.0


class BatchAccumulator:
    """Wraps a DestState snapshot with a per-destination running total of
    transactions seen earlier in THIS batch, so row N's dest features can
    reflect rows 0..N-1 of the same request in addition to the snapshot."""

    def __init__(self, dest_state: DestState):
        self._dest_state = dest_state
        self._batch: dict[str, _Accumulated] = {}

    def prior_state_for(self, name_dest: str) -> tuple[dict, bool]:
        """Merged (snapshot + in-batch-so-far) state for `name_dest`, and
        whether EITHER source had prior history (state_hit)."""
        snapshot, snapshot_hit = self._dest_state.lookup(name_dest)
        acc = self._batch.get(name_dest)

        if acc is None or acc.count == 0:
            return dict(snapshot), snapshot_hit

        merged_count = snapshot["prior_txn_count"] + acc.count
        merged_amount_total = (
            snapshot["prior_avg_amount"] * snapshot["prior_txn_count"] + acc.amount_sum
        )
        merged = {
            "prior_txn_count": merged_count,
            "prior_avg_amount": merged_amount_total / merged_count if merged_count else 0.0,
            "txn_count_24h": snapshot["txn_count_24h"] + acc.count,
            "amount_sum_24h": snapshot["amount_sum_24h"] + acc.amount_sum,
        }
        return merged, True

    def observe(self, name_dest: str, amount: float) -> None:
        """Records that a transaction to `name_dest` was just scored, so
        later rows in the same batch see it as prior history."""
        acc = self._batch.setdefault(name_dest, _Accumulated())
        acc.count += 1
        acc.amount_sum += amount
