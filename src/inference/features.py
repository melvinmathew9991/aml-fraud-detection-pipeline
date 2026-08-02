"""
inference/features.py

Raw transaction dict -> ordered feature vector, matching training's
`features.feature_query()` bit-for-bit (ARCHITECTURE.md §4). This is the
"plumbing" half of the training/serving skew problem (ARCHITECTURE.md §2):
13 of the 18 features are pure arithmetic on the request payload; the
other 5 (dest_prior_txn_count, dest_prior_avg_amount,
dest_amount_to_prior_avg_ratio, dest_txn_count_24h, dest_amount_sum_24h)
consult a resolved destination-state dict instead of a live SQL window,
because a single incoming transaction has no window of its own to look
back through.

Split into two halves so the batch endpoint can supply MERGED state
(snapshot + in-batch accumulation, per ARCHITECTURE.md §5's "aggregates
accumulate within the batch") without re-deriving the 13 stateless
features:

  compute_stateless_features(txn)                -> the 13 non-dest features
  compute_dest_features(amount, dest_state_values) -> the 5 dest features,
                                                       from an ALREADY-
                                                       RESOLVED state dict
  compute_features(txn, dest_state)               -> single-transaction
                                                       convenience wrapper
                                                       used by /score; does
                                                       its own snapshot-only
                                                       lookup via dest_state

Deliberately imports FEATURE_COLUMNS from training's `features.py` rather
than hand-maintaining a second copy of the column order: that module is
pure Python with no heavy imports of its own (no pandas/duckdb/sklearn), so
importing it costs nothing at serving time and structurally prevents the
two feature lists from drifting apart.

Raw field names match schema.py's RAW_TRANSACTION_SCHEMA (and the original
PaySim columns) rather than being renamed to snake_case, so there is no
silent translation layer between what the API accepts and what this module
reads.
"""

from typing import TypedDict

from features import FEATURE_COLUMNS

from .state import COLD_START, DestState

VALID_TYPES = ("CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER")

# PaySim's `step` column is one simulated hour; hour_of_day wraps every 24.
HOUR_OF_DAY_MODULUS = 24
NIGHT_HOUR_END = 5  # is_night true for hour_of_day in [0, NIGHT_HOUR_END]
BALANCE_MISMATCH_TOLERANCE = 0.01


class RawTransaction(TypedDict):
    step: int
    type: str
    amount: float
    nameOrig: str
    oldbalanceOrg: float
    newbalanceOrig: float
    nameDest: str
    oldbalanceDest: float
    newbalanceDest: float


def compute_stateless_features(txn: RawTransaction) -> dict:
    """The 13 features computable from the request payload alone."""
    hour_of_day = txn["step"] % HOUR_OF_DAY_MODULUS
    orig_balance_delta = txn["oldbalanceOrg"] - txn["newbalanceOrig"]
    dest_balance_delta = txn["newbalanceDest"] - txn["oldbalanceDest"]

    return {
        "amount": txn["amount"],
        "hour_of_day": hour_of_day,
        "is_night": 1 if hour_of_day <= NIGHT_HOUR_END else 0,
        "orig_balance_delta": orig_balance_delta,
        "dest_balance_delta": dest_balance_delta,
        "orig_balance_mismatch": (
            1 if abs(orig_balance_delta - txn["amount"]) > BALANCE_MISMATCH_TOLERANCE else 0
        ),
        "orig_emptied": (
            1 if txn["oldbalanceOrg"] > 0 and txn["newbalanceOrig"] == 0 else 0
        ),
        "amount_to_balance_ratio": txn["amount"] / (txn["oldbalanceOrg"] + 1),
        "dest_is_merchant": 1 if txn["nameDest"].startswith("M") else 0,
        "is_transfer": 1 if txn["type"] == "TRANSFER" else 0,
        "is_cash_out": 1 if txn["type"] == "CASH_OUT" else 0,
        "is_cash_in": 1 if txn["type"] == "CASH_IN" else 0,
        "is_debit": 1 if txn["type"] == "DEBIT" else 0,
    }


def compute_dest_features(amount: float, dest_state_values: dict) -> dict:
    """The 5 destination-history features, from an already-resolved state
    dict (snapshot-only for /score, snapshot+in-batch for /score/batch)."""
    prior_avg_amount = dest_state_values["prior_avg_amount"]
    return {
        "dest_prior_txn_count": dest_state_values["prior_txn_count"],
        "dest_prior_avg_amount": prior_avg_amount,
        "dest_amount_to_prior_avg_ratio": amount / (prior_avg_amount + 1),
        "dest_txn_count_24h": dest_state_values["txn_count_24h"],
        "dest_amount_sum_24h": dest_state_values["amount_sum_24h"],
    }


def assemble(values: dict) -> list[float]:
    """FEATURE_COLUMNS-ordered vector from a name-keyed feature dict."""
    return [values[name] for name in FEATURE_COLUMNS]


def compute_features(txn: RawTransaction, dest_state: DestState) -> tuple[list[float], dict, bool]:
    """
    Single-transaction convenience wrapper for /score: resolves dest state
    from the snapshot ONLY (no in-batch accumulation -- there is no batch).

    Returns (feature_vector, feature_values, state_hit).

    feature_vector is ordered exactly per FEATURE_COLUMNS (training's
    canonical order) -- what score.py feeds the model. feature_values is
    the same data keyed by name -- what rules.py's hard-block layer reads,
    so a rule survives a change to FEATURE_COLUMNS' order. state_hit
    reports whether the 5 destination features came from the snapshot
    (True) or a cold-start default (False) -- ARCHITECTURE.md §5 requires
    this be surfaced on every /score response, since silently defaulting is
    the dangerous failure mode section 2 describes.
    """
    dest_state_values, state_hit = dest_state.lookup(txn["nameDest"])
    values = {
        **compute_stateless_features(txn),
        **compute_dest_features(txn["amount"], dest_state_values),
    }
    return assemble(values), values, state_hit


__all__ = [
    "COLD_START",
    "VALID_TYPES",
    "RawTransaction",
    "assemble",
    "compute_dest_features",
    "compute_features",
    "compute_stateless_features",
]
