"""
features.py

Feature engineering matching the JD's exact framing: "temporal, behavioral,
aggregated features" on "structured, semi-structured, and unstructured data."

All features are computed leakage-safely: aggregates are built only from
information available strictly BEFORE the transaction in question (or from
static account-level history), never from the transaction's own outcome.
"""

import pandas as pd
import numpy as np


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """'step' in PaySim = hour of simulation. Extract cyclical time features."""
    df["hour_of_day"] = df["step"] % 24
    df["is_night"] = df["hour_of_day"].between(0, 5).astype(np.int8)
    return df


def add_behavioral_features(df: pd.DataFrame) -> pd.DataFrame:
    """Per-transaction behavioral signals -- classic fraud tells in PaySim-style data."""
    # Balance-consistency checks: fraud often produces balances that don't
    # reconcile (e.g. origin account emptied regardless of transfer amount)
    df["orig_balance_delta"] = df["oldbalanceOrg"] - df["newbalanceOrig"]
    df["dest_balance_delta"] = df["newbalanceDest"] - df["oldbalanceDest"]
    df["orig_balance_mismatch"] = (
        (df["orig_balance_delta"] - df["amount"]).abs() > 0.01
    ).astype(np.int8)

    # Empties the account entirely -- a strong real-world fraud signal
    df["orig_emptied"] = (
        (df["oldbalanceOrg"] > 0) & (df["newbalanceOrig"] == 0)
    ).astype(np.int8)

    # Amount relative to sender's available balance
    df["amount_to_balance_ratio"] = df["amount"] / (df["oldbalanceOrg"] + 1)

    # Destination is a merchant (PaySim convention: nameDest starts with 'M')
    df["dest_is_merchant"] = df["nameDest"].str.startswith("M").astype(np.int8)

    return df


def add_aggregated_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Account-level aggregates computed from EACH ACCOUNT'S OWN PRIOR HISTORY
    ONLY, never using the current row's own label or same-timestamp peer
    transactions -- this is the leakage-safety pattern already used in the
    Clinical EMR and Nectar projects.

    Vectorized via groupby cumsum/cumcount rather than groupby().apply(): in
    PaySim, nameOrig is ~unique per row (6.35M unique senders across 6.36M
    rows), so a per-group .apply(lambda ...) means ~6M individual Python-level
    calls -- effectively unrunnable at this scale. cumsum/cumcount do the
    same computation as a single vectorized pass over the sorted frame.
    """
    df = df.sort_values(["nameOrig", "step"]).reset_index(drop=True)

    grp_amount = df.groupby("nameOrig", sort=False)["amount"]
    prior_count = grp_amount.cumcount()
    prior_sum = grp_amount.cumsum() - df["amount"]

    df["orig_prior_txn_count"] = prior_count
    df["orig_prior_avg_amount"] = (
        prior_sum / prior_count.replace(0, np.nan)
    ).fillna(0)

    return df


def build_feature_table(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = add_temporal_features(df)
    df = add_behavioral_features(df)
    df = add_aggregated_features(df)
    return df


FEATURE_COLUMNS = [
    "amount", "hour_of_day", "is_night",
    "orig_balance_delta", "dest_balance_delta", "orig_balance_mismatch",
    "orig_emptied", "amount_to_balance_ratio", "dest_is_merchant",
    "orig_prior_txn_count", "orig_prior_avg_amount",
]
