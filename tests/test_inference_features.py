"""
Unit tests for inference/features.py against hand-computed values, mirroring
test_features.py's style but exercising the RAW-TRANSACTION-DICT path
(inference's job) rather than the SQL path (training's job) -- the two are
compared for agreement in test_skew_state.py / test_skew_plumbing.py.
"""

import pytest

from inference.features import compute_dest_features, compute_features, compute_stateless_features
from inference.state import COLD_START, DestState
from features import FEATURE_COLUMNS


def _empty_dest_state() -> DestState:
    import numpy as np
    return DestState(
        keys=np.array([], dtype="uint64"),
        count=np.array([], dtype="int32"),
        avg=np.array([], dtype="float32"),
        c24=np.array([], dtype="int32"),
        s24=np.array([], dtype="float32"),
    )


BASE_TXN = {
    "step": 26,  # hour_of_day = 2 -> is_night
    "type": "TRANSFER",
    "amount": 100.0,
    "nameOrig": "C_ORIG",
    "oldbalanceOrg": 1000.0,
    "newbalanceOrig": 900.0,
    "nameDest": "C_DEST",
    "oldbalanceDest": 0.0,
    "newbalanceDest": 100.0,
}


def test_stateless_features_hand_computed():
    values = compute_stateless_features(BASE_TXN)
    assert values["amount"] == 100.0
    assert values["hour_of_day"] == 2
    assert values["is_night"] == 1
    assert values["orig_balance_delta"] == 100.0
    assert values["dest_balance_delta"] == 100.0
    assert values["orig_balance_mismatch"] == 0
    assert values["orig_emptied"] == 0
    assert values["amount_to_balance_ratio"] == pytest.approx(100.0 / 1001.0)
    assert values["dest_is_merchant"] == 0
    assert values["is_transfer"] == 1
    assert values["is_cash_out"] == 0
    assert values["is_cash_in"] == 0
    assert values["is_debit"] == 0


@pytest.mark.parametrize("hour_step,expected_night", [(0, 1), (5, 1), (6, 0), (23, 0)])
def test_is_night_boundary_matches_sql_between_0_and_5(hour_step, expected_night):
    txn = {**BASE_TXN, "step": hour_step}
    values = compute_stateless_features(txn)
    assert values["hour_of_day"] == hour_step
    assert values["is_night"] == expected_night


def test_orig_balance_mismatch_flags_amount_disagreeing_with_balance_delta():
    # oldbalanceOrg - newbalanceOrig = 100, but amount = 50 -> mismatch.
    txn = {**BASE_TXN, "oldbalanceOrg": 1000.0, "newbalanceOrig": 900.0, "amount": 50.0}
    values = compute_stateless_features(txn)
    assert values["orig_balance_mismatch"] == 1


def test_orig_emptied_true_only_when_balance_swept_to_zero():
    swept = {**BASE_TXN, "oldbalanceOrg": 500.0, "newbalanceOrig": 0.0}
    assert compute_stateless_features(swept)["orig_emptied"] == 1

    not_swept = {**BASE_TXN, "oldbalanceOrg": 500.0, "newbalanceOrig": 10.0}
    assert compute_stateless_features(not_swept)["orig_emptied"] == 0

    zero_to_zero = {**BASE_TXN, "oldbalanceOrg": 0.0, "newbalanceOrig": 0.0}
    assert compute_stateless_features(zero_to_zero)["orig_emptied"] == 0


def test_amount_to_balance_ratio_div_by_zero_guard():
    txn = {**BASE_TXN, "oldbalanceOrg": 0.0, "amount": 250.0}
    values = compute_stateless_features(txn)
    assert values["amount_to_balance_ratio"] == 250.0  # 250 / (0 + 1)


def test_dest_is_merchant_from_name_prefix():
    assert compute_stateless_features({**BASE_TXN, "nameDest": "M12345"})["dest_is_merchant"] == 1
    assert compute_stateless_features({**BASE_TXN, "nameDest": "C12345"})["dest_is_merchant"] == 0


@pytest.mark.parametrize("txn_type,flag", [
    ("TRANSFER", "is_transfer"), ("CASH_OUT", "is_cash_out"),
    ("CASH_IN", "is_cash_in"), ("DEBIT", "is_debit"),
])
def test_type_indicators_one_hot(txn_type, flag):
    values = compute_stateless_features({**BASE_TXN, "type": txn_type})
    all_flags = ["is_transfer", "is_cash_out", "is_cash_in", "is_debit"]
    for f in all_flags:
        assert values[f] == (1 if f == flag else 0)


def test_payment_type_is_reference_level_all_flags_zero():
    values = compute_stateless_features({**BASE_TXN, "type": "PAYMENT"})
    for f in ["is_transfer", "is_cash_out", "is_cash_in", "is_debit"]:
        assert values[f] == 0


def test_dest_features_cold_start_matches_first_transaction_convention():
    values = compute_dest_features(amount=100.0, dest_state_values=dict(COLD_START))
    assert values["dest_prior_txn_count"] == 0
    assert values["dest_prior_avg_amount"] == 0
    assert values["dest_txn_count_24h"] == 0
    assert values["dest_amount_sum_24h"] == 0
    # +1 guard: ratio degrades to `amount` when there's no prior average.
    assert values["dest_amount_to_prior_avg_ratio"] == 100.0


def test_dest_features_div_by_zero_guard_with_history():
    state = {"prior_txn_count": 3, "prior_avg_amount": 199.0, "txn_count_24h": 2,
             "amount_sum_24h": 300.0}
    values = compute_dest_features(amount=100.0, dest_state_values=state)
    assert values["dest_amount_to_prior_avg_ratio"] == pytest.approx(100.0 / 200.0)


def test_compute_features_cold_start_destination_reports_state_hit_false():
    vector, values, state_hit = compute_features(BASE_TXN, _empty_dest_state())
    assert state_hit is False
    assert values["dest_prior_txn_count"] == 0
    assert len(vector) == len(FEATURE_COLUMNS)
    assert vector == [values[name] for name in FEATURE_COLUMNS]


def test_compute_features_vector_order_matches_feature_columns():
    vector, values, _ = compute_features(BASE_TXN, _empty_dest_state())
    for name, v in zip(FEATURE_COLUMNS, vector):
        assert v == values[name]
