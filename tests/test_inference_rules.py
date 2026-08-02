"""
Unit tests for inference/rules.py's hard-block layer: BLOCK takes
precedence over the model's REVIEW/PASS flag, and the rule only fires on
its documented predicate (full-balance sweep on TRANSFER/CASH_OUT above
the amount threshold).
"""

from inference.rules import HARD_BLOCK_AMOUNT_THRESHOLD, decide, evaluate_hard_block

BASE_FEATURES = {
    "is_transfer": 0, "is_cash_out": 0, "is_cash_in": 0, "is_debit": 0,
    "orig_emptied": 0, "amount": 0.0,
}


def test_no_rule_fires_on_ordinary_payment():
    features = {**BASE_FEATURES, "amount": 50.0}
    result = evaluate_hard_block(features)
    assert result.decision == "PASS"
    assert result.rule is None


def test_full_balance_sweep_transfer_above_threshold_blocks():
    features = {**BASE_FEATURES, "is_transfer": 1, "orig_emptied": 1,
                "amount": HARD_BLOCK_AMOUNT_THRESHOLD}
    result = evaluate_hard_block(features)
    assert result.decision == "BLOCK"
    assert result.rule == "full_balance_sweep_large_amount"


def test_full_balance_sweep_cash_out_above_threshold_blocks():
    features = {**BASE_FEATURES, "is_cash_out": 1, "orig_emptied": 1,
                "amount": HARD_BLOCK_AMOUNT_THRESHOLD + 1}
    result = evaluate_hard_block(features)
    assert result.decision == "BLOCK"


def test_sweep_below_threshold_does_not_block():
    features = {**BASE_FEATURES, "is_transfer": 1, "orig_emptied": 1,
                "amount": HARD_BLOCK_AMOUNT_THRESHOLD - 1}
    result = evaluate_hard_block(features)
    assert result.decision == "PASS"


def test_large_amount_without_sweep_does_not_block():
    features = {**BASE_FEATURES, "is_transfer": 1, "orig_emptied": 0,
                "amount": HARD_BLOCK_AMOUNT_THRESHOLD * 10}
    result = evaluate_hard_block(features)
    assert result.decision == "PASS"


def test_sweep_on_non_transfer_non_cashout_type_does_not_block():
    features = {**BASE_FEATURES, "orig_emptied": 1, "amount": HARD_BLOCK_AMOUNT_THRESHOLD}
    result = evaluate_hard_block(features)
    assert result.decision == "PASS"


def test_decide_block_takes_precedence_over_model_flag():
    features = {**BASE_FEATURES, "is_transfer": 1, "orig_emptied": 1,
                "amount": HARD_BLOCK_AMOUNT_THRESHOLD}
    result = decide(features, flagged=False)
    assert result.decision == "BLOCK"


def test_decide_review_when_model_flags_and_no_rule_fires():
    features = {**BASE_FEATURES, "amount": 10.0}
    result = decide(features, flagged=True)
    assert result.decision == "REVIEW"
    assert result.rule is None


def test_decide_pass_when_model_does_not_flag_and_no_rule_fires():
    features = {**BASE_FEATURES, "amount": 10.0}
    result = decide(features, flagged=False)
    assert result.decision == "PASS"
