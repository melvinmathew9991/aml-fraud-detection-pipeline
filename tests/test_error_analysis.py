"""
Unit tests for error_analysis.py against hand-computed values -- flagged as
a coverage gap in the Sprint 4 audit (every other non-trivial src/ module
had a dedicated test file; this one didn't, despite segment_profile's
std-gap formula and missed_fraud_ranking's tie-sensitive rank cutoff both
being exactly the kind of arithmetic this project's test culture otherwise
insists on pinning).
"""

import numpy as np
import pandas as pd
import pytest

from error_analysis import missed_fraud_ranking, segment_profile


# Six rows: TP = {0, 1}, FP = {2}, FN = {4}, TN = {3, 5} (not flagged, not fraud).
X = pd.DataFrame({
    "amount": [100.0, 200.0, 150.0, 50.0, 300.0, 80.0],
    "flag_feat": [1.0, 1.0, 0.0, 0.0, 1.0, 0.0],
})
Y_TRUE = np.array([1, 1, 0, 0, 1, 0])
Y_PROBA = np.array([0.9, 0.8, 0.7, 0.2, 0.3, 0.1])
THRESHOLD = 0.5


def test_segment_profile_group_membership_and_counts():
    profile = segment_profile(X, Y_TRUE, Y_PROBA, THRESHOLD)
    row = profile.set_index("feature")
    assert row.loc["amount", "true_positive_n"] == 2
    assert row.loc["amount", "false_positive_n"] == 1
    assert row.loc["amount", "false_negative_n"] == 1


def test_segment_profile_group_means_hand_computed():
    profile = segment_profile(X, Y_TRUE, Y_PROBA, THRESHOLD)
    row = profile.set_index("feature")
    # TP = rows 0,1 -> amount (100+200)/2=150, flag_feat (1+1)/2=1.0
    assert row.loc["amount", "true_positive_mean"] == pytest.approx(150.0)
    assert row.loc["flag_feat", "true_positive_mean"] == pytest.approx(1.0)
    # FP = row 2 -> amount=150, flag_feat=0
    assert row.loc["amount", "false_positive_mean"] == pytest.approx(150.0)
    assert row.loc["flag_feat", "false_positive_mean"] == pytest.approx(0.0)
    # FN = row 4 -> amount=300, flag_feat=1
    assert row.loc["amount", "false_negative_mean"] == pytest.approx(300.0)
    assert row.loc["flag_feat", "false_negative_mean"] == pytest.approx(1.0)


def test_segment_profile_std_gap_matches_formula():
    profile = segment_profile(X, Y_TRUE, Y_PROBA, THRESHOLD)
    row = profile.set_index("feature")

    overall_std = X.std(axis=0, ddof=1)  # pandas' default ddof, matches X.std(axis=0)
    expected_fp_vs_tp = (150.0 - 150.0) / overall_std["amount"]  # amount: FP mean == TP mean
    assert row.loc["amount", "fp_vs_tp_std_gap"] == pytest.approx(expected_fp_vs_tp)

    expected_flag_gap = (0.0 - 1.0) / overall_std["flag_feat"]
    assert row.loc["flag_feat", "fp_vs_tp_std_gap"] == pytest.approx(expected_flag_gap)


def test_segment_profile_sorted_by_absolute_gap_descending():
    profile = segment_profile(X, Y_TRUE, Y_PROBA, THRESHOLD)
    gaps = profile["fp_vs_tp_std_gap"].abs().to_numpy()
    assert list(gaps) == sorted(gaps, reverse=True)


def test_segment_profile_no_false_positives_reports_nan_gap_not_crash():
    # Threshold high enough that nothing is flagged at all -> FP group empty.
    profile = segment_profile(X, Y_TRUE, Y_PROBA, threshold=0.95)
    row = profile.set_index("feature")
    assert row.loc["amount", "false_positive_n"] == 0
    assert np.isnan(row.loc["amount", "false_positive_mean"])
    assert np.isnan(row.loc["amount", "fp_vs_tp_std_gap"])


def test_segment_profile_zero_variance_feature_gap_is_nan_not_inf():
    constant_X = pd.DataFrame({"amount": X["amount"], "constant": [5.0] * 6})
    profile = segment_profile(constant_X, Y_TRUE, Y_PROBA, THRESHOLD)
    row = profile.set_index("feature")
    # std=0 -> replace(0, nan) in the source -> gap must be nan, not +/-inf.
    assert np.isnan(row.loc["constant", "fp_vs_tp_std_gap"])


# ------------------------------------------------------------- missed_fraud_ranking

def test_missed_fraud_ranking_no_fraud_returns_empty_dict():
    y_true = np.zeros(10, dtype=int)
    y_proba = np.linspace(0, 1, 10)
    assert missed_fraud_ranking(y_true, y_proba, k=3) == {}


def test_missed_fraud_ranking_hand_computed_ranks_and_misses():
    # 8 rows, scores strictly descending by index so rank == index + 1.
    y_proba = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2])
    y_true = np.array([1, 0, 1, 0, 1, 0, 1, 0])  # fraud at rank 1, 3, 5, 7
    k = 4  # top 4 flagged -> rank 1 and 3 caught, rank 5 and 7 missed

    stats = missed_fraud_ranking(y_true, y_proba, k=k)
    assert stats["total_fraud"] == 4
    assert stats["fraud_in_top_k"] == 2
    assert stats["fraud_missed"] == 2
    assert stats["missed_rank_min"] == 5
    assert stats["missed_rank_max"] == 7
    assert stats["missed_rank_median"] == pytest.approx(6.0)
    assert stats["k_to_catch_half_of_missed"] == 6
    assert stats["k_to_catch_all_missed"] == 7
    # within 2*k=8 -> both misses (rank 5, 7) qualify
    assert stats["missed_within_2x_k"] == 2
    # bottom half of 8 rows is rank > 4 -> both misses (5, 7) qualify
    assert stats["missed_in_bottom_half"] == 2


def test_missed_fraud_ranking_uses_n_flagged_not_requested_capacity():
    # Docstring's tie-handling contract: k must be n_flagged (actual count
    # at/above threshold), not the requested capacity K, since ties can make
    # them differ. Passing a smaller k (as if it were the raw capacity K
    # instead of n_flagged) would count a caught fraud as missed.
    y_proba = np.array([0.9, 0.8, 0.8, 0.8, 0.5])  # 3-way tie at 0.8
    y_true = np.array([1, 1, 0, 1, 0])
    # Suppose the threshold admits all three 0.8-tied rows -> n_flagged=4,
    # even though the "requested" capacity K might have been 2.
    n_flagged = 4
    stats = missed_fraud_ranking(y_true, y_proba, k=n_flagged)
    assert stats["fraud_missed"] == 0  # all 3 fraud rows (rank 1,2-or-3,2-or-3) are within top 4

    # Using the smaller requested K=2 instead would wrongly count a tied
    # fraud row as missed, demonstrating why the caller must pass n_flagged.
    stats_wrong_k = missed_fraud_ranking(y_true, y_proba, k=2)
    assert stats_wrong_k["fraud_missed"] > 0


def test_missed_fraud_ranking_zero_when_all_fraud_in_queue():
    y_proba = np.array([0.9, 0.8, 0.1, 0.05])
    y_true = np.array([1, 1, 0, 0])
    stats = missed_fraud_ranking(y_true, y_proba, k=2)
    assert stats["fraud_missed"] == 0
    assert "missed_rank_min" not in stats  # only populated when there ARE misses
