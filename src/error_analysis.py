"""
error_analysis.py

What the review queue actually contains, beyond a precision number.

Precision@K says 1 in N alerts is fraud. It does not say whether the false
positives are near-misses that a human would also have flagged, or obvious
noise the model should never have surfaced -- and those two situations call
for completely different follow-up work (better features vs. a better
threshold vs. a rule carve-out). Same for the misses: fraud ranked just below
the cutoff means "raise capacity", fraud ranked in the bottom decile means
"the model has no idea what this pattern is".

Everything here is computed on ONE fold's test window (the final, largest
one) using the already-scored predictions -- no refitting, no extra passes
over the full dataset.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger("train_pipeline.error_analysis")


def segment_profile(X: pd.DataFrame, y_true: np.ndarray, y_proba: np.ndarray,
                    threshold: float) -> pd.DataFrame:
    """
    Per-feature means for the three groups that matter operationally:

      TP  -- flagged and fraud       (the queue working as intended)
      FP  -- flagged and legitimate  (the review cost)
      FN  -- not flagged but fraud   (the losses that get through)

    Also reports a standardized gap between FP and TP: (mean_FP - mean_TP)
    divided by the pooled std of the whole window. That normalization is what
    makes the columns comparable -- raw `amount` differences run to six
    figures while indicator features move between 0 and 1, so unnormalized
    gaps would just rank features by their units.
    """
    flagged = y_proba >= threshold
    is_fraud = y_true == 1

    groups = {
        "true_positive": flagged & is_fraud,
        "false_positive": flagged & ~is_fraud,
        "false_negative": ~flagged & is_fraud,
    }

    overall_std = X.std(axis=0).replace(0, np.nan)

    profile = pd.DataFrame({"feature": X.columns})
    profile["overall_mean"] = X.mean(axis=0).to_numpy()

    for label, mask in groups.items():
        profile[f"{label}_n"] = int(mask.sum())
        profile[f"{label}_mean"] = (
            X.loc[mask].mean(axis=0).to_numpy() if mask.any() else np.nan
        )

    profile["fp_vs_tp_std_gap"] = (
        (profile["false_positive_mean"] - profile["true_positive_mean"])
        / overall_std.to_numpy()
    )
    profile["fn_vs_tp_std_gap"] = (
        (profile["false_negative_mean"] - profile["true_positive_mean"])
        / overall_std.to_numpy()
    )

    return profile.sort_values("fp_vs_tp_std_gap", key=np.abs, ascending=False) \
                  .reset_index(drop=True)


def missed_fraud_ranking(y_true: np.ndarray, y_proba: np.ndarray, k: int) -> dict:
    """
    Where do the missed frauds sit in the ranking?

    Distinguishes "just outside the queue" from "invisible to the model",
    which is the difference between a capacity problem and a modelling
    problem. Percentile is expressed so that 100 = top of the ranking.

    `k` must be the number of alerts ACTUALLY flagged (`operating_point`'s
    `n_flagged`), not the requested capacity K. The two differ whenever scores
    tie at the threshold, and passing the requested K would count the tied
    rows as misses here while `segment_profile` -- which cuts on the threshold
    -- counts them as flagged, so the two halves of the error analysis would
    disagree about the same queue. Cutting at rank <= n_flagged reproduces the
    threshold-flagged set exactly, since those rows are by construction the
    top n_flagged by score.
    """
    is_fraud = y_true == 1
    n_fraud = int(is_fraud.sum())
    if n_fraud == 0:
        return {}

    # Rank 1 = highest score. argsort of -proba gives descending order.
    order = np.argsort(-y_proba, kind="stable")
    rank_of_row = np.empty(len(y_proba), dtype=np.int64)
    rank_of_row[order] = np.arange(1, len(y_proba) + 1)

    fraud_ranks = rank_of_row[is_fraud]
    missed_ranks = fraud_ranks[fraud_ranks > k]

    stats = {
        "total_fraud": n_fraud,
        "fraud_in_top_k": int((fraud_ranks <= k).sum()),
        "fraud_missed": len(missed_ranks),
        "n_rows": len(y_proba),
        "k": int(k),
    }
    if len(missed_ranks):
        stats.update({
            "missed_rank_min": int(missed_ranks.min()),
            "missed_rank_median": float(np.median(missed_ranks)),
            "missed_rank_max": int(missed_ranks.max()),
            # How far past the cutoff would capacity have to stretch to catch
            # half / all of the misses?
            "k_to_catch_half_of_missed": int(np.median(missed_ranks)),
            "k_to_catch_all_missed": int(missed_ranks.max()),
            "missed_within_2x_k": int((missed_ranks <= 2 * k).sum()),
            "missed_in_bottom_half": int((missed_ranks > len(y_proba) / 2).sum()),
        })
    return stats


def log_error_analysis(profile: pd.DataFrame, missed: dict, top_n: int = 6) -> None:
    tp_n = int(profile["true_positive_n"].iloc[0])
    fp_n = int(profile["false_positive_n"].iloc[0])
    fn_n = int(profile["false_negative_n"].iloc[0])

    logger.info("  Review queue composition: %d true fraud, %d false positives, %d fraud missed",
                tp_n, fp_n, fn_n)

    if fp_n == 0:
        logger.info("    No false positives in the queue -- nothing to profile.")
    else:
        logger.info("    Features most separating FALSE POSITIVES from TRUE fraud in the queue")
        logger.info("    (gap in pooled std units; +ve = higher in false positives):")
        for _, row in profile.head(top_n).iterrows():
            if not np.isfinite(row["fp_vs_tp_std_gap"]):
                continue
            logger.info("      %-32s FP mean %12.3f vs TP mean %12.3f   gap %+.2f sd",
                        row["feature"], row["false_positive_mean"],
                        row["true_positive_mean"], row["fp_vs_tp_std_gap"])

    if missed.get("fraud_missed"):
        logger.info("    Missed fraud: %d of %d, ranked %d..%d out of %d rows (median rank %.0f)",
                    missed["fraud_missed"], missed["total_fraud"],
                    missed["missed_rank_min"], missed["missed_rank_max"],
                    missed["n_rows"], missed["missed_rank_median"])
        logger.info("      %d of the %d misses fall within 2x the review capacity "
                    "(a staffing gap); %d sit in the bottom half of the ranking "
                    "(a modelling gap).",
                    missed["missed_within_2x_k"], missed["fraud_missed"],
                    missed["missed_in_bottom_half"])
    elif missed:
        logger.info("    Missed fraud: none -- every fraud in this window ranked inside the queue.")
