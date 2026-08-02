"""
inference/rules.py

A minimal hard-block rule layer, evaluated before the model
(ARCHITECTURE.md §1/§5). Realistic and cheap: a short list of predicates
over already-computed features, checked in order, first match wins. The ML
score stays the primary signal -- this layer exists to demonstrate the
architecture pattern (some decisions should never wait on a model call),
not to replace scoring.

ONE_RULE, measured against the real 6.36M-row table (`features` materialized
table, 2026-08-02): type in {TRANSFER, CASH_OUT} AND the origin account was
swept to zero AND the amount clears a threshold fires on 0.247% of rows at
a 10.1% precision (1,588 of 15,723 fraud). That is deliberately reported
here rather than left implicit: PaySim's fraud generator always produces a
full-balance sweep on TRANSFER/CASH_OUT (see features.py's ACCOUNT
STRUCTURE note), so this predicate is a near-necessary but nowhere-near-
sufficient condition on this dataset -- adding "destination has no prior
history" barely moves precision (most PaySim destinations have none
regardless of fraud status). A real deployment would calibrate
HARD_BLOCK_AMOUNT_THRESHOLD against a labeled false-block budget; here it
is a placeholder chosen to keep the block rate low, not a tuned value.

Decisions:
  BLOCK  -- a hard rule fired. The transaction is denied without waiting
            for a model score (though score.py still runs, for the audit
            record -- see api/main.py).
  REVIEW -- no rule fired, and the model score is at or above the bundled
            decision_threshold.
  PASS   -- no rule fired and the model score is below threshold.
"""

from dataclasses import dataclass
from typing import Callable

HARD_BLOCK_AMOUNT_THRESHOLD = 1_000_000


def _full_balance_sweep(features: dict) -> bool:
    return (
        (features["is_transfer"] == 1 or features["is_cash_out"] == 1)
        and features["orig_emptied"] == 1
        and features["amount"] >= HARD_BLOCK_AMOUNT_THRESHOLD
    )


# Ordered; the first matching rule wins. Each predicate reads the same
# feature dict inference/features.py's compute_features() assembles before
# reordering it into a vector, keyed by name rather than position so a rule
# survives a change to FEATURE_COLUMNS' order.
HARD_BLOCK_RULES: list[tuple[str, Callable[[dict], bool]]] = [
    ("full_balance_sweep_large_amount", _full_balance_sweep),
]


@dataclass(frozen=True)
class RuleResult:
    decision: str  # "BLOCK" | "REVIEW" | "PASS"
    rule: str | None  # name of the rule that fired, or None


def evaluate_hard_block(features: dict) -> RuleResult:
    """Checks HARD_BLOCK_RULES in order; returns the first match, if any."""
    for name, predicate in HARD_BLOCK_RULES:
        if predicate(features):
            return RuleResult(decision="BLOCK", rule=name)
    return RuleResult(decision="PASS", rule=None)


def decide(features: dict, flagged: bool) -> RuleResult:
    """
    Full decision combining the hard-block layer with the model's flag:
    BLOCK (rule fired) takes precedence over REVIEW (model flagged) takes
    precedence over PASS.
    """
    hard = evaluate_hard_block(features)
    if hard.decision == "BLOCK":
        return hard
    return RuleResult(decision="REVIEW" if flagged else "PASS", rule=None)
