"""
monitoring/drift.py

Population Stability Index, and the decision rules layered on top of it
(ARCHITECTURE.md §10).

PSI compares two distributions of the same quantity by binning both against a
shared set of cut points and summing, over bins,

    (comparison_share - reference_share) * ln(comparison_share / reference_share)

which is the symmetric (Jeffreys) divergence between the two binned
distributions. Symmetric matters here: "the new window looks unlike training"
and "training looks unlike the new window" are the same statement, and a metric
that answered them differently would need an argument for which direction to
report.

Three implementation choices are worth stating, because each is a place where a
plausible-looking alternative quietly changes the number.

**Binning strategy is chosen from the reference, not assumed.** Quantile bins
are the right default for a continuous feature and meaningless for a binary
indicator -- deciles of a 0/1 column collapse onto duplicate edges, and whatever
survives is an artifact of tie-breaking rather than a description of the data.
`prepare_reference` counts the reference's distinct values first and uses exact
per-level shares when there are few enough of them, so `is_night` and `amount`
are each binned the way they should be, and every result records which strategy
produced it.

**An empty bin is floored at half an observation, not at a constant.** PSI is
undefined when a bin is empty on one side (ln 0), and the conventional fix is to
substitute a small fixed number, typically 1e-4. That makes the reported PSI
depend on an arbitrary constant: with ten bins and one of them empty, 1e-6 and
1e-4 differ by roughly 2x on identical data. The floor used here is `0.5 / n` --
half an observation in a sample of that size -- which is the natural stand-in
for "we saw none of these" and, unlike a constant, scales correctly: an empty
bin in 500,000 rows is far stronger evidence of a shift than an empty bin in 114
rows, and produces a larger PSI, which is the answer a fraud lead wants. The
cost is that PSI becomes sample-size dependent in exactly the regime where it is
least trustworthy anyway, which is why every result also carries `n_comparison`
and `sufficient_sample` rather than being reported bare.

**Bins empty on BOTH sides are dropped, not floored.** They carry no
information, and floring them would have each side land on a slightly different
floor (0.5/n_reference vs 0.5/n_comparison) and contribute a small nonzero term
to PSI out of nothing. That matters because the categorical path always carries
an "unseen level" bin, which is empty on both sides for most windows.

Deliberately pure numpy: no pandas, no duckdb, no lightgbm. The tests that prove
this detector fires (tests/test_drift.py -- injected mean, variance and
category-reweighting shifts) construct distributions directly, and a detector
that could only be exercised by running a 6.36M-row job is a detector that would
not be tested.
"""

from dataclasses import dataclass

import numpy as np

# Conventional PSI bands (Siddiqi, credit-risk scorecard practice), used here as
# the starting points they are rather than as anything fitted to this data.
# ARCHITECTURE.md §10 and the dashboard both label them as such.
PSI_STABLE_BELOW = 0.10
PSI_SIGNIFICANT_AT = 0.25

# Per-feature drift is called at the "significant" band. The score distribution
# -- the single series a reviewer actually watches day to day, and one that
# moves only when something upstream already has -- is called at the tighter
# "moderate" band.
PSI_FEATURE_THRESHOLD = PSI_SIGNIFICANT_AT
PSI_SCORE_THRESHOLD = PSI_STABLE_BELOW

DEFAULT_N_BINS = 10

# Below this many rows a PSI number is noise rather than a measurement. Such
# windows are reported with `sufficient_sample=False` rather than suppressed:
# PaySim's own volume collapses to 114 rows in its final day, and hiding that
# window would hide a real property of the data. 1,000 rows across 10 bins is
# ~100 expected per bin, the usual rule of thumb for a stable share estimate.
MIN_COMPARISON_ROWS = 1_000


@dataclass(frozen=True)
class ReferenceBins:
    """A reference distribution reduced to bins, ready to compare windows against.

    Prepared once and reused, because the job compares 18 features against ~32
    windows each: re-deriving deciles from 5.09M reference rows inside every one
    of those 576 comparisons is the difference between a job that runs in
    seconds and one that runs in minutes, and it would compute the identical
    edges every time.
    """

    binning: str  # "quantile" | "categorical"
    counts: np.ndarray
    n_reference: int
    edges: np.ndarray | None = None   # quantile binning
    levels: np.ndarray | None = None  # categorical binning

    @property
    def n_bins(self) -> int:
        return int(self.counts.size)


@dataclass(frozen=True)
class DriftResult:
    """One PSI measurement, with everything needed to judge whether to believe it."""

    psi: float
    binning: str  # "quantile" | "categorical" | "undefined"
    n_bins: int
    n_reference: int
    n_comparison: int
    threshold: float
    breached: bool
    sufficient_sample: bool

    @property
    def band(self) -> str:
        return classify(self.psi)


def classify(psi_value: float) -> str:
    """The conventional three-band reading of a PSI number."""
    if not np.isfinite(psi_value):
        return "undefined"
    if psi_value < PSI_STABLE_BELOW:
        return "stable"
    if psi_value < PSI_SIGNIFICANT_AT:
        return "moderate"
    return "significant"


def quantile_bin_edges(reference: np.ndarray, n_bins: int = DEFAULT_N_BINS) -> np.ndarray:
    """
    Bin edges at the reference's own quantiles, open at both ends.

    Open ends (-inf / +inf) are not decoration: a comparison window may contain
    values beyond anything the reference held, and those rows must land in the
    outermost bins rather than be dropped by `np.histogram`. Silently discarding
    the most extreme rows in a window is precisely the wrong behaviour for a
    drift detector.

    Duplicate interior edges are collapsed. A feature whose reference is 60%
    zeros produces several identical low quantiles; keeping them would create
    empty-by-construction bins contributing a fixed, meaningless term to every
    window's PSI.
    """
    interior = np.quantile(reference, np.linspace(0, 1, n_bins + 1)[1:-1])
    return np.concatenate([[-np.inf], np.unique(interior), [np.inf]])


def prepare_reference(reference: np.ndarray, n_bins: int = DEFAULT_N_BINS) -> ReferenceBins:
    """
    Bin a reference distribution, choosing the strategy from its cardinality.

    The categorical layout carries one extra trailing bin, always empty on the
    reference side, which collects comparison values at levels the reference
    never held. A level that appears only in the comparison window is a new
    population -- the clearest form of drift there is -- and a scheme that
    restricted itself to the reference's levels would drop those rows and report
    the window as stable.
    """
    reference = np.asarray(reference, dtype="float64").ravel()
    levels = np.unique(reference)

    if levels.size <= n_bins:
        # Few enough levels that exact per-level shares beat any binning --
        # every indicator feature and every small-integer count lands here.
        counts = np.array(
            [np.count_nonzero(reference == lv) for lv in levels] + [0.0], dtype="float64"
        )
        return ReferenceBins(binning="categorical", counts=counts,
                             n_reference=reference.size, levels=levels)

    edges = quantile_bin_edges(reference, n_bins)
    counts, _ = np.histogram(reference, bins=edges)
    return ReferenceBins(binning="quantile", counts=counts.astype("float64"),
                         n_reference=reference.size, edges=edges)


def comparison_counts(prepared: ReferenceBins, comparison: np.ndarray) -> np.ndarray:
    """Bin a comparison window into `prepared`'s bins."""
    comparison = np.asarray(comparison, dtype="float64").ravel()

    # The two asserts pin ReferenceBins' invariant: `binning` decides which of
    # `edges` / `levels` is populated, and nothing outside prepare_reference
    # constructs one. Without them a hand-built ReferenceBins would bin every
    # window against `None` and np.histogram would silently re-derive its own
    # edges from the comparison data -- a comparison against itself, which
    # reports every window as perfectly stable.
    if prepared.binning == "quantile":
        assert prepared.edges is not None
        counts, _ = np.histogram(comparison, bins=prepared.edges)
        return counts.astype("float64")

    assert prepared.levels is not None
    per_level = np.array(
        [np.count_nonzero(comparison == lv) for lv in prepared.levels], dtype="float64"
    )
    unseen = float(comparison.size - per_level.sum())
    return np.concatenate([per_level, [unseen]])


def _floored_shares(counts: np.ndarray, n: int) -> np.ndarray:
    """Bin counts -> shares, with empty bins floored at half an observation.

    Renormalised after flooring so the result is still a distribution; without
    that the two sides sum to slightly different totals and the per-bin PSI
    terms stop being comparable.
    """
    shares = counts / n
    shares = np.maximum(shares, 0.5 / n)
    return shares / shares.sum()


def psi_from_counts(reference_counts: np.ndarray, comparison_counts_: np.ndarray) -> float:
    """PSI from two aligned bin-count vectors. This is the arithmetic core;
    everything around it is only deciding what the bins are."""
    reference_counts = np.asarray(reference_counts, dtype="float64")
    comparison_counts_ = np.asarray(comparison_counts_, dtype="float64")
    if reference_counts.shape != comparison_counts_.shape:
        raise ValueError(
            f"bin counts must align: reference has {reference_counts.shape}, "
            f"comparison has {comparison_counts_.shape}"
        )

    n_ref = reference_counts.sum()
    n_comp = comparison_counts_.sum()
    if n_ref <= 0 or n_comp <= 0:
        return float("nan")

    # A bin nobody ever occupied says nothing about stability. Keeping it would
    # floor each side at its own 0.5/n and manufacture a small PSI term from an
    # empty pair -- see the module docstring.
    occupied = (reference_counts > 0) | (comparison_counts_ > 0)
    reference_counts = reference_counts[occupied]
    comparison_counts_ = comparison_counts_[occupied]

    ref_share = _floored_shares(reference_counts, int(n_ref))
    comp_share = _floored_shares(comparison_counts_, int(n_comp))
    return float(np.sum((comp_share - ref_share) * np.log(comp_share / ref_share)))


def compare(prepared: ReferenceBins, comparison: np.ndarray,
            threshold: float = PSI_FEATURE_THRESHOLD,
            min_comparison_rows: int = MIN_COMPARISON_ROWS) -> DriftResult:
    """
    PSI of one comparison window against an already-prepared reference.

    An empty window returns a NaN PSI rather than raising: a monitoring job that
    dies on a quiet hour is worse than one that records the hour as unmeasurable.
    """
    comparison = np.asarray(comparison, dtype="float64").ravel()
    n_comp = comparison.size
    sufficient = n_comp >= min_comparison_rows

    if prepared.n_reference == 0 or n_comp == 0:
        return DriftResult(float("nan"), "undefined", 0, prepared.n_reference, n_comp,
                           threshold, False, sufficient)

    counts = comparison_counts(prepared, comparison)
    value = psi_from_counts(prepared.counts, counts)

    return DriftResult(
        psi=value,
        binning=prepared.binning,
        n_bins=prepared.n_bins,
        n_reference=prepared.n_reference,
        n_comparison=n_comp,
        threshold=threshold,
        # A breach is only claimed on a window large enough to mean it. The PSI
        # is still reported for undersized windows; the flag is what a
        # retraining decision keys on (MONITORING.md), and firing it on 114 rows
        # would make that criterion unusable.
        breached=bool(np.isfinite(value) and value >= threshold and sufficient),
        sufficient_sample=sufficient,
    )


def compare_score(prepared: ReferenceBins, comparison: np.ndarray,
                  min_comparison_rows: int = MIN_COMPARISON_ROWS) -> DriftResult:
    """`compare` at the score threshold.

    This exists so PSI_SCORE_THRESHOLD is named in exactly one place on the
    shipped path. It previously was not: `run_drift.py` passed the constant at
    its own call site, and no test could catch that call site being changed to
    the feature threshold, because no window in this dataset has a score PSI
    between the two bands (0.10-0.25) -- every breach flag in the CSV would have
    been identical and the whole suite would still have passed. A threshold that
    only one un-exercised line decides is a threshold nobody is checking.
    """
    return compare(prepared, comparison, threshold=PSI_SCORE_THRESHOLD,
                   min_comparison_rows=min_comparison_rows)


def compute_drift(reference: np.ndarray, comparison: np.ndarray,
                  n_bins: int = DEFAULT_N_BINS,
                  threshold: float = PSI_FEATURE_THRESHOLD,
                  min_comparison_rows: int = MIN_COMPARISON_ROWS) -> DriftResult:
    """One-shot `prepare_reference` + `compare`, for callers holding a single
    pair of distributions (every test, and any ad-hoc check)."""
    reference = np.asarray(reference, dtype="float64").ravel()
    if reference.size == 0:
        return DriftResult(float("nan"), "undefined", 0, 0,
                           int(np.asarray(comparison).size), threshold, False,
                           int(np.asarray(comparison).size) >= min_comparison_rows)
    prepared = prepare_reference(reference, n_bins=n_bins)
    return compare(prepared, comparison, threshold=threshold,
                   min_comparison_rows=min_comparison_rows)


def compute_score_drift(reference_scores: np.ndarray, comparison_scores: np.ndarray,
                        n_bins: int = DEFAULT_N_BINS,
                        min_comparison_rows: int = MIN_COMPARISON_ROWS) -> DriftResult:
    """One-shot `prepare_reference` + `compare_score`. Model scores are
    continuous by construction, so this always takes the quantile path."""
    reference_scores = np.asarray(reference_scores, dtype="float64").ravel()
    if reference_scores.size == 0:
        return compute_drift(reference_scores, comparison_scores, n_bins=n_bins,
                             threshold=PSI_SCORE_THRESHOLD,
                             min_comparison_rows=min_comparison_rows)
    return compare_score(prepare_reference(reference_scores, n_bins=n_bins),
                         comparison_scores, min_comparison_rows=min_comparison_rows)
