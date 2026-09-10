"""
Tests src/monitoring/drift.py.

These are the tests ARCHITECTURE.md §10 calls "what proves the implementation".
The PSI series the job produces over PaySim runs on data whose drift nobody
controls -- if the dataset happened not to shift, a flat chart would be
indistinguishable from a detector that never fires. So the detector is proven
here instead, against distributions with a shift deliberately injected: a mean
shift, a variance shift, and a category re-weighting, each asserted to push PSI
past the threshold while an unshifted sample of the same size stays under it.

The second group pins the two judgement calls in the module -- the
half-observation floor for empty bins, and the cardinality-based choice between
quantile and categorical binning -- because both are places where a later
"simplification" would silently change every number the dashboard shows.
"""

import numpy as np
import pytest

from monitoring.drift import (
    MIN_COMPARISON_ROWS,
    PSI_FEATURE_THRESHOLD,
    PSI_SCORE_THRESHOLD,
    PSI_STABLE_BELOW,
    classify,
    compare,
    compare_score,
    compute_drift,
    compute_score_drift,
    prepare_reference,
    psi_from_counts,
    quantile_bin_edges,
)

N_REFERENCE = 50_000
N_COMPARISON = 20_000


@pytest.fixture
def rng():
    return np.random.default_rng(20260910)


# --------------------------------------------------------------- injected shifts


def test_no_shift_stays_stable(rng):
    """The control. Two draws from the same distribution must not fire, or every
    other assertion in this file is measuring sampling noise rather than drift."""
    result = compute_drift(rng.normal(0, 1, N_REFERENCE), rng.normal(0, 1, N_COMPARISON))

    assert result.psi < PSI_STABLE_BELOW
    assert result.band == "stable"
    assert not result.breached
    assert result.sufficient_sample


def test_injected_mean_shift_fires(rng):
    reference = rng.normal(0, 1, N_REFERENCE)
    comparison = rng.normal(0, 1, N_COMPARISON) + 1.0  # one standard deviation

    result = compute_drift(reference, comparison)

    assert result.breached
    assert result.psi >= PSI_FEATURE_THRESHOLD
    assert result.band == "significant"


def test_injected_variance_shift_fires(rng):
    """A pure spread change with the mean held at zero. Worth its own test: a
    detector comparing means alone -- a plausible-looking simplification -- would
    report this window as perfectly stable."""
    reference = rng.normal(0, 1, N_REFERENCE)
    comparison = rng.normal(0, 1, N_COMPARISON) * 3.0

    result = compute_drift(reference, comparison)

    assert np.isclose(comparison.mean(), 0, atol=0.1), "the injected shift must be variance-only"
    assert result.breached
    assert result.psi >= PSI_FEATURE_THRESHOLD


def test_injected_category_reweighting_fires(rng):
    """The categorical path: same levels, different mix. This is what a change in
    transaction-type composition looks like, and it is the shift the indicator
    features (is_transfer, is_night, ...) can express at all."""
    reference = rng.choice([0.0, 1.0, 2.0], size=N_REFERENCE, p=[0.7, 0.2, 0.1])
    comparison = rng.choice([0.0, 1.0, 2.0], size=N_COMPARISON, p=[0.2, 0.3, 0.5])

    result = compute_drift(reference, comparison)

    assert result.binning == "categorical"
    assert result.breached


def test_psi_grows_monotonically_with_shift_size(rng):
    """Not just "it fires" but "it measures". A detector whose output did not
    order shifts by size would be a boolean with extra steps, and the dashboard
    plots the value, not the flag."""
    reference = rng.normal(0, 1, N_REFERENCE)
    prepared = prepare_reference(reference)

    values = [
        compare(prepared, rng.normal(0, 1, N_COMPARISON) + delta).psi
        for delta in (0.0, 0.25, 0.5, 1.0, 2.0)
    ]

    assert values == sorted(values)


def test_a_level_absent_from_the_reference_is_drift(rng):
    """A category the training data never contained is the clearest drift there
    is. Binning restricted to the reference's own levels would drop those rows
    and call the window stable -- hence the trailing unseen-level bin."""
    reference = rng.choice([0.0, 1.0], size=N_REFERENCE, p=[0.5, 0.5])
    comparison = np.concatenate([
        rng.choice([0.0, 1.0], size=N_COMPARISON // 2, p=[0.5, 0.5]),
        np.full(N_COMPARISON // 2, 9.0),  # a level the reference never held
    ])

    result = compute_drift(reference, comparison)

    assert result.binning == "categorical"
    assert result.breached


def test_values_beyond_the_reference_range_are_not_dropped(rng):
    """The open-ended outer bins. `np.histogram` with closed edges would discard
    every out-of-range row, so a window consisting entirely of extreme values
    would report as unchanged."""
    reference = rng.normal(0, 1, N_REFERENCE)
    comparison = rng.normal(0, 1, N_COMPARISON) + 50.0  # far outside the reference

    result = compute_drift(reference, comparison)

    assert result.breached
    edges = quantile_bin_edges(reference)
    assert edges[0] == -np.inf and edges[-1] == np.inf


# --------------------------------------------------- score drift and its threshold


def test_score_drift_uses_the_tighter_threshold(rng):
    """The score band is 0.10, not 0.25 -- a shift big enough to move the score
    distribution has already moved something upstream."""
    reference = rng.beta(0.5, 20, N_REFERENCE)
    comparison = rng.beta(0.6, 20, N_COMPARISON)

    result = compute_score_drift(reference, comparison)

    assert result.threshold == PSI_SCORE_THRESHOLD
    assert result.threshold < PSI_FEATURE_THRESHOLD


def test_compare_score_is_the_path_the_job_takes(rng):
    """`run_drift.py` prepares the reference once and calls `compare_score` per
    window, so that is the function that has to carry the score band -- not the
    one-shot helper above.

    This test exists because of an audit finding. The job used to pass the
    constant at its own call site, and nothing could catch that line being
    changed to the feature threshold: no window in the dataset has a score PSI
    between 0.10 and 0.25, so every breach flag in the committed CSV would have
    been byte-identical and the whole suite would still have passed.
    """
    reference = rng.beta(0.5, 20, N_REFERENCE)
    comparison = rng.beta(0.6, 20, N_COMPARISON)
    prepared = prepare_reference(reference)

    from_job_path = compare_score(prepared, comparison)

    assert from_job_path.threshold == PSI_SCORE_THRESHOLD
    assert from_job_path.psi == pytest.approx(compute_score_drift(reference, comparison).psi)
    assert from_job_path.breached == compute_score_drift(reference, comparison).breached


def test_the_same_distribution_at_both_thresholds_can_disagree(rng):
    """The point of having two thresholds: a moderate shift is a breach for the
    score series and not for a feature. If this ever stops being true the two
    constants have collapsed into one."""
    reference = rng.normal(0, 1, N_REFERENCE)
    comparison = rng.normal(0, 1, N_COMPARISON) + 0.4

    as_score = compute_score_drift(reference, comparison)
    as_feature = compute_drift(reference, comparison)

    assert as_score.psi == as_feature.psi
    assert as_score.breached and not as_feature.breached


# ------------------------------------------------------------- sample-size rules


def test_an_undersized_window_reports_psi_but_never_breaches(rng):
    """PaySim's final day is 114 rows. Its PSI is real arithmetic and worth
    plotting; treating it as a retraining trigger is not."""
    reference = rng.normal(0, 1, N_REFERENCE)
    comparison = rng.normal(5, 1, MIN_COMPARISON_ROWS - 1)

    result = compute_drift(reference, comparison)

    assert np.isfinite(result.psi)
    assert result.psi >= PSI_FEATURE_THRESHOLD, "the shift is real"
    assert not result.sufficient_sample
    assert not result.breached, "but it must not fire on a sample this small"


def test_an_empty_window_is_undefined_not_an_error(rng):
    """A quiet hour must not kill the job."""
    result = compute_drift(rng.normal(0, 1, N_REFERENCE), np.array([]))

    assert np.isnan(result.psi)
    assert result.band == "undefined"
    assert not result.breached


def test_the_empty_bin_floor_scales_with_sample_size():
    """The half-observation floor, pinned. The same *shape* of miss -- one bin
    empty out of ten equal reference bins -- must produce a larger PSI in a
    larger window, because an empty bin in 100,000 rows is stronger evidence
    than an empty bin in 1,000. A fixed epsilon would make these two equal.
    """
    reference_counts = np.full(10, 1000.0)

    small = np.concatenate([np.full(9, 1000.0 / 9), [0.0]])
    large = np.concatenate([np.full(9, 100_000.0 / 9), [0.0]])

    assert psi_from_counts(reference_counts, large) > psi_from_counts(reference_counts, small)


def test_bins_empty_on_both_sides_contribute_nothing():
    """Otherwise the always-present unseen-level bin would add a small PSI term
    to every categorical comparison out of nothing at all."""
    reference = np.array([500.0, 500.0])
    comparison = np.array([200.0, 200.0])

    with_dead_bin = psi_from_counts(
        np.concatenate([reference, [0.0]]), np.concatenate([comparison, [0.0]])
    )

    assert with_dead_bin == pytest.approx(psi_from_counts(reference, comparison))


# ------------------------------------------------------------- binning strategy


def test_binary_features_take_the_categorical_path(rng):
    """Deciles of a 0/1 column collapse onto duplicate edges; whatever survives
    describes tie-breaking, not the data."""
    prepared = prepare_reference((rng.random(N_REFERENCE) < 0.3).astype(float))

    assert prepared.binning == "categorical"
    assert prepared.n_bins == 3  # two levels plus the unseen-level bin


def test_continuous_features_take_the_quantile_path(rng):
    prepared = prepare_reference(rng.normal(0, 1, N_REFERENCE))

    assert prepared.binning == "quantile"
    assert prepared.n_bins == 10


def test_duplicate_quantile_edges_are_collapsed(rng):
    """A feature that is mostly zeros -- amount_to_balance_ratio, and most of the
    destination aggregates -- produces several identical low quantiles. Keeping
    them would add empty-by-construction bins to every window."""
    reference = np.concatenate([np.zeros(35_000), rng.random(15_000) + 1])

    edges = quantile_bin_edges(reference)

    assert len(edges) == len(np.unique(edges)), "duplicate edges survived"


def test_a_constant_reference_does_not_divide_by_zero():
    """Degenerate but reachable: a feature can be constant inside the reference
    fold and vary later."""
    unchanged = compute_drift(np.zeros(N_REFERENCE), np.zeros(N_COMPARISON))
    changed = compute_drift(np.zeros(N_REFERENCE), np.ones(N_COMPARISON))

    assert unchanged.psi == pytest.approx(0.0)
    assert changed.breached


# ------------------------------------------------------------------ metric shape


def test_psi_is_symmetric(rng):
    """(comp - ref) * ln(comp/ref) is symmetric by construction, and the reading
    of the number depends on it: "this window is unlike training" and "training
    is unlike this window" are one statement."""
    a = rng.normal(0, 1, N_COMPARISON)
    b = rng.normal(0.5, 1, N_COMPARISON)

    counts_a, edges = np.histogram(a, bins=quantile_bin_edges(a))
    counts_b, _ = np.histogram(b, bins=edges)

    assert psi_from_counts(counts_a, counts_b) == pytest.approx(
        psi_from_counts(counts_b, counts_a)
    )


def test_psi_is_zero_for_identical_distributions(rng):
    values = rng.normal(0, 1, N_REFERENCE)

    assert compute_drift(values, values).psi == pytest.approx(0.0)


def test_classify_bands_match_the_documented_cutoffs():
    assert classify(0.0) == "stable"
    assert classify(0.099) == "stable"
    assert classify(0.10) == "moderate"
    assert classify(0.249) == "moderate"
    assert classify(0.25) == "significant"
    assert classify(float("nan")) == "undefined"


def test_mismatched_bin_counts_raise():
    """Aligning two differently-binned histograms would produce a plausible
    number from meaningless pairs."""
    with pytest.raises(ValueError, match="align"):
        psi_from_counts(np.ones(10), np.ones(8))


def test_prepared_reference_matches_the_one_shot_call(rng):
    """The job prepares once and compares many times; the tests call the one-shot
    helper. They must agree, or the tested path is not the shipped path."""
    reference = rng.normal(0, 1, N_REFERENCE)
    comparison = rng.normal(0.3, 1.2, N_COMPARISON)

    assert compare(prepare_reference(reference), comparison).psi == pytest.approx(
        compute_drift(reference, comparison).psi
    )
