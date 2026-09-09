"""
Pins dashboard/common.py's FEATURE_LABELS to src/features.py:FEATURE_COLUMNS.

The labels are display-only, so nothing breaks loudly when they drift -- a
renamed or added feature simply falls back to its raw name in the UI, which is
correct behaviour but silent. Sprint 11 adds graph features and Sprint 3
already removed two, so drift is expected rather than hypothetical, and the
failure mode is a page that quietly shows `dest_amount_to_prior_avg_ratio` to
the audience the labels exist for.

This is the same class of gap that let pages/4_Model_Card.py stay broken on
main for six weeks (see tests/test_dashboard_page_imports.py): correct code,
no test reaching it.
"""

import sys
from pathlib import Path

import common

SRC = Path(common.__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from features import FEATURE_COLUMNS  # noqa: E402


def test_every_model_feature_has_a_plain_english_label():
    unlabelled = [f for f in FEATURE_COLUMNS if f not in common.FEATURE_LABELS]
    assert not unlabelled, (
        f"{unlabelled} have no entry in FEATURE_LABELS, so the dashboard will show "
        f"the raw feature name to a non-technical reader."
    )


def test_no_label_describes_a_feature_that_no_longer_exists():
    stale = [f for f in common.FEATURE_LABELS if f not in FEATURE_COLUMNS]
    assert not stale, (
        f"FEATURE_LABELS still describes {stale}, which src/features.py no longer "
        f"produces."
    )


def test_feature_label_falls_back_to_the_raw_name():
    # The fallback is what keeps an unmapped Sprint 11 feature rendering as
    # itself instead of raising inside a page.
    assert common.feature_label("some_future_graph_feature") == "some_future_graph_feature"


def test_labels_are_not_just_the_feature_name_reformatted():
    # A label that is the snake_case name with underscores swapped for spaces
    # adds nothing for the reader it exists for.
    lazy = [
        name for name, label in common.FEATURE_LABELS.items()
        if label.lower().replace(" ", "_") == name.lower()
    ]
    assert not lazy, f"{lazy} are labelled with their own field name"
