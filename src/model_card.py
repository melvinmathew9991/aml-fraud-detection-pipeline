"""
model_card.py

Facts about the currently committed model bundle that both the FastAPI
service (api/main.py, for the live /model-info response) and the
Streamlit dashboard's static Model card page need to state identically.
Zero dependencies -- importable from either the serving image or the
dashboard's own lightweight requirements.txt without pulling in anything
either wasn't already using.

Kept here instead of duplicated in both places for the same reason
inference/features.py imports FEATURE_COLUMNS from training's features.py
rather than hand-copying it: a second copy is how the two silently drift
apart.
"""

MODEL_LIMITATIONS = [
    "The destination-state snapshot is frozen at a single training-time step; "
    "it is a point-in-time approximation, not an online feature store.",
    "Scoring the same transaction twice via /score returns the same answer -- "
    "state does not accumulate. /score/batch accumulates within one batch only.",
    "Serving state is as-of one instant; training features were as-of each "
    "row's own timestamp. The two agree only for transactions arriving after "
    "the snapshot step.",
    "The hard-block rule layer is illustrative, not tuned: measured precision "
    "on the training data is low (~10%) at a ~0.25% block rate. The model "
    "score is the primary signal.",
]

# The dashboard's Model card page is static (ARCHITECTURE.md §6: pages 3-5
# render with no API call), so it cannot read this live off a running
# bundle the way /model-info does. Mirrors the currently committed
# model_bundle/v1/dest_state.parquet's embedded snapshot_step metadata
# (verified via /model-info against the live bundle, not just asserted
# here) -- update this if the bundle is ever rebuilt against a different
# snapshot.
CURRENT_BUNDLE_SNAPSHOT_STEP = 743
