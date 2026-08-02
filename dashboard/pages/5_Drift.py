"""
5_Drift.py

Placeholder (ARCHITECTURE.md §6, page 5 -- explicitly "placeholder until
Sprint 8"). Static; no fabricated charts or numbers. What ships here later
is real: PSI computed over successive slices of the dataset's own 743
simulated hours (not scored demo traffic, which is statistically
meaningless at portfolio-demo volume -- ARCHITECTURE.md §10 explains why
that framing was deliberately rejected), against thresholds of PSI > 0.25
per feature / > 0.10 on the score.
"""

import streamlit as st

st.set_page_config(page_title="Drift", page_icon="🔍", layout="wide")
st.title("Drift")

st.info(
    "Not live yet — this page ships with Sprint 8. Nothing below is a mock of "
    "real output; it's the design this page will implement.",
    icon="🚧",
)

st.markdown(
    """
### What this page will show

- **PSI per feature**, reference = the final training fold's distribution,
  comparison = successive time windows of the dataset itself replayed as if
  arriving live — not scored demo traffic, which at the volume a portfolio
  deployment actually receives would be statistically meaningless (a few
  dozen requests can't estimate a distribution).
- **PSI on the score distribution**, the single number a fraud lead would
  actually watch day to day.
- **Thresholds**: PSI > 0.25 on any feature, > 0.10 on the score —
  conventional starting points, not fitted to this data, and labelled as
  such rather than presented as calibrated.
- **A detector unit test** using deliberately injected shifts (mean, variance,
  category re-weighting) that prove the detector fires correctly, since the
  real dataset may not drift in a way a demo can rely on.

### Why it isn't here now

Sprint 5 ships the dashboard shell and the four pages that have real,
already-computed data to show. Drift needs the Sprint 8 monitoring job
(a scheduled PSI computation, `ARCHITECTURE.md` §10) to exist first — this
page is where its output will land, not a place to fake that output early.

Live scored-traffic volume and score-distribution telemetry (not a drift
signal at demo volume, but real information) will also land here once the
persistence layer (Sprint 10) exists to log it.
"""
)
