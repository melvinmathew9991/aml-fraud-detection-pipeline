"""
5_Drift.py

Monitoring page (ARCHITECTURE.md §6, page 5). Sprint 8 replaced the Sprint 5
placeholder with the real output of `src/run_drift.py`.

Two things live here, and the page keeps them visibly apart because they are not
the same kind of evidence:

- **Drift**, computed offline over the dataset's own 743 simulated hours and
  read from committed CSVs. This is the measurement.
- **Live telemetry** from the deployed API's /metrics. Real, but a portfolio
  endpoint serves tens of requests, and PSI on that sample would be noise. It is
  labelled as volume and latency information, never as a drift signal.

Static except for the telemetry panel, which is the only thing on this page that
touches the API -- and it degrades to a notice when the service is asleep.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from common import (
    call_api_with_wake_retry,
    feature_label,
    load_csv,
    load_json,
)

st.set_page_config(page_title="Drift", page_icon="🔍", layout="wide")
st.title("Drift")

reference = load_json("drift_reference")
score_psi = load_csv("drift_score_psi")
feature_psi = load_csv("drift_feature_psi")

FEATURE_THRESHOLD = reference["thresholds"]["feature_psi"]
SCORE_THRESHOLD = reference["thresholds"]["score_psi"]
MIN_ROWS = reference["thresholds"]["min_comparison_rows"]
REF = reference["reference"]

st.caption(
    f"Population Stability Index over dataset time. Reference: {REF['definition']} "
    f"— steps {REF['step_min']}–{REF['step_max']}, {REF['n_rows']:,} rows. "
    f"Comparison: each successive simulated day of the same dataset, scored with "
    f"the deployed bundle `{reference['bundle_version']}`."
)

measured = score_psi[score_psi["sufficient_sample"]]
breaching_features = feature_psi[feature_psi["breached"]]["feature"].nunique()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Windows", len(score_psi),
          help=f"One simulated day each ({reference['windows']['steps_per_window']} steps).")
c2.metric("Score PSI breaches", int(measured["score_breached"].sum()),
          help=f"Windows where the score distribution moved past PSI {SCORE_THRESHOLD:.2f}, "
               f"counting only windows with at least {MIN_ROWS:,} rows.")
c3.metric("Features ever breaching", f"{breaching_features} / {len(reference['features'])}",
          help=f"Features that passed PSI {FEATURE_THRESHOLD:.2f} in at least one window.")
c4.metric("Undersized windows", int((~score_psi["sufficient_sample"]).sum()),
          help=f"Fewer than {MIN_ROWS:,} rows. PSI is still shown; it just cannot "
               "raise a flag.")

st.info(
    "**These thresholds are conventions, not calibration.** PSI < 0.10 stable, "
    "0.10–0.25 moderate, ≥ 0.25 significant are the standard credit-risk bands. "
    "They were not fitted to this data, and nothing here claims they are the right "
    "numbers for it.",
    icon="ℹ️",
)

# --------------------------------------------------------------------- score PSI

st.markdown("---")
st.subheader("Score PSI over dataset time")
st.caption(
    "The single series a fraud lead watches day to day: has the distribution of "
    "the model's own output moved? Shaded windows fall inside the reference "
    "period — their PSI is the detector's noise floor, measured on the data the "
    "reference was built from rather than asserted."
)

fig = go.Figure()
in_reference = score_psi[score_psi["in_reference_period"]]
if len(in_reference):
    fig.add_vrect(
        x0=in_reference["window"].min() - 0.5, x1=in_reference["window"].max() + 0.5,
        fillcolor="LightSkyBlue", opacity=0.18, line_width=0,
        annotation_text="reference period", annotation_position="top left",
    )
fig.add_trace(go.Scatter(
    x=score_psi["window"], y=score_psi["score_psi"],
    mode="lines+markers", name="Score PSI",
    marker={"size": [10 if not ok else 7 for ok in score_psi["sufficient_sample"]],
            "symbol": ["x" if not ok else "circle"
                       for ok in score_psi["sufficient_sample"]]},
    customdata=score_psi[["step_start", "step_end", "n_rows", "sufficient_sample"]],
    hovertemplate=("Window %{x}<br>steps %{customdata[0]}–%{customdata[1]}"
                   "<br>%{customdata[2]:,} rows<br>PSI %{y:.4f}"
                   "<br>usable sample: %{customdata[3]}<extra></extra>"),
))
fig.add_hline(y=SCORE_THRESHOLD, line_dash="dash", line_color="firebrick",
              annotation_text=f"score threshold {SCORE_THRESHOLD:.2f}")
fig.update_layout(height=420, yaxis_type="log", xaxis_title="Window (simulated day)",
                  yaxis_title="PSI (log scale)", showlegend=False)
st.plotly_chart(fig, width='stretch')
st.caption(
    "Log scale — the flat stretch sits three orders of magnitude below the "
    "breaches, and a linear axis would render it as a straight line at zero. "
    "✕ marks a window too small to raise a flag."
)

# ------------------------------------------------------------------- feature PSI

st.markdown("---")
st.subheader("Feature PSI")
st.caption(
    "Every feature against every window. Read it by row: a band of colour across "
    "a feature means that input, not the model, is what moved."
)

heat = feature_psi.pivot(index="feature", columns="window", values="psi")
heat = heat.loc[heat.max(axis=1).sort_values(ascending=False).index]
heat.index = [feature_label(name) for name in heat.index]

heat_fig = px.imshow(
    heat, color_continuous_scale="Reds", zmin=0, zmax=FEATURE_THRESHOLD * 4,
    aspect="auto", labels={"x": "Window (simulated day)", "y": "", "color": "PSI"},
)
heat_fig.update_layout(height=560)
heat_fig.update_traces(
    hovertemplate="%{y}<br>window %{x}<br>PSI %{z:.3f}<extra></extra>")
st.plotly_chart(heat_fig, width='stretch')
st.caption(
    f"Colour saturates at {FEATURE_THRESHOLD * 4:.1f}; anything at "
    f"or above {FEATURE_THRESHOLD:.2f} counts as a breach on a window with a "
    "usable sample."
)

with st.expander("Per-window detail"):
    chosen = st.selectbox(
        "Window",
        score_psi["window"].tolist(),
        format_func=lambda w: (
            f"Window {w} — steps {score_psi.loc[score_psi['window'] == w, 'step_start'].iloc[0]}"
            f"–{score_psi.loc[score_psi['window'] == w, 'step_end'].iloc[0]}"
            f" ({score_psi.loc[score_psi['window'] == w, 'n_rows'].iloc[0]:,} rows)"
        ),
    )
    detail = feature_psi[feature_psi["window"] == chosen].copy()
    detail["feature"] = detail["feature"].map(feature_label)
    st.dataframe(
        detail[["feature", "psi", "band", "breached", "binning", "n_bins"]]
        .sort_values("psi", ascending=False),
        hide_index=True, width='stretch',
    )

# ------------------------------------------------------- the operating point

st.markdown("---")
st.subheader("What the deployed threshold would have done")
st.caption(
    "The same windows, scored at the bundle's own fixed decision threshold — not "
    "at a threshold re-fitted per window. This is the operational half of "
    "monitoring, and on this dataset it moves far more than drift does."
)

alerts = go.Figure()
alerts.add_trace(go.Bar(
    x=score_psi["window"], y=score_psi["alerts_per_day"], name="Alerts per day",
    marker_color="steelblue",
))
alerts.add_trace(go.Scatter(
    x=score_psi["window"], y=score_psi["capacity"], name="Analyst capacity",
    mode="lines", line={"color": "firebrick", "dash": "dash"},
))
alerts.update_layout(height=380, yaxis_type="log", barmode="overlay",
                     xaxis_title="Window (simulated day)",
                     yaxis_title="Alerts (log scale)")
st.plotly_chart(alerts, width='stretch')

# The final window is 23 steps, not 24, and its capacity is clamped to its own
# row count -- so it reads as "over capacity" for an arithmetic reason rather
# than an operational one. Counting only the full windows keeps the claim honest.
full_windows = score_psi[score_psi["window"] < score_psi["window"].max()]
over = full_windows[full_windows["alerts_per_day"] > full_windows["capacity"]]
st.markdown(
    f"""
The fixed threshold produces **{full_windows['alerts_per_day'].min():,.0f} to
{full_windows['alerts_per_day'].max():,.0f} alerts a day** across these windows,
against a configured capacity of {int(score_psi['capacity'].max())}. It exceeds
capacity in **{len(over)} of the {len(full_windows)} full windows** — every one of
them a high-volume window, and none of them a window where drift fired.

A score threshold fixes a *score*, not a queue length. Volume across these
windows spans {full_windows['n_rows'].min():,} to {full_windows['n_rows'].max():,}
rows, and the alert queue moves with it. That is the operational consequence of
the fold-3-specific threshold the model card already lists as a limitation,
measured here across the whole dataset rather than argued — and it is a larger
effect than any drift on this page.
"""
)

st.markdown("#### Is the model itself still ranking?")
h1, h2, h3 = st.columns(3)
h1.metric("Recall at capacity",
          f"{score_psi['recall_at_capacity'].min():.3f} – "
          f"{score_psi['recall_at_capacity'].max():.3f}",
          help="Share of the window's fraud that lands inside the queue analysts "
               "can actually work. This is the retraining trigger.")
h2.metric("Windows on the precision ceiling",
          f"{int(score_psi['at_ceiling'].sum())} / {len(score_psi)}",
          help="Precision at capacity is bounded above by fraud count / queue "
               "size. Sitting on that bound means the ranking is perfect at this "
               "capacity.")
h3.metric("Precision at capacity (mean)",
          f"{full_windows['precision_at_capacity'].mean():.3f}",
          help="The bundle ships expected_precision 0.5257, measured on fold 3 "
               "alone.")

st.markdown(
    f"""
Recall at capacity is **{score_psi['recall_at_capacity'].min():.3f} in every
window** and precision at capacity sits **exactly on its own ceiling in
{int(score_psi['at_ceiling'].sum())} of {len(score_psi)}**. Every fraud lands
inside the top-{int(score_psi['capacity'].max())} queue, every day, across windows
whose volume differs by more than 500x. The bundle's shipped
`expected_precision` of 0.5257 — measured on fold 3 alone — sits inside the
0.432–0.640 range these windows produce.

This is also why **precision at capacity is not itself a retraining trigger**. A
model on the ceiling reports `fraud_count / queue_size`, so the number tracks how
much fraud occurred that day rather than how well it was ranked. What matters is
whether the model *stays* on the ceiling. `MONITORING.md` §4 sets the criteria
out.
"""
)

st.dataframe(
    score_psi[["window", "step_start", "step_end", "n_rows", "n_fraud",
               "score_psi", "score_band", "features_breached", "n_flagged",
               "alerts_per_day", "capacity", "precision_at_capacity",
               "recall_at_capacity", "at_ceiling",
               "precision_at_deployed_threshold"]],
    hide_index=True, width='stretch',
)
st.caption(
    "Precision and recall need labels, which a live system does not have on the "
    "day. They are hindsight measures — which is exactly how the retraining "
    "criteria in MONITORING.md use them."
)

# --------------------------------------------------------------- live telemetry

st.markdown("---")
st.subheader("Live traffic")
st.warning(
    "**Not a drift signal.** This is what the deployed service has actually seen "
    "since its last restart. A portfolio endpoint serves a handful of requests, "
    "and a distribution estimated from a handful of requests is noise. It is here "
    "because volume, latency and flag rate are real operational information — not "
    "because it measures drift.",
    icon="⚠️",
)

if st.button("Read live /metrics"):
    try:
        response = call_api_with_wake_retry("GET", "/metrics")
        response.raise_for_status()
        metrics = response.json()

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Requests", f"{metrics['request_count']:,}")
        m2.metric("Scored", f"{metrics['score_count']:,}")
        m3.metric("Flag rate", f"{metrics['flag_rate']:.1%}")
        m4.metric("p95 latency", f"{metrics['latency_p95_ms']:.1f} ms")

        histogram = pd.DataFrame(
            {"bucket": list(metrics["score_histogram"]),
             "count": list(metrics["score_histogram"].values())}
        )
        if histogram["count"].sum():
            st.plotly_chart(
                px.bar(histogram, x="bucket", y="count",
                       labels={"bucket": "Score", "count": "Requests"}),
                width='stretch',
            )
        else:
            st.caption("No transactions scored on this instance yet.")
        st.caption(
            "Per-instance and in-memory: Cloud Run scales to zero, so these reset "
            "on every cold start and do not aggregate across instances."
        )
    except requests.RequestException as exc:
        st.error(f"Could not reach the scoring service: {exc}")

# ----------------------------------------------------------------- what it means

st.markdown("---")
st.subheader("What this actually found")
st.markdown(
    f"""
**The detector's noise floor is real and low.** Windows 5–13 — nine consecutive
high-volume windows inside the reference period — score PSI 0.0005–0.0048 with no
feature breaching at all. Those windows are drawn from the data the reference was
built from, so a number near zero is the correct answer, and getting it is what
makes the non-zero numbers elsewhere worth reading. The two remaining high-volume
reference windows, 0 and 1, breach only the two velocity features and only just
(0.29–0.43 against a 0.25 band): they are the busiest windows in the dataset, so
destination velocity runs about three times the reference mean.

That floor is measured *in sample* — the reference scores come from scoring the
training fold with a model fitted on it. Windows 14–16, high-volume and just past
the reference cut, put the size of that bias at 2.8x (mean 0.0075 against
0.0027). It changes nothing below, and it is the honest number to quote.

**The features that drift are the two velocity features, and volume is why.**
`dest_txn_count_24h` and `dest_amount_sum_24h` count what reached a destination
account in the previous 24 simulated hours, so they are functions of transaction
volume by construction. PaySim's volume spans 1,070 to 574,255 rows across these
windows, and the share of rows with no 24-hour destination history runs from
48.6% in the busiest window to 99.0% in a collapsed one, against a reference
share of 58.7%. The features move because the simulation's throughput does.

**`hour_of_day` drifts for the same underlying reason.** In the reference period
the busiest six hours hold 49.8% of rows; in a collapsed window they hold 91.3%.
Activity concentrates into fewer hours when there is less of it.

**The score distribution mostly does not follow.** In every one of those
low-volume windows the score PSI stays under {SCORE_THRESHOLD:.2f} — three to
five features breach while the model's output does not. Feature drift and score
drift are different questions, and this dataset separates them cleanly.

**The three genuine score breaches are a PaySim artifact, not degradation.**
Windows 2–4 (steps 49–120) are the simulation's early collapse, where one window
is 1,070 rows of which 310 are fraud — a 29.0% fraud rate against 0.047–0.068%
in the high-volume windows around it. PaySim has stretches where little but the fraud
agent is active. That is a property of the generator, and it sits *inside* the
training reference, which is the honest thing to notice about it.

The retraining criteria these series feed, and the champion/challenger promotion
path, are in `MONITORING.md`.
"""
)

st.caption(
    f"Generated {reference['generated_at'][:10]} from commit "
    f"`{reference['git_commit']}` · `python tasks.py drift`"
)
