"""
3_Capacity_Economics_Explorer.py

The most important page in the project (ARCHITECTURE.md §6, page 3): the
Sprint 2 capacity sweep made interactive, with the §0 net-value curve
layered on top via three adjustable business rates. Computed entirely
client-side from the committed capacity_sweep.csv via common.py's
compute_net_value_curve (which wraps src/economics.py) -- no API call, so
this page loads instantly and works while the scoring API is asleep.

reviews_per_day is a SELECT over the exact rows capacity_sweep.csv
already contains, not a continuous slider -- Sprint 5's DoD requires this
page reproduce that CSV exactly at 500/day, which a client-side
interpolation between sweep points could not guarantee.
"""

import plotly.graph_objects as go
import streamlit as st

from common import AVG_FRAUD_AMOUNT, SENSITIVITY_RECOVERY_RATE_GRID, capacity_sweep_rows, compute_net_value_curve

st.set_page_config(page_title="Capacity & economics explorer", page_icon="🔍", layout="wide")
st.title("Capacity & economics explorer")
st.caption(
    "Static page -- everything below is computed from `data/processed/capacity_sweep.csv`, "
    "committed to the repo. No API call."
)

sweep_rows = capacity_sweep_rows()
sweep_by_rpd = {row["reviews_per_day"]: row for row in sweep_rows}
rpd_options = sorted(sweep_by_rpd.keys())

st.subheader("1. Operating point vs. review capacity")
selected_rpd = st.select_slider("Reviews per day", options=rpd_options, value=500)
row = sweep_by_rpd[selected_rpd]

m1, m2, m3, m4 = st.columns(4)
m1.metric("Precision", f"{row['precision']:.1%}")
m2.metric("Recall", f"{row['recall']:.1%}")
m3.metric("False positives", f"{row['false_positives']:,}")
m4.metric("Precision ceiling", f"{row['precision_ceiling']:.1%}",
         help="total_fraud / n_flagged -- once recall saturates, this is the arithmetic "
              "maximum precision can reach at this capacity. No model can exceed it.")

fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=rpd_options, y=[sweep_by_rpd[r]["precision"] for r in rpd_options],
                          name="Precision", mode="lines+markers"))
fig1.add_trace(go.Scatter(x=rpd_options, y=[sweep_by_rpd[r]["recall"] for r in rpd_options],
                          name="Recall", mode="lines+markers"))
fig1.add_trace(go.Scatter(x=rpd_options, y=[sweep_by_rpd[r]["precision_ceiling"] for r in rpd_options],
                          name="Precision ceiling", mode="lines", line=dict(dash="dot")))
fig1.add_vline(x=selected_rpd, line_dash="dash", line_color="gray")
fig1.update_layout(xaxis_title="Reviews per day", yaxis_title="Rate", xaxis_type="log",
                   legend=dict(orientation="h", y=1.1), height=400)
st.plotly_chart(fig1, width='stretch')

st.caption(
    f"At {selected_rpd}/day: {row['true_positives']:,} fraud caught, "
    f"{row['false_positives']:,} false positives, {row['false_negatives']:,} missed, "
    f"out of {row['total_fraud']:,} total fraud in this window."
)

st.markdown("---")
st.subheader("2. What is it worth? Net value under adjustable business rates")
st.caption(
    "net_value(K) = frauds_caught × avg_fraud_amount × recovery_rate "
    "− alerts_reviewed × cost_per_review − frauds_missed × avg_fraud_amount × liability_rate  "
    f"(avg_fraud_amount fixed at the measured final-fold value, {AVG_FRAUD_AMOUNT:,})."
)

c1, c2, c3 = st.columns(3)
# Upper bound deliberately reaches past the README-documented break-even
# range (84,942-161,795) rather than stopping at a "realistic" review cost
# -- dragging past it is how a user discovers that finding themselves
# instead of just reading it as an assertion.
cost_per_review = c1.slider("Cost per review", min_value=10, max_value=200_000, value=200, step=100)
recovery_rate = c2.slider("Recovery rate on caught fraud", min_value=0.0, max_value=1.0,
                          value=0.5, step=0.05)
liability_rate = c3.slider("Liability rate on missed fraud", min_value=0.0, max_value=1.0,
                           value=1.0, step=0.05)

primary_curve = compute_net_value_curve(cost_per_review, recovery_rate, liability_rate)
optimum = primary_curve.loc[primary_curve["net_value"].idxmax()]

band_curves = {
    rr: compute_net_value_curve(cost_per_review, rr, liability_rate)
    for rr in SENSITIVITY_RECOVERY_RATE_GRID
}

fig2 = go.Figure()
low_curve = band_curves[min(SENSITIVITY_RECOVERY_RATE_GRID)]
high_curve = band_curves[max(SENSITIVITY_RECOVERY_RATE_GRID)]
fig2.add_trace(go.Scatter(
    x=list(low_curve["reviews_per_day"]) + list(high_curve["reviews_per_day"])[::-1],
    y=list(low_curve["net_value"]) + list(high_curve["net_value"])[::-1],
    fill="toself", fillcolor="rgba(100,100,200,0.15)", line=dict(width=0),
    name=f"Sensitivity band (recovery_rate {min(SENSITIVITY_RECOVERY_RATE_GRID)}–"
        f"{max(SENSITIVITY_RECOVERY_RATE_GRID)})",
    hoverinfo="skip",
))
fig2.add_trace(go.Scatter(x=primary_curve["reviews_per_day"], y=primary_curve["net_value"],
                          name="Net value (your settings)", mode="lines+markers",
                          line=dict(color="firebrick", width=3)))
fig2.add_trace(go.Scatter(x=[optimum["reviews_per_day"]], y=[optimum["net_value"]],
                          name="Optimum", mode="markers",
                          marker=dict(size=14, symbol="star", color="gold",
                                     line=dict(width=1, color="black"))))
fig2.update_layout(xaxis_title="Reviews per day", yaxis_title="Net value", xaxis_type="log",
                   legend=dict(orientation="h", y=1.15), height=420)
st.plotly_chart(fig2, width='stretch')

# The real degeneracy question (README "The naive 'find the optimum' version
# is degenerate") is NOT "does the optimum sit at the highest capacity
# swept" -- recall is already 1.0 by 500/day, so every point past it can
# only add cost with zero benefit and the optimum can never legitimately
# exceed the recall-saturation point. The actual question is whether
# staffing UP TO that saturation point wins under every plausible business
# rate, i.e. whether the optimum reaches it even at the pessimistic
# (low-recovery-rate) end of the sensitivity band.
saturation_rpd = min(r for r in rpd_options if sweep_by_rpd[r]["recall"] >= 0.9999)
band_optima = {rr: curve.loc[curve["net_value"].idxmax(), "reviews_per_day"]
              for rr, curve in band_curves.items()}
always_reaches_saturation = all(opt >= saturation_rpd for opt in band_optima.values())

if always_reaches_saturation:
    st.warning(
        f"**Staffing up to {saturation_rpd:,}/day — the point where recall saturates — wins under "
        f"every recovery-rate assumption in the sensitivity band "
        f"({min(SENSITIVITY_RECOVERY_RATE_GRID)}–{max(SENSITIVITY_RECOVERY_RATE_GRID)}), at these "
        "cost/liability settings.** This is the project's headline economics finding: at PaySim's "
        "real average fraud amount, full recall wins under every plausible assumption, so the naive "
        "\"find the optimum\" framing never actually trades recall for cost. `src/economics.py`'s "
        "`degeneracy_check` names this explicitly rather than reporting an optimum that means "
        "nothing. Raise the cost-per-review slider toward the 85,000–162,000 break-even range "
        "(README \"Results\") to see this stop holding, or see `capacity_economics.csv`'s "
        "ticket-size crossover for where a smaller average fraud amount changes the answer.",
        icon="⚠️",
    )
else:
    st.success(
        f"At these settings, the optimum does **not** reach full recall "
        f"({saturation_rpd:,}/day) across the whole sensitivity band — recovery rate genuinely "
        "changes the staffing answer here. That's a real tradeoff, not a degenerate one.",
        icon="✅",
    )
