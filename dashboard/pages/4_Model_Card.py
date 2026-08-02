"""
4_Model_Card.py

Governance page (ARCHITECTURE.md §6, page 4): features, SHAP global
importance, ablation, per-fold metrics, known limitations, snapshot
as-of. Static -- everything here comes from committed CSVs plus
src/model_card.py's shared facts, no API call.
"""

import plotly.express as px
import streamlit as st
from common import (
    CURRENT_BUNDLE_SNAPSHOT_STEP,
    FEATURE_COLUMNS,
    MODEL_LIMITATIONS,
    load_csv,
)

st.set_page_config(page_title="Model card", page_icon="🔍", layout="wide")
st.title("Model card")
st.caption("Static page — everything below is read from committed CSVs. No API call.")

comparison = load_csv("model_comparison")
best = comparison.sort_values("capacity_precision_mean", ascending=False).iloc[0]

st.subheader("Summary")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Model", best["model"])
c2.metric("PR-AUC (mean ± std)", f"{best['pr_auc_mean']:.4f} ± {best['pr_auc_std']:.4f}")
c3.metric("Feature count", len(FEATURE_COLUMNS))
c4.metric("Snapshot as-of step", CURRENT_BUNDLE_SNAPSHOT_STEP,
         help="The destination-state snapshot (ARCHITECTURE.md §2) is frozen at this "
              "simulated hour -- the end of the training dataset.")

st.markdown("---")
st.subheader("Features")
st.caption(f"All {len(FEATURE_COLUMNS)} features fed to the model, in the exact order "
          "the bundle's scaler expects (src/features.py:FEATURE_COLUMNS).")
st.code(", ".join(FEATURE_COLUMNS), language=None)

st.markdown("---")
st.subheader("SHAP global importance")
shap_df = load_csv("shap_global_importance").sort_values("mean_abs_shap", ascending=True)
fig = px.bar(shap_df, x="mean_abs_shap", y="feature", orientation="h",
            color="mean_signed_shap", color_continuous_scale="RdBu_r",
            color_continuous_midpoint=0,
            labels={"mean_abs_shap": "Mean |SHAP|", "feature": "",
                    "mean_signed_shap": "Mean signed SHAP"})
fig.update_layout(height=500, coloraxis_colorbar_title="Direction")
st.plotly_chart(fig, width='stretch')
st.caption("Color shows average direction: red pushes risk up, blue pushes it down.")

st.markdown("---")
st.subheader("Feature ablation")
st.caption("Metric movement attributed to each feature group added (LightGBM defaults, "
          "final fold, K=8,083) -- src/train_pipeline.py's ablation arms.")
ablation_df = load_csv("feature_ablation")
st.dataframe(ablation_df, hide_index=True, width='stretch')

st.markdown("---")
st.subheader("Per-fold metrics")
by_fold = load_csv("model_comparison_by_fold")
selected_model = st.selectbox("Model", sorted(by_fold["model"].unique()),
                              index=sorted(by_fold["model"].unique()).index(best["model"]))
st.dataframe(
    by_fold[by_fold["model"] == selected_model][
        ["fold", "pr_auc", "roc_auc", "weighted_bce", "capacity_precision",
         "capacity_recall", "capacity_false_positives", "capacity_false_negatives"]
    ],
    hide_index=True, width='stretch',
)

st.markdown("---")
st.subheader("Known limitations")
for limitation in MODEL_LIMITATIONS:
    st.markdown(f"- {limitation}")
