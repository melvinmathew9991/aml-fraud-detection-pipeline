"""
Home.py

Landing page. Renders entirely from committed CSVs -- ARCHITECTURE.md §6's
cold-start rule: Streamlit Community Cloud sleeps after 12h idle (~30s
wake) and Cloud Run scales to zero (~3-5s cold start); chained, a first
API call on page load could leave a first-time visitor staring at nothing
for ~35s. No `requests` call happens anywhere in this file.
"""

import streamlit as st
from common import load_csv

st.set_page_config(page_title="AML Fraud Detection", page_icon="🔍", layout="wide")

st.title("AML Fraud Detection — Dashboard")
st.caption(
    "PaySim mobile-money fraud detection: a cost-sensitive model, a capacity-based "
    "operating point, and a scoring API, walked through from five angles. "
    "This page and pages 3-5 render entirely from files committed to the repo; "
    "only pages 1-2 call the live scoring API."
)

comparison = load_csv("model_comparison")
sweep = load_csv("capacity_sweep")
best = comparison.sort_values("capacity_precision_mean", ascending=False).iloc[0]
row_500 = sweep[sweep["reviews_per_day"] == 500].iloc[0]

col1, col2, col3, col4 = st.columns(4)
col1.metric("Selected model", best["model"])
col2.metric("PR-AUC (mean)", f"{best['pr_auc_mean']:.4f}")
col3.metric("Recall @ 500 reviews/day", f"{row_500['recall']:.1%}")
col4.metric("Precision @ 500 reviews/day", f"{row_500['precision']:.1%}")

st.markdown("---")

st.markdown(
    """
### Pages

1. **Score a transaction** — enter one transaction, get a score, a flag/pass
   decision, and the features that drove it. Calls the live API.
2. **Batch upload** — upload a CSV (or use the bundled 50k-row sample) and
   get a scored queue back, downloadable. Calls the live API.
3. **Capacity & economics explorer** — the project's headline finding, made
   interactive: move the review-queue size and three business rates, watch
   precision/recall and net value respond. Static — works with the API
   asleep.
4. **Model card** — features, SHAP global importance, ablation, per-fold
   metrics, and this model's known limitations. Static.
5. **Drift** — placeholder; live PSI monitoring ships in Sprint 8.

Pages 1-2 need the scoring API running (`uvicorn api.main:app --app-dir src`
locally, or `API_BASE_URL` pointed at a deployed instance). If it's asleep,
the first request wakes it — that can take up to a minute on Cloud Run's
free tier, and the page says so rather than looking stuck.
"""
)

st.info(
    "Every number on this page and pages 3-5 comes from a file already committed "
    "to the repo (`data/processed/*.csv`) — nothing here was computed for this "
    "page load.",
    icon="📄",
)
