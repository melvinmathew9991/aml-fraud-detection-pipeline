"""
2_Batch_Upload.py

CSV in, scored queue out (ARCHITECTURE.md §6, page 2). Falls back to the
bundled 50k-row stratified sample when nothing is uploaded, so the page
works with zero upload. Chunks requests to the API's own 10,000-row cap
(api/schemas.py's BATCH_MAX_ROWS) -- within-batch destination-state
accumulation (ARCHITECTURE.md §5) only applies within a single chunk, a
limitation stated on the page rather than left implicit.
"""

import io

import pandas as pd
import requests
import streamlit as st
from common import (
    RAW_TRANSACTION_FIELDS,
    call_api_with_wake_retry,
    load_sample_transactions,
    render_sidebar_api_status,
)

st.set_page_config(page_title="Batch upload", page_icon="🔍", layout="wide")
st.title("Batch upload")
render_sidebar_api_status()

API_BATCH_MAX_ROWS = 10_000

st.caption(
    "Upload a CSV of raw transactions (columns: "
    f"`{', '.join(RAW_TRANSACTION_FIELDS)}`), or use the bundled sample below. "
    "Requests larger than 10,000 rows are chunked -- the API's own cap "
    "(api/schemas.py) -- so destination-state accumulation "
    "(ARCHITECTURE.md §5) only carries across rows within the same chunk, "
    "not across the whole upload."
)

uploaded = st.file_uploader("Upload CSV", type="csv")

if uploaded is not None:
    df = pd.read_csv(uploaded)
    missing = set(RAW_TRANSACTION_FIELDS) - set(df.columns)
    if missing:
        st.error(f"Uploaded CSV is missing required column(s): {sorted(missing)}")
        st.stop()
    source_label = f"uploaded file ({len(df)} rows)"
else:
    sample = load_sample_transactions()
    n_rows = st.slider("Rows to score from the bundled sample", min_value=100,
                       max_value=len(sample), value=1000, step=100)
    df = sample.head(n_rows).copy()
    source_label = f"bundled sample, first {n_rows} of {len(sample)} rows"

st.write(f"Scoring: **{source_label}**")
has_ground_truth = "isFraud" in df.columns

if st.button("Score this batch", type="primary"):
    request_rows = df[RAW_TRANSACTION_FIELDS].to_dict(orient="records")
    chunks = [request_rows[i:i + API_BATCH_MAX_ROWS]
             for i in range(0, len(request_rows), API_BATCH_MAX_ROWS)]

    progress = st.progress(0.0, text=f"Scoring {len(request_rows)} rows in {len(chunks)} request(s)...")
    all_results = []
    summary_totals = {"n_scored": 0, "n_blocked": 0, "n_flagged": 0, "n_passed": 0}

    for i, chunk in enumerate(chunks):
        try:
            response = call_api_with_wake_retry(
                "POST", "/score/batch", json={"transactions": chunk})
        except requests.RequestException as exc:
            st.error(f"Could not reach the scoring API on chunk {i + 1}/{len(chunks)}: {exc}")
            st.stop()

        if response.status_code != 200:
            detail = response.json().get("detail", response.text) if response.content else response.text
            st.error(f"API returned {response.status_code} on chunk {i + 1}/{len(chunks)}: {detail}")
            st.stop()

        body = response.json()
        all_results.extend(body["results"])
        for key in summary_totals:
            summary_totals[key] += body["summary"][key]
        progress.progress((i + 1) / len(chunks),
                          text=f"Scored {min((i + 1) * API_BATCH_MAX_ROWS, len(request_rows))}"
                               f"/{len(request_rows)} rows...")

    progress.empty()

    results_df = pd.DataFrame(all_results)
    scored = pd.concat([df.reset_index(drop=True),
                        results_df[["probability", "flagged", "decision", "rule", "state_hit"]]],
                       axis=1)

    st.markdown("---")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Scored", summary_totals["n_scored"])
    m2.metric("Blocked", summary_totals["n_blocked"])
    m3.metric("Flagged for review", summary_totals["n_flagged"])
    m4.metric("Passed", summary_totals["n_passed"])

    if has_ground_truth:
        reviewed = scored[scored["decision"].isin(["BLOCK", "REVIEW"])]
        caught = int((reviewed["isFraud"] == 1).sum())
        total_fraud = int(scored["isFraud"].sum())
        st.caption(
            f"Ground truth available in this data (demo only -- a real upload "
            f"wouldn't have it): {caught} of {total_fraud} actual fraud "
            f"transactions ended up BLOCK/REVIEW."
        )

    st.subheader("Scored queue")
    st.dataframe(scored, hide_index=True, width='stretch')

    csv_buffer = io.StringIO()
    scored.to_csv(csv_buffer, index=False)
    st.download_button("Download scored results (CSV)", csv_buffer.getvalue(),
                       file_name="scored_transactions.csv", mime="text/csv")
