"""
1_Score_a_Transaction.py

The analyst's page: enter one transaction, get a score, a decision, and
the reason codes that drove it (ARCHITECTURE.md §6, page 1). Talks to the
live API's POST /score -- no model or feature logic here, the API owns
that entirely (ARCHITECTURE.md §4).
"""

import pandas as pd
import streamlit as st
from common import (
    RAW_TRANSACTION_FIELDS,
    call_api_with_wake_retry,
    render_sidebar_api_status,
)

st.set_page_config(page_title="Score a transaction", page_icon="🔍", layout="wide")
st.title("Score a transaction")
render_sidebar_api_status()

st.caption(
    "Enter a transaction and send it to the live scoring API. The response "
    "echoes the exact decision threshold it was judged against and whether "
    "the destination's history came from the bundled snapshot or a cold-start "
    "default -- both are what ARCHITECTURE.md §5 requires every /score "
    "response to carry."
)

TYPE_OPTIONS = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]

with st.form("score_form"):
    c1, c2, c3 = st.columns(3)
    with c1:
        step = st.number_input("step (simulated hour)", min_value=0, value=743, step=1)
        txn_type = st.selectbox("type", TYPE_OPTIONS, index=TYPE_OPTIONS.index("TRANSFER"))
        amount = st.number_input("amount", min_value=0.0, value=250000.0, step=1000.0)
    with c2:
        name_orig = st.text_input("nameOrig", value="C1231006815")
        old_balance_org = st.number_input("oldbalanceOrg", min_value=0.0, value=250000.0)
        new_balance_orig = st.number_input("newbalanceOrig", min_value=0.0, value=0.0)
    with c3:
        name_dest = st.text_input("nameDest", value="C1979787155")
        old_balance_dest = st.number_input("oldbalanceDest", min_value=0.0, value=0.0)
        new_balance_dest = st.number_input("newbalanceDest", min_value=0.0, value=250000.0)

    submitted = st.form_submit_button("Score transaction", type="primary")

if submitted:
    payload = {
        "step": int(step), "type": txn_type, "amount": float(amount),
        "nameOrig": name_orig, "oldbalanceOrg": float(old_balance_org),
        "newbalanceOrig": float(new_balance_orig), "nameDest": name_dest,
        "oldbalanceDest": float(old_balance_dest), "newbalanceDest": float(new_balance_dest),
    }
    assert set(payload.keys()) == set(RAW_TRANSACTION_FIELDS)

    try:
        response = call_api_with_wake_retry("POST", "/score", json=payload)
    except Exception as exc:  # noqa: BLE001 -- degrade to an error message, never crash the page
        st.error(f"Could not reach the scoring API: {exc}")
        st.stop()

    if response.status_code == 503:
        st.error("Service not ready yet — the bundle may still be loading. Try again shortly.")
        st.stop()
    elif response.status_code != 200:
        detail = response.json().get("detail", response.text) if response.content else response.text
        st.error(f"API returned {response.status_code}: {detail}")
        st.stop()

    body = response.json()
    decision = body["decision"]
    decision_color = {"BLOCK": "🔴", "REVIEW": "🟡", "PASS": "🟢"}[decision]

    st.markdown("---")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Decision", f"{decision_color} {decision}")
    m2.metric("Score", f"{body['probability']:.6f}")
    m3.metric("Threshold", f"{body['decision_threshold']:.2e}")
    m4.metric("Destination history", "Snapshot hit" if body["state_hit"] else "Cold start")

    if body["rule"]:
        st.warning(f"Hard-block rule fired: `{body['rule']}` — the model score below is "
                   "informational; the block decision did not wait for it.", icon="🚫")

    st.caption(f"model: {body['model_version']}  ·  bundle: {body['bundle_version']}  ·  "
              f"request: `{body['request_id']}`  ·  {body['latency_ms']:.1f}ms")

    st.subheader("Why")
    reasons_df = pd.DataFrame(body["reasons"])
    reasons_df = reasons_df.rename(columns={
        "feature": "Feature", "contribution": "SHAP contribution", "value": "Raw value",
    })
    st.dataframe(reasons_df, hide_index=True, width='stretch')
