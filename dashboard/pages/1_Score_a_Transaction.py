"""
1_Score_a_Transaction.py

The analyst's page: enter one transaction, get a score, a decision, and
the reason codes that drove it (ARCHITECTURE.md §6, page 1). Talks to the
live API's POST /score -- no model or feature logic here, the API owns
that entirely (ARCHITECTURE.md §4).

Written for two readers at once. The default view is for someone who has
never seen PaySim: one-click examples, plain-English field labels, and a
verdict in words. Everything a technical reader needs -- the raw score, the
threshold it was judged against, SHAP contributions, the snapshot-hit flag,
the request id -- is still on the page, moved into expanders rather than
removed. Neither audience is served by making the other scroll past noise.
"""

import pandas as pd
import streamlit as st
from common import (
    DECISION_PLAIN,
    EXAMPLE_TRANSACTIONS,
    FIELD_HELP,
    RAW_TRANSACTION_FIELDS,
    call_api_with_wake_retry,
    feature_label,
    render_sidebar_api_status,
)

st.set_page_config(page_title="Score a transaction", page_icon="🔍", layout="wide")
st.title("Score a transaction")
render_sidebar_api_status()

st.write(
    "Send one payment to the live fraud model and see what it decides — and why. "
    "If you don't know what to type, load one of the examples below."
)

# Every input is keyed on `f_<api field name>`, and the preset buttons write
# straight into those keys. This is the only arrangement that works: a keyed
# Streamlit widget takes its value from session state and ignores `value=` once
# the key exists, so presets cannot be implemented by re-passing `value=`. The
# `on_click` callback runs before the rerun, so the widgets are constructed
# from the new state rather than the old.
FIELD_KEY = "f_{}".format


def _load_example(name: str) -> None:
    for field, value in EXAMPLE_TRANSACTIONS[name].items():
        st.session_state[FIELD_KEY(field)] = value


for _field, _value in EXAMPLE_TRANSACTIONS["fraud"].items():
    st.session_state.setdefault(FIELD_KEY(_field), _value)

b1, b2, _ = st.columns([1, 1, 2])
b1.button("Load a suspicious example", width='stretch',
          on_click=_load_example, args=("fraud",))
b2.button("Load an ordinary example", width='stretch',
          on_click=_load_example, args=("legitimate",))

TYPE_OPTIONS = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]

with st.form("score_form"):
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**The payment**")
        st.number_input("Amount sent", min_value=0.0, step=1000.0,
                        key=FIELD_KEY("amount"), help=FIELD_HELP["amount"])
        st.selectbox("Kind of payment", TYPE_OPTIONS,
                     key=FIELD_KEY("type"), help=FIELD_HELP["type"])
        st.number_input("Hour of the simulation", min_value=0, step=1,
                        key=FIELD_KEY("step"), help=FIELD_HELP["step"])
    with c2:
        st.markdown("**Who sent it**")
        st.text_input("Sender ID", key=FIELD_KEY("nameOrig"),
                      help=FIELD_HELP["nameOrig"])
        st.number_input("Balance before", min_value=0.0,
                        key=FIELD_KEY("oldbalanceOrg"), help=FIELD_HELP["oldbalanceOrg"])
        st.number_input("Balance after", min_value=0.0,
                        key=FIELD_KEY("newbalanceOrig"), help=FIELD_HELP["newbalanceOrig"])
    with c3:
        st.markdown("**Who received it**")
        st.text_input("Receiver ID", key=FIELD_KEY("nameDest"),
                      help=FIELD_HELP["nameDest"])
        st.number_input("Balance before", min_value=0.0,
                        key=FIELD_KEY("oldbalanceDest"), help=FIELD_HELP["oldbalanceDest"])
        st.number_input("Balance after", min_value=0.0,
                        key=FIELD_KEY("newbalanceDest"), help=FIELD_HELP["newbalanceDest"])

    submitted = st.form_submit_button("Check this payment", type="primary")

if submitted:
    payload = {
        "step": int(st.session_state[FIELD_KEY("step")]),
        "type": st.session_state[FIELD_KEY("type")],
        "amount": float(st.session_state[FIELD_KEY("amount")]),
        "nameOrig": st.session_state[FIELD_KEY("nameOrig")],
        "oldbalanceOrg": float(st.session_state[FIELD_KEY("oldbalanceOrg")]),
        "newbalanceOrig": float(st.session_state[FIELD_KEY("newbalanceOrig")]),
        "nameDest": st.session_state[FIELD_KEY("nameDest")],
        "oldbalanceDest": float(st.session_state[FIELD_KEY("oldbalanceDest")]),
        "newbalanceDest": float(st.session_state[FIELD_KEY("newbalanceDest")]),
    }
    assert set(payload.keys()) == set(RAW_TRANSACTION_FIELDS)

    with st.spinner("Asking the model… the first request of the day can take a few "
                    "seconds while the service wakes up."):
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
    icon, headline, explanation = DECISION_PLAIN[decision]

    st.markdown("---")
    st.subheader(f"{icon} {headline}")
    st.write(explanation)

    if body["rule"]:
        st.warning(f"Hard-block rule fired: `{body['rule']}` — the model score is "
                   "informational here; the block decision did not wait for it.", icon="🚫")

    # The model's own reasons, as sentences. A SHAP contribution is a
    # log-odds push, and its magnitude is not something to hand a
    # non-technical reader as a number -- but the ORDER is exactly what they
    # want: what mattered most about this payment. So rank, and describe the
    # direction, and keep the figures one expander away.
    st.markdown("**What stood out about this payment**")
    reasons = body["reasons"]
    if reasons:
        for r in reasons:
            direction = "raised" if r["contribution"] > 0 else "lowered"
            st.markdown(f"- **{feature_label(r['feature'])}** — {direction} the suspicion")
    else:
        st.write("Nothing unusual enough to single out.")

    st.caption(
        f"Decided in {body['latency_ms']:.0f} milliseconds by {body['model_version']}."
    )

    with st.expander("The numbers behind this"):
        m1, m2, m3 = st.columns(3)
        m1.metric("Decision", decision)
        m2.metric("Fraud probability", f"{body['probability']:.6f}",
                  help="The model's raw output for this transaction.")
        m3.metric("Decision threshold", f"{body['decision_threshold']:.2e}",
                  help="Chosen by review capacity, not by maximising a metric — "
                       "see the capacity & economics page.")

        st.markdown("**SHAP contributions** — each feature's push on the log-odds, "
                    "signed and ranked. This is the model's own attribution, not a "
                    "post-hoc explanation.")
        reasons_df = pd.DataFrame(reasons)
        if not reasons_df.empty:
            reasons_df.insert(0, "In plain English", reasons_df["feature"].map(feature_label))
            reasons_df = reasons_df.rename(columns={
                "feature": "Feature", "contribution": "SHAP contribution",
                "value": "Raw value",
            })
        st.dataframe(reasons_df, hide_index=True, width='stretch')

        st.caption(
            f"model: {body['model_version']}  ·  bundle: {body['bundle_version']}  ·  "
            f"request: `{body['request_id']}`  ·  "
            f"destination history: {'snapshot hit' if body['state_hit'] else 'cold start'}"
        )
        st.caption(
            "The echoed threshold and the snapshot/cold-start flag are what "
            "ARCHITECTURE.md §5 requires every /score response to carry, so a "
            "decision can be reconstructed after the fact."
        )
