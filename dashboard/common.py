"""
common.py

Shared helpers for every dashboard page (ARCHITECTURE.md §6). Streamlit
puts this file's directory (dashboard/) on sys.path automatically when
`streamlit run dashboard/Home.py` starts, so pages/*.py can `import common`
without any path setup of their own.

Contains no model or feature-engineering logic -- the dashboard talks to
the API over HTTP and reads precomputed CSVs, exactly the constraint
ARCHITECTURE.md §6 states ("contains no model and no feature logic").

net_value_curve is imported from src/economics.py rather than
reimplemented here: that module has zero heavy dependencies (numpy/pandas
only, both already in dashboard/requirements.txt), and importing it is
what keeps the capacity explorer's math and the training pipeline's own
economics analysis from silently drifting apart -- the same reasoning
inference/features.py uses to import FEATURE_COLUMNS from training's
features.py instead of hand-copying it.
"""

import os
import sys
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

DASHBOARD_DIR = Path(__file__).resolve().parent
REPO_ROOT = DASHBOARD_DIR.parent
DATA_PROCESSED = REPO_ROOT / "data" / "processed"
SAMPLE_CSV = DASHBOARD_DIR / "sample_transactions.csv"

SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
from economics import net_value_curve  # noqa: E402

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000").rstrip("/")
API_TIMEOUT_SECONDS = 5
API_COLD_START_TIMEOUT_SECONDS = 60  # Cloud Run cold start, ARCHITECTURE.md §6

# The final fold's measured average fraud amount (README.md "Results" --
# "against the real final-fold average fraud amount (1,572,443, measured,
# not assumed)"). Fixed rather than adjustable: the ticket-size crossover
# question this number feeds is already answered by
# capacity_economics.csv / src/economics.py's ticket_size_crossover, and
# ARCHITECTURE.md §6 lists exactly three adjustable rates for this page
# (cost per review, recovery rate, liability rate) -- not four.
AVG_FRAUD_AMOUNT = 1_572_443

# Same grid src/economics.py's degeneracy_check sweeps, reused here to draw
# the sensitivity band around the primary net-value curve.
SENSITIVITY_RECOVERY_RATE_GRID = [0.05, 1.0]

RAW_TRANSACTION_FIELDS = [
    "step", "type", "amount", "nameOrig", "oldbalanceOrg", "newbalanceOrig",
    "nameDest", "oldbalanceDest", "newbalanceDest",
]


@st.cache_data
def load_csv(name: str) -> pd.DataFrame:
    """Loads a committed data/processed/<name>.csv. Cached per Streamlit
    session -- these are small, static, read-only artifacts."""
    return pd.read_csv(DATA_PROCESSED / f"{name}.csv")


def capacity_sweep_rows() -> list[dict]:
    """capacity_sweep.csv as the list-of-dicts shape src/economics.py's
    functions expect (the same shape src/threshold.py's capacity_sweep()
    returns at training time) -- kept as a plain function, not inlined in
    the page script, so it's unit-testable without Streamlit's AppTest
    machinery (see tests/test_dashboard_common.py)."""
    return load_csv("capacity_sweep").to_dict(orient="records")


def compute_net_value_curve(cost_per_review: float, recovery_rate: float,
                            liability_rate: float, avg_fraud_amount: float = AVG_FRAUD_AMOUNT):
    """The capacity explorer's net-value curve for one set of business
    rates -- a thin wrapper over src/economics.py's net_value_curve so the
    page script and any test can share the exact same call."""
    return net_value_curve(capacity_sweep_rows(), avg_fraud_amount,
                           cost_per_review, recovery_rate, liability_rate)


@st.cache_data
def load_sample_transactions() -> pd.DataFrame:
    return pd.read_csv(SAMPLE_CSV)


def api_url(path: str) -> str:
    return f"{API_BASE_URL}{path}"


def check_api_ready() -> tuple[bool, str]:
    """Quick, short-timeout /ready probe. Returns (ready, detail) -- never
    raises, since every caller needs to degrade gracefully rather than
    crash the page when the API is asleep or unreachable."""
    try:
        r = requests.get(api_url("/ready"), timeout=API_TIMEOUT_SECONDS)
        body = r.json()
        return body.get("status") == "ready", body.get("detail") or ""
    except requests.RequestException as exc:
        return False, str(exc)


def call_api_with_wake_retry(method: str, path: str, **kwargs):
    """
    POSTs/GETs against the API with a short timeout first; on timeout or
    connection failure, shows an honest "waking the scoring service"
    spinner and retries once with a long timeout -- the graceful-
    degradation behavior ARCHITECTURE.md §6 requires for Cloud Run's
    ~3-5s cold start (and Streamlit's own ~30s wake, which has already
    happened by the time this code runs).

    Raises requests.RequestException if the retry also fails; callers
    show that as an error rather than letting the page crash.
    """
    try:
        return requests.request(method, api_url(path), timeout=API_TIMEOUT_SECONDS, **kwargs)
    except requests.RequestException:
        with st.spinner("Waking the scoring service (cold start can take up to a minute)..."):
            return requests.request(method, api_url(path),
                                     timeout=API_COLD_START_TIMEOUT_SECONDS, **kwargs)


def render_sidebar_api_status() -> None:
    """A small, non-blocking API status indicator -- checked lazily (only
    when a page that needs the API renders it), never on the landing page
    (ARCHITECTURE.md §6's cold-start rule)."""
    with st.sidebar:
        st.caption(f"API: `{API_BASE_URL}`")
        ready, detail = check_api_ready()
        if ready:
            st.success("Scoring service ready", icon="✅")
        else:
            st.warning("Scoring service asleep or unreachable — "
                       "the first request below may take a moment.", icon="⏳")
            if detail:
                st.caption(detail)
