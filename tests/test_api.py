"""
API contract + integration tests via FastAPI's TestClient (ARCHITECTURE.md
§7's "Contract" and part of "Integration" layers -- the container-up
Docker integration test is deferred to Sprint 6's CI, per ROADMAP.md; this
is everything reachable without a built image).

Each test builds its OWN app via api.main.create_app() rather than
importing the shared module-level `app` -- avoids cross-test contamination
of the in-memory rate limiter and metrics counters that a shared singleton
would cause.
"""

import json
import shutil

import pytest
from fastapi.testclient import TestClient

from api.main import DEFAULT_BUNDLE_DIR, create_app
from features import FEATURE_VERSION

VALID_TXN = {
    "step": 10,
    "type": "TRANSFER",
    "amount": 1000.0,
    "nameOrig": "C_ORIG",
    "oldbalanceOrg": 5000.0,
    "newbalanceOrig": 4000.0,
    "nameDest": "C_DEST",
    "oldbalanceDest": 0.0,
    "newbalanceDest": 1000.0,
}


def _require_real_bundle():
    if not (DEFAULT_BUNDLE_DIR / "bundle_meta.json").exists():
        pytest.skip("model_bundle/v1 not present -- run export_bundle.py first.")


@pytest.fixture
def client():
    _require_real_bundle()
    app = create_app(rate_limit_max_requests=10_000)  # effectively unlimited for these tests
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------- health/ready

def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_ready_ok(client):
    r = client.get("/ready")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ready"
    assert body["bundle_version"] == "v1"


def test_ready_fails_on_corrupted_bundle(tmp_path):
    _require_real_bundle()
    # Copy a real bundle, then corrupt one file so its checksum no longer
    # matches bundle_meta.json -- ROADMAP.md's Sprint 4 DoD: "/ready fails
    # correctly on a corrupted bundle."
    for name in ["bundle_meta.json", "scaler.json", "threshold.json",
                 "model.txt", "dest_state.parquet"]:
        shutil.copy(DEFAULT_BUNDLE_DIR / name, tmp_path / name)
    with open(tmp_path / "threshold.json") as f:
        payload = json.load(f)
    payload["decision_threshold"] = 0.999999
    with open(tmp_path / "threshold.json", "w") as f:
        json.dump(payload, f)

    app = create_app(bundle_dir=tmp_path, rate_limit_max_requests=10_000)
    with TestClient(app) as c:
        r = c.get("/ready")
        assert r.status_code == 503
        assert r.json()["status"] == "not_ready"

        # Every other endpoint that depends on the bundle must also refuse,
        # not silently score against nothing.
        r = c.post("/score", json=VALID_TXN)
        assert r.status_code == 503
        assert r.json()["type"] == "about:blank"


def test_ready_fails_on_missing_bundle_dir(tmp_path):
    app = create_app(bundle_dir=tmp_path / "does_not_exist", rate_limit_max_requests=10_000)
    with TestClient(app) as c:
        r = c.get("/ready")
        assert r.status_code == 503


# --------------------------------------------------------------------- score

def test_score_valid_transaction_returns_full_contract(client):
    r = client.post("/score", json=VALID_TXN)
    assert r.status_code == 200
    body = r.json()
    for field in ["request_id", "decision", "rule", "probability", "flagged",
                  "decision_threshold", "model_version", "bundle_version",
                  "state_hit", "reasons", "latency_ms"]:
        assert field in body
    assert body["decision"] in {"BLOCK", "REVIEW", "PASS"}
    assert 0.0 <= body["probability"] <= 1.0
    assert isinstance(body["reasons"], list)
    assert len(body["reasons"]) > 0
    for reason in body["reasons"]:
        assert {"feature", "contribution", "value"} <= reason.keys()


def test_score_echoes_threshold_and_versions(client):
    r = client.post("/score", json=VALID_TXN)
    body = r.json()
    assert body["decision_threshold"] > 0
    assert body["bundle_version"] == "v1"
    assert isinstance(body["model_version"], str) and body["model_version"]


def test_score_unknown_destination_reports_state_hit_false(client):
    txn = {**VALID_TXN, "nameDest": "C_definitely_never_seen_12345"}
    r = client.post("/score", json=txn)
    assert r.status_code == 200
    assert r.json()["state_hit"] is False


def test_score_hard_block_rule_fires_and_wins_over_review(client):
    txn = {**VALID_TXN, "type": "TRANSFER", "amount": 5_000_000.0,
           "oldbalanceOrg": 5_000_000.0, "newbalanceOrig": 0.0}
    r = client.post("/score", json=txn)
    assert r.status_code == 200
    body = r.json()
    assert body["decision"] == "BLOCK"
    assert body["rule"] == "full_balance_sweep_large_amount"


@pytest.mark.parametrize("field,bad_value", [
    ("step", -1),
    ("amount", -100.0),
    ("oldbalanceOrg", -1.0),
    ("newbalanceOrig", -1.0),
    ("oldbalanceDest", -1.0),
    ("newbalanceDest", -1.0),
])
def test_score_rejects_negative_numeric_fields(client, field, bad_value):
    txn = {**VALID_TXN, field: bad_value}
    r = client.post("/score", json=txn)
    assert r.status_code == 422
    assert r.json()["type"] == "about:blank"


def test_score_rejects_invalid_type_enum(client):
    r = client.post("/score", json={**VALID_TXN, "type": "NOT_A_TYPE"})
    assert r.status_code == 422


@pytest.mark.parametrize("field", [
    "step", "type", "amount", "nameOrig", "oldbalanceOrg",
    "newbalanceOrig", "nameDest", "oldbalanceDest", "newbalanceDest",
])
def test_score_rejects_missing_required_field(client, field):
    txn = {k: v for k, v in VALID_TXN.items() if k != field}
    r = client.post("/score", json=txn)
    assert r.status_code == 422


def test_score_rejects_empty_string_names(client):
    r = client.post("/score", json={**VALID_TXN, "nameDest": ""})
    assert r.status_code == 422


def test_score_rejects_unknown_extra_field(client):
    r = client.post("/score", json={**VALID_TXN, "totally_unexpected_field": 1})
    assert r.status_code == 422


def test_score_rejects_wrong_type_for_amount(client):
    r = client.post("/score", json={**VALID_TXN, "amount": "not-a-number"})
    assert r.status_code == 422


def test_score_boundary_zero_amount_is_valid(client):
    r = client.post("/score", json={**VALID_TXN, "amount": 0.0})
    assert r.status_code == 200


def test_score_boundary_step_zero_is_valid(client):
    r = client.post("/score", json={**VALID_TXN, "step": 0})
    assert r.status_code == 200


# --------------------------------------------------------------- score/batch

def test_batch_scores_multiple_rows_and_summary(client):
    payload = {"transactions": [VALID_TXN, {**VALID_TXN, "amount": 50.0}]}
    r = client.post("/score/batch", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert len(body["results"]) == 2
    summary = body["summary"]
    assert summary["n_scored"] == 2
    assert summary["n_blocked"] + summary["n_flagged"] + summary["n_passed"] == 2


def test_batch_accumulates_destination_state_within_request(client):
    base = {**VALID_TXN, "nameDest": "C_repeat_dest"}
    payload = {"transactions": [base, {**base, "amount": 200.0}, {**base, "amount": 300.0}]}
    r = client.post("/score/batch", json=payload)
    results = r.json()["results"]
    assert results[0]["state_hit"] is False
    assert results[1]["state_hit"] is True
    assert results[2]["state_hit"] is True


def test_batch_rejects_empty_transaction_list(client):
    r = client.post("/score/batch", json={"transactions": []})
    assert r.status_code == 422


def test_batch_rejects_over_max_rows(client):
    payload = {"transactions": [VALID_TXN] * 10_001}
    r = client.post("/score/batch", json=payload)
    assert r.status_code == 422


def test_batch_rejects_oversized_content_length_before_parsing(client):
    r = client.post(
        "/score/batch", content=b"{}",
        headers={"Content-Length": str(20 * 1024 * 1024), "Content-Type": "application/json"},
    )
    assert r.status_code == 413


def test_batch_one_invalid_row_rejects_whole_batch(client):
    payload = {"transactions": [VALID_TXN, {**VALID_TXN, "type": "NOT_VALID"}]}
    r = client.post("/score/batch", json=payload)
    assert r.status_code == 422


# ----------------------------------------------------------------- metrics

def test_metrics_reflects_prior_requests(client):
    client.post("/score", json=VALID_TXN)
    client.post("/score", json=VALID_TXN)
    r = client.get("/metrics")
    assert r.status_code == 200
    body = r.json()
    assert body["score_count"] >= 2
    assert body["request_count"] >= 2
    assert "latency_p50_ms" in body
    assert "score_histogram" in body


# ------------------------------------------------------------- model-info

def test_model_info_contract(client):
    r = client.get("/model-info")
    assert r.status_code == 200
    body = r.json()
    assert body["bundle_version"] == "v1"
    assert len(body["feature_names"]) == 18
    # The SERVED bundle's feature-schema version must match the TRAINING code's.
    # A mismatch means the committed bundle is stale relative to features.py --
    # precisely the train/serve skew this project is built to prevent. The old
    # assertion here compared feature_version against the feature count, which
    # passed only because export_bundle.py wrote the count into both fields.
    assert body["feature_version"] == FEATURE_VERSION
    assert isinstance(body["limitations"], list) and len(body["limitations"]) > 0
    assert isinstance(body["dest_state_snapshot_step"], int)


# ------------------------------------------------------------- rate limit

def test_rate_limit_returns_429_after_threshold(tmp_path):
    _require_real_bundle()
    app = create_app(rate_limit_max_requests=3, rate_limit_window_seconds=60)
    with TestClient(app) as c:
        codes = [c.get("/model-info").status_code for _ in range(5)]
    assert codes[:3] == [200, 200, 200]
    assert 429 in codes[3:]


def test_health_exempt_from_rate_limit(tmp_path):
    _require_real_bundle()
    app = create_app(rate_limit_max_requests=2, rate_limit_window_seconds=60)
    with TestClient(app) as c:
        codes = [c.get("/health").status_code for _ in range(10)]
    assert all(code == 200 for code in codes)
