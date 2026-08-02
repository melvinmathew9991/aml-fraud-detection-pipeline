"""
api/audit.py

Structured JSON prediction audit log (ARCHITECTURE.md §5's "Prediction
audit log"): one JSON line per scored transaction to stdout, picked up by
whatever log sink the deployment uses (Cloud Logging in ARCHITECTURE's
design, plain stdout locally). This is the AML traceability requirement
from the original roadmap.

Raw feature VALUES are hashed, not logged -- the log is a compliance trail
(what was scored, when, against which model/threshold, with what outcome)
without becoming a PII store of account balances and destination ids.
"""

import hashlib
import json
import logging
import time

logger = logging.getLogger("api.audit")


def feature_hash(feature_vector: list[float]) -> str:
    payload = json.dumps(feature_vector, sort_keys=False).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def log_prediction(*, request_id: str, model_version: str, bundle_version: str,
                    threshold: float, score: float, flagged: bool, decision: str,
                    state_hit: bool, feature_vector: list[float], latency_ms: float) -> None:
    record = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "event": "prediction",
        "request_id": request_id,
        "model_version": model_version,
        "bundle_version": bundle_version,
        "threshold": threshold,
        "score": score,
        "flagged": flagged,
        "decision": decision,
        "state_hit": state_hit,
        "feature_hash": feature_hash(feature_vector),
        "latency_ms": latency_ms,
    }
    logger.info(json.dumps(record))
