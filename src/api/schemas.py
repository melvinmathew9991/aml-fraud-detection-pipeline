"""
api/schemas.py

Pydantic v2 request/response models (ARCHITECTURE.md §5). Every request
model uses `extra="forbid"`: an unrecognized field is a client bug worth
surfacing as a 422, not something to silently drop. Field constraints
mirror schema.py's RAW_TRANSACTION_SCHEMA (the same validation the training
ingest path applies), so a payload that would fail pandera at training time
also fails Pydantic at serving time.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

TransactionType = Literal["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]

BATCH_MAX_ROWS = 10_000
BATCH_MAX_BYTES = 10 * 1024 * 1024


class TransactionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step: int = Field(ge=0, description="Simulated hour since dataset start.")
    type: TransactionType
    amount: float = Field(ge=0)
    nameOrig: str = Field(min_length=1)
    oldbalanceOrg: float = Field(ge=0)
    newbalanceOrig: float = Field(ge=0)
    nameDest: str = Field(min_length=1)
    oldbalanceDest: float = Field(ge=0)
    newbalanceDest: float = Field(ge=0)


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    transactions: list[TransactionRequest] = Field(min_length=1, max_length=BATCH_MAX_ROWS)


class ReasonCode(BaseModel):
    feature: str
    contribution: float
    value: float


class ScoreResponse(BaseModel):
    request_id: str
    decision: Literal["BLOCK", "REVIEW", "PASS"]
    rule: str | None
    probability: float
    flagged: bool
    decision_threshold: float
    model_version: str
    bundle_version: str
    state_hit: bool
    reasons: list[ReasonCode]
    latency_ms: float


class BatchSummary(BaseModel):
    n_scored: int
    n_blocked: int
    n_flagged: int
    n_passed: int
    alert_rate: float


class BatchResponse(BaseModel):
    results: list[ScoreResponse]
    summary: BatchSummary
    model_version: str
    bundle_version: str
    decision_threshold: float
    latency_ms: float


class ModelInfoResponse(BaseModel):
    model_name: str
    bundle_version: str
    feature_version: int
    git_commit: str
    feature_names: list[str]
    decision_threshold: float
    reviews_per_day: float
    expected_precision: float
    expected_recall: float
    precision_ceiling: float
    dest_state_rows: int
    dest_state_snapshot_step: int | None
    limitations: list[str]


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"


class ReadyResponse(BaseModel):
    status: Literal["ready", "not_ready"]
    detail: str | None = None
    bundle_version: str | None = None


class MetricsResponse(BaseModel):
    request_count: int
    score_count: int
    flagged_count: int
    blocked_count: int
    flag_rate: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    score_histogram: dict[str, int]


class ProblemDetail(BaseModel):
    """RFC 7807 problem+json body."""
    type: str = "about:blank"
    title: str
    status: int
    detail: str | None = None
    instance: str | None = None
