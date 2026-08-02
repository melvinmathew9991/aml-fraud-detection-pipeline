"""
api/main.py

FastAPI service (ARCHITECTURE.md §5). The only code path from raw
transaction to score is `inference/` -- this module is a thin HTTP shell
around it: parse, call inference.rules/score, log, respond. No feature
logic lives here.

Run locally with the bundle already committed at model_bundle/v1/:

    uvicorn api.main:app --app-dir src --reload

`create_app()` is a factory rather than a bare module-level `app` so tests
can build isolated instances (a corrupted-bundle /ready test, a tight
rate-limit test) without env-var/reload gymnastics; `app` below is just
`create_app()` called once, for uvicorn's benefit.
"""

import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import APIRouter, FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse

from features import FEATURE_COLUMNS
from inference import rules
from inference.bundle import Bundle, BundleError, load_bundle
from inference.features import (
    compute_dest_features,
    compute_features,
    compute_stateless_features,
)
from inference.score import score_batch as score_batch_vectors
from inference.score import score_one
from inference.state import DestState, load_dest_state
from model_card import MODEL_LIMITATIONS

from .audit import log_prediction
from .batch import BatchAccumulator
from .errors import register_exception_handlers
from .limits import BodySizeLimitMiddleware
from .metrics import MetricsTracker
from .rate_limit import (
    DEFAULT_MAX_REQUESTS,
    DEFAULT_WINDOW_SECONDS,
    RateLimitMiddleware,
)
from .schemas import (
    BatchRequest,
    BatchResponse,
    BatchSummary,
    HealthResponse,
    MetricsResponse,
    ModelInfoResponse,
    ReadyResponse,
    ReasonCode,
    ScoreResponse,
    TransactionRequest,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("api")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BUNDLE_DIR = PROJECT_ROOT / "model_bundle" / "v1"
REASONS_PER_ALERT = 4


class ServiceState:
    def __init__(self):
        self.bundle: Bundle | None = None
        self.dest_state: DestState | None = None
        self.ready: bool = False
        self.ready_detail: str | None = None
        self.metrics = MetricsTracker()


def load_service_state(bundle_dir: Path) -> ServiceState:
    state = ServiceState()
    try:
        state.bundle = load_bundle(bundle_dir)
        state.dest_state = load_dest_state(state.bundle.dest_state_path)
        state.ready = True
        logger.info('{"event":"startup","status":"ready","bundle_version":"%s"}',
                     state.bundle.bundle_version)
    except (BundleError, FileNotFoundError, OSError) as exc:
        state.ready = False
        state.ready_detail = str(exc)
        logger.error('{"event":"startup","status":"not_ready","detail":%r}', str(exc))
    return state


def _require_ready(service: ServiceState) -> None:
    if not service.ready or service.bundle is None or service.dest_state is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Service not ready: {service.ready_detail or 'bundle not loaded'}",
        )


def _score_response(request_id: str, decision, probability: float, flagged: bool,
                     threshold: float, model_version: str, bundle_version: str,
                     state_hit: bool, reasons, latency_ms: float) -> ScoreResponse:
    return ScoreResponse(
        request_id=request_id,
        decision=decision.decision,
        rule=decision.rule,
        probability=probability,
        flagged=flagged,
        decision_threshold=threshold,
        model_version=model_version,
        bundle_version=bundle_version,
        state_hit=state_hit,
        reasons=[ReasonCode(feature=r.feature, contribution=r.contribution, value=r.value)
                 for r in reasons],
        latency_ms=latency_ms,
    )


router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse()


@router.get("/ready", response_model=ReadyResponse)
async def ready(request: Request) -> ReadyResponse | JSONResponse:
    # FastAPI's response_model machinery assumes a 200; a "not ready" bundle
    # must surface as a 503 with a body, so that branch returns a plain
    # JSONResponse instead -- returning a Response subclass bypasses
    # response_model processing entirely, which is exactly what's wanted here.
    service: ServiceState = request.app.state.service
    if service.ready:
        return ReadyResponse(status="ready", bundle_version=service.bundle.bundle_version)
    body = ReadyResponse(status="not_ready", detail=service.ready_detail)
    return JSONResponse(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, content=body.model_dump())


@router.get("/model-info", response_model=ModelInfoResponse)
async def model_info(request: Request) -> ModelInfoResponse:
    service: ServiceState = request.app.state.service
    _require_ready(service)
    bundle = service.bundle
    return ModelInfoResponse(
        model_name=bundle.bundle_meta["model_name"],
        bundle_version=bundle.bundle_version,
        feature_version=bundle.feature_version,
        git_commit=bundle.bundle_meta["git_commit"],
        feature_names=bundle.feature_names,
        decision_threshold=bundle.threshold.decision_threshold,
        reviews_per_day=bundle.threshold.reviews_per_day,
        expected_precision=bundle.threshold.expected_precision,
        expected_recall=bundle.threshold.expected_recall,
        precision_ceiling=bundle.threshold.precision_ceiling,
        dest_state_rows=len(service.dest_state.keys),
        dest_state_snapshot_step=service.dest_state.snapshot_step,
        limitations=MODEL_LIMITATIONS,
    )


@router.post("/score", response_model=ScoreResponse)
async def score(txn: TransactionRequest, request: Request) -> ScoreResponse:
    service: ServiceState = request.app.state.service
    _require_ready(service)
    service.metrics.record_request()

    start = time.perf_counter()
    request_id = str(uuid.uuid4())

    vector, values, state_hit = compute_features(txn.model_dump(), service.dest_state)
    result = score_one(service.bundle, vector, top_n_reasons=REASONS_PER_ALERT)
    decision = rules.decide(values, flagged=result.flagged)

    latency_ms = (time.perf_counter() - start) * 1000
    service.metrics.record_score(result.probability, result.flagged, decision.decision, latency_ms)

    log_prediction(
        request_id=request_id,
        model_version=service.bundle.bundle_meta["model_name"],
        bundle_version=service.bundle.bundle_version,
        threshold=result.decision_threshold,
        score=result.probability,
        flagged=result.flagged,
        decision=decision.decision,
        state_hit=state_hit,
        feature_vector=vector,
        latency_ms=latency_ms,
    )

    return _score_response(
        request_id, decision, result.probability, result.flagged, result.decision_threshold,
        service.bundle.bundle_meta["model_name"], service.bundle.bundle_version,
        state_hit, result.reasons, latency_ms,
    )


@router.post("/score/batch", response_model=BatchResponse)
async def score_batch(payload: BatchRequest, request: Request) -> BatchResponse:
    service: ServiceState = request.app.state.service
    _require_ready(service)
    service.metrics.record_request()

    start = time.perf_counter()
    accumulator = BatchAccumulator(service.dest_state)

    feature_rows = []
    state_hits = []
    for txn in payload.transactions:
        txn_dict = txn.model_dump()
        dest_state_values, state_hit = accumulator.prior_state_for(txn_dict["nameDest"])
        values = {
            **compute_stateless_features(txn_dict),
            **compute_dest_features(txn_dict["amount"], dest_state_values),
        }
        feature_rows.append(values)
        state_hits.append(state_hit)
        accumulator.observe(txn_dict["nameDest"], txn_dict["amount"])

    vectors = [[values[name] for name in FEATURE_COLUMNS] for values in feature_rows]
    score_results = score_batch_vectors(service.bundle, vectors, top_n_reasons=REASONS_PER_ALERT)

    results = []
    n_blocked = n_flagged = n_passed = 0
    for i, (values, state_hit, result) in enumerate(
            zip(feature_rows, state_hits, score_results, strict=True)):
        decision = rules.decide(values, flagged=result.flagged)
        if decision.decision == "BLOCK":
            n_blocked += 1
        elif decision.decision == "REVIEW":
            n_flagged += 1
        else:
            n_passed += 1

        row_latency_ms = (time.perf_counter() - start) * 1000
        service.metrics.record_score(result.probability, result.flagged, decision.decision,
                                      row_latency_ms)
        log_prediction(
            request_id=f"batch-{i}",
            model_version=service.bundle.bundle_meta["model_name"],
            bundle_version=service.bundle.bundle_version,
            threshold=result.decision_threshold,
            score=result.probability,
            flagged=result.flagged,
            decision=decision.decision,
            state_hit=state_hit,
            feature_vector=vectors[i],
            latency_ms=row_latency_ms,
        )
        results.append(_score_response(
            f"batch-{i}", decision, result.probability, result.flagged,
            result.decision_threshold, service.bundle.bundle_meta["model_name"],
            service.bundle.bundle_version, state_hit, result.reasons, row_latency_ms,
        ))

    latency_ms = (time.perf_counter() - start) * 1000
    n_scored = len(results)
    return BatchResponse(
        results=results,
        summary=BatchSummary(
            n_scored=n_scored, n_blocked=n_blocked, n_flagged=n_flagged, n_passed=n_passed,
            alert_rate=(n_flagged + n_blocked) / n_scored if n_scored else 0.0,
        ),
        model_version=service.bundle.bundle_meta["model_name"],
        bundle_version=service.bundle.bundle_version,
        decision_threshold=service.bundle.threshold.decision_threshold,
        latency_ms=latency_ms,
    )


@router.get("/metrics", response_model=MetricsResponse)
async def metrics(request: Request) -> MetricsResponse:
    service: ServiceState = request.app.state.service
    return MetricsResponse(**service.metrics.snapshot())


def create_app(bundle_dir: Path = DEFAULT_BUNDLE_DIR,
               rate_limit_max_requests: int = DEFAULT_MAX_REQUESTS,
               rate_limit_window_seconds: float = DEFAULT_WINDOW_SECONDS) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.service = load_service_state(bundle_dir)
        yield

    app = FastAPI(title="AML Fraud Detection API", version="1.0.0", lifespan=lifespan)
    register_exception_handlers(app)
    app.add_middleware(RateLimitMiddleware, max_requests=rate_limit_max_requests,
                        window_seconds=rate_limit_window_seconds)
    app.add_middleware(BodySizeLimitMiddleware)
    app.include_router(router)
    return app


app = create_app()
