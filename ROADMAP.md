# Roadmap: Production-Readiness Plan

This document tracks what's needed to take this project from a working
scaffold (proven correct on the real 6.36M-row PaySim dataset) to a
production-shaped, end-to-end system suitable for a portfolio deep-dive.

## 1. Current state (as of 2026-08-02, end of Sprint 4)

| Area | State | Closed by |
|---|---|---|
| Version control | Git repo, branch-per-sprint, PR merges | Sprint 0 |
| Dependencies | Split into `requirements-train.txt` / `requirements-serve.txt` / `dashboard/requirements.txt` | Sprint 0, split Sprint 3 |
| Model artifacts | Persisted per run to `models/<run_id>/` + `metadata.json` | Sprint 0 |
| Config | `config.yaml` drives paths, CV, models, Optuna, capacity, MLflow, economics | Sprint 0, extended Sprint 3 |
| Logging | `logging` to console + `reports/train_<run_id>.log` | Sprint 0 |
| Data layer | DuckDB out-of-core; peak RSS 1,367MB, full run 28.0 min | post-Sprint 0 |
| Experiment tracking | MLflow local sqlite, every (model, fold) run | Sprint 1 |
| Validation | 3 expanding-window time-based CV folds | Sprint 1 |
| Tuning | Optuna, mean PR-AUC across all folds | Sprint 1 |
| Operating point | Capacity-based K + deployable `decision_threshold` | Sprint 2 |
| Explainability | SHAP global + per-alert reason codes (offline) | Sprint 2 |
| Error analysis | TP/FP/FN queue profile, missed-fraud ranking | Sprint 2 |
| **Testing** | `pytest` suite, 66 tests (features/metrics/threshold/cv/schema/economics/bundle/golden-file) | Sprint 3 |
| **Data validation** | `pandera` schema on raw ingest (dtypes, ranges, nullability, `type` enum) | Sprint 3 |
| **Business decision layer** | `src/economics.py` — net value, capacity-constraint cost, ticket-size crossover; naive optimum tested and found degenerate | Sprint 3 |
| **Serving artifact** | Versioned `model_bundle/v1/` (~9.2MB): LightGBM native format + pure-numpy scaler + `dest_state.parquet` — verified to reproduce the golden file in a `requirements-serve.txt`-only venv | Sprint 3 |
| Feature set | 18 features (two zero-SHAP features removed, `FEATURE_VERSION` 3) | Sprint 3 |
| **Serving** | `src/inference/` core (bundle/state/features/score/rules) + FastAPI service (`/health`, `/ready`, `/model-info`, `/score`, `/score/batch`, `/metrics`); both skew tests pass; verified end-to-end (real `uvicorn` process, real HTTP requests) in a `requirements-serve.txt`-only venv | Sprint 4 |
| **CI/CD** | **Nothing automated** | Sprint 6 |
| **Deployment** | **Nothing deployed** | Sprint 7 |
| **Monitoring** | **No drift detection** | Sprint 8 |
| **Governance** | Prediction audit log live (structured JSON, feature-hashed); model card still absent | Sprint 4 done / Sprint 9 |
| **Database** | **None — no persistent store anywhere** | Sprint 10 |
| **Auth/security** | **None — endpoint would be fully open** | Sprint 10 |
| **Graph analytics** | Origin-side proven impossible; destination-side unbuilt | Sprint 11 |
| **Significance testing** | "Tree models are tied" asserted, never tested | Sprint 11 |
| **GenAI/LLM** | **Absent entirely** | Sprint 12 |

Three defects found by the Sprint 1 and Sprint 2 audits are already fixed and
inform the plan below: evidence files cited but untracked, a stale metric claim
contradicted by its own CSV, and a tie-handling inconsistency between `k` and
`n_flagged`. Each remaining sprint therefore ends with an audit step.

## 2. SDLC mapping

| Phase | Artifact | Sprint | Status |
|---|---|---|---|
| Requirements | Problem framing, capacity-based operating point, success metric | 0-2 | Done |
| Requirements | **Business framing: expected-value optimum, not just an exchange rate** (ARCHITECTURE §0) | 3 | Done |
| Design | `ARCHITECTURE.md`: topology, skew resolution, bundle format, API surface, cost controls | — | Done |
| Design | Rendered architecture diagram, model card | 9 | Planned |
| Development | Config-driven modular pipeline, DuckDB data layer | 0-2 | Done |
| Development | `src/inference/` shared core, FastAPI service | 4 | Done |
| Development | Streamlit dashboard | 5 | Planned |
| Testing | Unit (features/metrics/threshold/cv), pandera ingest schema, golden file | 3 | Done |
| Testing | Training/serving skew tests (plumbing + state), contract | 4 | Done |
| Testing | Container integration test | 6 | Planned |
| Build/Release | Model bundle with checksums | 3 | Done |
| Build/Release | Multi-stage Docker image | 6 | Planned |
| CI/CD | Actions: lint -> type -> test -> smoke-train -> build -> scan -> deploy | 6-7 | Planned |
| Deployment | Cloud Run (API) + Streamlit Community Cloud (UI), keyless auth, rollback | 7 | Planned |
| Operations | Cost controls (rate limiting), structured audit log | 4 | Done |
| Operations | Budget alert | 7 | Planned |
| Monitoring | PSI drift on features + scores, scheduled job, dashboard page | 8 | Planned |
| Maintenance | Retraining trigger criteria, champion/challenger promotion path | 8 | Planned |
| Data/Persistence | Neon Postgres: prediction audit store + analyst feedback | 10 | Planned |
| Security | API-key auth, per-key rate limits, Secret Manager | 10 | Planned |
| Analytics | Destination graph features, leakage-controlled component stats | 11 | Planned |
| Statistics | Bootstrap CIs + McNemar on model selection | 11 | Planned |
| GenAI | Network-level STR narratives, groundedness guardrail, eval harness | 12 | Planned |

## 3. Sprint plan

### Sprint 0 — Engineering hygiene (foundation, blocks everything else)
- `git init`, `.gitignore` (exclude the 493MB CSV and `models/*.pkl`), first commit
- `requirements.txt` pinned to what's actually installed
- Move hardcoded paths/hyperparameters into a `config.yaml`
- Replace `print()` with `logging`
- Persist trained models to `models/` (joblib), with a filename that includes a run timestamp/git hash

### Sprint 1 — Experiment rigor
- Install and run the already-stubbed XGBoost/LightGBM sections
- Multiple time-based CV folds instead of one 80/20 split (fraud counts in a single test window are small and noisy — 4,250 events)
- Hyperparameter tuning (Optuna, optimizing PR-AUC)
- Lightweight experiment tracking (MLflow local mode is enough — compare runs over time instead of overwriting `model_comparison.csv`)

### Sprint 2 — Sharpen an already-strong ranker
Originally framed as "close the Precision@K gap" (PR-AUC ~0.9996 but
Precision@21250 only ~0.20). Re-examined 2026-07-24: K=21250 was hardcoded
as 5x the test set's true fraud count (4,250), and Recall@21250 is ~99.95%
— the model finds virtually every fraud in that window. Precision@K is
mathematically capped at 1/5=0.20 whenever K=5x the fraud count and recall
is ~100%, regardless of model quality. At K=1x fraud count (the realistic
"K = actual daily fraud volume" operating point), HistGradientBoosting hits
Precision@K=Recall@K=0.9988. See `data/processed/precision_recall_at_k.csv`
for precision/recall at K in {1x, 2x, 5x, 10x} fraud count, per model.
So the ranker itself is not the weak point; this sprint is about tightening
an already-strong model and making the operating point realistic:
- ~~Velocity features (transactions/hour per account), graph features
  (fan-in/fan-out across `nameOrig`/`nameDest`)~~ — **revised during the
  sprint**: per-origin velocity and fan-out are not computable on PaySim
  (`nameOrig` is 99.85% unique) and distinct-sender fan-in is arithmetically
  identical to prior-transaction count. Built on the destination side
  instead. See Status for the full finding.
- SHAP explainability per prediction
- Threshold tuning at a K set from actual review capacity, not a multiple of the (unknowable in production) true fraud count
- Error analysis report (what the false positives in the top-K actually look like)

### Sprints 3-9 — the plan to production

Planned 2026-08-01 against `ARCHITECTURE.md`, which decides the system design
these sprints implement. Sized for **8-10 hrs/week over 8 weeks (~72h)**.
Ordering rationale, since it differs from the original roadmap:

- **Tests come before serving.** Sprint 4 refactors feature construction into a
  shared inference module. That refactor is unsafe without the unit and
  golden-file tests from Sprint 3 already in place -- Sprint 4 adds the
  skew tests the refactor itself needs, so testing moved earlier overall
  (was Sprint 4 in the original roadmap).
- **The model bundle comes before the API.** The API's contract depends on the
  artifact format; deciding it mid-Sprint-4 would mean rework.
- **Deployment comes before monitoring.** Drift monitoring needs somewhere to
  observe, and PSI reference distributions come from real scored traffic.

Every sprint has an explicit Definition of Done. A sprint is not complete until
its DoD passes, including the audit step — the Sprint 1 and Sprint 2 audits both
found real defects that would otherwise have shipped.

**Tooling baseline (verified 2026-08-01).** Present: `git`, `python` 3.11.
Absent: `docker`, `gh`, `gcloud`; no `.github/workflows/`. The plan is built
around that reality rather than assuming it away:

| Tool | Decision |
|---|---|
| Docker | **Do not install.** Desktop's WSL2 backend costs ~2GB idle on a dual-core/8GB machine. Images are built and integration-tested in CI only (Sprint 6). |
| `gcloud` | **Install in Sprint 7** (~150MB, no daemon). Needed for project + Workload Identity Federation setup. |
| `gh` | Optional convenience in Sprint 6; `git` + the web UI is sufficient. |
| Neon account | Sprint 10. Free tier, no card. |
| Gemini API key | Sprint 12. Free tier, no card. Never commit it — Secret Manager + a local `.env` that is gitignored. |

Consequence to plan around: local development is `uvicorn` + `streamlit` run
directly, never containers. Anything container-shaped is a CI round-trip.

### Sprint 3 — Test foundation, economics & model bundle (Weeks 1-2, ~14h)

**Goal:** make the codebase safe to refactor, answer the actual business
question, and freeze the serving artifact format.

- Commit and merge the outstanding Sprint 2 work (branch `sprint-2-*` -> PR).
- **`src/economics.py` — the business deliverable** (ARCHITECTURE §0). Net value
  = recovered fraud − review cost − missed-fraud liability, over the capacity
  sweep. **The naive "find the optimum" version was tested against real data and
  is degenerate** — at PaySim's 1,572,443 average fraud amount the break-even
  review cost is 85k–162k per alert, so full recall always wins and the answer
  never changes. Two non-degenerate outputs instead:
  (a) **cost of a capacity constraint** — staffing 250/day instead of 500 misses
  208 frauds ≈ 327M exposure, i.e. the marginal value of an analyst seat;
  (b) **fraud-ticket-size crossover** — sweep `avg_fraud_amount` to find where
  reviewing every alert stops paying for itself. PaySim averages 1.57M; UPI fraud
  is high-volume/low-ticket, so this is the sweep with real domain relevance.
  Emits `capacity_economics.csv`. Pure function of counts and rates — unit-tests
  without a model.
- **Remove the two dead features and retrain** (`orig_prior_txn_count`,
  `orig_prior_avg_amount`, both exactly zero SHAP). Feature count 20 -> 18,
  `FEATURE_VERSION` 2 -> 3. Costs one 28-min run plus re-verification of every
  published number. Reversal of an earlier "keep for comparability" decision —
  that argument covers the research narrative, not a deployed artifact.
- `pytest` suite: `features.py`, `custom_metrics.py`, `threshold.py`, `cv.py`
  — including a regression test for the `n_flagged > k` tie path fixed in the
  Sprint 2 audit, which is currently exercised only by chance.
- `pandera` schema validating the raw table on ingest (dtypes, ranges,
  nullability, `type` enum).
- Split into `requirements-train.txt` / `requirements-serve.txt` /
  `dashboard/requirements.txt` (Streamlit Cloud resolves per app directory).
- `.gitignore` negation `!model_bundle/` — the bundle is a committed release
  artifact, not a run output. Forgetting this is the exact defect class the
  Sprint 1 audit caught.
- `src/export_bundle.py`: emit `model.txt` (LightGBM native), `scaler.json`,
  `threshold.json`, `bundle_meta.json` with per-file sha256.
- `src/build_dest_state.py`: emit `dest_state.parquet` (571,961 non-merchant
  destinations; merchants resolve to cold-start defaults).
- Golden file: 200 canned transactions + expected scores, committed.

**DoD:** `pytest` green; **a throwaway venv containing only
`requirements-serve.txt` loads the bundle and reproduces the golden file** (the
earlier DoD said "loads with no scikit-learn installed", which is unverifiable
from inside the training environment); `dest_state.parquet` < 20MB; golden scores
reproduce `predict_proba` to 1e-9; economics reports the per-seat marginal value
and the ticket-size crossover, with the degeneracy of the naive optimum stated
rather than hidden.

**Risk:** the bundle must reproduce the pickled model exactly. Already validated
(ARCHITECTURE §12, exact to 0.000e+00) — the golden file makes it a standing
regression test rather than a one-off check.

### Sprint 4 — Inference core & FastAPI (Weeks 3-4, ~18h)

**Goal:** one code path from raw transaction to score, wrapped in an API.

- `src/inference/`: `bundle.py`, `state.py`, `features.py`, `score.py`
  (see ARCHITECTURE §4). No feature logic anywhere else.
- **Skew test (plumbing)**: training feature matrix -> `inference/score.py`
  matches `predict_proba` to 1e-9.
- **Skew test (state)**: `inference/state.py` matches the DuckDB window query
  at a fixed step for a sample of destinations. Kept separate so a failure
  identifies *which* half broke.
- FastAPI: `/health`, `/ready`, `/model-info`, `/score`, `/score/batch`,
  `/metrics`.
- **Live reason codes via LightGBM `pred_contrib=True`** (native TreeSHAP, no
  `shap`/`numba`/`llvmlite` in the image). Removes the precomputed reason-code
  table and the `/explain` endpoint an earlier draft of this plan required.
- Hard-block rule layer evaluated before the model; response carries
  `decision: BLOCK/REVIEW/PASS` and the rule that fired.
- Pydantic v2 schemas; reject rather than coerce; batch capped at 10k rows/10MB.
- **`/score/batch` accumulates destination aggregates within the batch** — this
  is the project's honest demonstration of incremental feature state, and the
  reason the separate streaming sprint was cut (ARCHITECTURE §16).
- `state_hit` on every scoring response, and `decision_threshold` +
  `model_version` echoed on every score.
- Structured JSON prediction audit log (feature values hashed, not logged).
- Per-IP rate-limit middleware; RFC 7807 error responses.

**DoD:** both skew tests pass; contract tests cover every field's invalid case;
`/score` p95 < 100ms locally; `/ready` fails correctly on a corrupted bundle.

**Risk:** this is the sprint where training/serving skew hides. The two skew
tests are the control and are non-negotiable.

### Sprint 5 — Streamlit dashboard (Week 5, ~9h)

**Goal:** the artifact a non-engineer actually looks at.

- Five pages per ARCHITECTURE §6: Score a transaction, Batch upload, **Capacity
  & economics explorer**, Model card, Drift (placeholder until Sprint 8).
- The explorer layers the §0 expected-value curve over the capacity sweep: three
  adjustable business rates -> net-value curve, marked optimum, sensitivity band.
  Computed client-side from committed CSVs, so it loads instantly and works with
  the API asleep.
- Talks to the API over HTTP; contains no model and no feature logic.
- Bundled 50k-row stratified sample so every page works with zero upload.
- `API_BASE_URL` configurable; graceful degradation when the API is asleep.
- **Landing page renders from committed CSVs only — no API call on load.**
  Streamlit sleep (~30s) chained to Cloud Run cold start (~3-5s) would
  otherwise show a first-time visitor ~35s of nothing.

**DoD:** all pages render under 1GB RAM against a local API; capacity explorer
reproduces `capacity_sweep.csv` exactly at 500/day; no page loads the full
dataset; landing page renders with the API deliberately stopped.

**Priority note:** if this sprint overruns, ship the **capacity explorer** and
cut pages 4-5. It carries the project's headline finding.

### Sprint 6 — Containerization & CI, **CI-first** (Week 6, ~9h)

**Goal:** reproducible build and an automated quality gate — **without
installing Docker on this machine.**

Verified 2026-08-01: `docker`, `gh` and `gcloud` are all absent, and there is no
`.github/workflows/`. Docker Desktop's WSL2 backend costs ~2GB idle on a
dual-core/8GB box (see hardware notes), which is a bad trade for a container
this project never needs to run locally. So the image is **built and tested
exclusively in CI**, and local development runs `uvicorn` and `streamlit`
directly. This is a deliberate constraint-driven choice, not a shortcut.

- Multi-stage `Dockerfile` on `python:3.12-slim`, non-root user, serving deps
  only. Target < 400MB uncompressed.
- GitHub Actions PR gate: `ruff` -> `mypy src/inference` -> `pytest` ->
  smoke-train on `generate_sample_data.py` output (never the gitignored CSV).
- **Serving-isolation job**: install *only* `requirements-serve.txt` and run the
  inference tests. This is what stops a training-only import (sklearn, shap,
  duckdb) from silently entering the serving path and breaking the deploy.
- **Container integration test runs in CI**, not locally: build image -> run ->
  `/ready` -> `/score` -> assert schema, latency, and image size.
- `trivy` image scan.
- `docker-compose.yml` is still committed for anyone cloning with Docker
  available, but is **explicitly marked as CI-verified, not locally verified**.

**DoD:** CI green on a PR; the container integration job passes in CI; image
size and cold-start time recorded in the README from the CI run's output.

**Accepted cost:** every container fix is a CI round-trip (~3-5 min) rather than
a local rebuild. Budgeted into this sprint's 9h.

**Optional (~30 min, do it if the round-trips get annoying):** install the `gh`
CLI — it is lightweight, has no daemon, and makes watching CI runs from the
terminal much faster than the web UI.

### Sprint 7 — Cloud deployment (Week 7, ~9h)

**Goal:** a live public URL that costs $0.

- **Install the `gcloud` CLI first (~150MB, no daemon — unlike Docker Desktop
  it is cheap on this hardware).** Verified absent 2026-08-01. It is needed for
  the initial project/WIF setup, which is materially harder through the web
  console alone. Everything after setup runs from CI.
- GCP project, Artifact Registry repo, **cleanup policy keeping 3 versions**
  (0.5GB free tier breaches on the 4th build otherwise).
- Workload Identity Federation — no service-account JSON in repo secrets.
- Deploy Cloud Run `us-central1`: `--min-instances=0 --max-instances=2
  --memory=512Mi --cpu=1 --concurrency=80 --timeout=30s`.
- **Budget alert at $1/month before the first deploy, not after.**
- Deploy Streamlit Community Cloud from the public repo, pointed at the API.
- Post-deploy smoke test in CI; auto-rollback if `/ready` fails or
  `bundle_version` mismatches.

**DoD:** public URL serves `/score`; cold-start latency measured and documented;
budget alert confirmed active; a deliberately broken deploy rolls back.

**Risk:** a public endpoint on a billing-enabled account. Every control in
ARCHITECTURE §9 ships with the first deploy.

### Sprint 8 — Monitoring & drift (Week 8, ~9h)

**Goal:** evidence the system is still working after deployment.

- `src/monitoring/drift.py`: PSI per feature + on the score distribution;
  reference = final training fold.
- **Comparison windows are successive time slices of the dataset itself, not
  demo traffic.** A portfolio API sees tens of requests; PSI on that sample is
  meaningless and would have produced noise dressed as monitoring. PaySim's 743
  hourly steps give a genuine PSI-over-time series on 6.4M real rows.
- Live traffic is still surfaced, but as **volume/score telemetry**, explicitly
  labelled as not a drift signal.
- Thresholds: PSI > 0.25 (feature) or > 0.10 (score) raises a flag.
- **Detector unit test with injected shifts** (mean, variance, category
  re-weighting) — this is what proves the implementation works.
- Cloud Scheduler (3 jobs free) runs it; output CSV feeds the dashboard Drift page.
- **Documented retraining trigger criteria** (PSI breach / precision-at-capacity
  drop / 90 days). Criteria are the deliverable; automated retraining is
  explicitly out of scope on free tier and labelled as such.
- Champion/challenger: document the Cloud Run revision traffic-split promotion path.

**DoD:** drift job runs on schedule; injecting a shifted distribution raises the
expected flag; Drift page reads live output.

### Sprint 9 — Portfolio polish (Week 9, ~9h)

**Goal:** make an outsider understand it in five minutes.

- Architecture diagram (rendered, not ASCII).
- **Model card**: intended use, features, metrics per fold, the capacity ceiling
  finding, the two dead features, snapshot limitation, fold-3-specific threshold.
- Business-impact write-up built on Sprint 3's `economics.py`: the recommended
  staffing level, the net value at that optimum, and how far the recommendation
  moves under ±50% shifts in each business rate. **Lead with the optimum, not the
  exchange rate** — the 18-FP-per-marginal-fraud figure is the mechanism, the
  staffing recommendation is the answer.
- **State the PaySim separability caveat plainly** (ARCHITECTURE §0): the model
  is near-perfect because the data is synthetic and rule-generated. The system,
  not the score, is the achievement. Saying this first is more credible than
  letting a reviewer discover it.
- README rewrite: lead with the live demo and the capacity finding.
- Demo GIF; cold-start expectations stated honestly.
- Final end-to-end audit (as run for Sprints 1 and 2).

**DoD:** a stranger can go from README to scoring a transaction in under 5
minutes; every numeric claim re-verified against its source file.

---

### Sprints 10-12 — market alignment (Weeks 10-12)

Added 2026-08-01 after auditing the plan against 2026 data-scientist hiring
requirements, then trimmed by a senior-review pass. Sprints 3-9 build a complete,
shippable system; these three close gaps a current JD would flag. **Sprint 9
remains the "always have something finished" checkpoint** — if this phase stalls,
the project is still whole.

Why each is here, with the market signal:

| Gap | Signal |
|---|---|
| No GenAI/LLM anywhere | Entry-level GenAI postings +200% YoY; RAG/LangChain expected at fresher level; LLM+MLOps named the highest-paid specialisations in India 2026 |
| No graph analytics | Fintech fraud JDs explicitly name graph analytics for "fraud patterns and relationships" |
| No database at all | Blocks the feedback loop; leaves the stack without a persistence skill |
| No significance testing | Stats are heavily interviewed; also closes a Sprint 2 audit finding |

Streaming was a fourth item and is **cut** — see the struck-out Sprint 13 below
for why the design did not survive review.

**Sizing: 12 weeks end to end (~104h at 8-10h/week).** Sprint 3 grew to two weeks
when it absorbed `economics.py` and the dead-feature retrain; Sprint 13's removal
gave that week back. If the timeline needs compressing, cut Sprint 12 (GenAI)
before Sprint 11 (graph) — the graph work feeds Sprint 12, so the reverse order
is not available.

### Sprint 10 — Persistence, auth & the feedback loop (Week 10, ~9h)

**Goal:** give the system a database, and close the loop that a database was
blocking.

- **Neon Postgres** (0.5GB, 100 CU-hours, scale-to-zero). Chosen over Supabase,
  whose free projects **pause after 7 days idle** — disqualifying for a demo
  link that must answer on first click.
- Tables `predictions` and `analyst_feedback` (ARCHITECTURE §13).
- Audit-log writes are **fire-and-forget with a bounded queue**: a database
  outage must never fail a scoring request.
- API-key auth + per-key rate limits, replacing the per-IP bucket. Keys in
  Secret Manager; separate read/score and write keys.
- `POST /feedback`; dashboard review queue with confirm/reject actions.

**DoD:** feedback written from the dashboard is queryable in Postgres; scoring
still succeeds with the database deliberately unreachable; unauthenticated
requests are rejected.

### Sprint 11 — Graph analytics & statistical rigor (Week 11, ~9h)

**Goal:** the fraud-specific analytics signal, and evidence for a claim the
project currently only asserts.

- Bipartite origin -> destination graph over the 571,961 non-merchant
  destinations; `networkx`, training-only.
- Features: component size, degree, **flagged-neighbour count** (the mule-network
  analogue), component fraud rate.
- **Leakage control**: all component statistics computed from training-fold
  edges only. Verified with the same self-join restatement approach Sprint 2 used.
- 4-arm ablation as in Sprint 2. **A null result is a likely and acceptable
  outcome** given the feature set was already at PR-AUC 0.9995 — it gets reported
  as one, not buried.
- **Bootstrap CIs** on capacity precision + **McNemar** between the top tree
  models. This closes the Sprint 2 audit finding that "the choice among trees is
  noise" was asserted without a test.
- Graph output is also the **substrate for Sprint 12's narratives** — account
  inbound patterns, component membership, flagged neighbours. This ordering is
  load-bearing, not incidental.

**DoD:** graph features attributed by ablation; leakage probe passes; the
tree-model tie is either confirmed or refuted with a p-value.

**Cut on review:** Vertex AI Model Registry. Registering a model in Vertex while
serving it from Cloud Run and using no other Vertex capability is a keyword, not
an architecture — an interviewer asks "why is it there?" and there is no good
answer. The versioned checksummed bundle plus MLflow is coherent on its own.

### Sprint 12 — GenAI: network-level STR narratives (Week 12, ~9h)

**Goal:** a production-shaped LLM feature, not a chatbot demo.

**Reframed on review.** An earlier draft generated a narrative per *transaction*
from 20 numeric features — where an LLM adds essentially nothing over a string
template, and the groundedness check would have been verifying that the model did
a template's job. That version was theatre. This one operates on the **account
and its network**, where the content is a genuine synthesis of velocity, graph
and model evidence that a template cannot easily produce.

- `POST /narrative/{account}`: account inbound pattern (distinct originators,
  time span, amounts) + Sprint 11 graph context (component size, flagged
  neighbours) + how many contributing originators were themselves emptied + the
  flagged transactions with SHAP drivers -> Gemini 2.5 Flash (free tier, ~15 RPM
  / ~1k RPD, no card) -> Pydantic-validated
  `{summary, typology, evidence[], recommended_action, confidence}`.
- **Explicitly not real-time.** Case narratives are an analyst workflow; the
  endpoint is never called from the scoring path, so `/score` latency is
  untouched and ~1k requests/day is ample for work a human reads.
- **Groundedness check in code**: every number in the narrative must appear in
  the input payload, else reject. This is the anti-hallucination control and it
  is not a prompt instruction.
- **Deterministic template fallback** on API error, rate-limit, or failed
  groundedness — the endpoint never fails because an LLM was unavailable.
- Eval harness: ~30 fixed alerts scored for groundedness, schema validity,
  completeness.
- The LLM **never influences the score or the flag**; it narrates a decision
  already made. Stated in the model card.
- **Governance note in the model card**: Google's free tier may train on
  submitted requests. Acceptable here *only* because PaySim is synthetic; a real
  deployment needs a VPC-hosted or enterprise no-training model. Documenting that
  boundary is part of the deliverable.

**DoD:** eval harness reports groundedness on the fixed set; fallback verified by
disabling the API key; `/score` latency unchanged (narratives are on-demand only).

### ~~Sprint 13 — Streaming replay & online state~~ — **CUT on review**

The design did not work. Cloud Run runs 0-2 **stateless** instances that scale to
zero, so "rolling online state" held in-process would diverge between instances,
vanish on scale-to-zero, and return different answers depending on which instance
served the request — a demonstration of a bug, not of streaming. Moving the state
to Postgres means one write per transaction against a 100 CU-hour/month free tier
across ~1.2M rows: neither cheap nor fast.

The concept is already covered honestly elsewhere: `/score/batch` accumulates
destination aggregates within a batch (Sprint 4), and ARCHITECTURE §16 documents
the full scaling path — including exactly where Kafka/Flink and a real online
store enter, and what forces them. That is a better interview answer than an
unused broker in a compose file.

**Net effect: one week saved, one incoherent component removed.**

## 4. Cross-cutting workstreams — dispositions

Each of these was on the original roadmap as an open idea. Every one now has a
decision, so none of them resurfaces as undefined scope mid-sprint.

| Workstream | Disposition | Where |
|---|---|---|
| **Feature store simulation** | **In scope, Sprint 3.** Becomes `dest_state.parquet` — a point-in-time per-destination state snapshot. This is the resolution to the training/serving skew problem, not just a performance optimisation. | ARCHITECTURE §2 |
| **Hybrid decisioning** (rules over ML) | **In scope, Sprint 4**, minimal form: a hard-block rule layer evaluated before the model, returning `decision: BLOCK/REVIEW/PASS` with the rule that fired. Realistic and cheap. Kept small — the ML score stays the primary signal. | API `/score` |
| **Feedback loop** (analyst labels) | **Now in scope, Sprint 10.** Originally cut for lack of a database — the market-alignment review showed the missing database was itself the gap. Neon Postgres (scale-to-zero, no idle pause) makes the loop real: dashboard review queue -> `analyst_feedback` -> a labelled set a retraining run can consume. Retraining *execution* stays out of scope. | ARCHITECTURE §13 |
| **Alerting** | **Reduced scope.** The Streamlit queue view is the review queue. No Slack webhook — it adds a secret to manage and demonstrates nothing new. | Sprint 5 |
| **Champion/challenger** | **In scope, Sprint 8**, as a documented promotion path using Cloud Run revision traffic splitting, with `bundle_version` exposed on `/model-info`. Not an automated A/B harness. | ARCHITECTURE §10 |
| **Drop the two dead features** | **Deferred to a v2 bundle, deliberately.** `orig_prior_txn_count` and `orig_prior_avg_amount` have exactly zero SHAP attribution, but removing them forces a retrain and invalidates every published number. Removed with a before/after, never silently. | ARCHITECTURE §11 |

### Explicitly out of scope

Named so they read as decisions rather than gaps:

- **Kafka or any message bus** — `/score/batch` demonstrates incremental state
  within a batch, and ARCHITECTURE §16 documents the scaling path that forces a
  broker and where it enters. One added to be named answers no interview
  question well.
- **Kubernetes** — Cloud Run covers the deployment story at this scale.
- **A hosted online feature store** (Redis/Bigtable) — no free tier survives.
- **Automated retraining execution** — the trigger criteria and the labelled
  feedback set are the deliverables (Sprints 8, 10).
- **Multi-region deployment**, multi-tenancy.
- **PySpark, deep-learning baselines, and a Prefect/Airflow DAG** — reviewed
  during the market-alignment pass and deliberately deferred. Each is a keyword
  win, none is better engineering than what is already here at 6.4M rows, and
  all three together would add ~3 weeks for no new capability. Revisit only if a
  specific JD names them.

Two items previously listed here have **moved into scope**: authentication
(Sprint 10), and live per-request SHAP — LightGBM's native `pred_contrib=True`
provides it with no `shap`/`numba`/`llvmlite` dependency, so the image-size
objection that excluded it no longer applies. A hosted model registry moved *in*
and then back *out*: Vertex AI was added for keyword coverage and cut on review
as theatre. The versioned checksummed bundle plus MLflow remains the registry
story.

## Status

- [x] Real dataset (PaySim, 6.36M rows) swapped in and pipeline verified end-to-end
- [x] Sprint 0 -- git init, requirements.txt, config.yaml, logging, model persistence
      (also fixed liblinear/RandomForest performance bugs found along the way:
      total run time 20+ min -> ~80s via saga solver + HistGradientBoosting +
      train-side undersampling)
- [x] Data-layer hardening (post-Sprint 0, pre-Sprint 1, not itself a sprint):
      full runs were hanging/crashing the 8GB dev machine. Rewrote the
      account-history aggregate in features.py as a DuckDB SQL window
      function (was a full-frame pandas sort + groupby cumsum/cumcount that
      transiently duplicated the whole wide dataframe), and moved
      train_pipeline.py's data loading off pandas.read_csv onto a persistent
      DuckDB store (data/processed/paysim.duckdb, gitignored, cached by raw
      file mtime/size + FEATURE_VERSION). Peak RSS: unmeasured multi-GB spike
      -> ~900MB-1GB. Cached reruns: ~33s. Results verified to match the old
      pandas pipeline within noise.
- [x] Metric-framing fix (post-data-layer-hardening, pre-Sprint 1, not itself
      a sprint): added `recall_at_k` to `custom_metrics.py` and a
      Precision/Recall@K curve (`data/processed/precision_recall_at_k.csv`)
      across K in {1x, 2x, 5x, 10x} true fraud count, per model. Revealed
      the "Precision@21250 ~0.20" number was an artifact of K=5x fraud
      count at ~99.95% recall, not a model weakness — see corrected Sprint 2
      framing above.
- [x] Sprint 1 -- expanding-window 3-fold time-based CV (`src/cv.py`),
      XGBoost/LightGBM wired up and evaluated across all folds, Optuna
      tuning, MLflow local (sqlite) tracking of every (model, fold) result.
      Tree models land at PR-AUC ~0.9965-0.9971 (mean across folds).
- [x] Post-Sprint-1 quick fixes (not itself a sprint): (1) Optuna's
      objective originally scored a single designated fold, which happened
      to be near-perfectly separable (see below) and gave it no signal to
      discriminate trials -- changed to mean PR-AUC across all 3 CV folds,
      after which XGBoost (tuned) does land ahead on PR-AUC (0.9971 vs
      0.9968 untuned). (2) `scale_pos_weight` for XGBoost/LightGBM was set
      to the true deployment-time cost ratio (~300-1700:1) even though
      training data was already undersampled to 50:1 -- double-correcting
      on top of undersampling; fixed to use the post-undersampling ratio,
      mirroring what `class_weight='balanced'` already used for the other
      models. (3) Fixed a LightGBM `eval_set` deprecation warning
      (`eval_X`/`eval_y` instead). (4) Each Optuna trial now costs 3
      fold-fits instead of 1 (from fix (1)), pushing a full run to ~30min
      -- cut `n_trials` from 25 to 10 per model to bring it back to
      ~15min, verified to not change which model wins on either metric.
      Fixing (1) surfaced a new, genuine finding: Optuna's PR-AUC-only
      objective found XGBoost hyperparameters that rank fraud slightly
      better but calibrate worse -- weighted-BCE loss (`custom_metrics.py`)
      over 2.5x worse than the untuned default, and marginally behind on
      the realistic Precision@K=1x operating
      point. "Best mean PR-AUC" and "best model" are not the same model
      here, so `best_model` selection was changed from mean PR-AUC to
      mean Precision@K at K=1x fraud count (the operationally relevant
      metric this project's thesis is built around) -- under that
      criterion `best_model` is LightGBM (default params), not the
      higher-PR-AUC XGBoost (tuned). Optuna's tuning result is still
      reported (`xgboost_best_params`/`top_pr_auc_model` in
      `metadata.json`) for comparison, just no longer auto-selected as
      the winner. Full writeup in `README.md` Results.
      **CV surfaced something a single split couldn't**: fold 2's test
      window was PR-AUC=1.0000 for every tree model but PR-AUC=0.06-0.08
      for every linear model. Checked directly against the raw transaction
      table (not our engineered features, so not a leak): in that window
      98.8% of fraud has `amount_to_balance_ratio` exactly 1.00 (account
      drained to the cent) and 100% has `dest_is_merchant=0` -- a known
      PaySim construction characteristic. That's a narrow, nonlinear
      value-band rule trees split out trivially and a single linear
      hyperplane structurally cannot express regardless of class
      weighting. **Those two figures are the Sprint 1 feature set's**, and
      no longer match the regenerated
      `data/processed/model_comparison_by_fold.csv` -- Sprint 2's features
      moved the linear models on this fold to 0.24-0.41 and weakened the
      conclusion. See the Sprint 2 entry below and `README.md` Results.
- [x] End-to-end audit of all Sprint 1 + post-Sprint-1 changes (not itself
      a sprint), performed before anything was staged to git. Verified
      clean: early-stopping/`predict_proba` correctness for both XGBoost
      and LightGBM (confirmed empirically that early stopping actually
      truncates predictions, not just training), `train_pos_weight` vs
      `true_pos_weight` wiring (grepped every call site), `cv.py` fold
      boundary/leakage correctness, `config.yaml`/`requirements.txt`
      consistency, MLflow artifact persistence, and every numeric claim
      in this file/README cross-checked against the actual output CSVs.
      One real finding: `model_comparison_by_fold.csv` and
      `precision_recall_at_k_by_fold.csv` were untracked in git despite
      being cited by name in README/ROADMAP as evidence -- **still needs
      `git add`ing at staging time**, since these files get regenerated
      (and re-untracked relative to the index) by every verification
      rerun done since the audit.
- [x] Sprint 1 and all post-Sprint-1 work committed (`0772bb9`) and merged
      to `main` via PR #2 (`e8eb7ff`) on 2026-07-25, including the audit's
      fix for the two `_by_fold.csv` files being untracked while cited as
      evidence -- both are tracked now.
- [x] Sprint 2 -- capacity-based operating point (`src/threshold.py`),
      destination-side velocity/graph features, SHAP (`src/explain.py`),
      and top-K error analysis (`src/error_analysis.py`). Feature count
      11 -> 20. Outcomes, including the negative ones:
      * **Two of the three planned feature types were not computable.**
        Profiling first (rather than building first) showed `nameOrig` is
        99.85% unique -- 6,344,009 of 6,353,307 origin accounts appear
        exactly once -- so per-origin velocity would be zero for 99.85% of
        rows and fan-out is degenerate. Every `(nameOrig, nameDest)` pair
        is unique, so "distinct prior senders" fan-in is arithmetically
        identical to prior-transaction count. Velocity/graph aggregates
        were built on the destination side instead, which is where the
        repeat structure lives and is the more AML-relevant direction
        (mule accounts taking rapid inflows). Fraud destinations are
        measurably quieter: 2.7 prior transactions vs 5.1.
      * Closed an unrelated real gap found while profiling: **transaction
        `type` was not a feature at all**, despite fraud occurring in only
        2 of 5 types.
      * **The new features bought very little**: 37 fewer false positives
        out of ~3,870 (0.96%) and 2 fewer missed frauds. Reported as such.
        The Sprint 1 feature set had already reached PR-AUC 0.9995, so
        there was almost nothing left to recover. Attribution is via a
        4-arm ablation (`feature_ablation.csv`), not assertion.
      * **SHAP confirmed two features are dead**: `orig_prior_txn_count`
        and `orig_prior_avg_amount` have exactly zero attribution -- the
        two the profiling predicted would be near-dead. Kept for now to
        preserve comparability with the Sprint 1 baseline; flagged for
        removal.
      * **Sprint 2 partly falsified Sprint 1's fold-2 conclusion.** Sprint
        1 recorded that fold 2's window was PR-AUC 0.06-0.08 for every
        linear model and read that as a structural limit of a single
        hyperplane against a nonlinear value band. On the Sprint 2 feature
        set the same models on the same fold score 0.24-0.41 (Lasso 0.239,
        Ridge 0.304, LogReg 0.410) -- a 3-7x move with only the features
        changed. The transaction-type indicators are the cause: fraud is
        confined to TRANSFER/CASH_OUT, and "type is one of two values" is
        expressible as a hyperplane constraint where the ratio band is not.
        The structural claim survives in weaker form (0.41 is still far
        from 1.0000), but a good share of what looked like a model-class
        limitation was a feature-representation gap. Recorded because the
        Sprint 1 writeup asserted the strong version and the regenerated
        CSV it cited now contradicts it.
      * **The headline result is the capacity sweep.** On the final fold
        the model sits on the precision ceiling (`total_fraud / K`) at
        every staffing level swept, so precision at capacity is bounded by
        queue size, not model quality. At 250 reviews/day: precision 1.000,
        recall 0.951, zero false positives. At 500/day: recall 1.000 but
        3,833 false positives -- i.e. the last 208 frauds cost ~18 false
        positives each. That exchange rate, not a single precision number,
        is the deliverable. Scoped to the final fold deliberately: on fold
        1 the same model is *not* at the ceiling (9 of 887 frauds missed,
        precision 0.5621 vs a 0.5678 ceiling).
      * `best_model` selection moved from Precision@K=1x-fraud-count to
        precision at review capacity, since the former is defined by the
        labels and could not be evaluated at deployment time. Caveat
        recorded honestly: the five tree models span 0.5281-0.5291 on this
        metric against a ±0.032 fold std, so the choice among them is
        noise; the metric separates trees from linear baselines, not trees
        from each other.
      * **Known limitation of that selection metric**: it averages three
        folds that are not measuring the same thing. Folds are cut on
        `step` quantiles, so each test window holds ~20% of rows, but
        PaySim's volume collapses in later steps -- folds 1/2 span 3.1 days
        at ~401k/~419k transactions per day while fold 3 spans 16.2 days at
        ~77k. Since `K = reviews_per_day x days`, fold 3 gets 5.2-5.4x more
        review slots per transaction and carries ~5x the fraud prevalence
        (pos_weight 293 vs 1411/1679), so it structurally permits higher
        precision at identical staffing. The ±0.032 std is therefore mostly
        window density, not model variance, and the mean is an average over
        three operating points rather than three estimates of one quantity.
        Model *ranking* is unaffected (all models see the same three
        windows), so selection still works -- but the absolute number
        should not be quoted without naming the fold. Fraud *per day* is
        stable across windows (250-284), which is what keeps a
        headcount-based K defensible. Full table in `README.md`.
      * Leakage verified by restating the window-function features as
        correlated self-joins and checking exact agreement (8,471 rows,
        300 busy destinations, zero disagreements), plus first-transaction
        and same-step-peer probes.
      * Hardening: DuckDB now runs under an explicit 2GB `memory_limit`
        instead of its default ~80% of system RAM. Full run 28.0 min
        measured (up from ~15), peak RSS 1,367 MB.
- [x] End-to-end architecture and sprint plan for Sprints 3-9 (2026-08-01, not
      itself a sprint). Free-tier research across GCP/Azure/AWS resolved the
      deployment target to **Cloud Run (API) + Streamlit Community Cloud (UI)**:
      AWS is disqualified because accounts created after 2025-07-15 expire at 6
      months, and Streamlit cannot run on Cloud Run because its websocket would
      bill CPU continuously and exhaust the 180k vCPU-second monthly grant.
      Azure Container Apps is a documented drop-in fallback (identical grant).
      Design recorded in `ARCHITECTURE.md`; sprint plan, SDLC mapping and
      workstream dispositions rewritten in sections 1-4 above.
      A review pass over the first draft corrected six defects before any code
      was written: (1) loading 571,961 state rows as a dict risked OOM on a
      512MiB instance -- replaced with sorted-hash numpy arrays + searchsorted,
      ~18MB; (2) LightGBM's native `pred_contrib=True` gives exact TreeSHAP with
      no `shap`/`numba`/`llvmlite`, which deleted a precomputed reason-code
      table and an `/explain` endpoint from the design and made per-request
      explanation live; (3) the bundle had no defined path into the image, since
      `models/**` is gitignored -- now a committed `model_bundle/v1/` with a
      `.gitignore` negation; (4) Streamlit Cloud resolves `requirements.txt` per
      app directory, so the dependency split needed three files, not two;
      (5) Streamlit sleep chained to Cloud Run cold start could show a first
      visitor ~35s of nothing -- landing page is now static-only; (6) nothing
      prevented a training-only import entering the serving path -- added a CI
      job installing only `requirements-serve.txt`; (7) Sprints 6-7 assumed
      `docker`/`gh`/`gcloud`, all verified absent, so containerization was
      restructured to be CI-only (Docker Desktop's ~2GB idle cost is unaffordable
      here) with `gcloud` installed in Sprint 7.
      Five load-bearing technical claims were then validated against the real
      trained artifacts rather than assumed: numpy StandardScaler and LightGBM
      text round-trip both reproduce predictions **exactly** (0.000e+00), native
      `pred_contrib` sums to the raw margin to 8e-14, and the state snapshot
      measures 5.68MB on disk / 13.7MB resident -- both under the planned budget.
      See `ARCHITECTURE.md` §12.
- [x] Sprint 3 -- test foundation, economics & model bundle. Merged to `main`
      via PR #4 on 2026-08-01. Outcomes, including the negative/uncertain ones:
      * **Removed the two zero-SHAP features** (`orig_prior_txn_count`,
        `orig_prior_avg_amount`) and retrained. Feature count 20 -> 18,
        `FEATURE_VERSION` 2 -> 3. Every published number was re-verified
        against the retrained CSVs rather than left stale.
      * **`src/economics.py`** (ARCHITECTURE §0): confirmed the naive
        "find the optimum" framing is genuinely degenerate against this
        project's own data -- break-even review cost 84,942-161,795 across
        recovery_rate 0.05-1.00, so full recall wins under every plausible
        assumption. Reports two non-degenerate questions instead: cost of a
        capacity constraint (208 frauds, ~327,068,118 exposure, ~1,308,272
        per marginal review-seat) and the fraud-ticket-size crossover
        (recommended staffing changes at avg_fraud_amount ~500 and ~5,000 --
        PaySim's own 1,572,443 average sits nowhere near either).
      * **66-test `pytest` suite** across `features.py` (leakage safety,
        div-by-zero guards, the ROWS-vs-RANGE tie/velocity distinction,
        verified against a hand-built DuckDB fixture), `custom_metrics.py`,
        `threshold.py` (including a regression test for the `n_flagged > k`
        tie path, previously exercised only by chance), `cv.py`, the new
        `pandera` ingest schema, `economics.py`, and the bundle/golden-file
        pipeline.
      * **Versioned `model_bundle/v1/`** (~9.2MB, committed via a
        `.gitignore` negation): LightGBM native text format, a pure-numpy
        `StandardScaler` reimplementation, and `dest_state.parquet` (571,961
        rows, mean 7.36 / max 113 prior transactions -- exact match to the
        ARCHITECTURE §2 profiling). Verified by reproducing a 200-row golden
        file to **exact** floating-point equality (0.000e+00) in a
        throwaway venv containing *only* `requirements-serve.txt` --
        scikit-learn, duckdb, mlflow, and pandas confirmed absent.
      * **A real reproducibility bug found and fixed while investigating an
        anomaly.** `config.yaml` documented that `random_state` was
        "injected separately... so it stays consistent across all models,"
        but `train_pipeline.py` never actually passed it to any of the
        three `LogisticRegression` constructions. Harmless for the plain
        and Ridge-penalized models (`lbfgs` is deterministic regardless of
        seed), but Lasso's `solver: saga` is stochastic, so its coefficients
        were not reproducible run to run. Fixed, with a regression test
        (`test_train_pipeline_determinism.py`) pinning it.
      * **A finding flagged rather than papered over.** The linear models'
        fold-2 PR-AUC came back substantially lower than Sprint 2's
        published 0.24-0.41 (now 0.085-0.113). Ruled out solver randomness
        as the cause -- Logistic Regression and Ridge are deterministic and
        reproduced bit-identically across three separate retrains in this
        environment -- which leaves the two-feature removal and a possible
        training-environment difference (different sklearn/scipy build than
        whatever produced the original numbers) as the remaining candidates,
        not distinguished here. Recorded in `README.md` rather than quietly
        overwritten, matching how the Sprint 1/2 audits handled similar
        surprises.
      * Split dependencies into `requirements-train.txt` /
        `requirements-serve.txt` / `dashboard/requirements.txt`, per
        ARCHITECTURE §3's isolation requirement.
- [x] Sprint 4 -- inference core & FastAPI. Not yet merged (branch work,
      2026-08-02). Outcomes, including the negative/uncertain ones:
      * **A real, pre-existing bundle-integrity bug found and fixed.**
        `model_bundle/v1/model.txt`'s sha256 in `bundle_meta.json` matched
        neither the file on disk NOR the git-committed blob -- two separate
        defects layered on top of each other. (1) `core.autocrlf=true` with
        no `.gitattributes` was silently rewriting the LightGBM native-text
        model file's LF line endings to CRLF on every Windows checkout,
        which broke LightGBM's parser outright (`Model format error`) and
        changed its checksum. (2) Independently, the checksum actually
        recorded in `bundle_meta.json` at Sprint 3's commit didn't match
        even the correct (git blob) content -- the manifest was stale
        before the CRLF issue ever entered the picture. Fixed both: added
        `.gitattributes` (`model_bundle/** -text`, disabling EOL conversion
        unconditionally) and regenerated the correct sha256 into
        `bundle_meta.json`. Root-caused by treating `load_bundle()`'s
        checksum failure as a real signal rather than working around it --
        the whole point of ARCHITECTURE §3's integrity check is to catch
        exactly this class of defect, and it did.
      * **`src/inference/`** (ARCHITECTURE §4): `bundle.py` (load +
        sha256-verify), `state.py` (dest-state snapshot as five parallel
        numpy arrays keyed by a sorted 64-bit blake2b hash, `np.searchsorted`
        lookup, measured **13.09MB resident** for 571,961 destinations
        against a 13.7MB budget), `features.py` (raw transaction dict ->
        18-feature vector, split into stateless/dest halves so
        `/score/batch` can inject merged snapshot+in-batch state), `score.py`
        (probability + live TreeSHAP reason codes via
        `booster.predict(pred_contrib=True)`), and `rules.py` (a hard-block
        layer -- one illustrative rule, full-balance-sweep on
        TRANSFER/CASH_OUT above 1,000,000, measured 0.247% block rate /
        10.1% precision against the real table; documented as illustrative,
        not tuned, since the model score stays the primary signal).
      * **Both skew tests pass.** Plumbing (`test_skew_plumbing.py`):
        `inference/score.py` against the golden file's training-computed
        `expected_score`, max abs diff **1.1e-16**. State
        (`test_skew_state.py`): `inference/state.py`'s lookups against an
        independently-written DuckDB query over the live table, for a
        sampled 30 real destinations plus a merchant sample (all correctly
        cold-start).
      * **FastAPI service** (ARCHITECTURE §5): `/health`, `/ready`
        (fails correctly on a corrupted or missing bundle -- verified),
        `/model-info`, `/score`, `/score/batch`, `/metrics`. Pydantic v2,
        `extra="forbid"`, RFC 7807 problem+json errors, per-IP rate limiting,
        a Content-Length-based 10MB body cap ahead of the 10k-row Pydantic
        cap, and a structured JSON prediction audit log (feature values
        hashed, not logged). `/score/batch` accumulates destination
        aggregates within the batch, verified directly: the first of three
        transactions to a new destination in one batch reports
        `state_hit=false`, the next two `true`.
      * **120 new tests** (42 inference unit + skew, 39 API contract/
        integration, on top of Sprint 3's 66 -- **147 total, all passing**)
        via a `create_app()` factory (not a shared module-level singleton)
        so rate-limit/metrics state can't leak between tests.
      * **Measured, not assumed: `/score` p95 = 4.9ms locally** (300
        requests after warmup) against the 100ms DoD ceiling -- LightGBM
        inference and the numpy state lookup are not the bottleneck at this
        scale.
      * **Verified end-to-end in a `requirements-serve.txt`-only venv**: a
        real `uvicorn` process (not just an import check) answering real
        HTTP requests to `/health`, `/ready`, and `/score`, with `pandas`,
        `scikit-learn`, `duckdb`, `shap`, `mlflow`, and `optuna` absent from
        the environment. `TestClient`-based tests needed an additional
        `httpx` install `starlette` doesn't require at runtime -- confirmed
        that dependency is test-only and does not appear in
        `requirements-serve.txt`.
- [ ] Sprint 5 -- Streamlit dashboard
- [ ] Sprint 6 -- containerization & CI
- [ ] Sprint 7 -- cloud deployment
- [ ] Sprint 8 -- monitoring & drift
- [ ] Sprint 9 -- portfolio polish **(project is complete and shippable here)**
- [x] Market-alignment review of the plan against 2026 DS hiring requirements
      (2026-08-01, not itself a sprint). Verdict: the MLOps spine is well
      targeted -- "model serving, monitoring, feature stores" are exactly the
      skills carrying a 30-50% pay premium, and BFSI+fintech is ~65% of Indian
      analytics roles, so the payments/AML framing is the right bet. Five gaps
      found: no GenAI anywhere (entry-level GenAI postings +200% YoY), no graph
      analytics (named explicitly in fintech fraud JDs), no streaming, **no
      database at all** (which was itself the reason the feedback loop had been
      scoped out), and no significance testing behind the "tree models are tied"
      claim. Added Sprints 10-13. PySpark, deep-learning baselines and
      orchestration were considered and deliberately deferred -- keyword wins
      that are not better engineering here. Free-tier choices verified: Neon over
      Supabase (whose free projects pause after 7 days idle), Gemini Flash free
      tier (~15 RPM / ~1k RPD, no card) with the caveat that it may train on
      submitted requests -- acceptable only because PaySim is synthetic, and
      documented as a production boundary.
- [x] Senior-review iteration of the whole plan (2026-08-01, not itself a
      sprint). Six changes, four of them removals -- the plan was carrying
      complexity that would not have survived an interview:
      (1) **The business case stopped one step short.** The capacity sweep gives
      an exchange rate (18 FPs per marginal fraud); an exchange rate is not a
      decision. Added `src/economics.py` (Sprint 3). **Then tested it against the
      real data and found the obvious version degenerate**: average fraud amount
      in the final fold is 1,572,443, so break-even review cost is 85k-162k per
      alert and full recall wins under every plausible assumption -- an "optimum"
      that never moves. Reframed to the two questions that do have content: the
      **marginal value of an analyst seat** (250/day vs 500/day = 208 missed
      frauds ~ 327M exposure) and the **fraud-ticket-size crossover** below which
      reviewing every alert stops paying -- the latter being directly relevant to
      UPI's high-volume/low-ticket profile versus PaySim's 1.57M average.
      (2) **The premise was unstated and slightly dishonest.** PaySim is
      near-trivially separable by construction, so PR-AUC 0.9995 is a property
      of the dataset, not an achievement. ARCHITECTURE §0 now says so first, and
      repositions the deliverable as an ML *systems* project.
      (3) **Sprint 13 (streaming) CUT** -- the design did not work. Cloud Run is
      stateless with 0-2 instances and scale-to-zero, so in-process "rolling
      online state" would diverge across instances and vanish on scale-down.
      Replaced by an honest scaling path (§16) plus the batch endpoint's
      within-batch accumulation.
      (4) **GenAI reframed from transaction-level to network-level.** An LLM
      narrating 20 numeric features adds nothing over a template; narrating an
      account's fan-in pattern plus graph context is a real synthesis. Makes
      Sprint 11 -> 12 ordering load-bearing.
      (5) **Vertex AI Model Registry CUT** as theatre -- registering a model you
      serve from Cloud Run, using no other Vertex capability.
      (6) **Drift monitoring was fictional** -- PSI against a portfolio API's
      handful of requests is noise. Now computed over successive time windows of
      the 6.4M-row dataset, with injected-shift unit tests proving the detector.
      Also reversed: the two zero-SHAP features are now **removed in Sprint 3**
      (feature count 20 -> 18) rather than deferred to a v2 -- comparability is a
      research-narrative argument, not a reason to ship inert inputs.
- [x] Git workflow & migration plan (2026-08-01, not itself a sprint) --
      `GIT_WORKFLOW.md`. Branch-per-sprint model, Conventional Commits, tagging
      (`v1.0.0` = Sprint 7 first deploy), branch protection, and safety hooks.
      Two findings from auditing the actual history: (1) commit `0772bb9`
      (Sprint 1, on `main`, already pushed) carries `Co-Authored-By: Claude` and
      `Claude-Session` trailers -- a dry-run-validated `git filter-repo`
      migration to strip them is documented, and it force-pushes `main`, so it
      is the user's call to run; (2) the repo's recurring defect -- docs citing
      evidence files that were never tracked -- now has a one-line pre-merge
      check that catches all 9 currently outstanding files.
      Also plans the migration of the current working tree into two branches:
      `sprint-2-capacity-shap-error-analysis` (code + audit + results) and
      `docs/architecture-and-plan-sprints-3-12`, the split needing one
      `git add -p` pass over `ROADMAP.md`.
- [ ] Sprint 10 -- persistence, auth & feedback loop
- [ ] Sprint 11 -- graph analytics & statistical rigor
- [ ] Sprint 12 -- GenAI network-level STR narratives
