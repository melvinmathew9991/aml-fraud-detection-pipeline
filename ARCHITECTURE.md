# Architecture

System design for the end-to-end fraud detection platform: training, serving,
dashboard, deployment, and monitoring. Written before Sprint 3 so the
remaining sprints implement a decided design rather than discovering one.

Companion to `ROADMAP.md` (what gets built when) and `README.md` (results).

---

## 0. Business framing — read this first

Every design choice below follows from this section. It was rewritten on
2026-08-01 after a review found the plan was building a production system around
an unstated and slightly dishonest premise.

### The model is not the product

PaySim's fraud is close to trivially separable: it is synthetically generated
with near-deterministic rules (source account drained to the cent, confined to
TRANSFER and CASH_OUT). The pipeline reaches PR-AUC 0.9995+ and, at 250
reviews/day, precision 1.000 / recall 0.951 with **zero false positives**.

A model that good is not a research result — it is a property of the dataset.
Presenting it as a modelling achievement would be the single most obvious thing
for an interviewer to puncture. So the project does not claim it. **The model is
a component; the decision system around it is the product.**

### The actual business question

A fraud team does not ask "what is your PR-AUC?" It asks:

> **How many analysts should we staff, and what do we lose if we staff fewer?**

Sprint 2 got most of the way there with the capacity sweep — precision and
recall as a function of reviews/day, bounded by the `total_fraud / K` ceiling,
with the finding that the last 208 frauds cost ~18 false positives each.

**It stopped one step short of the answer.** An exchange rate is not a decision.
The decision needs money on both sides:

```
  net_value(K) = (frauds_caught(K) x avg_fraud_amount x recovery_rate)
               - (alerts_reviewed(K) x cost_per_review)
               - (frauds_missed(K)  x avg_fraud_amount x liability_rate)
```

### The naive version of this is degenerate — measured, not assumed

Computing it against the real data (2026-08-01) gives average fraud amount
**1,572,443** in the final fold, and a break-even review cost of **85,000 to
161,875 per alert** across recovery rates from 0.05 to 1.00. No alert review
costs that. Net value is therefore maximised at full recall under *every*
plausible assumption, and the "optimum" is a foregone conclusion.

Shipping a module whose answer never changes would be worse than shipping
nothing. So the framing is inverted into the two questions that do have content:

**1. What does a capacity constraint cost?** If you can only staff 250/day
instead of 500, you miss 208 frauds — about **327M** in exposure. Expressed per
seat, that is the marginal value of an analyst, and it is a staffing business
case a manager can actually take to a budget meeting. This is linear, honest, and
does not pretend to an optimum that isn't there.

**2. At what ticket size do the economics start to bind?** This is the
interesting sweep, and it is **directly relevant to Indian payments**. PaySim's
fraud averages 1.57M, which swamps review cost by four orders of magnitude. UPI
fraud is the opposite profile — high volume, low ticket (hundreds to a few
thousand rupees). Sweeping `avg_fraud_amount` finds the crossover where
reviewing every alert stops paying for itself and an under-staffed,
higher-threshold policy becomes correct.

So the deliverable is **the sensitivity, not the optimum**: cost per analyst
seat, plus the fraud-ticket-size threshold below which the whole
review-everything strategy inverts. That is a finding; "staff to full recall" is
arithmetic.

`src/economics.py` (Sprint 3) is surfaced in the dashboard's capacity explorer.
The three business rates stay user-adjustable — they are what a fraud lead would
argue about — but the headline is the crossover analysis.

### What this project demonstrates, honestly stated

An ML **systems** project: leakage-safe feature engineering at 6.4M rows,
time-aware validation, a deployable operating point derived from capacity rather
than labels, an economic decision layer, and the serving/monitoring/CI apparatus
around it. The modelling is competent and deliberately not the headline.

---

## 1. Target topology

```
                          ┌─────────────────────────────────────┐
   OFFLINE (laptop)       │  GitHub Actions (CI/CD)             │
   ┌──────────────────┐   │  lint -> test -> smoke-train ->     │
   │ data/raw/*.csv   │   │  build image -> push AR -> deploy   │
   │   (493MB, local) │   └──────────────┬──────────────────────┘
   └────────┬─────────┘                  │ Workload Identity Federation
            │                            │ (keyless, no JSON key in repo)
            v                            v
   ┌──────────────────┐          ┌──────────────────────┐
   │ paysim.duckdb    │          │ Artifact Registry    │
   │  (684MB, local)  │          │  (0.5GB free tier)   │
   └────────┬─────────┘          └──────────┬───────────┘
            │ train_pipeline.py             │
            v                               v
   ┌──────────────────┐          ┌──────────────────────────────┐
   │ MODEL BUNDLE     │  baked   │  Cloud Run: fraud-api        │
   │  model.txt       │  into    │  FastAPI + uvicorn           │
   │  scaler.json     │─ image ─>│  scale-to-zero, max 2 inst   │
   │  threshold.json  │          │  512MiB / 1 vCPU             │
   │  dest_state.pq   │          │  public HTTPS endpoint       │
   └──────────────────┘          └──────────┬───────────────────┘
                                    │       │       │
                        HTTPS(JSON)│       │       │
            ┌───────────────────────┘       │       └──────────────┐
            v                               v                      v
 ┌──────────────────────────┐  ┌────────────────────┐  ┌────────────────────┐
 │ Streamlit Community Cloud│  │ Neon Postgres      │  │ Gemini Flash API   │
 │  dashboard (public repo) │  │  predictions       │  │  SAR narrative     │
 │  1GB RAM, sleeps @12h    │  │  analyst_feedback  │  │  15 RPM / ~1k RPD  │
 └──────────────────────────┘  │  0.5GB, scale-to-0 │  │  free, no card     │
                               └────────────────────┘  └────────────────────┘
```

**Why Neon, not Supabase.** Supabase free projects **pause after 7 days idle**
and only wake on visit — disqualifying for a link that must answer on first
click. Neon scales to zero without pausing, and 0.5GB / 100 CU-hours is far
beyond what an audit log needs.

**Why the frontend is not on Cloud Run.** Streamlit holds a websocket open for
the life of a browser session. Cloud Run bills CPU for the full duration of a
request, so one tab left open for ~50 hours consumes the entire 180,000
vCPU-second monthly free grant. Streamlit Community Cloud is free, purpose-built
for this, and requires only a public repo. The split is a cost decision.

**Why not AWS.** Accounts created after 2025-07-15 expire 6 months in and then
run on expiring credits. A portfolio link has to outlive the job hunt.

**Azure fallback.** Azure Container Apps offers an identical free grant
(180k vCPU-s / 360k GiB-s / 2M requests, non-expiring). If the target roles are
Azure-shops, swap Cloud Run -> Container Apps and Artifact Registry -> GHCR
(ACR has no free tier). Everything else in this document is unchanged; the
service is a plain container listening on `$PORT`.

---

## 2. The central problem: training/serving skew

This is the hardest correctness issue in the project and the reason the serving
layer is designed the way it is.

Five of the 18 serving features (20 before Sprint 3 drops the two zero-SHAP
origin features — see §11) are **stateful aggregates over destination account
history**, computed at training time as SQL window functions over the full
6.36M-row table:

| Feature | Training-time definition |
|---|---|
| `dest_prior_txn_count` | `COUNT(*) OVER w_dest` (rows strictly before this one) |
| `dest_prior_avg_amount` | `AVG(amount) OVER w_dest` |
| `dest_amount_to_prior_avg_ratio` | `amount / (prior_avg + 1)` |
| `dest_txn_count_24h` | `COUNT(*) OVER w_dest_velocity` (RANGE 24 PRECEDING) |
| `dest_amount_sum_24h` | `SUM(amount) OVER w_dest_velocity` |

At serving time a single incoming transaction has no window to look back
through. Recomputing them requires the destination's history, which lives in a
684MB DuckDB file that cannot ship in a container. **A model that silently
receives zeros for these five features at serving time is a broken model that
still returns plausible-looking scores** — the most dangerous failure mode in
the whole system, because nothing errors.

### Resolution: a bundled point-in-time state snapshot

Ship a read-only per-destination state table, built once at training time:

```
dest_state.parquet
  name_dest             VARCHAR   -- non-merchant destinations only
  prior_txn_count       INTEGER   -- cumulative, as of snapshot step
  prior_avg_amount      DOUBLE
  txn_count_24h         INTEGER   -- trailing 24h as of snapshot step
  amount_sum_24h        DOUBLE
```

Sizing (measured, not estimated):
- 571,961 non-merchant destinations need a row.
- 2,150,401 merchant (`M%`) destinations average **1.0005** transactions each,
  so their prior-history is ~always empty. They are **not stored**; they resolve
  to the cold-start default. This drops 79% of the rows for negligible fidelity
  loss.
- **Measured artifact: 5.68MB** parquet (zstd), built and sized against the real
  table on 2026-08-01. Ships inside the image.

### In-memory representation (this matters on 512MiB)

Do **not** load the parquet into a dict or a pandas DataFrame. 571,961 Python
dict entries cost >100MB of object overhead and would put a 512MiB Cloud Run
instance at genuine OOM risk alongside numpy and LightGBM.

Load instead as five parallel numpy arrays, sorted by a 64-bit hash of
`name_dest`, and look up with `np.searchsorted`:

```
keys   uint64[571961]   sorted blake2b-64 of name_dest
count  int32[571961]
avg    float32[571961]
c24    int32[571961]
s24    float32[571961]
```

**13.7MB resident** (measured: 571,961 x 24 bytes), O(log n) lookup, no per-row
Python objects. Hash collisions are
resolved by storing the collision set separately (expected count at 2^64 over
572k keys is effectively zero, but it is checked at build time and the build
fails rather than serving a silently wrong row).

### Cold-start policy (unknown destination)

Return `prior_txn_count=0, prior_avg_amount=0, txn_count_24h=0,
amount_sum_24h=0`, and let `dest_amount_to_prior_avg_ratio` degrade to
`amount / 1`. **This is not a fallback hack — it is exactly what the training
SQL produces** for an account's first transaction (`COUNT` over an empty window
is 0; `COALESCE(AVG(...), 0)`; the `+1` guard makes the ratio equal `amount`).
Training and serving agree by construction, which is the property that matters.

This case is also the common one, not the exception: 42.8% of rows have no
prior transaction to the same destination.

### Honest limitations, to be stated in the model card

1. The snapshot is **frozen at step 743** (end of dataset). A real system needs
   an online feature store updated per transaction; this is a point-in-time
   approximation and is labelled as such in `/model-info`.
2. Scoring the same transaction twice returns the same answer — the snapshot
   does not accumulate. The batch endpoint *does* accumulate within the
   submitted batch, which is the closer analogue of production behaviour.
3. Serving state is as-of a single instant, whereas training features were
   as-of each row's own timestamp. The two are only equivalent for
   transactions arriving after step 743.

---

## 3. Model bundle format

Serving does **not** load `*.joblib`. Pickled sklearn/LightGBM estimators are
tied to the exact library versions that wrote them, which turns every dependency
bump into a silent deserialization risk in production.

The training pipeline emits a self-describing, version-portable bundle:

```
model_bundle/
  model.txt          LightGBM native text format (booster_.save_model())
  scaler.json        {"feature_names": [...], "mean": [...], "scale": [...]}
  threshold.json     {"decision_threshold": ..., "reviews_per_day": 500,
                      "fold": 3, "expected_precision": ..., "expected_recall": ...,
                      "precision_ceiling": ...}
  dest_state.parquet per-destination state snapshot (section 2)
  bundle_meta.json   {"bundle_version", "feature_version", "git_commit",
                      "run_id", "model_name", "trained_at", "sha256": {...}}
```

Consequences, all deliberate:

- **`scikit-learn` is not a serving dependency.** `StandardScaler` is just
  `(x - mean) / scale`; exporting the two arrays to JSON and applying them in
  numpy is exact and removes ~100MB (sklearn + its scipy pull-in) from the image.
- **`shap`, `numba`, `llvmlite` are not serving dependencies** — and reason
  codes are still live, not precomputed. LightGBM implements TreeSHAP natively:
  `booster.predict(X, pred_contrib=True)` returns exact per-feature SHAP
  contributions plus a base value, using only LightGBM itself. The `shap`
  package (which drags in `numba` and `llvmlite`, the two largest entries in
  `requirements.txt`) is needed **only** for the offline global-importance
  study. This removes the need for a precomputed reason-code table entirely and
  makes `/score` genuinely explainable per request.
- Serving requirements collapse to: `fastapi, uvicorn, pydantic, numpy,
  lightgbm, pyarrow`. Target image < 400MB uncompressed.
- `bundle_meta.json` carries sha256 per file. The API verifies them at startup
  and refuses to serve a tampered or partial bundle.

### Dependency split

Three files, because three environments have genuinely different needs:

| File | Consumer | Contents |
|---|---|---|
| `requirements-train.txt` | laptop, CI smoke-train | everything today (incl. shap/numba/mlflow/optuna), plus `networkx`, `scipy.stats` for Sprint 11 |
| `requirements-serve.txt` | Docker image | `fastapi, uvicorn, pydantic, numpy, lightgbm, pyarrow`; **+ Sprint 10-12**: `psycopg[binary]`, `google-genai` |
| `dashboard/requirements.txt` | Streamlit Community Cloud | `streamlit, pandas, plotly, requests` |

The Sprint 10-12 serving additions are the only growth to the image, and both
are small pure-Python-plus-driver packages. The rule that `scikit-learn`,
`shap`, `numba`, `llvmlite`, `duckdb`, `mlflow`, `optuna` and `pandas` never
enter the serving path is unchanged and still enforced by the isolation job.

Versions are pinned identically across files where they overlap, and CI has a
job that installs **only** `requirements-serve.txt` and runs the inference tests
— the check that keeps a training-only import from sneaking into the serving
path. The dashboard file lives in `dashboard/` because Streamlit Community
Cloud resolves `requirements.txt` relative to the app directory.

### How the bundle reaches the image

`models/**` is gitignored (run-specific, regenerable). The bundle is not — it is
a **release artifact**, versioned deliberately:

```
model_bundle/v1/    <- committed, un-ignored
```

**Measured total ~7.3MB** (`model.txt` 1.57MB + `dest_state.parquet` 5.68MB +
three small JSON files), comfortably under GitHub's limits and small enough that
the repo stays self-contained: CI, `docker build`, and a fresh clone all work
with no external fetch and no credentials. If the bundle ever exceeds ~40MB,
move `dest_state.parquet` to a GitHub Release asset and have the Dockerfile
fetch it by tag — the loader already verifies sha256, so the integrity check is
unchanged.

A `.gitignore` negation (`!model_bundle/`) is required in Sprint 3; forgetting
it is exactly the class of defect the Sprint 1 audit caught (evidence cited but
untracked).

---

## 4. Inference core

A single module, `src/inference/`, is the **only** code path that turns a raw
transaction into a score. Both the API and the batch scorer import it. Nothing
reimplements feature construction.

```
src/inference/
  bundle.py     load + checksum-verify a model bundle
  state.py      destination state lookup (searchsorted), cold-start defaults
  features.py   raw transaction dict -> ordered 18-feature vector
  score.py      vector -> probability -> flag/no-flag against threshold
src/economics.py  capacity -> expected net value, optimum, sensitivity (§0)
```

`economics.py` is separate from `inference/` deliberately: it depends only on
counts and business rates, not on the model or the bundle, so it is unit-testable
in isolation and reusable against any model's capacity sweep.

`inference/features.py` must produce **bit-identical** output to the DuckDB
`feature_query()` for the same inputs. This is enforced by a golden-file test
(section 7), not by inspection. The 15 stateless features (amount, hour_of_day,
is_night, balance deltas/mismatch/emptied, ratio, dest_is_merchant, the four
type indicators, and the two dead origin features) are pure arithmetic on the
request payload; only the 5 destination features consult state.

---

## 5. API surface (FastAPI)

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Liveness. No dependencies touched. |
| GET | `/ready` | Readiness: bundle loaded, checksums verified, state resolvable. |
| GET | `/model-info` | Model name, bundle version, git commit, feature list, threshold, expected precision/recall + **the ceiling**, snapshot as-of step, known limitations. |
| POST | `/score` | Single transaction -> score, flag, threshold, reasons, state-hit indicator. |
| POST | `/score/batch` | Up to N transactions; aggregates accumulate **within** the batch. Returns per-row results + queue summary. |
| GET | `/metrics` | Request counts, latency percentiles, score histogram, flag rate. |

`/score` returns reason codes inline (top-N features by `|contribution|` with
raw values), computed live via LightGBM `pred_contrib=True`. There is no
separate `/explain` endpoint and no precomputed reason-code table — an earlier
draft of this design had both, before the native TreeSHAP path removed the need.

Design rules:

- **Every scoring response echoes `decision_threshold`, `model_version`, and
  `bundle_version`.** A score without the threshold that produced it is not
  auditable.
- **`state_hit: true|false` on every response.** The caller must be able to
  tell whether the five destination features came from the snapshot or from
  cold-start defaults. Silently defaulting is the failure mode from section 2.
- Pydantic v2 request models validate types, ranges, and `type` enum
  (`TRANSFER|CASH_OUT|CASH_IN|PAYMENT|DEBIT`). Reject, never coerce.
- Batch size capped (default 10,000 rows / 10MB) so a single request cannot
  exhaust a 512MiB instance.
- Errors return RFC 7807 problem+json, never a stack trace.

### Prediction audit log

Every scored transaction emits one structured JSON line to stdout (picked up by
Cloud Logging, 50GiB/month free):

```json
{"ts":"...","event":"prediction","request_id":"...","model_version":"...",
 "bundle_version":"...","threshold":2.74e-05,"score":0.83,"flagged":true,
 "state_hit":true,"feature_hash":"sha256:...","latency_ms":12.4}
```

Raw feature values are hashed rather than logged, so the log is a compliance
trail without becoming a PII store. This is the AML traceability requirement
from the original roadmap, satisfied at near-zero cost.

---

## 6. Dashboard (Streamlit)

Five pages, each answering a question a different stakeholder asks:

1. **Score a transaction** (analyst) — form -> score, flag, reason codes,
   `state_hit` badge, and the threshold it was judged against.
2. **Batch upload** (ops) — CSV in, scored queue out, download results.
   Bundled 50k-row stratified sample so the page works with no upload.
3. **Capacity & economics explorer** (fraud lead) — **the most important page in
   the project.** The Sprint 2 capacity sweep made interactive (move reviews/day,
   watch precision/recall/FP against the `total_fraud/K` ceiling), plus the §0
   expected-value curve layered on top: three adjustable business rates
   (cost per review, recovery rate, liability rate) drive a net-value curve with
   a marked optimum and a sensitivity band. Computed client-side from the
   committed capacity-sweep CSV — no API call, so it loads instantly and works
   while the API is asleep. If every other page is cut, this one ships.
4. **Model card** (governance) — features, SHAP global importance, ablation,
   per-fold metrics, known limitations, snapshot as-of.
5. **Drift** (MLOps) — PSI per feature and on the score distribution.

Constraints: 1GB RAM, sleeps after 12h idle (~30s wake), no secrets beyond the
API base URL. All heavy data is precomputed CSV/parquet in the repo — the
dashboard never loads the 6.36M-row dataset and never trains.

**Cold-start rule: the landing page must render entirely from committed CSVs
and must never call the API on load.** Streamlit Community Cloud sleeps after
12h idle (~30s wake) and Cloud Run scales to zero (~3-5s cold start). Chained,
a first-time visitor could wait ~35s staring at nothing. Pages 3-5 are fully
static and become the default view; the API is called only when the user
actively scores something, at which point a spinner with an honest "waking the
scoring service" message is correct rather than embarrassing. The capacity
explorer — the strongest page — therefore loads instantly.

---

## 7. Test strategy

Testing precedes serving in the sprint order because the refactor into
`src/inference/` is unsafe without it.

| Layer | What it pins |
|---|---|
| Unit — `features.py` | Each engineered feature against hand-computed values, including div-by-zero and first-transaction cases. |
| Unit — `custom_metrics.py` | Precision@K/Recall@K/weighted-BCE against hand-computed values; the float32 and groupby-scaling bugs found historically. |
| Unit — `threshold.py` | `window_days` endpoint inclusivity, `capacity_k` clamping, **the `n_flagged` > `k` tie path** (fixed in the Sprint 2 audit — it needs a regression test, currently exercised only by chance). |
| Unit — `cv.py` | Fold boundaries, no train/test overlap, chronological ordering. |
| Contract | Pydantic schemas: valid payload, each invalid field, boundary values. |
| **Skew — plumbing** | Feed the exact feature matrix from the training pipeline through `inference/score.py`; assert probabilities match `predict_proba` to 1e-9. Isolates model/scaler/threshold plumbing. |
| **Skew — state** | For a fixed step, assert `inference/state.py` returns values identical to the DuckDB window query for a sample of destinations. Isolates state resolution. |
| Golden file | 200 canned transactions -> expected scores, committed. Any change to features, bundle, or model that moves a score fails CI loudly. |
| Integration | Docker container up -> `/ready` -> `/score` -> assert schema + latency. |
| Data validation | pandera schema on ingest: dtypes, ranges, nullability, enum on `type`. |

The two skew tests are deliberately separate. A combined test would pass or fail
without telling you whether the model plumbing or the state lookup was wrong.

---

## 8. CI/CD

GitHub Actions, unlimited free minutes on a public repo.

```
PR:    ruff -> mypy(src/inference) -> pytest -> smoke-train on synthetic sample
main:  above -> docker build -> trivy scan -> push Artifact Registry
              -> deploy Cloud Run -> post-deploy smoke -> auto-rollback on fail
```

- **Smoke-train uses `src/generate_sample_data.py`**, not the real CSV — the
  493MB file is gitignored and CI must never need it.
- **Keyless auth** via Workload Identity Federation. No service-account JSON in
  repo secrets.
- **Artifact Registry cleanup policy**: keep the 3 most recent versions. The
  free tier is 0.5GB and a ~150MB compressed image would breach it by the 4th
  build.
- Deploy is gated on `/ready` returning 200 with matching `bundle_version`;
  otherwise traffic stays on the previous revision.

### CI is also the only container runtime

Verified 2026-08-01: `docker`, `gh` and `gcloud` are absent from the dev machine.
Docker Desktop's WSL2 backend costs ~2GB idle on a dual-core/8GB box, which this
project cannot spare while training. So **the image is never built locally** —
CI is the build environment *and* the integration-test environment. Local
development runs `uvicorn` and `streamlit` as plain processes.

This is a real constraint with a real cost: container defects surface only on a
CI round-trip (~3-5 min), never at a local prompt. It is accepted deliberately
because the alternative is a permanent 2GB tax on a machine that peaks at
1,367MB during training and has 8GB total.

`gcloud` is installed in Sprint 7 — it has no daemon and is cheap, unlike Docker.

---

## 9. Cost controls (non-negotiable)

A public URL on a billing-enabled account is a financial risk. All of these
ship with the first deploy, not later:

| Control | Setting |
|---|---|
| `--max-instances` | **2** — hard ceiling on concurrent billing |
| `--min-instances` | 0 — scale to zero; cold start accepted |
| `--memory` / `--cpu` | 512Mi / 1 |
| `--concurrency` | 80 |
| `--timeout` | 30s |
| Budget alert | $1/month, email at 50%/90%/100% |
| App rate limit | per-IP token bucket in middleware |
| Request size cap | 10MB |
| Region | `us-central1` (Always Free eligible) |

Expected steady-state cost: **$0**. Portfolio traffic is far below 2M
requests/month, and 180,000 vCPU-seconds is ~50 hours of active request time —
unreachable with scale-to-zero and sub-second requests.

---

## 10. Monitoring

### Drift is measured over dataset time, not demo traffic

An earlier draft specified PSI against "rolling scored traffic." A portfolio demo
receives tens of requests; PSI on that sample is statistically meaningless and
the job would have produced noise dressed as monitoring.

The real analysis uses the data that exists. PaySim spans 743 hourly steps, so:

- **Reference**: the final training fold's feature distribution.
- **Comparison**: successive time windows of the *dataset itself*, replayed as
  if they were arriving. This produces a genuine PSI-over-time series on 6.4M
  real rows and can show whether PaySim's own generative process shifts.
- **Thresholds**: PSI > 0.25 on any feature, > 0.10 on the score. These are
  conventional starting points, labelled as such.
- **Detector unit test**: deliberately injected shifts (mean shift, variance
  shift, category re-weighting) assert the detector fires. That is what proves
  the implementation, not the demo traffic.

Live scored traffic is still logged to Postgres and surfaced, but as **volume and
score-distribution telemetry**, not as a drift signal — the sample is too small
and the distinction is stated on the dashboard rather than blurred.
- **Retraining trigger**: documented criteria (PSI breach, precision-at-capacity
  drop, or 90 days elapsed). The *criteria* are the deliverable; automated
  retraining is out of scope on free tier and is stated as such rather than
  faked.
- **Cloud Scheduler** free tier (3 jobs/month) runs the drift job; results land
  in a CSV the dashboard reads.
- **Champion/challenger**: `/model-info` exposes `bundle_version`, and the
  deploy flow supports Cloud Run revision traffic splitting, so a challenger can
  take 10% before promotion.

---

## 11. Decisions deliberately deferred

Recorded so they are visible choices, not oversights:

1. ~~The two dead features stay in v1.~~ **Reversed on review — they are removed
   in Sprint 3.** `orig_prior_txn_count` and `orig_prior_avg_amount` have exactly
   zero SHAP attribution. The original argument for keeping them (comparability
   with the Sprint 1 baseline) applies to the *research narrative*, not to the
   *deployed artifact* — and shipping a production model with two provably inert
   inputs is not defensible in a project claiming production readiness. Cost is
   one 28-minute retrain plus re-verification of the published numbers. The
   before/after is itself the better story: profiling predicted them dead, SHAP
   confirmed it, they were removed, and the metrics did not move. Serving
   feature count becomes **18**.
2. **No hosted online feature store.** Section 2's snapshot is the documented
   approximation; a real store (Redis/Bigtable) has no free tier that survives.
   `/score/batch` demonstrates incremental state within a batch instead, and §16
   documents the scaling path that would force a real store.
3. **Model registry**: MLflow local + the versioned, checksummed bundle. An
   earlier draft added Vertex AI Model Registry; it was **cut on review as
   theatre** — registering a model in Vertex while serving it from Cloud Run and
   using no other Vertex capability is a keyword, not an architecture, and an
   interviewer would ask "why is it there?" with no good answer available. The
   bundle + MLflow story is coherent on its own.
4. **Threshold is fold-3-specific.** Per the Sprint 2 audit, the capacity metric
   is not comparable across folds. `/model-info` states which fold produced the
   threshold; the dashboard lets users move it rather than presenting one number
   as universal.

---

## 12. Feasibility validation (run 2026-08-01, before Sprint 3)

This design makes four claims that would be expensive to discover were false in
Sprint 4. All four were tested against the real trained artifacts in
`models/20260728T172950Z/` before the plan was accepted.

| Claim | Result |
|---|---|
| `StandardScaler` is reproducible in pure numpy, so `scikit-learn` can leave the serving image | **max abs diff 0.000e+00** (exact) |
| LightGBM native text round-trip preserves predictions, so `joblib` pickles can leave the serving path | **max abs diff 0.000e+00** (exact); `model.txt` = 1.57MB |
| `pred_contrib=True` yields exact TreeSHAP without the `shap` package | contributions sum to raw margin, **max abs diff 8.08e-14**; shape (n, 21) = 20 features + base value |
| `sigmoid(raw_margin)` equals `predict_proba` | **max abs diff 0.000e+00** (exact) |
| `dest_state` snapshot is small enough to bake into the image | **5.68MB** parquet / **13.7MB** resident, both under the planned budget |

Measured against the current 20-feature model in `models/20260728T172950Z/`;
Sprint 3's 18-feature retrain re-runs the same checks (the contribution matrix
becomes (n, 19)).

Consequence: the serving image needs none of `scikit-learn`, `shap`, `numba`,
`llvmlite`, `duckdb`, `mlflow`, `optuna`, or `pandas`. The Sprint 3 DoD asserts
this by building a throwaway venv from `requirements-serve.txt` alone.

The claims **not** yet validated, and where they get tested:

- Feature-construction parity between `inference/features.py` and the DuckDB
  `feature_query()` — Sprint 4 skew tests. This is the highest remaining risk in
  the project.
- Cloud Run cold-start latency with this image — measured in Sprint 7 and
  recorded in the README rather than predicted here.
- PSI thresholds (0.25 feature / 0.10 score) are conventional defaults and will
  be recalibrated against observed traffic in Sprint 8.

---

## 13. Persistence, auth & the feedback loop (Sprint 10)

Added after a market-alignment review: the design had no database at all, which
is why the analyst feedback loop was originally scoped out.

**Neon Postgres** (free: 0.5GB, 100 CU-hours/month, scale-to-zero, no idle
pause). Two tables:

```sql
predictions(                        analyst_feedback(
  id, ts, request_id,                 prediction_id -> predictions.id,
  model_version, bundle_version,      label,          -- CONFIRMED_FRAUD | FALSE_POSITIVE
  threshold, score, flagged,          analyst, ts, note
  state_hit, decision,              )
  feature_hash, latency_ms
)
```

The stdout JSON audit line stays (Cloud Logging is the cheap, always-on trail);
Postgres becomes the queryable store that makes the loop possible. Writes are
fire-and-forget with a bounded queue — **a database outage must never fail a
scoring request**, since the model's answer does not depend on the log.

**Auth**: API-key header with per-key rate limits, replacing the per-IP bucket.
Keys live in Secret Manager (6 free secret versions). The dashboard holds a
read/score key; the feedback endpoint requires a separate write key.

**The loop**: dashboard review queue -> analyst marks an alert -> row in
`analyst_feedback` -> a labelled set accumulates that a retraining run can
consume. The retraining *execution* stays out of scope; the labelled data and
the documented path are the deliverable.

---

## 14. Graph analytics (Sprint 11)

Sprint 2 established that **origin-side** graph features are impossible here —
`nameOrig` is 99.85% unique, so fan-out is degenerate and per-origin velocity is
zero for almost every row. That finding stands and is not being reversed.

The **destination side** supports a real graph. Build the bipartite
origin -> destination transaction graph over the 571,961 non-merchant
destinations (up to 113 transactions each) and derive:

- `dest_component_size` — connected-component size containing the destination
- `dest_degree` / `dest_weighted_degree`
- `dest_flagged_neighbours` — count of same-component destinations already
  flagged, the direct analogue of a **mule network**
- `dest_component_fraud_rate` — computed **train-fold-only** to avoid leakage

Leakage is the live risk here: any component-level statistic computed over the
full graph leaks test-window information backwards. All component statistics are
built from training-fold edges only, and the existing correlated-self-join
verification approach is extended to cover them.

Attribution is via the same 4-arm ablation pattern Sprint 2 used, so "graph
features helped" is measured, not asserted. Given Sprint 2's finding that the
feature set is already near-saturated (PR-AUC 0.9995 before Sprint 2's
additions), **a null result is a likely and acceptable outcome** and will be
reported as one.

`networkx` is training-only. Graph features resolve at serving time from the
same state snapshot mechanism as §2.

---

## 15. GenAI: network-level STR narrative generation (Sprint 12)

AML analysts must write a Suspicious Transaction Report narrative for every
escalated case. Drafting it from structured evidence is one of the few genuinely
production-grade GenAI use cases in this domain.

### Why this operates at network level, not transaction level

An earlier draft generated a narrative per *transaction*, from 20 numeric
features. Reviewing it honestly: **an LLM adds almost nothing over a string
template for one row of numbers.** The groundedness check would have been
verifying that the model did a template's job. That version was theatre.

The version worth building operates on the **account and its network**, which is
where the substrate actually is and where a template genuinely struggles:

**Input** (assembled from Sprint 11's graph output plus existing artifacts):
- the destination account and its inbound pattern — count of distinct
  originators, time span, total and per-transaction amounts
- graph context: component size, flagged neighbours, component fraud rate
- how many contributing originators were themselves emptied
- the flagged transactions in the case, with their scores and SHAP drivers

That produces narratives a template cannot easily write, because the content is
a *synthesis*: "Account C4471 received 47 inbound transfers from 47 distinct
originators over 18 hours totalling 12.4M. 31 of those originators were emptied
by the transfer. The account sits in a 63-node component with 12 previously
flagged destinations. This pattern is consistent with fan-in mule layering."

That is a real AML typology assessment, and it requires joining velocity, graph
and model evidence — exactly the kind of multi-source summarisation LLMs are
actually good at. **This is why Sprint 11 (graph) must precede Sprint 12.**

**Output**: a Pydantic-validated structure —
`{summary, typology, evidence[], recommended_action, confidence}` — not free
text. Structured output is what makes it testable.

**Model**: Gemini 2.5 Flash free tier (~15 RPM, ~1,000 RPD, no card required).

**Not real-time.** Case narratives are an analyst workflow, not a scoring-path
concern. The endpoint is explicitly asynchronous-friendly and rate-limited; it
is never called during `/score`. A free tier of ~1,000 requests/day is ample for
a workflow where a human reads every output.

### Guardrails, which are the actual engineering

1. **Groundedness check**: every numeric value in the generated narrative must
   appear in the input payload. A regex/set comparison rejects the draft
   otherwise. This is the anti-hallucination control and it is enforced in code,
   not in the prompt.
2. **No decisioning**: the LLM narrates an alert the model already produced. It
   never influences the score or the flag. Stated explicitly in the model card.
3. **Fallback**: on API error, rate-limit, or failed groundedness, return a
   deterministic template-filled narrative. The endpoint never fails because an
   LLM was unavailable.
4. **Evaluation**: a fixed set of ~30 alerts scored for groundedness, schema
   validity, and completeness — an eval harness, not vibes.
5. **Cost/latency**: narratives are generated on demand, never in the scoring
   path. `/score` latency is unaffected.

### Data-governance caveat (important, and stated in the model card)

Google's free tier explicitly may use submitted requests to improve its models.
That is acceptable here **only because PaySim is synthetic** — there is no real
customer data anywhere in this project. In a real AML deployment this feature
would require a VPC-hosted or enterprise-tier model with a no-training
guarantee. Documenting that boundary is the point: it demonstrates awareness
that "call an LLM API" is not a deployable answer in regulated finance.

---

## 16. Scaling path — and why streaming was cut

An earlier draft of this plan had a Sprint 13 that streamed a replay into the
deployed API with "rolling online state." **That design does not work on this
architecture, and the review that caught it is worth recording.**

Cloud Run runs 0-2 stateless instances that scale to zero. In-process
accumulating state would therefore (a) diverge between the two instances, (b)
vanish on scale-to-zero, and (c) give different answers depending on which
instance served the request. It would have been a demonstration of a bug. Moving
the state to Postgres means a write per transaction against a 100 CU-hour/month
free tier, replaying ~1.2M rows — neither cheap nor fast.

The concept it was meant to show — incremental feature state — is already
demonstrated honestly by `/score/batch`, which accumulates destination
aggregates **within** a submitted batch (§5). That is the same idea at a scale
the architecture actually supports. The sprint was cut; this section replaces it.

### What changes as volume grows

| Scale | Binding constraint | Change required |
|---|---|---|
| **Today** (~6.4M rows offline, demo traffic) | None | Current design |
| **10x** (~65M rows, low real-time traffic) | Feature build time; state snapshot ~60MB | Keep DuckDB; move `dest_state` out of the image to Cloud Storage, memory-mapped |
| **100x** (~640M rows, sustained traffic) | Snapshot no longer fits in RAM; staleness becomes material | Real online store (Redis/Bigtable); feature writes become a separate ingestion path; Cloud Run min-instances > 0 |
| **1000x** (real payments volume, sub-second SLA) | Single-node feature computation | Streaming ingestion (Kafka/Pub-Sub + Flink/Beam) writing to the online store; model serving separates from feature serving; multi-region |

The consistent theme: **the model server is the easy part to scale; the feature
state is the hard part.** That is the honest answer to "how would you scale
this?", and it is more useful than a Kafka container that was never load-tested.

### Why no Kafka anywhere

It has no free tier that survives, and at the scale this project actually
operates it would be a service added to be named rather than used. The table
above states where it enters and what forces it. An unused broker in a
`docker-compose.yml` is not evidence of streaming experience.
