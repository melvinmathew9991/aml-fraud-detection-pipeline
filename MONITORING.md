# Monitoring & Drift (Sprint 8)

What is watched after deployment, what each signal is evidence of, and what
would actually trigger a retrain. Design rationale lives in `ARCHITECTURE.md`
§10; this file is the runbook and the record of what the job measured.

---

## 1. What is measured, and what it is evidence of

| Signal | Source | What it is evidence of | Where |
|---|---|---|---|
| **Feature PSI** | `src/run_drift.py` over the dataset's 743 simulated hours | An *input* distribution moved | `data/processed/drift_feature_psi.csv` |
| **Score PSI** | same job | The *model's output* distribution moved | `data/processed/drift_score_psi.csv` |
| **Recall at capacity** | same job | Whether fraud is still landing inside the queue analysts can work — the model-quality signal | same CSV |
| **Alerts per day vs capacity** | same job | Whether the fixed threshold is producing a workable queue | same CSV |
| **Live volume / latency / flag rate** | deployed API `/metrics` | The service is up and being used | Dashboard page 5 |

The last row is deliberately not a drift signal, and the dashboard says so on
the page. A portfolio endpoint serves tens of requests; a distribution estimated
from tens of requests is noise. ARCHITECTURE.md §10 records why that framing was
rejected before it was built rather than after.

**PSI thresholds are conventions, not calibration.** PSI < 0.10 stable,
0.10–0.25 moderate, ≥ 0.25 significant are the standard credit-risk bands. They
were not fitted to this data and nothing here claims they are the right numbers
for it. Features are called at 0.25; the score series is called at the tighter
0.10, because a shift large enough to move the model's output has already moved
something upstream.

---

## 2. Running it

```bash
python tasks.py drift          # or: python src/run_drift.py
```

Reads the `features` table `train_pipeline.py` materializes into
`data/processed/paysim.duckdb`, scores all 6,362,620 rows with `model_bundle/v1`
through `src/inference/`, and writes three artifacts under `data/processed/`:

| File | Shape |
|---|---|
| `drift_feature_psi.csv` | one row per (window, feature) — 558 rows |
| `drift_score_psi.csv` | one row per window — score PSI plus the operating point |
| `drift_reference.json` | what the reference was, so a stale run is detectable |

**Measured cost**, across three runs on 2026-09-10 on the 8GB dual-core reference
machine: **2 min 15 s to 5 min 18 s** end to end, peak RSS **643–751 MB**.

Reporting a range rather than the first run's figure, because the spread is not
noise: almost all of it is the DuckDB feature fetch (44 s to 4 min), which
depends on whether the 796 MB store is already warm in the OS page cache. The
scoring stage is stable at 49–71 s for all 6,362,620 rows. It runs in 500k-row
chunks because `scaler.transform` promotes to float64, and one call over the full
matrix would allocate ~915 MB for the scaled copy alone. Per-run timings and
memory land in `reports/drift_<run_id>.log`, which is generated, not committed.

The job does **not** build the feature table. Feature construction is a
6.36M-row window-function query owned by the training pipeline, and a second
place that interprets `FEATURE_VERSION` is a second place for it to go stale. If
the table is missing or was built at a different feature version, the job exits
with the instruction to re-run training.

**Reference = the final training fold**, taken by calling `src/cv.py`'s own fold
builder rather than re-deriving the 0.8 quantile: steps 1–355, 5,113,884 rows,
3,963 fraud. **Comparison = each successive simulated day**, 24 steps per window,
31 windows covering steps 1–743.

Windows start at step 1, not after the reference cut. The first 14 windows sit
*inside* the reference period, so their PSI is the detector's noise floor
measured on the data the reference was built from rather than asserted.

---

## 3. What the job found (run 2026-09-10)

**The noise floor is real and low.** Windows 5–13 — nine consecutive high-volume
windows inside the reference period — score PSI 0.0005–0.0048 with **no feature
breaching at all**. That is the control: a number near zero on data drawn from
the reference is the correct answer, and getting it is what makes the non-zero
numbers elsewhere worth reading.

The two remaining high-volume reference windows, 0 and 1, breach only the two
velocity features and only just — PSI 0.29–0.43 against a 0.25 band. They are the
busiest windows in the dataset (574,255 and 455,238 rows), so destination
velocity runs about three times the reference mean (3.78 and 3.16 against 1.26).
That the same two features breach at *both* ends of the volume range, in opposite
directions, is what identifies volume rather than elapsed time as the driver.

**The reference is in-sample, and that is worth stating.** The reference score
distribution is produced by scoring the training fold with a model fitted on that
fold, so it is sharper than the model's output on unseen data — which makes the
in-sample noise floor flatter than an honest out-of-sample one. The size of the
effect is measurable here, because windows 14–16 are high-volume windows just
past the reference cut: they run score PSI 0.0073–0.0078, mean 0.0075, against
0.0005–0.0048, mean 0.0027 for the nine in-sample control windows — **2.8x
higher**. Both sit an order of magnitude below the 0.10 band, so no conclusion on
this page changes; but "the noise floor is 0.0005–0.0048" is an in-sample
statement, and the out-of-sample floor is about 0.0075.

The alternative — holding out a slice of the training fold purely to serve as a
drift reference — would cost model quality to buy a cleaner monitoring baseline,
which is the wrong trade at this scale. Naming the bias is the cheaper fix.

Note also that "the final training fold" here means the fold's **full**
population, 5,113,884 rows. The model was fitted on an undersampled 50:1 draw
from it (`config.yaml`, `sampling`). Full population is the right reference for
drift — it is the distribution production actually sees — but it is not
literally the set of rows the model was shown.

**Three features carry nearly all the drift, and volume is why.**

| Feature | Windows breaching 0.25 (of 31) |
|---|---|
| `dest_txn_count_24h` | 20 |
| `dest_amount_sum_24h` | 17 |
| `hour_of_day` | 16 |
| `is_night` | 3 |
| everything else | ≤ 2 each |

The two velocity features count what reached a destination account in the
previous 24 simulated hours, so they are functions of transaction volume by
construction. Volume across the full windows spans **1,070 to 574,255 rows**, and
the share of rows with no 24-hour destination history runs from **48.6%** in the
busiest window to **99.0%** in a collapsed one, against a reference share of
**58.7%**. `hour_of_day` moves for the same reason: the busiest six hours hold
**49.8%** of rows in the reference and **91.3%** in a collapsed window — activity
concentrates into fewer hours when there is less of it.

This is drift in PaySim's generative process, not in a payment population. It is
reported as what it is.

**Feature drift and score drift come apart cleanly.** In windows 17–29 — the
low-volume tail, entirely outside the reference period — three to five features
breach in every window while score PSI never exceeds **0.0906**, under the 0.10
band. The model's output is not sensitive to the input shift the feature
detector is correctly reporting.

**The three real score breaches are inside the training reference.** Windows 2–4
(steps 49–120) are PaySim's early collapse. Window 2 is 1,070 rows of which 310
are fraud — a **29.0%** fraud rate against 0.047–0.068% in the high-volume
windows around it. The simulation has stretches where little but the fraud agent is
active. That is a property of the generator, it sits inside the data the model
was trained on, and it is not degradation.

**The undersized-window rule earns its place immediately.** Window 30 (steps
721–743) is 272 rows, all of them fraud, and scores PSI **5.57** — the largest
number the job produces. It is reported and plotted, and it raises no flag,
because 272 rows cannot support a distributional claim. Suppressing it would
have hidden a real property of the data; flagging it would have made the
retraining criteria unusable.

**The largest effect on this page is not drift at all.** At the bundle's fixed
decision threshold the queue runs **272 to 4,594 alerts per day** against a
configured capacity of 500, exceeding capacity in **14 of the 30 full windows** —
every one of them a high-volume window, and none of them a window where drift
fired. A score threshold fixes a score, not a queue length. This is the
operational consequence of the fold-3-specific threshold `ARCHITECTURE.md` §11
already lists as a limitation, now measured across the whole dataset instead of
argued.

**And the model itself is not degrading anywhere.** Recall at capacity is **1.000
in all 31 windows**, and precision at capacity equals its own ceiling in **all 31**.
Precision at capacity runs 0.432–0.640 with a mean of **0.529** across the full
windows, against the bundle's shipped `expected_precision` of **0.5257** — which
was measured on fold 3 alone. The shipped operating-point estimate generalises
across windows whose volume differs by more than 500x.

---

## 4. Retraining triggers

The criteria are the deliverable. **Automated retraining execution is out of
scope on free tier** and is stated as such rather than faked — there is no
scheduled trainer, and `ROADMAP.md` §4 lists it under "explicitly out of scope".

A retrain is warranted when **any** of these holds:

**A. Ranking degradation — the primary trigger.**
`recall_at_capacity < 1.0` on three consecutive windows with a usable sample
(≥ 1,000 rows). Today it is 1.000 in every window. Falling below it means fraud
has started escaping the queue analysts can actually work, which is the failure
that matters and the one no other signal here detects.

**B. Sustained score drift.**
`score_psi ≥ 0.10` on three consecutive windows with a usable sample.
*Consecutive* is load-bearing: single-window breaches on this dataset are volume
artifacts — windows 2, 3 and 4 all breach, and all three are PaySim's early
collapse inside the training reference.

**C. Sustained feature drift outside the known volume-driven set.**
Any feature other than `dest_txn_count_24h`, `dest_amount_sum_24h` and
`hour_of_day` breaching 0.25 on three consecutive usable windows. Those three are
excluded because their drift is measured, explained and expected (§3); including
them would make this criterion fire permanently and therefore mean nothing.

**D. Elapsed time.**
90 days since the bundle's `trained_at`, regardless of the above. A model nobody
has re-examined in a quarter is a model nobody can vouch for.

### Precision at capacity is deliberately *not* a trigger

It looks like the obvious health metric and it is not one here. Precision at
capacity is bounded above by `fraud_count / queue_size`, and this model sits
exactly on that bound in all 31 windows. So the number reports **how much fraud
occurred that day**, not how well the model ranked it: a quiet fraud day lowers
precision at capacity without anything about the model changing. Its useful form
is the *boolean* — `at_ceiling`, which is what trigger A tests through recall.

Precision at the *deployed fixed threshold* is a worse candidate still: it swings
0.059 to 0.968 across these windows, driven almost entirely by transaction
volume. It is in the CSV because it describes what the review queue would really
have looked like, not because it is a health signal.

**Both label-based criteria are hindsight measures.** A live system does not know
the labels on the day. They are evaluated when labels arrive — which is exactly
what the Sprint 10 analyst-feedback loop exists to provide, and why the trigger
criteria and not an automated trainer are the deliverable here.

---

## 5. Champion / challenger

Cloud Run revision traffic splitting is the promotion path. No A/B harness is
built; the mechanism is the deploy pipeline that already exists.

1. **A challenger deploys with no traffic.** `.github/workflows/ci.yml`'s deploy
   job already does exactly this — `gcloud run deploy --no-traffic
   --tag=candidate` — so a challenger is addressable at its own tagged URL while
   the champion keeps 100% of traffic. This is not new machinery for
   champion/challenger; it is the same mechanism that makes a failed deploy a
   no-op (`DEPLOY.md` §7.3).
2. **Verify on the tagged URL.** `/ready` must report `status: ready` and the
   expected `bundle_version`; `/score` must return a decision. Both already run
   against the candidate before any traffic moves.
3. **Split traffic.**
   ```bash
   gcloud run services update-traffic fraud-api --region=us-central1 \
     --to-revisions=CHALLENGER=10,CHAMPION=90
   ```
4. **Compare.** `/model-info` exposes `bundle_version`, so every scored response
   is attributable to a revision, and the prediction audit log
   (`src/api/audit.py`) records which. At portfolio traffic volume this
   comparison is a *mechanism demonstration*, not a powered experiment — the
   sample is nowhere near enough to separate two models, and claiming otherwise
   would be the same mistake as running PSI on demo traffic.
5. **Promote or roll back.** `--to-revisions=CHALLENGER=100`, or drop the
   challenger's share to zero. The rollback drill in `DEPLOY.md` §7.3 found
   something stronger than rollback: a revision that never took traffic has
   nothing to undo.

---

## 6. The scheduled check — and why it is not Cloud Scheduler

`ROADMAP.md` Sprint 8 planned Cloud Scheduler (1 of 3 free jobs) to run the
drift job. **That was reconsidered during implementation and changed.** The
reasoning, recorded because the change is a deviation from the plan:

**The drift analysis is deterministic over a fixed historical dataset.** PaySim's
743 steps do not change. Re-running the PSI job on a schedule returns a
byte-identical answer every time. Scheduling that computation would produce a
cron entry, a green checkmark, and no information — the same class of thing as
running PSI over demo traffic, which this sprint's design already rejected.

**And it could not run in the cloud anyway.** The comparison data is the 493 MB
raw dataset, which is gitignored and not in the serving image — deliberately: the
image is 518.9 MB compressed to 172.7 MB and Artifact Registry's free tier holds
two versions (`GCP.md` §6). Getting the data to a scheduled cloud job would mean
Cloud Storage plus a Cloud Run job, i.e. new billing surface for a job whose
answer never changes.

**What does change on a schedule is the deployed service**, so that is what the
scheduled job watches. `.github/workflows/monitoring.yml` runs daily and:

1. Runs `tests/test_drift.py` — the injected-shift suite, so a detector that
   silently stopped firing is caught.
2. Probes the live service: `/health`, `/ready`, `/model-info`, `/metrics`.
3. **Checks the drift reference is not stale.** It compares `/model-info`'s
   `bundle_version`, `feature_version`, `decision_threshold` and feature list
   against `data/processed/drift_reference.json`. If the deployed model moves and
   the reference does not, every PSI number on the dashboard describes a model
   nobody is serving. That is a real failure with a real trigger, and it is the
   reason this job exists.
4. Writes the live telemetry snapshot to the run summary.

`GCP.md` §5④ had already anticipated this option — "alternatively move it to a
GitHub Actions cron schedule, which is free for public repositories and removes
the GCP dependency entirely". The consequence is that Sprint 8 consumes **0 of
the 3 free Cloud Scheduler jobs** and adds no GCP billing surface at all.

The workflow is inert until configured, the same pattern the deploy pipeline
uses: with the `CLOUD_RUN_URL` repository variable unset, the probe steps skip
and the detector tests still run.

---

## 7. What this does not cover

Named so they read as decisions rather than gaps:

- **No automated retraining.** §4 is criteria only. Out of scope on free tier
  (`ROADMAP.md` §4), and stated rather than faked.
- **No alerting channel.** A breach shows on the dashboard and, for the service
  checks, as a failed scheduled workflow. No Slack webhook — it adds a secret to
  manage and demonstrates nothing new (`ROADMAP.md` §4).
- **No drift on live traffic.** By design, measured and explained above.
- **No persisted telemetry history.** `/metrics` is per-instance and in-memory;
  Cloud Run scales to zero, so it resets on every cold start. Persisting it needs
  the database Sprint 10 introduces.
- **Concept drift is not separated from covariate drift.** PSI measures input and
  output distributions. Distinguishing "the population changed" from "the
  relationship between features and fraud changed" requires labels at a latency
  this dataset cannot simulate honestly.

---

## 8. Sprint 8 audit (2026-09-10)

Run end to end before the sprint was committed, in the manner of `AUDIT.md`.
**Verdict: three defects found, all fixed; two limitations documented rather
than fixed; no numeric claim in the docs contradicted its artifact.**

### 8.1 Method

Nine checks, all executed rather than reasoned about:

1. **Dead code / call-site sweep** over `monitoring/drift.py` — every public and
   private function traced to a caller.
2. **PSI verified against an independent implementation** — a separate textbook
   version (fixed decile cuts, plain share arithmetic) plus one hand-computed
   two-bin case.
3. **Mutation testing of the detector** — eight deliberate breaks applied to
   `drift.py`, suite run against each.
4. **Mutation testing of the artifact guards** — two `config.yaml` mutations.
5. **Dependency isolation** — the scheduled job's detector step run in a venv
   containing only `numpy` and `pytest`.
6. **The stale-reference check run against the live deployment**, not a fixture.
7. **Determinism** — the job re-run and its outputs byte-compared.
8. **Internal consistency** — the two notions of "window length" reconciled
   against the data.
9. **Every numeric claim in README / ROADMAP / ARCHITECTURE / this file
   re-derived from the committed CSVs.**

### 8.2 Defects found and fixed

**8.2.1 The score threshold was decided by a line no test could reach.**
`run_drift.py` passed `threshold=PSI_SCORE_THRESHOLD` at its own call site.
Changing that one line to the feature threshold would have produced a
**byte-identical** `drift_score_psi.csv` and a fully passing suite, because no
window in this dataset has a score PSI between 0.10 and 0.25 — the largest
non-breaching value is 0.0906 and the smallest breach is 0.2652. The band that
separates "the model's output moved" from "it did not" was therefore unguarded.
Fixed by adding `monitoring.drift.compare_score`, which names the threshold once
on the shipped path, and `test_compare_score_is_the_path_the_job_takes`. The
regenerated artifacts are byte-identical, confirming a pure refactor.

**8.2.2 The scheduled job invoked `python`, which the runner may not provide.**
The staleness check ran `python - <<'PY'`. That step deliberately runs no
`actions/setup-python`, and the Ubuntu runner image ships no unversioned
`python`. The monitoring job would have failed on its first scheduled run —
and a monitoring job that fails for its own reasons is worse than none, because
the failure looks like the thing it was watching. Changed to `python3`.

**8.2.3 The same job installed unpinned `numpy`.** Every other dependency in
this repo is pinned; a job whose purpose is to still work in six months had a
floating one. Pinned to `numpy==2.4.6`, matching `requirements-train.txt` and
`requirements-serve.txt`. The pip cache was also keyed on requirements files the
job does not install from, and was removed rather than left as decoration.

### 8.3 Test gaps closed

- **Nothing pinned the reference to the fold config actually in use.**
  `run_drift.py` derives the cut by calling `src/cv.py`, which is right, but
  re-cutting `cv.fold_boundaries` and not re-running the job would leave a
  reference describing a fold that no longer exists. Added
  `test_the_reference_is_the_fold_config_actually_defines`; verified it fails
  under a mutated `config.yaml`.
- **Nothing pinned the capacity arithmetic.** Added
  `test_capacity_is_the_configured_staffing_applied_to_the_window`, likewise
  mutation-verified. It also guards the latent coupling in §8.4.2.
- **`in_reference_period` was not required to contain both classes**, so a
  reference covering every window (or none) would have passed silently. Added.

### 8.4 Limitations documented, not fixed

**8.4.1 The reference is in-sample.** Covered in §3 with its measured size
(2.8x). Fixing it would mean holding rows out of training to serve monitoring,
which costs model quality to buy a cleaner baseline — the wrong trade here.

**8.4.2 Two notions of window length coexist.** `window_days` in the CSV is the
window's nominal step span; `capacity` comes from `threshold.capacity_k`, which
measures the span of the steps actually present. **Verified equal for all 31
windows** — every window has rows at both edges — so there is no current impact,
and the new capacity test fails the moment that stops being true. Not unified,
because `capacity_k` is Sprint 2 code with its own defensible semantics and
churning it to remove a coupling that no longer bites is not an improvement.

**8.4.3 `drift_reference.json` records the commit it was generated at**, which
is necessarily the pre-commit state (`-dirty`, or the parent). The same is true
of `model_bundle/v1/bundle_meta.json` (`b874804-dirty`). An artifact cannot
record the commit that contains it.

### 8.5 What held up

- **PSI arithmetic is exact.** Four distribution pairs matched an independent
  implementation to 0.00e+00, and a hand-computed two-bin case matched to ten
  decimals.
- **The detector's tests have teeth.** All eight mutations were caught: removing
  the open-ended bins, swapping the half-observation floor for a fixed epsilon,
  narrowing the categorical cutoff, dropping the unseen-level bin, letting
  undersized windows breach, keeping both-empty bins, substituting KL for the
  symmetric form, and collapsing the two thresholds.
- **The detector really is numpy-only.** `pytest tests/test_drift.py` passes in a
  venv holding only `numpy` and `pytest` — verified, not asserted.
- **The stale-reference check works against the real deployment.** Run against
  `https://fraud-api-amj2cl4jhq-uc.a.run.app/model-info`: the served bundle,
  feature version, feature list and decision threshold all match the committed
  reference, and the float comparison holds **exactly** — so the strict equality
  the check uses will not false-alarm on serialisation.
- **The job is deterministic.** Re-run produces byte-identical CSVs; the manifest
  differs only in `generated_at` and `run_id`.
- **Every documented number re-derives from the artifacts.** Thirty-one
  assertions over the committed CSVs, all passing.

### 8.6 What this audit did not cover

- **The scheduled workflow has never executed on GitHub.** Its YAML parses, its
  heredoc terminates at column 0, its Python and staleness logic were run
  locally, and its detector step was run in an isolated venv — but cron
  workflows only run from the default branch, so the first real execution will
  be after merge. Treat the first scheduled run as the verification step, the
  way `DEPLOY.md` §0 treats the first deploy.
- **`smoke-train` was not run locally.** It regenerates the synthetic sample over
  `data/raw/paysim_transactions.csv`, the path holding the real dataset. CI runs
  it; this machine must not.
- **The dashboard was verified with Streamlit's `AppTest`, not a browser.** All
  three charts, both tables and all seven metrics render with no exception, but
  Sprint 7 is the reminder that Community Cloud can differ from local — it found
  a Python-version difference and a requirements-resolution surprise.
- **No load or concurrency testing** of the drift job; it is a single-process
  batch script by design.
