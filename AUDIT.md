# Pre-Deployment Audit — 2026-08-03

End-to-end audit performed at the Sprint 6 → Sprint 7 boundary, before any
cloud resource is created. Scope: results, architecture, workflow, project
structure, data governance, and requirements. Every claim below was checked
against the artifact that produces it, not against another document.

This file is the project's defect register and history record. `ROADMAP.md`
remains the plan-of-record; `ARCHITECTURE.md` the design; `GIT_WORKFLOW.md`
the git policy.

**Sections 0–7 are the 2026-08-03 pre-deployment audit and describe the project
as it stood at the Sprint 6 → Sprint 7 boundary. They are left as written.**
`§8` is a second, independent audit performed on **2026-09-12** over the whole
project after Sprint 9 — different scope, different reviewer framing, and one
finding that materially changes how the results in §3.1 should be read.

---

## 0. Verdict

**Cleared for Sprint 7**, after five defects found here were fixed and verified.

| Gate | State |
|---|---|
| Test suite | **182 passed** (174 pre-audit + 8 new regression tests) |
| `ruff` | All checks passed |
| `mypy` (`src/inference`) | Success, no issues in 6 source files |
| CI on `main` | Green — all three jobs |
| Published numbers vs source CSVs | **Zero mismatches** |
| Secrets in history | **None**, verified across all refs |

One defect (§2.1) was a live correctness bug in the serving artifact's
metadata. One (§2.4) had silently split a third of the project's experiment
history across two databases. Neither was caught by 174 existing tests.

---

## 1. Method

The audit did not trust documentation as evidence. For each claim:

1. Locate the artifact that generates the number or behaviour.
2. Recompute or re-read it directly.
3. Compare to what the docs say.
4. Where a test asserted the invariant, verify the test **can actually fail**
   by reintroducing the bug and observing a red run.

Step 4 mattered: two tests were found asserting an invariant that held only
*because* of the bug they were supposed to guard (§2.1).

---

## 2. Defects found and fixed

### 2.1 `feature_version` in the serving bundle was the feature *count* — **live bug**

`src/export_bundle.py` wrote the number of features into both `feature_version`
and `n_features`:

```python
"feature_version": len(metadata["feature_names"]),   # -> 18
"n_features":      len(metadata["feature_names"]),   # -> 18
```

The true value is `features.FEATURE_VERSION = 3`. Consequences:

- `bundle_meta.json` reported `feature_version: 18` for a v3 feature set.
- `train_pipeline.py:574` writes the **correct** value into `metadata.json`, so
  the same field name meant *version* in training and *count* in serving — the
  two artifacts disagreed.
- `/model-info` published the wrong value to every client (`api/main.py:152`).
- The field is the natural place to detect train/serve feature-schema skew —
  the project's stated central problem — and it could not, being a constant
  that changes only when the feature count does.

**Why 174 tests missed it.** Two tests asserted the bug:

```python
# tests/test_api.py            assert len(body["feature_names"]) == body["feature_version"]
# tests/test_inference_bundle.py  assert len(bundle.feature_names) == bundle.feature_version
```

Both passed *only because* count and version were the same number.

**Fix.** `export_bundle.py` now reads `metadata["feature_version"]` — the
version the model was actually trained with, not `features.FEATURE_VERSION`,
which would stamp an old bundle with whatever the current code says. A run
predating the field raises rather than guessing. The committed
`bundle_meta.json` was corrected to `3` (safe: `bundle_meta.json` is not inside
its own sha256 manifest — verified — so the other four checksums are
unaffected, and were re-verified OK after the edit).

Both tests were rewritten to assert the real invariants:

- serving side (`test_inference_bundle.py`, which runs in the
  serving-isolation CI job and therefore may not import training modules):
  `len(feature_names) == n_features == 18` and `feature_version == 3`
- API side (`test_api.py`): `feature_version == FEATURE_VERSION` — this now
  actively detects a committed bundle going stale against `features.py`

**Verification.** Reintroducing `feature_version = 18` fails both tests
(`assert 18 == 3`). The old assertion could not have. Restored to 3; green.

Correctness of `3` was independently established: the bundle's 18 feature names
match the current code's feature list exactly and in order.

### 2.2 The MLflow tracking URI was resolved against the working directory — **live bug**

`config.yaml` sets `mlflow.tracking_uri: "sqlite:///mlflow.db"` — a *relative*
URI — and `train_pipeline.py` passed it straight to `mlflow.set_tracking_uri()`,
which resolves relative sqlite paths against the **process working directory**.

Every other path in that module (`RAW_PATH`, `PROCESSED_DIR`, `MODEL_DIR`,
`REPORTS_DIR`) is anchored to `PROJECT_ROOT`. The tracking URI was the one that
was not.

**Impact, measured:** running the pipeline from `src/` wrote to `src/mlflow.db`.
That orphan store holds **54 runs across 2 experiments**; the canonical root
store holds 108. **A third of the project's experiment history was in a
database nothing reads** — while Sprint 1's deliverable is "experiment
tracking of every (model, fold) result."

**Fix.** `resolve_tracking_uri()` added to `src/config.py` (its proper home —
it resolves a config value, as `PROJECT_ROOT` does, and unlike
`train_pipeline.py` it has no import-time side effects, so it is unit-testable).
Absolute URIs and non-sqlite backends pass through untouched.

**Verification.** New `tests/test_config.py`, 8 tests, including one asserting
the resolved URI is identical when called from `tmp_path` and from `src/` —
the precise property that was broken.

**Not fixed, deliberately:** the 54 orphaned runs were left in place. Merging
MLflow sqlite stores is error-prone and destroying run history to tidy a
directory is a bad trade. See §6.

### 2.3 Documentation claims contradicted their own sources

| Location | Claimed | Actual |
|---|---|---|
| `ROADMAP.md` state table | "66 tests" | 174 (now 182) |
| `README.md` tree + status para | "166 tests" | 174 (now 182) |
| `ROADMAP.md` Sprint 2 entry | "3,833 false positives" at 500/day | **3,835** (`capacity_sweep.csv`); 3,833 is the *ablation* table's number for a different feature set |
| `ROADMAP.md` state table | "model card still absent" | Exists: `src/model_card.py`, dashboard page 4, limitations served on `/model-info` |
| `ROADMAP.md` Sprint 6 entry | cites `reports/train_20260801T130131Z.log` | `reports/` is gitignored **and that log no longer exists** |

All corrected. The log citation was marked unverifiable in place rather than
deleted — this repo's most-repeated defect is documentation citing evidence
that cannot be checked, and silently removing the evidence of that pattern
would defeat the point.

### 2.4 `GIT_WORKFLOW.md` §1 described a completed migration as pending

§1 documented the `git filter-repo` pass to strip AI trailers as work still to
be done, gating §6's branch protection on it ("protection blocks force-push").

**It had already been executed.** Verified: every commit reachable from `main`
is clean of trailers, and the Sprint 1 commit is `947e8ea`, not `0772bb9`.
Pre-rewrite history survives on `origin/backup/pre-rewrite-2026-08-01`
(14 commits).

This mattered operationally — the stale text was actively deferring branch
protection for a reason that no longer existed. §1 now records completion and
§6 records that nothing gates it.

### 2.5 No `.gitignore` rule for secrets

`.gitignore` covered data, models, logs, MLflow, Python and OS artifacts — but
had no rule for `.env`, service-account JSON, or key material.

Nothing secret has ever been committed (verified against the full history, all
refs). The gap was purely prospective — and Sprint 7 introduces GCP while
Sprint 10 introduces Neon Postgres and API keys.

**Fix.** Rules added for `.env`, `.env.*` (with `!.env.example`),
`*-service-account*.json`, `*credentials*.json`, `gha-creds-*.json`, `*.pem`,
`*.key`. Verified effective, and verified that the `!model_bundle/` negation
still holds — all 5 bundle files remain tracked. **The point of a secrets rule
is to predate the secret.**

---

## 3. Audit by dimension

### 3.1 Results — clean

Every published number was recomputed from `data/processed/*.csv`:

- Model comparison table: all 5 tree rows match `model_comparison.csv` to 4 dp,
  including the "five tree models span 0.5254–0.5293" tie claim.
- `best_model` = **LightGBM (tuned)**, matching `bundle_meta.json`'s
  `model_name`, and it now leads on *both* PR-AUC (0.9974) and capacity
  precision (0.5293). The "best PR-AUC ≠ best model" divergence documented in
  Sprints 1–2 has resolved under the 18-feature set — the historical narrative
  about it is correctly scoped in past tense and was left alone.
- Capacity sweep: 250/day → 4,042 TP / 0 FP / 208 FN; 500/day → precision
  0.5257 = 4250/8085 exactly; ~18.4 FPs per marginal fraud.
- Economics: `327,068,118` = 208 × the unrounded 1,572,442.875. Correct — the
  docs quote the rounded 1,572,443 alongside the unrounded product, which is
  consistent, not contradictory.
- `model_comparison.csv` parses cleanly (8 rows × 14 columns) despite commas
  inside quoted model names.

### 3.2 Architecture — clean

- **Serving isolation holds.** No `sklearn`, `shap`, `joblib`, `duckdb`,
  `pandas`, `mlflow`, `optuna` or `xgboost` import anywhere in `src/api/` or
  `src/inference/` (ARCHITECTURE §3). Independently enforced by the
  `serving-isolation` CI job.
- **Dashboard is display-only** (ARCHITECTURE §4). Imports no model or feature
  library; reaches the API over HTTP via a configurable `API_BASE_URL`.
- **Bundle integrity** verified: all four sha256 entries match, both in the
  working tree and as git blobs (what CI checks out). `model_bundle/** -text`
  in `.gitattributes` overrides `core.autocrlf=true`, which is what prevents
  the CRLF corruption fixed in `8e5cff5` from recurring.
- Bundle size 9.18 MB; `dest_state.parquet` 7.48 MB, against a `<20MB` DoD.

### 3.3 Workflow and git history — one finding (§2.4), otherwise sound

- 28 commits on `main`, 5 merge commits (PRs #1–#5, all merged).
- Conventional Commits used from `8e5cff5` onward; earlier commits use a
  "Sprint N: …" convention. Mixed but internally consistent by era.
- `main` carries zero AI trailers.
- **Deviation, historical:** Sprints 4 and 5 landed as direct commits on `main`
  (`c7e52bb`, `af73b46`, `c1c4f0a`) with no PR and no CI, contrary to
  `GIT_WORKFLOW.md`'s branch-per-sprint rule. Not correctable retroactively;
  it is precisely what branch protection would prevent (§6).
- **Stale branches:** `sprint-1-cv-optuna-mlflow` and
  `sprint-2-capacity-shap-error-analysis` point at *pre-rewrite* commits and so
  report as "not merged". Their trees are **byte-identical** to their merge
  commits on `main` (`d26d354`, `1359f7d`) — verified — so deleting them loses
  nothing and removes the last non-backup copies of the trailer commit.
  `data-layer-hardening` and `sprint-6-containerization-ci` are cleanly merged.

### 3.4 Project structure — clean

- All **55** entries in README's documented tree exist on disk.
- No dead or orphaned source files found.
- `src/mlflow.db` and `src/mlruns/` are the artifacts of §2.2. Both are already
  matched by `.gitignore` (patterns without a leading slash match at any depth),
  so they were never committable — clutter, not a governance risk.

### 3.5 Data governance — one finding (§2.5), otherwise well designed

- The 493 MB PaySim CSV is ignored via `data/raw/*.csv`; the regenerable DuckDB
  store via `data/processed/*.duckdb`.
- The 11 evidence CSVs under `data/processed/` **are** tracked — this is
  deliberate, and is what closed the Sprint 1 audit finding about documents
  citing untracked evidence.
- `model_bundle/` is tracked via an explicit `!` negation as a release artifact.
- The prediction audit log hashes feature values rather than logging them
  (`api/audit.py`), and writes to stdout — which also means the non-root
  container user needs no writable path.
- No secret-shaped file has ever been added in the repo's history, across all
  refs.
- PaySim is synthetic, which is what makes the Sprint 12 Gemini free-tier
  decision (requests may be used for training) acceptable — recorded in
  ROADMAP as a production boundary, correctly.

### 3.6 Requirements — clean

- 29 pinned packages across four files, **zero version conflicts**, zero
  unpinned entries.
- The four-way split (`train` / `serve` / `dashboard` / `dev`) is enforced, not
  merely declared: the `serving-isolation` CI job installs only
  `requirements-serve.txt` and boots a real `uvicorn` process over HTTP.
- **Known environment gap:** there is no project venv. Tests run against a
  global Python 3.11 that was missing `pandera` and `httpx2` — both pinned in
  `requirements-train.txt` — until they were installed during this work. This
  drift is precisely what hid the Sprint 6 CI failure (`streamlit` was present
  locally, absent in CI). See §6.

---

## 4. Complete project history

28 commits, 2026-07-22 → 2026-08-03. Merge commits marked ⑃.

| # | Commit | Date | Subject |
|---|---|---|---|
| 1 | `fc3c2af` | 07-22 | Sprint 0: engineering hygiene for production readiness |
| 2 | `4567900` | 07-24 | Data-layer hardening: DuckDB-backed storage and features |
| 3 | `b10136c` ⑃ | 07-24 | Merge PR #1 — data-layer-hardening |
| 4 | `ff6684d` | 07-24 | edited README.md |
| 5 | `12fc905` | 07-24 | Add Recall@K and a precision/recall-at-K curve to correct a metric misread |
| 6 | `9d5aa88` | 07-24 | Flag dirty working tree in `git_commit_hash()` for accurate run provenance |
| 7 | `947e8ea` | 07-25 | Sprint 1: XGBoost/LightGBM, time-based CV, Optuna tuning, MLflow tracking |
| 8 | `d26d354` ⑃ | 07-25 | Merge PR #2 — sprint-1-cv-optuna-mlflow |
| 9 | `2b0b991` | 08-01 | Sprint 2: capacity-based operating point, SHAP explainability, top-K error analysis |
| 10 | `0d21a8e` | 08-01 | Sprint 2 results and audit corrections in README and ROADMAP |
| 11 | `bb497d1` | 08-01 | Architecture and git workflow documentation for Sprints 3-12 |
| 12 | `1359f7d` ⑃ | 08-01 | Merge PR #3 — sprint-2-capacity-shap-error-analysis |
| 13 | `bc48fcb` | 08-01 | Sprint 3: tests, economics module, and versioned model bundle |
| 14 | `5b1af5d` ⑃ | 08-01 | Merge PR #4 — sprint-3-tests-economics-bundle |
| 15 | `70092fc` | 08-02 | Sprint 3 status update |
| 16 | `8e5cff5` | 08-02 | fix(bundle): correct model.txt checksum and CRLF corruption |
| 17 | `c7e52bb` | 08-02 | feat(inference,api): add serving core and FastAPI service |
| 18 | `af73b46` | 08-02 | test(error-analysis,explain): close coverage gap, fix audit findings |
| 19 | `c1c4f0a` | 08-02 | Sprint 5: Streamlit dashboard and snapshot-step tracking |
| 20 | `0c86106` | 08-02 | Sprint 6: containerization and CI |
| 21 | `e50a2d9` | 08-02 | Sprint 6: record image size and cold-start numbers |
| 22 | `dc35581` | 08-03 | fix(ci): repair the PR gate and remove unmeasured image claims |
| 23 | `f480736` | 08-03 | fix(sample-data): create the output directory before writing |
| 24 | `cdb73b8` | 08-03 | fix(ci): pin trivy-action to an existing tag |
| 25 | `f350833` | 08-03 | fix(ci): make the readiness wait fail instead of faking a measurement |
| 26 | `39c6eaf` | 08-03 | fix(docker): install libgomp1 for LightGBM's native library |
| 27 | `4435d69` | 08-03 | docs: record the real CI measurements and the missed image target |
| 28 | `1ec0719` ⑃ | 08-03 | Merge PR #5 — sprint-6-containerization-ci |

Note commits 17–19: Sprints 4 and 5 bypassed the PR workflow entirely (§3.3).

---

## 5. Cumulative defect register

Defects found by each sprint's audit, oldest first. The recurring theme is
stated plainly because it recurs: **documentation asserting evidence that does
not exist.**

| Sprint | Defect | Class |
|---|---|---|
| 1 | Evidence CSVs cited by name in README but never tracked in git | evidence |
| 2 | "fold 2 linear PR-AUC 0.06–0.08" stale; type indicators had moved it to 0.24–0.41 | stale claim |
| 2 | `capacity_precision_mean` averaged 3 folds spanning 3.1/3.1/16.2 days — not comparable | methodology |
| 2 | Config's "0.24% alert rate" was a dataset average matching no fold | stale claim |
| 2 | Peak RSS misstated (actual 1,367 MB) | stale claim |
| 2 | README cited a gitignored log as evidence | evidence |
| 2 | `k` vs `n_flagged` tie inconsistency across `threshold.py` / `error_analysis.py` | correctness |
| 3 | `random_state` never passed to the three `LogisticRegression` constructions — saga is stochastic, so results were irreproducible | correctness |
| 6 | `model.txt` checksum + CRLF corruption on Windows checkout | correctness |
| 6 | `streamlit` missing from CI's `lint-test` install → pytest died at collection | CI |
| 6 | `--no-cache-dir` contradicted `cache: pip` → post-run cache save failed a passing job | CI |
| 6 | `generate_sample_data.py` never created `data/raw/` → could not run on a clean clone | correctness |
| 6 | `trivy-action@0.28.0` does not exist (tags are `v`-prefixed) | CI |
| 6 | **Readiness poll loop exited 0 on timeout** — reported its own 30 s ceiling as a cold-start measurement and passed green while the container was dead | CI / evidence |
| 6 | `libgomp.so.1` absent from `python:3.12-slim` → container exited (1) every run | correctness |
| 6 | Image size / cold start recorded as "measured in CI" for a job that had never executed (wrong by 2.8× and 8×) | evidence |
| **Audit** | `feature_version` = feature count in the serving bundle (§2.1) | correctness |
| **Audit** | MLflow tracking URI resolved against cwd; 54 runs orphaned (§2.2) | correctness |
| **Audit** | Five documentation claims contradicting their sources (§2.3) | stale claim |
| **Audit** | `GIT_WORKFLOW.md` §1 described a completed migration as pending (§2.4) | stale claim |
| **Audit** | No `.gitignore` rule for secrets (§2.5) | governance |
| **Senior** | `/score/batch` records cumulative batch-elapsed time as *each row's* latency, corrupting `/metrics` p50/p95/p99 and every batch audit record (§8.4) | correctness |
| **Senior** | Live `/score` returns `BLOCK` from a rule measured at 10.1% precision (§8.4) | judgment |
| **Senior** | No retraining trigger keys on queue volume — the one failure actually occurring, in 14 of 30 windows (§8.4) | monitoring |
| **Senior** | `suggest_pos_weight` docstring cites "~0.026% fraud … 3800:1"; the real rate is 0.1291%, giving ~774:1 (§8.4) | stale claim |
| **Senior** | `capacity_k` clamps K to `n_rows`, silently turning a capacity constraint into "flag everything" on collapsed windows (§8.4) | correctness |
| **Senior** | Snapshot stores `avg` as float32 against training's float64; 39 of 571,961 differ at epsilon (§8.4) | methodology |

**Two lessons this register supports.** First, a green check that cannot go red
is worse than no check — the readiness loop and the two `feature_version` tests
both actively manufactured false confidence. Second, several defects were
reachable only after the one ahead of them was fixed, so "CI is green" is a
statement about the last barrier reached, not about everything behind it.

---

## 6. Outstanding before Sprint 7

Not blockers, but decisions that should be made deliberately rather than by
default:

1. **Enable branch protection.** Nothing gates it any more (§2.4). Ruleset
   `main-protection`, empty bypass list, required approvals `0`, all three CI
   checks required, linear history **off**. Directly prevents the Sprint 4/5
   direct-to-`main` deviation from recurring.
2. **Create a project venv** from `requirements-train.txt`. The absence of one
   is why environment drift hid a CI failure (§3.6).
3. **Delete the stale pre-rewrite branches**, local and remote. Content is
   provably preserved on `main` (§3.3). Shared refs, so this is your call.
4. **Decide on the 54 orphaned MLflow runs** (§2.2): merge into the canonical
   store, or leave and document. Left in place for now.

   *[RESOLVED 2026-09-12 — merged.* The orphan store turned out to be the more
   valuable of the two, which is why deleting it was rejected: `src/mlflow.db`
   held 54 runs ending **2026-07-28 17:54**, while the canonical store's newest
   run was **2026-07-24 14:45**, and the orphan carried the only MLflow record
   of run `20260728T172950Z` — the newest training run still surviving under
   `models/`. In a project that already cannot trace its deployed bundle to a
   surviving run (item 5 below), discarding that was the wrong direction.

   Both stores were backed up first. The merge was built as a copy and verified
   before being swapped in: run ids do not overlap (108 + 54 = 162, zero
   collisions), both databases were at the same schema revision
   (`alembic_version b7e4c1a90f23`) with identical columns on every merged
   table, and all five row counts landed exactly
   (runs 162, params 234, metrics 1504, latest_metrics 1504, tags 1290) with
   `PRAGMA integrity_check` clean and no duplicate `run_uuid`. The orphan's
   `artifact_uri` values were rewritten from `/src/mlruns/` to `/mlruns/`, and
   its one artifact directory relocated and confirmed byte-identical to the
   backup (8 files, sha256 match). Verified functionally afterwards, not just by
   row count: `mlflow.search_runs` returns **162 runs**, and the surviving
   provenance tag is present as `mlflow.runName=20260728T172950Z`.

   `src/mlflow.db` and `src/mlruns/` are now gone, so the working-directory bug
   in §2.2 leaves no residue.]*
5. **Bundle provenance gap — DECIDED 2026-08-03: ship as-is, regenerate later.**
   `bundle_meta.json` claims `run_id: 20260801T130131Z`, but neither
   `models/20260801T130131Z/` nor its `reports/` log still exists — the newest
   local run is `20260728T172950Z`, and the recorded `git_commit` is
   `b874804-dirty`. The bundle in `model_bundle/v1/` therefore traces to no
   surviving artifact and **cannot be regenerated byte-for-byte**.

   The obvious fix — re-run training from a clean commit — was considered and
   **rejected for now**, because its costs land in the wrong place:

   - **It resets the golden file.** `tests/golden/golden_transactions.csv` is
     the standing regression control pinning serving behaviour to 1e-9. A new
     model means new expected scores, so the file is regenerated and its value
     as a *continuity* check is lost precisely when a deployment is about to
     start exercising the serving path.
   - **It shifts every published metric.** LightGBM is not bit-reproducible
     across runs here: the Sprint 1 record shows PR-AUC drifting 0.9976 →
     0.9971 between identical invocations from multi-threaded float
     non-determinism. The README results table, capacity sweep and economics
     figures would all need re-verification — reintroducing the documentation
     drift this audit just spent a full pass eliminating, days before a first
     deployment.

   What a deployment actually requires is a **verified** artifact, not a
   *reproducible* one, and this bundle is verified along all three axes that
   matter at serving time: sha256-checked on load, pinned to 1e-9 by the golden
   file, and exercised end-to-end by the container integration job in CI.
   Reproducibility is a research-narrative property here, not a serving one.

   **Regenerate at the next legitimate training run** — Sprint 8's drift work,
   or any feature change — when the metric refresh is happening anyway. At that
   point `export_bundle.py` will also write `feature_version` correctly of its
   own accord, rather than carrying the hand-correction applied in §2.1.

   **Accepted risk, stated plainly:** until then, the deployed model cannot be
   rebuilt from source. If `model_bundle/v1/` were lost, it would have to be
   retrained, and the resulting model would not be identical. The committed
   bundle is the only copy, which is why it is tracked in git rather than
   treated as a build output.

   *[2026-09-12: the path is now `model_bundle/v2/` — v1 was repackaged on
   2026-09-10 (parquet → `.npz`) and is no longer in the repo. **The risk is
   unchanged, not resolved**: v2 was converted from v1's arrays rather than
   rebuilt from source, so it carries the same `run_id`, `trained_at` and
   `git_commit: b874804-dirty`, and still traces to no surviving training
   artifact. The regeneration this item calls for has not happened.]*
6. **Image size 510.6 MB against a `<400 MB` target** (28% over). Carried from
   Sprint 6 as a documented deviation. `pyarrow` — present solely to read
   `dest_state.parquet` — is the prime suspect, alongside the `scipy` that
   `lightgbm` pulls in. Dropping it means changing the bundle's storage format.

   *[Closed 2026-09-10 by PR #21, recorded 2026-09-12: the storage format was
   changed. The v2 bundle ships `dest_state.npz` and `pyarrow` is gone from
   `requirements-serve.txt`; `scipy` stays, measured at 112.7 MB and required
   by `lightgbm`. CI measured 367.5 MB uncompressed / 125.0 MB compressed —
   target met. This item is no longer outstanding.]*

---

## 7. What this audit did not cover

Stated so the clearance above is not read as broader than it is:

- **No full training re-run.** Published metrics were verified against the
  committed CSVs, not regenerated from the 6.36 M-row dataset (~28 min).
- **No container run.** This machine had no Docker at the time of this audit;
  container behaviour is known only from CI. *[2026-09-12: Docker Desktop has
  since been installed (2026-09-09). It does not change what this audit
  covered, and CI remains the authoritative container evidence — it is the only
  build that enforces the 512MiB Cloud Run ceiling and runs `trivy`.]*
- **No load or soak testing.** `/score` p95 is asserted locally by the test
  suite; nothing has been tested under concurrency.
- **No adversarial or security review** of the API beyond the existing
  rate-limit middleware and `trivy` image scan.
- **Model quality was not re-litigated.** PaySim is near-trivially separable by
  construction and ARCHITECTURE §0 says so; this audit checked that the
  reported numbers are true, not that they are impressive.

---

# 8. Senior review — whole-project audit (2026-09-12)

Performed against `main` at `2f91d24`, after Sprint 9. Read as a senior data
scientist in payments fraud / AML auditing this as if it were proposed for
production at a bank or PSP — a different question from §0–§7, which asked
whether the project's own claims were true.

**Method.** Every figure below was re-derived from the DuckDB store, the
committed CSVs, or the running service. Nothing was taken from the project's
prose, including prose this audit's own earlier sections wrote. Where a
conclusion depended on a definition ("drained to the cent"), the alternatives
were computed and compared rather than one being assumed.

**Verdict.** As an ML *systems* project this is in the top few percent of what a
reviewer sees: the engineering discipline is real and unusual. As a fraud/AML
*modelling* project it rests on a dataset that does not require a model, and
§8.1 states that far more precisely than the project previously did.

---

## 8.1 A three-predicate rule beats the deployed model

Using only raw columns, with no model:

```
type ∈ {TRANSFER, CASH_OUT}
  AND  |(oldbalanceOrg − newbalanceOrig) − amount| ≤ 0.01   -- balances reconcile
  AND  oldbalanceOrg > 0 AND newbalanceOrig = 0             -- origin emptied
```

| Scope | Flagged | True positives | False positives | Precision | Recall |
|---|---|---|---|---|---|
| **All 6,362,620 rows** | 8,008 | 8,008 | **0** | **1.0000** | **0.9750** |
| **Fold 3 test window** | 4,125 | 4,125 | **0** | **1.0000** | **0.9706** |
| *Deployed model @ 500/day* | *8,085* | *4,250* | *3,835* | *0.5257* | *1.0000* |

The rule catches 97% of fraud with **zero false positives in 6.36M transactions**,
using half the review capacity the model's operating point consumes. The model's
entire marginal contribution on the deployed fold is the last **125** frauds,
bought with **3,835** false positives.

**Why.** `orig_balance_mismatch` is the top feature by SHAP (0.2518 of total
attribution) and its behaviour is inverted from intuition:

| Population | Rows | Balance mismatch |
|---|---|---|
| Non-fraud, all types | 6,354,407 | **79.73%** |
| Fraud | 8,213 | **0.55%** |
| Non-fraud, TRANSFER/CASH_OUT only | 2,762,196 | **90.47%** |

By type: CASH_IN 100.00%, TRANSFER 95.47%, CASH_OUT 88.95%, PAYMENT 53.72%,
DEBIT 29.79%. PaySim writes arithmetically consistent balances for the fraud it
constructs and leaves the rest of its ledger inconsistent, so the model's
largest single driver is detecting **which simulator code path wrote the row**.

This is a stronger statement than the one ARCHITECTURE §0 and MODEL_CARD §10
make. They cite the drained-to-the-cent rule (99.1% of fold-2 fraud) — a
*behavioural* artifact a real fraudster might also produce. The
balance-reconciliation inversion is a *bookkeeping* artifact with no real-world
analogue, and it is what makes the problem degenerate.

**What follows.** PR-AUC 0.9974 is fully explained and needs no attribution to
modelling skill. The capacity/economics analysis rests on a precision/recall
tradeoff that largely does not exist here. None of this invalidates the
engineering: the serving path, skew tests, CI, monitoring and deployment would
be identical for a model that mattered — which is the honest defence, and a good
one. **Recommendation: lead with this finding.** "A three-line rule beats my
model, here is the evidence, here is why the system is still the deliverable" is
a far stronger position than reporting 0.9974.

## 8.2 This is fraud detection, not AML

PaySim's `isFraud` marks account takeover and cash-out. Anti-money-laundering
concerns placement, layering and integration — structuring, smurfing,
round-tripping, mule networks — visible only *across* transactions and entities
over time. Absent here: entity resolution, typology coverage, multi-hop graph
features (planned, Sprint 11), case management / SAR-STR workflow, sanctions and
PEP screening, and model-risk governance (independent validation sign-off, a
production challenger, a documented backtesting protocol).

The destination-side velocity and fan-in features are the right instinct and the
most AML-shaped part of the system. The criticism is the label on the tin: an
AML interviewer raises this within five minutes. Either rename it, or state the
distinction in README's first section.

## 8.3 The threshold/queue mismatch — the one real design flaw

K is derived from capacity (correct: staffing is known in advance, the fraud
count is not), then the *score at position K* ships as a fixed constant. A fixed
score does not produce a fixed queue. The project's own drift output proves it:
across 31 windows the queue runs **272 to 4,594 alerts/day against a capacity of
500, exceeding it in 14 of the 30 full windows** — a 9× swing in analyst workload
from an operating point chosen to control analyst workload.

Standard fixes: a rank-based cutoff per scoring period; a **rolling-percentile
threshold** recomputed from the last 24–48h and shipped as a percentile rather
than a score (usually the right answer for single-transaction decisions); or a
closed-loop controller holding queue length near target. **No retraining trigger
in MONITORING §4 keys on queue volume**, so the failure actually occurring would
never raise a flag.

Two related weaknesses. `capacity_k` clamps K to `n_rows`, so a collapsed window
flags everything (window 30: 272 rows, all 272 flagged) — a capacity constraint
silently becoming "review everything". And the destination snapshot is frozen at
**step 743**: velocity is precisely the mule signal and the feature set that goes
stale fastest, with no refresh path short of rebuilding the bundle. `state_hit`
reports *coverage*, not *staleness*; a `snapshot_age_days` field would be its
honest companion.

**What is genuinely strong**, stated at the same detail as the criticisms:
leakage safety is real and verifiable (`ROWS BETWEEN UNBOUNDED PRECEDING AND 1
PRECEDING` for history, `RANGE BETWEEN 24 PRECEDING AND 1 PRECEDING` for
velocity — the RANGE frame correctly excludes same-`step` peers, which most
implementations get wrong); the skew treatment is better than most production
systems (golden-file test at exact floating-point equality, plus a CI job
installing *only* `requirements-serve.txt`, which already caught a real
`libgomp` failure); the PSI implementation is unusually careful (symmetric
divergence, binning chosen from reference cardinality, open-ended edges,
doubly-empty bins dropped, a `0.5/n` floor with an argument behind it rather
than the conventional arbitrary 1e-4); the audit log does what it claims —
checked specifically, because that claim is often aspirational: `audit.py`
hashes to sha256 and never logs raw balances or account ids; and negative
results are retained rather than tidied away.

## 8.4 Defects found

| # | Severity | Finding |
|---|---|---|
| 1 | **High** | `/score/batch` computes `row_latency_ms = (perf_counter() − start)` *inside* the loop with `start` fixed at batch start, then passes it to `MetricsTracker.record_score` and `log_prediction`. A 10,000-row batch injects 10,000 monotonically increasing values, so `/metrics` p50/p95/p99 report batch position rather than latency, and the audit log inherits the same wrong number. |
| 2 | **Medium** | The live `/score` endpoint returns **`BLOCK`** from a rule measured at **10.1% precision** — nine legitimate customers blocked per fraud. Either gate it behind a config flag (default off) or add the missing predicate from §8.1, which takes the same rule to 100% precision on this data. |
| 3 | **Medium** | No retraining trigger keys on queue volume (§8.3). |
| 4 | **Low** | `custom_metrics.suggest_pos_weight` docstring claims "~0.026% fraud … roughly 3800:1"; the real rate is **0.1291%**, giving ~774:1. |
| 5 | **Low** | `capacity_k` clamps to `n_rows` (§8.3). |
| 6 | **Low** | Snapshot stores `avg` as float32 against training's float64; 39 of 571,961 differ at epsilon — documented, but a permanent skew source in the five stateful features. |

Nothing found here contradicts a claim the project makes about itself, except
§8.1 (where the project understates its own problem) and defect 4.

## 8.5 Methodological critiques

**PSI bands are borrowed from a regime that does not apply.** The 0.10/0.25
thresholds come from credit-scorecard practice, where population sizes are
broadly stable. Here windows span **272 to 574,255 rows — a 2,000× range** — and
the deliberate `0.5/n` empty-bin floor makes PSI explicitly sample-size
dependent. A PSI of 0.25 therefore does not mean the same thing in window 2 as
in window 16. `sufficient_sample` mitigates the worst of it but does not make
the values comparable; a size-normalised statistic, or bands calibrated per
volume decile, would be defensible.

**The "tree models are tied" claim is honest but untestable as constructed.**
Five models spanning 0.5254–0.5293 with ±0.03 fold-to-fold std across **three**
folds gives no test any power. Sprint 11's planned significance testing will not
fix n=3; more folds or a blocked bootstrap within folds would.

**No embargo between CV train and test windows.** With a 24-hour velocity
feature, a test row one hour past the boundary draws history from the training
period. Not label leakage, and it *is* what a production feature store would
serve — but purging/embargoing is standard in financial time-series ML, and its
absence should be a stated choice.

**Monitoring is distributional, not performance-based.** With no label feedback
loop, trigger A (`recall_at_capacity < 1.0`) cannot fire in production; it can
only be evaluated retrospectively. Until Sprint 10 lands analyst dispositions,
"monitoring" means a drift detector over a fixed historical dataset plus a
liveness check. MONITORING §4 says this plainly, to its credit — but the gap
between "monitoring exists" and "we would know if the model degraded tomorrow"
is the largest remaining distance between this and production.

## 8.6 Production readiness

| Area | State |
|---|---|
| **Authentication** | **None.** Public unauthenticated scoring endpoint; rate limiting and body-size caps exist, authentication does not. Sprint 10. |
| **Persistence** | None. `/metrics` is per-instance and resets on cold start. |
| **Reproducibility** | **Broken by a known gap.** The bundle records `run_id 20260801T130131Z` / `git_commit b874804-dirty`; those artifacts no longer exist. The deployed model cannot be rebuilt. In a regulated deployment this alone blocks approval. |
| **Rollback** | Genuinely good — deploy-without-traffic, verify, migrate, drill actually performed. |
| **Cost controls** | Budget alert, max-instances ceiling, scale-to-zero, registry retention — verified against the billing API. |
| **Data validation** | pandera schema at ingest over a 50k reservoir sample; explicitly traded off. A rare violation can still pass. |
| **Fairness** | Impossible on PaySim (no protected attributes) and correctly not claimed. A real deployment needs it before launch. |

## 8.7 What I would do next, ranked

1. **State §8.1 prominently** and ship the three-predicate rule as a documented
   baseline the model must beat. Highest value per hour in the backlog, and a
   credibility multiplier rather than an admission.
2. **Fix the batch latency defect** (§8.4 #1) — it corrupts the only performance
   telemetry the system has.
3. **Replace the fixed threshold with a rolling-percentile operating point** and
   add a queue-volume retraining trigger (§8.3).
4. **Gate or fix the hard-block rule** (§8.4 #2) before anyone clicks the demo.
5. **Rename, or explicitly scope, the AML framing** (§8.2) — one paragraph.
6. **Then** Sprints 10–12 as planned. Graph analytics (Sprint 11) would most
   change what this project *is*, because it is where AML starts and where
   PaySim's degeneracy stops helping.

Explicitly **not** recommended: retraining, tuning, or chasing a better score.
The dataset cannot distinguish model quality at this point, and §8.1 shows the
ranking problem is not the bottleneck.

## 8.8 What this review did not cover

- **No training re-run.** Metrics verified against committed CSVs and the DuckDB
  store, not regenerated.
- **No load, soak or concurrency testing.**
- **No security review** beyond noting the absence of authentication — no
  dependency CVE audit beyond the existing trivy scan, no input fuzzing, no
  review of the rate limiter under distributed load.
- **No review of the dashboard's Streamlit code** beyond the landing page.
- **Sprint 9's own corrections were taken as verified** by that sprint's audit
  and not independently re-derived, except the balance-mismatch figures in §8.1,
  which are new measurements.
