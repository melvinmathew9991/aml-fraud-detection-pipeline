# Pre-Deployment Audit — 2026-08-03

End-to-end audit performed at the Sprint 6 → Sprint 7 boundary, before any
cloud resource is created. Scope: results, architecture, workflow, project
structure, data governance, and requirements. Every claim below was checked
against the artifact that produces it, not against another document.

This file is the project's defect register and history record. `ROADMAP.md`
remains the plan-of-record; `ARCHITECTURE.md` the design; `GIT_WORKFLOW.md`
the git policy.

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
6. **Image size 510.6 MB against a `<400 MB` target** (28% over). Carried from
   Sprint 6 as a documented deviation. `pyarrow` — present solely to read
   `dest_state.parquet` — is the prime suspect, alongside the `scipy` that
   `lightgbm` pulls in. Dropping it means changing the bundle's storage format.

---

## 7. What this audit did not cover

Stated so the clearance above is not read as broader than it is:

- **No full training re-run.** Published metrics were verified against the
  committed CSVs, not regenerated from the 6.36 M-row dataset (~28 min).
- **No container run.** This machine has no Docker by design; container
  behaviour is known only from CI.
- **No load or soak testing.** `/score` p95 is asserted locally by the test
  suite; nothing has been tested under concurrency.
- **No adversarial or security review** of the API beyond the existing
  rate-limit middleware and `trivy` image scan.
- **Model quality was not re-litigated.** PaySim is near-trivially separable by
  construction and ARCHITECTURE §0 says so; this audit checked that the
  reported numbers are true, not that they are impressive.
