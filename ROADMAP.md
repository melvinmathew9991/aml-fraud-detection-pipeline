# Roadmap: Production-Readiness Plan

This document tracks what's needed to take this project from a working
scaffold (proven correct on the real 6.36M-row PaySim dataset) to a
production-shaped, end-to-end system suitable for a portfolio deep-dive.

## 1. Current state audit

| Area | Current state |
|---|---|
| Version control | Not a git repo yet — no history, no branching, no diffs |
| Dependencies | No `requirements.txt`/`pyproject.toml` — env isn't reproducible |
| Model artifacts | `models/` is empty — training doesn't persist anything; every run starts from scratch |
| Config | Hyperparameters and paths are hardcoded in `train_pipeline.py` |
| Logging | `print()` statements, no structured logs |
| Experiment tracking | No record of past runs/metrics beyond the single `model_comparison.csv` that gets overwritten |
| Testing | Zero tests |
| Serving | No way to score a new transaction — it's a script, not a service |
| Explainability | No SHAP/feature-importance output for analyst trust or audit |
| Monitoring | No drift detection — fraud patterns shift over time, this is the #1 real-world failure mode |
| CI/CD | Nothing automated |
| Governance | No audit log of predictions (who/what/when/which model version) — a real compliance requirement in AML |

## 2. SDLC mapping

| Phase | Artifact to add |
|---|---|
| Requirements/Design | Architecture diagram, data flow diagram, model card |
| Development | Config-driven pipeline, modular structure, pre-commit hooks |
| Testing | Unit tests (features, metrics), data validation schema, integration test on the trained pipeline |
| CI/CD | GitHub Actions: lint → test → smoke-train on a small sample on every push |
| Deployment | Packaged model + FastAPI service + Dockerfile |
| Monitoring/Maintenance | Drift detection, retraining trigger, prediction audit log |

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
- Velocity features (transactions/hour per account), graph features (fan-in/fan-out across `nameOrig`/`nameDest`)
- SHAP explainability per prediction
- Threshold tuning at a K set from actual review capacity, not a multiple of the (unknowable in production) true fraud count
- Error analysis report (what the false positives in the top-K actually look like)

### Sprint 3 — Serving
- FastAPI service wrapping the best model, pydantic request/response schemas
- Dockerfile + docker-compose
- Batch-scoring script for historical data alongside the single-transaction endpoint

### Sprint 4 — CI/CD & testing
- pytest unit tests for `features.py`/`custom_metrics.py` (would have caught the groupby-scaling bug and the float32-precision bug found while porting this project to the real dataset)
- Data validation on ingest (pandera/Great Expectations schema check)
- GitHub Actions: lint (ruff) → test → smoke-train on sample data on every push

### Sprint 5 — MLOps/monitoring
- Model registry/versioning (MLflow registry or a simple versioned artifact convention)
- Drift detection (PSI on feature distributions and prediction scores over time)
- Scheduled retraining job (cron or Prefect)
- Prediction audit log (model version, feature values, timestamp — for compliance traceability)

### Sprint 6 — Portfolio polish
- Architecture diagram, demo (Streamlit app: upload a transaction batch → get fraud scores + SHAP explanation)
- Business-impact write-up (estimated $ fraud caught vs. false-positive review cost at the chosen threshold)

## 4. End-to-end features/workflow to add

- **Feature store simulation**: precompute account-level rolling stats once instead of recomputing per run — also addresses the scale/memory constraints already hit on the 8GB dev machine
- **Hybrid decisioning**: business rule layer (hard blocks) on top of the ML score — realistic in real fraud systems, rarely pure ML
- **Feedback loop**: analyst-confirmed labels (fraud/false-positive) feeding back into the next retraining cycle
- **Alerting**: push high-score transactions to a review queue (Slack webhook or simple dashboard)
- **Champion/challenger evaluation**: compare a new model against the currently deployed one before promoting it

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
      window is PR-AUC=1.0000 for every tree model but PR-AUC=0.06-0.08 for
      every linear model. Checked directly against the raw transaction
      table (not our engineered features, so not a leak): in that window
      98.8% of fraud has `amount_to_balance_ratio` exactly 1.00 (account
      drained to the cent) and 100% has `dest_is_merchant=0` -- a known
      PaySim construction characteristic. That's a narrow, nonlinear
      value-band rule trees split out trivially and a single linear
      hyperplane structurally cannot express regardless of class
      weighting. See `README.md` Results for the full writeup and
      `data/processed/model_comparison_by_fold.csv` for the numbers.
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
- [ ] **Nothing from Sprint 1 onward is staged/committed yet.** Working
      tree has: `.gitignore`, `README.md`, `ROADMAP.md`, `config.yaml`,
      `data/processed/model_comparison.csv`, `data/processed/
      precision_recall_at_k.csv`, `requirements.txt`, `src/
      train_pipeline.py` modified; `src/cv.py` and the two `_by_fold.csv`
      files untracked. Held back deliberately per the project's git
      workflow (see below) -- only stage once iteration is genuinely
      done, not mid-cycle. If a future session picks this back up,
      `git status`/`git diff` are the source of truth for exactly what's
      pending; this file documents the *why* behind each pending change.
- [ ] Sprint 2
- [ ] Sprint 3
- [ ] Sprint 4
- [ ] Sprint 5
- [ ] Sprint 6
