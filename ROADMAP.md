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

### Sprint 2 — Close the Precision@K gap
The real weak point in current results: PR-AUC is ~0.9996 but Precision@21250
is only ~0.20 — 4 out of 5 flagged transactions are false alarms.
- Velocity features (transactions/hour per account), graph features (fan-in/fan-out across `nameOrig`/`nameDest`)
- SHAP explainability per prediction
- Threshold tuning at the actual operating point instead of ranking by raw probability
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
- [ ] Sprint 0
- [ ] Sprint 1
- [ ] Sprint 2
- [ ] Sprint 3
- [ ] Sprint 4
- [ ] Sprint 5
- [ ] Sprint 6
