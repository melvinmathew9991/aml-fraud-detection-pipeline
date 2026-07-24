# AML Fraud Detection Pipeline

A cost-sensitive fraud-detection pipeline for imbalanced payment transaction
data. Trains and compares several classifiers (Logistic Regression,
Ridge- and Lasso-penalized variants, HistGradientBoosting, XGBoost, LightGBM)
under a custom weighted loss, evaluates them with Precision@K/Recall@K
instead of relying on accuracy or ROC-AUC alone, and uses expanding-window
time-based cross-validation to avoid leakage from future transactions into
training data. XGBoost/LightGBM are additionally tuned via Optuna, and every
run is tracked in a local MLflow store.

## Dataset

[PaySim](https://www.kaggle.com/datasets/ealaxi/paysim1) (Lopez-Rojas,
Elmir & Axelsson, 2016), a mobile-money transaction simulator: 6.36M
transactions, ~0.13% fraud rate. Its schema (sender/receiver balances,
transfer types) maps closely onto real payments-infrastructure data,
including sender/receiver balance reconciliation and account-emptying
patterns that are strong real-world fraud signals.

## Results

Metrics below are mean ± std across **3 expanding-window time-based CV
folds** (`src/cv.py`), not a single train/test split — see "A CV finding
worth flagging" below for why only the tree models are summarized this way.

| Model | PR-AUC (mean ± std) | Precision@K (mean, K=1x fraud count) | Recall@K (mean) |
|---|---|---|---|
| XGBoost (tuned) | 0.9971 ± 0.0048 | 0.996 | 0.996 |
| XGBoost (default params) | 0.9968 ± 0.0052 | 0.995 | 0.995 |
| LightGBM (tuned) | 0.9968 ± 0.0053 | 0.996 | 0.996 |
| LightGBM (default params) | 0.9968 ± 0.0053 | 0.996 | 0.996 |
| HistGradientBoosting (class_weight=balanced) | 0.9965 ± 0.0057 | 0.996 | 0.996 |

XGBoost (tuned) leads on mean PR-AUC (ranking quality) but is *not* the
best pick by Precision@K, and its calibration is markedly worse than the
others — see "Optuna improved ranking but hurt calibration" below.
**The pipeline's `best_model` selection is Precision@K-based, not
PR-AUC-based, for exactly this reason** — it currently picks LightGBM
(default params), not the higher-PR-AUC XGBoost (tuned).

Precision@K depends on both the model and where K is set relative to the
true fraud count, so it's reported here at K = the test set's actual fraud
volume (the realistic "review queue sized to daily fraud volume" operating
point) rather than at an arbitrary multiple. Full curve — Precision@K and
Recall@K at K in {1x, 2x, 5x, 10x} true fraud count, per model and per
fold — is in `data/processed/precision_recall_at_k.csv` (fold-aggregated)
and `..._by_fold.csv` (raw per-fold rows).

### A CV finding worth flagging

Moving from a single 80/20 split to 3 CV folds surfaced something a single
split never would have. Fold 2's test window (simulated hours 282–355) is
almost perfectly separable by the tree models (PR-AUC = 1.0000) but not by
the linear ones (PR-AUC collapses to 0.06–0.08 in that same window — full
breakdown in `model_comparison_by_fold.csv`). Checked directly against the
raw, un-engineered transaction table (not a leak from our own feature
pipeline): in that window, 98.8% of fraud transactions have
`amount_to_balance_ratio` exactly 1.00 (the source account drained to the
cent) and 100% have `dest_is_merchant = 0` — a known characteristic of how
PaySim constructs its synthetic fraud. That's a narrow, nonlinear
value-band rule: tree models split it out trivially, but logistic
regression's single hyperplane can't express "ratio in a tight band" no
matter how the positive class is weighted. It's a genuine, fold-specific
property of the data, not a bug — and a concrete illustration of why tree
ensembles beat linear models on fraud specifically, so the linear
baselines' 3-fold PR-AUC means aren't reported above (their per-fold
variance from this one window is too large to summarize as a single
number honestly).

### Optuna improved ranking but hurt calibration — and that's informative too

10-trial Optuna tuning per model (search spaces in `train_pipeline.py`;
cut from 25 to keep full-run time reasonable — see Running It) scores
each trial's PR-AUC averaged across all 3 CV folds, not a single fold (an
earlier version scored against one fold that turned out to be trivially
separable — see the finding above — which gave Optuna no real signal to
discriminate between trials). With the fix, XGBoost (tuned) does land
ahead on mean PR-AUC: 0.9971 vs. 0.9968 untuned.

But PR-AUC only measures rank ordering, and Optuna's objective is PR-AUC
alone — nothing in the search rewards well-calibrated probabilities. The
custom weighted-BCE loss (`custom_metrics.py`), which *is*
calibration-sensitive, tells a different story: XGBoost (tuned) has a mean
weighted-BCE of 0.332 versus 0.127 for the untuned default — over 2.5x
worse, and at the realistic Precision@K=1x operating point it's actually
marginally *behind* LightGBM, not ahead. Two
separate cost-sensitivity mechanisms are in tension here: `scale_pos_weight`
during training now correctly reflects the resampled training ratio
(~50:1, matching what `class_weight='balanced'` uses for the other
models — see the fix below), while the weighted-BCE *evaluation* metric
uses the true deployment-time ratio (~300–1700:1). Optuna's hyperparameter
search, chasing PR-AUC only, found a configuration that ranks fraud
slightly better while drifting further from well-calibrated probabilities
under that evaluation-time weighting. Concretely: **"best mean PR-AUC" and
"best model" aren't the same model here**, so `train_pipeline.py` no
longer selects `best_model` by mean PR-AUC. It now selects by mean
Precision@K at the realistic K=1x-fraud-count operating point — this
project's own thesis throughout is that Precision@K is the operationally
relevant metric, not PR-AUC alone (a fraud team reviews a fixed-size
queue, not a probability ranking). Under that criterion, **LightGBM
(default params)** is `best_model`, not the higher-PR-AUC XGBoost
(tuned) — a deliberate resolution, not a workaround: Optuna's tuning
still ran and its result is still reported for comparison
(`xgboost_best_params`/`top_pr_auc_model` in `metadata.json`), it just
isn't automatically treated as "the" winner anymore.

A related, now-fixed calibration bug from the same investigation:
`scale_pos_weight` for XGBoost/LightGBM used to be set to the *true*
deployment-time cost ratio (~300–1700:1) even though training data was
already undersampled to 50:1 — double-correcting for imbalance on top of
undersampling. It's now set to the post-undersampling ratio actually seen
by the trained model (mirroring `class_weight='balanced'`'s behavior on
the other models), with the true ratio reserved for evaluation only.

Sprint 2 (velocity/graph features, SHAP-driven error analysis, threshold
tuning at a capacity-based K) is still the next body of work — see
`ROADMAP.md`.

## Project structure

```
fraud-detection-project/
├── data/
│   ├── raw/                    # source CSV (not committed -- see .gitignore)
│   └── processed/
│       ├── paysim.duckdb                       # cached raw + engineered feature tables (not committed)
│       ├── model_comparison.csv                # fold-aggregated (mean/std) metrics, latest run
│       ├── model_comparison_by_fold.csv        # raw per-(model,fold) metrics
│       ├── precision_recall_at_k.csv           # fold-aggregated Precision/Recall@K curve
│       └── precision_recall_at_k_by_fold.csv   # raw per-(model,fold,K) curve rows
├── models/                     # trained model artifacts per run (not committed)
│   └── <run_id>/                 scaler + one .joblib per model (final CV fold) + metadata.json
├── mlflow.db                   # local MLflow tracking store (not committed)
├── reports/                    # per-run training logs (generated)
├── src/
│   ├── config.py               # loads config.yaml
│   ├── generate_sample_data.py # builds a schema-accurate synthetic sample for local dev
│   ├── features.py             # feature engineering as a DuckDB SQL query
│   ├── cv.py                   # expanding-window time-based CV fold generator
│   ├── custom_metrics.py       # weighted BCE loss + Precision@K / Recall@K
│   └── train_pipeline.py       # main training + CV + Optuna tuning + MLflow pipeline
├── config.yaml                 # paths, CV, model, Optuna, and MLflow config
├── requirements.txt
├── ROADMAP.md                  # production-readiness plan, sprint by sprint
└── README.md
```

## Running it

```
pip install -r requirements.txt
python src/train_pipeline.py
```

Paths, CV folds, and model/Optuna/MLflow settings live in `config.yaml`, not
hardcoded in the script. Data loading and feature engineering run through
DuckDB (`data/processed/paysim.duckdb`) rather than pandas, so peak memory
stays under ~1GB even with 3 CV folds, 6 models, and 20 Optuna trials in a
single run. Each Optuna trial fits across all 3 CV folds (its objective is
mean PR-AUC across folds, not a single fold — see Results), which is the
dominant cost — `optuna.n_trials` was cut from 25 to 10 per model
specifically to keep this reasonable (25 pushed a full run to ~30 minutes;
10 brings it to ~15) on this project's 8GB/dual-core dev machine, once the
raw CSV is cached. Each run persists the final CV fold's
trained models plus a `metadata.json` (feature list, git commit, Optuna best
params, per-fold metrics) to `models/<run_id>/`, a log to
`reports/train_<run_id>.log`, and every (model, fold) result to the local
MLflow store (`mlflow.db`) — browse with:

```
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## Design notes

- **Leakage-safe features**: account-level aggregates (prior transaction
  count, prior average amount) are computed from each account's history
  strictly *before* the current transaction, via a SQL window function —
  never from same-timestamp or future rows.
- **Expanding-window time-based CV**: 3 sequential folds (`src/cv.py`) —
  each trains on everything up to a cutoff and tests on the next slice —
  rather than one 80/20 split or a random shuffle. A random split would let
  the model see "future" account behavior during training; a single split
  leaves too few fraud rows in the test window to trust one PR-AUC number.
- **Train-side undersampling**: the majority (non-fraud) class is
  downsampled only in training, independently within each fold, keeping
  every fold's test set at the real fraud rate so evaluation reflects
  deployment conditions.
- **Cost-sensitive evaluation**: Precision@K/Recall@K over raw
  accuracy/ROC-AUC, since a fraud team reviews a fixed-size queue of
  top-ranked alerts, not every transaction.
- **Optuna tuning + MLflow tracking**: XGBoost/LightGBM hyperparameters are
  tuned via Optuna (sequential trials — this dev machine has no spare
  cores), and every (model, fold) result plus tuning outcome is logged to
  a local MLflow store instead of overwriting a single CSV run to run.

## Roadmap

See `ROADMAP.md` for the full sprint-by-sprint plan — closing the
Precision@K gap, a served API, CI/CD, drift monitoring, and further
polish.
