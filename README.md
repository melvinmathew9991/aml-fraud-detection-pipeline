# AML Fraud Detection Pipeline

A cost-sensitive fraud-detection pipeline for imbalanced payment transaction
data. Trains and compares several classifiers (Logistic Regression,
Ridge- and Lasso-penalized variants, HistGradientBoosting) under a custom
weighted loss, evaluates them with Precision@K instead of relying on
accuracy or ROC-AUC alone, and uses a time-based train/test split to avoid
leakage from future transactions into training data.

## Dataset

[PaySim](https://www.kaggle.com/datasets/ealaxi/paysim1) (Lopez-Rojas,
Elmir & Axelsson, 2016), a mobile-money transaction simulator: 6.36M
transactions, ~0.13% fraud rate. Its schema (sender/receiver balances,
transfer types) maps closely onto real payments-infrastructure data,
including sender/receiver balance reconciliation and account-emptying
patterns that are strong real-world fraud signals.

## Results

| Model | PR-AUC | ROC-AUC | Precision@21,250 |
|---|---|---|---|
| HistGradientBoosting (class_weight=balanced) | 0.9995 | 0.9999 | 0.200 |
| Logistic Regression (class_weight=balanced) | 0.9946 | 0.9991 | 0.199 |
| Ridge-penalized Logistic Regression (L2) | 0.9946 | 0.9990 | 0.199 |
| Lasso-penalized Logistic Regression (L1) | 0.9945 | 0.9986 | 0.199 |

**PR-AUC alone overstates how usable this is.** At the operating point that
would flag the top 21,250 highest-risk transactions, only ~1 in 5 flagged
transactions is actually fraud — 80% false positives. Ranking is strong;
the decision threshold isn't yet tuned to a realistic review-queue budget.
Closing that gap (velocity/graph features, SHAP-driven error analysis,
threshold tuning) is the next body of work — see `ROADMAP.md`.

## Project structure

```
fraud-detection-project/
├── data/
│   ├── raw/                    # source CSV (not committed -- see .gitignore)
│   └── processed/
│       ├── paysim.duckdb        # cached raw + engineered feature tables (not committed)
│       └── model_comparison.csv # latest run's metrics
├── models/                     # trained model artifacts per run (not committed)
│   └── <run_id>/                 scaler + one .joblib per model + metadata.json
├── reports/                    # per-run training logs (generated)
├── src/
│   ├── config.py               # loads config.yaml
│   ├── generate_sample_data.py # builds a schema-accurate synthetic sample for local dev
│   ├── features.py             # feature engineering as a DuckDB SQL query
│   ├── custom_metrics.py       # weighted BCE loss + Precision@K
│   └── train_pipeline.py       # main training + evaluation pipeline
├── config.yaml                 # paths, split, and model hyperparameters
├── requirements.txt
├── ROADMAP.md                  # production-readiness plan, sprint by sprint
└── README.md
```

## Running it

```
pip install -r requirements.txt
python src/train_pipeline.py
```

Paths and model hyperparameters live in `config.yaml`, not hardcoded in the
script. Data loading and feature engineering run through DuckDB
(`data/processed/paysim.duckdb`) rather than pandas, so a full 6.36M-row run
holds under ~1GB peak memory and completes in well under a minute once the
raw CSV has been loaded once and cached. Each run persists trained models
plus a `metadata.json` (feature list, git commit, per-model metrics) to
`models/<run_id>/`, and a log to `reports/train_<run_id>.log`.

## Design notes

- **Leakage-safe features**: account-level aggregates (prior transaction
  count, prior average amount) are computed from each account's history
  strictly *before* the current transaction, via a SQL window function —
  never from same-timestamp or future rows.
- **Time-based split**: train on earlier transactions, test on later ones,
  rather than a random shuffle — a random split would let the model see
  "future" account behavior during training, which doesn't reflect how the
  model will actually be used.
- **Train-side undersampling**: the majority (non-fraud) class is
  downsampled only in training, keeping the test set at the real fraud
  rate so evaluation reflects deployment conditions.
- **Cost-sensitive evaluation**: Precision@K over raw accuracy/ROC-AUC,
  since a fraud team reviews a fixed-size queue of top-ranked alerts, not
  every transaction.

## Roadmap

See `ROADMAP.md` for the full sprint-by-sprint plan — closing the
Precision@K gap, a served API, CI/CD, drift monitoring, and further
polish.
