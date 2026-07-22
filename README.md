# Fraud/AML Detection on Imbalanced Payment Transaction Data

A cost-sensitive fraud-detection pipeline built to close specific gaps identified
against the NPCI Associate Data Science JD: payments/fraud domain exposure,
imbalanced-data handling, custom loss functions, Lasso/Ridge as named baselines,
and Precision@K -- a metric this author's own MEDBOT project audit had flagged
as missing.

## ⚠️ Status: scaffold built and tested on sample data — swap in real dataset before publishing

Everything here runs end-to-end right now on a **schema-accurate synthetic
sample** (`data/raw/paysim_transactions.csv`, 50,000 rows, ~0.026% fraud rate)
so the pipeline, custom loss, and Precision@K implementation are all proven
correct. Before publishing results or putting numbers on a resume:

1. Download the real dataset: https://www.kaggle.com/datasets/ealaxi/paysim1
   (~6.3M rows, same schema, same column names)
2. Replace `data/raw/paysim_transactions.csv` with the real file
3. Re-run: `python3 src/train_pipeline.py`
4. Install XGBoost/LightGBM locally (`pip install xgboost lightgbm`) and
   uncomment those sections in `train_pipeline.py` -- they're written
   correctly but couldn't be installed in this sandbox (no network access)

## ⚠️ A known artifact in the current sample results — read before quoting numbers

Every model currently reports **PR-AUC = 1.0000**. This is **not a real
result** — it's an artifact of the synthetic generator, and I'm flagging it
explicitly rather than letting it stand, the same way Nectar's own audit
flagged its 0.997 ROC-AUC as inflated by clean synthetic fault ramps.

**Root cause**: `generate_sample_data.py`'s fraud injection deterministically
sets `newbalanceOrig = 0` for every fraud case. The `orig_emptied` feature
then perfectly separates fraud from non-fraud in this sample — every model
finds this trivially. Real PaySim data has the same *tendency* but not a
100%-deterministic rule, so this gap will close substantially on the real
6.3M-row dataset. **Do not report PR-AUC = 1.0 as a real result** — treat it
as confirmation the pipeline runs, not as a finding.

## Dataset

PaySim mobile-money transaction simulator (Lopez-Rojas, Elmir & Axelsson,
2016), chosen over IEEE-CIS specifically because its transaction schema
(sender/receiver balances, mobile-money transfer types) maps more directly
onto a payments-infrastructure domain than IEEE-CIS's anonymized card
features -- easier to speak to in an interview about UPI-style rails.

## Project structure

```
fraud-detection-project/
├── data/
│   ├── raw/                    # source CSV (not committed -- see .gitignore)
│   └── processed/              # model comparison output (generated)
├── models/                     # trained model artifacts per run (not committed)
│   └── <run_id>/                 scaler + one .joblib per model + metadata.json
├── reports/                    # per-run training logs (generated)
├── src/
│   ├── config.py               # loads config.yaml
│   ├── generate_sample_data.py # builds the placeholder sample
│   ├── features.py             # temporal, behavioral, aggregated features
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
script. Each run writes trained models + a metadata.json (feature list,
git commit, per-model metrics) to `models/<run_id>/`, and a log to
`reports/train_<run_id>.log`.

## What each module demonstrates, mapped to the JD

| File | JD requirement it answers |
|---|---|
| `features.py` | "temporal, behavioral, aggregated features" — verbatim JD phrase, three functions, one each |
| `custom_metrics.py::weighted_bce_loss` | "custom loss functions (weighted BCE, cost-sensitive)" |
| `custom_metrics.py::precision_at_k` | Precision@K — closes MEDBOT's own documented gap |
| `train_pipeline.py` — Lasso/Ridge baselines | Named explicitly in the JD, distinct from generic "Logistic Regression" |
| `train_pipeline.py` — time-based split | Same leakage-safety pattern as Clinical EMR/Deep Learning Call Center |
| `train_pipeline.py` — PR-AUC selection | Reuses the exact methodology from Nectar's model selection under class imbalance |

## Honest scope of what this does and doesn't close

**Closes for real:** payments/fraud domain, cost-sensitive learning, custom
loss functions, Lasso/Ridge, Precision@K, imbalanced-data handling.

**Does not close:** this is classical ML (Logistic Regression, Random Forest,
XGBoost/LightGBM once installed) — it does **not** touch Graph AI/GNNs,
PyTorch, or GPU/CUDA optimization. Those are Project 2's job (GNN on Nectar's
existing asset graph), not this one's.

## Next steps to finish the project

1. Swap in the real PaySim dataset and re-run — get real, non-artifact numbers
2. Install and run the XGBoost/LightGBM sections (already written)
3. Add a simple neural network (PyTorch) using the custom weighted-BCE loss
   directly as its training objective — bridges this project toward PyTorch
   exposure without needing Project 2's graph data
4. Write up final results with the real dataset before adding to resume/portfolio
