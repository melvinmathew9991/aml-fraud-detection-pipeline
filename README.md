# AML Fraud Detection Pipeline

A cost-sensitive fraud-detection pipeline for imbalanced payment transaction
data. Trains and compares several classifiers (Logistic Regression,
Ridge- and Lasso-penalized variants, HistGradientBoosting, XGBoost, LightGBM)
under a custom weighted loss, evaluates them with Precision@K/Recall@K
instead of relying on accuracy or ROC-AUC alone, and uses expanding-window
time-based cross-validation to avoid leakage from future transactions into
training data. XGBoost/LightGBM are additionally tuned via Optuna, and every
run is tracked in a local MLflow store.

The output is not just a ranking but a **deployable decision rule**: a score
threshold derived from analyst review capacity (not from the fraud labels),
shipped with SHAP reason codes per alert and an error-analysis profile of
what the review queue actually contains.

## Dataset

[PaySim](https://www.kaggle.com/datasets/ealaxi/paysim1) (Lopez-Rojas,
Elmir & Axelsson, 2016), a mobile-money transaction simulator: 6.36M
transactions, ~0.13% fraud rate. Its schema (sender/receiver balances,
transfer types) maps closely onto real payments-infrastructure data,
including sender/receiver balance reconciliation and account-emptying
patterns that are strong real-world fraud signals.

## Results

Metrics below are mean ± std across **3 expanding-window time-based CV
folds** (`src/cv.py`), not a single train/test split. "Precision@capacity"
is precision at a review queue sized by analyst headcount (500 reviews/day,
`config.yaml`) rather than by the true fraud count — see "Setting the
operating point from capacity, not from the labels" below for why that
distinction is the point of Sprint 2.

| Model | PR-AUC (mean ± std) | Weighted BCE | Precision@capacity (mean ± std) | Recall@capacity |
|---|---|---|---|---|
| LightGBM (tuned) — `best_model` | 0.9968 ± 0.0054 | 0.056 | 0.5291 ± 0.0315 | 0.997 |
| LightGBM (default params) | 0.9963 ± 0.0053 | 0.059 | 0.5288 ± 0.0320 | 0.997 |
| HistGradientBoosting (class_weight=balanced) | 0.9965 ± 0.0057 | **0.025** | 0.5284 ± 0.0306 | 0.996 |
| XGBoost (default params) | 0.9966 ± 0.0058 | 0.031 | 0.5283 ± 0.0316 | 0.997 |
| XGBoost (tuned) | **0.9971** ± 0.0050 | 0.791 | 0.5281 ± 0.0335 | 0.997 |
| Logistic Regression (class_weight=balanced) | 0.621 ± 0.293 | 0.076 | 0.5015 ± 0.0693 | 0.942 |
| Ridge-penalized Logistic Regression | 0.580 ± 0.346 | 0.070 | 0.4704 ± 0.1156 | 0.880 |
| Lasso-penalized Logistic Regression | 0.561 ± 0.389 | 0.082 | 0.4350 ± 0.1715 | 0.810 |

**Read the tree rows as a tie.** The five tree models span 0.5281–0.5291 on
the selection metric while the fold-to-fold std is ±0.032 — a spread ~30x
smaller than the noise. `best_model` is LightGBM (tuned), but nothing here
supports a claim that it is genuinely better than the other four; the metric
separates trees from linear baselines decisively and does not separate trees
from each other at all. Note also that the ±0.032 is *not* mostly model
variance — see "The capacity metric is not comparable across folds" below
for why the mean of these three folds hides three very different operating
points. XGBoost (tuned) still leads on PR-AUC while carrying
a weighted-BCE ~31x worse than the best-calibrated model
(HistGradientBoosting, 0.025), which is the
Sprint 1 calibration finding reproduced and amplified on the larger feature
set.

Full Precision@K/Recall@K curves at K in {1x, 2x, 5x, 10x} true fraud count
remain in `data/processed/precision_recall_at_k.csv` (fold-aggregated) and
`..._by_fold.csv` (per-fold rows), for model-vs-model comparison.

### Setting the operating point from capacity, not from the labels

Every Precision@K number this project reported before Sprint 2 used
K = some multiple of the test window's **true fraud count**. That is a
reasonable way to compare models offline and a bad way to deploy one: the
true fraud count for today's transactions is only knowable in hindsight,
which is exactly what the model exists to predict. An operating point tuned
against it is a quiet form of using the labels at decision time.

Analyst headcount, by contrast, is known in advance. So K is now
`reviews_per_day x days in the window` (`src/threshold.py`), and the score
cutoff is read off at that K. The result is a genuinely deployable artifact
— `decision_threshold` in `metadata.json`, applicable to a single incoming
transaction with no knowledge of any label.

Sweeping that capacity on the final fold (4,250 fraud in 16.2 days) is the
most decision-relevant output in the project:

| Reviews/day | K | Precision | Recall | Fraud caught | False positives |
|---|---|---|---|---|---|
| 100 | 1,617 | 1.0000 | 0.381 | 1,619 | 0 |
| 250 | 4,042 | 1.0000 | 0.951 | 4,042 | 0 |
| 500 (configured) | 8,083 | 0.5258 | 1.000 | 4,250 | 3,833 |
| 1,000 | 16,167 | 0.2629 | 1.000 | 4,250 | 11,917 |
| 2,000 | 32,333 | 0.1314 | 1.000 | 4,250 | 28,083 |

**The model is at the precision ceiling at every capacity level.** Once
recall saturates, precision at K is bounded above by `total_fraud / K`, so
every review slot beyond the fraud volume can only add a false positive.
At 500/day the reported 0.5258 precision is exactly 4250/8083 — the
arithmetic maximum, not a model shortfall. The 3,833 false positives exist
because the queue is sized ~1.9x actual fraud volume.

The operationally interesting line is 250/day: **zero false positives at
95.1% recall**. Going from 250 to 500 reviews/day buys the last 208 frauds
at a cost of 3,833 false positives — roughly **18 false positives per
additional fraud caught**. That is a staffing decision with an explicit
exchange rate attached, which is the actual deliverable; the pipeline
deliberately does not pick a point on this curve by itself.

Two honest caveats. First, sitting on the ceiling at *every* level means the
ranking is essentially perfect on this fold — that reflects how separable
PaySim is (see the fold-2 finding below), and is not a claim that a
real-world model would behave this way. It is also specific to this fold:
on fold 1 the same model is *not* at the ceiling, missing 9 of 887 frauds
(recall 0.990, precision 0.5621 against a ceiling of 887/1562 = 0.5678).
Second, choosing 250/day because this table says so is itself a
label-informed decision; it is legitimate capacity planning on historical
data rather than model tuning, but it will drift as fraud volume changes,
which is a Sprint 5 monitoring concern.

### The capacity metric is not comparable across folds

`capacity_precision_mean` is the metric `best_model` is selected on, and it
averages three folds that are not measuring the same thing. Folds are cut on
`step` quantiles, so each test window holds ~20% of the *rows* — but PaySim's
transaction volume collapses in later steps, so those equal row counts span
very unequal amounts of time:

| Fold | Test rows | Days | Fraud | Fraud/day | Txns/day | K at 500/day | Alert rate | Fraud rate |
|---|---|---|---|---|---|---|---|---|
| 1 | 1,252,147 | 3.1 | 887 | 284 | ~401k | 1,562 | 0.125% | 0.071% |
| 2 | 1,293,285 | 3.1 | 770 | 250 | ~419k | 1,542 | 0.119% | 0.060% |
| 3 | 1,248,736 | 16.2 | 4,250 | 263 | ~77k | 8,083 | 0.647% | 0.340% |

Because `K = reviews_per_day x days`, fold 3 gets **5.2–5.4x more review
slots per transaction** than folds 1–2, and it carries ~5x the fraud prevalence
(the pipeline's own `pos_weight` reads 293 there versus 1411 and 1679).
Precision at a fixed alert budget is bounded by prevalence, so fold 3
structurally permits far higher precision than folds 1–2 at identical
staffing — before any model is involved.

That has two consequences worth stating plainly. The ±0.032 fold-to-fold
std is mostly window density and prevalence, not model variance, so it
overstates the noise floor for *comparing models* while understating how
much the operating point itself moves. And the mean of the three is an
average over three different operating points rather than three estimates of
one quantity; the per-fold columns in `model_comparison_by_fold.csv` are the
more honest read. Model *ranking* is unaffected — every model is scored on
the same three windows — which is why selection still works, but the
absolute number should not be quoted as "the precision this model achieves
at 500 reviews/day" without naming the fold.

Fraud per day is roughly stable across the three windows (250–284), which is
what makes a headcount-based K defensible in the first place; it is
transaction volume, not fraud volume, that swings. A production system would
size the queue against a rolling estimate of both.

### A CV finding worth flagging — and what Sprint 2 did to it

Moving from a single 80/20 split to 3 CV folds surfaced something a single
split never would have. Fold 2's test window (simulated hours 282–355) is
almost perfectly separable by the tree models (PR-AUC 0.9987–1.0000) but
not by the linear ones. Checked directly against the raw, un-engineered
transaction table (not a leak from our own feature pipeline): in that
window, 98.8% of fraud transactions have `amount_to_balance_ratio` exactly
1.00 (the source account drained to the cent) and 100% have
`dest_is_merchant = 0` — a known characteristic of how PaySim constructs
its synthetic fraud. That's a narrow, nonlinear value-band rule: trees
split it out trivially, while logistic regression's single hyperplane
cannot express "ratio in a tight band" no matter how the positive class is
weighted.

**Sprint 2 partly falsified the strong version of that claim, which is the
more interesting result.** On the Sprint 1 feature set the linear models
scored PR-AUC 0.06–0.08 on fold 2 — a genuine collapse. On the Sprint 2
feature set the same models, on the same fold, score **0.24–0.41** (Lasso
0.239, Ridge 0.304, Logistic Regression 0.410): a 3–7x improvement with no
change to the models or the fold, only to the features. The cause is the
transaction-type indicators. Fraud in PaySim occurs only in TRANSFER and
CASH_OUT, and "type is one of two values" *is* expressible as a hyperplane
constraint, unlike the ratio band — so giving a linear model an
axis-aligned handle on the same pattern recovered a large part of what it
had been missing.

The structural point survives: 0.41 is still nowhere near 1.0000, and the
value band itself remains inexpressible to a linear model. But the honest
version is weaker than "linear models cannot do this window" — they could
not do it *with the Sprint 1 features*. A good share of what looked like a
model-class limitation was a feature-representation gap, which is a caution
worth carrying into any future "linear baseline underperforms" claim. Full
breakdown in `model_comparison_by_fold.csv`.

The linear baselines' 3-fold PR-AUC means are listed in the results table,
but read the ±0.29–0.39 standard deviations as the real content: Lasso's
0.56 ± 0.39 is one near-perfect window (0.993) averaged with two poor ones
(0.239, 0.450), not a performance estimate. Their precision@capacity
figures (0.44–0.50 ± 0.07–0.17) are the more stable comparison, and still
sit clearly below every tree model.

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
weighted-BCE of **0.791 versus 0.031** for the untuned default — ~25x
worse, and at the capacity operating point it is marginally *behind*
LightGBM, not ahead. (On the Sprint 1 feature set this gap was 0.332 vs
0.127; widening the feature set widened the gap rather than closing it,
which is what an unconstrained PR-AUC search should be expected to do given
more room to trade calibration for ranking.) Two
separate cost-sensitivity mechanisms are in tension here: `scale_pos_weight`
during training now correctly reflects the resampled training ratio
(~50:1, matching what `class_weight='balanced'` uses for the other
models — see the fix below), while the weighted-BCE *evaluation* metric
uses the true deployment-time ratio (~300–1700:1). Optuna's hyperparameter
search, chasing PR-AUC only, found a configuration that ranks fraud
slightly better while drifting further from well-calibrated probabilities
under that evaluation-time weighting. Concretely: **"best mean PR-AUC" and
"best model" aren't the same model here**, so `train_pipeline.py` does not
select `best_model` by mean PR-AUC.

Sprint 1 moved that selection to Precision@K at K=1x fraud count. Sprint 2
moved it once more, to precision at the review-capacity operating point,
for the reason given above: K=1x-fraud-count is still defined by the
labels, so "the model that wins at K=1x" is a criterion you could not
actually evaluate on the day you had to deploy. Capacity-based K has the
same operational shape and *is* computable in advance. Under that criterion
`best_model` is **LightGBM (tuned)** — though see the tie caveat under the
results table: that choice is not meaningfully better than the other tree
models, and the honest claim is only that a tree model should be selected
and that the PR-AUC leader should not be. Optuna's tuning still runs and
its result is still reported for comparison
(`xgboost_best_params`/`top_pr_auc_model` in `metadata.json`), it just
isn't automatically treated as "the" winner.

A related, now-fixed calibration bug from the same investigation:
`scale_pos_weight` for XGBoost/LightGBM used to be set to the *true*
deployment-time cost ratio (~300–1700:1) even though training data was
already undersampled to 50:1 — double-correcting for imbalance on top of
undersampling. It's now set to the post-undersampling ratio actually seen
by the trained model (mirroring `class_weight='balanced'`'s behavior on
the other models), with the true ratio reserved for evaluation only.

### The roadmap's velocity feature was not computable — and the data said so

Sprint 2 was specified as "velocity features (transactions/hour per account),
graph features (fan-in/fan-out across `nameOrig`/`nameDest`)". Profiling the
6.36M-row table before writing any of it showed two of those three cannot
exist on PaySim:

- **`nameOrig` is effectively a per-transaction identifier.** 6,344,009 of
  the 6,353,307 origin accounts appear exactly once, and only 0.15% of rows
  have any prior transaction from the same origin. Per-origin velocity would
  be zero for 99.85% of rows, and fan-*out* is degenerate for the same reason.
- **Fan-in as "distinct prior senders" is redundant here.** Because origins
  are never reused, every `(nameOrig, nameDest)` pair in the dataset is
  unique, so distinct-sender count is *identical* to prior-transaction count
  (verified: 0 rows differ). Shipping both would have been two names for one
  column.

So velocity and graph aggregates are built on the **destination** side, where
PaySim's repeat structure actually lives (571,961 non-merchant destinations
averaging 7.36 transactions, max 113) — and which is the more AML-relevant
direction anyway, since the pattern worth catching is a mule account taking
rapid inflows. Fraud destinations are measurably quieter: 2.7 prior
transactions on average versus 5.1 for non-fraud.

Sprint 2 also closed a plain gap: **transaction type was not a feature at
all**, despite fraud appearing in only 2 of the 5 types (TRANSFER 0.77%,
CASH_OUT 0.18%, and exactly zero across the 3.59M CASH_IN/PAYMENT/DEBIT
rows). The model had only been seeing it indirectly through
`dest_is_merchant`.

Feature count went 11 → 20. All new aggregates are strictly prior-only; the
window-frame implementation was verified against an independent
correlated-self-join restatement over 8,471 rows across 300 busy
destinations, with zero disagreements, zero first-transactions showing prior
history, and same-step peers correctly excluded from the velocity window.

### What the new features were actually worth: not much

The ablation (`data/processed/feature_ablation.csv`, LightGBM defaults, final
fold, K=8,083) attributes the movement rather than asserting it:

| Feature set | n | PR-AUC | Weighted BCE | False positives | Fraud missed |
|---|---|---|---|---|---|
| Sprint 1 features only | 11 | 0.99953 | 0.0074 | 3,870 | 2 |
| + transaction type | 15 | 0.99995 | 0.0024 | 3,867 | 0 |
| + destination aggregates | 16 | 0.99953 | 0.0735 | 3,846 | 2 |
| all Sprint 2 features | 20 | 0.99998 | 0.0023 | 3,833 | 0 |

Net effect of nine new features: **37 fewer false positives out of ~3,870
(a 0.96% reduction) and 2 fewer missed frauds**. That is a real but small
gain, and it is small for a defensible reason — the Sprint 1 feature set had
already pushed this dataset to PR-AUC 0.9995, leaving almost nothing to
recover. The two groups also do different jobs: transaction type is what
eliminates the missed frauds and improves calibration ~3x, while the
destination aggregates cut false positives but *worsen* calibration ~10x on
their own; only combined do both metrics land at their best.

### SHAP confirmed two features were dead weight

Global attribution (`shap_global_importance.csv`, mean |SHAP| over 20k
sampled test rows) is led by the balance-reconciliation features:
`orig_balance_mismatch` (30.1% of total attribution), `orig_emptied`
(14.5%), `dest_is_merchant` (8.7%), `amount_to_balance_ratio` (7.6%),
`orig_balance_delta` (6.6%). The nine Sprint 2 features together carry
~20.5%, led by `is_cash_out` (5.8%) and `dest_prior_txn_count` (3.5%).

The useful result is at the bottom of the table: **`orig_prior_txn_count`
and `orig_prior_avg_amount` have exactly zero attribution** — the model
makes no use of them whatsoever. Those are precisely the two features the
account-structure profiling predicted would be near-dead (nonzero on 0.15%
of rows). They were deliberately kept rather than dropped, so that the
explainability layer had a known-dead pair to catch; it caught them. They
are retained for now because removing them would break comparability with
the Sprint 1 baseline, and are flagged for removal in a later sprint.

Per-alert reason codes for the 50 highest-scoring alerts are in
`shap_alert_reasons.csv` — each alert's top contributing features with their
raw (un-scaled) values, which is the form an analyst or an AML audit needs,
rather than a bare score.

### Error analysis: the queue has no misses, and the false positives are small transfers

At 500 reviews/day on the final fold the queue is 4,250 true fraud, 3,833
false positives, **0 fraud missed**. The false positives separate from true
fraud most sharply on transaction size and account-draining behaviour
(`error_analysis_profile.csv`, gaps in pooled-std units): `orig_balance_delta`
−7.51 sd (FP mean 24,038 vs TP mean 1,555,748), `dest_amount_to_prior_avg_ratio`
−4.39 sd, `amount` −3.05 sd, `orig_emptied` −2.19 sd (FP 0.029 vs TP 0.971),
`orig_balance_mismatch` +1.95 sd.

In plain terms: the false positives are **small transfers that did not empty
the source account**, while true fraud is large and drains the account to the
cent. They are not random noise — they sit in the same region of feature
space as fraud, just at lower magnitude. Since there are no misses to
recover at this capacity, the lever is the threshold and the queue size, not
more features.

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
│       ├── precision_recall_at_k_by_fold.csv   # raw per-(model,fold,K) curve rows
│       ├── capacity_sweep.csv                  # operating point vs. analyst review capacity
│       ├── shap_global_importance.csv          # mean |SHAP| per feature
│       ├── shap_alert_reasons.csv              # per-alert reason codes for top alerts
│       ├── error_analysis_profile.csv          # TP/FP/FN feature profile of the review queue
│       ├── missed_fraud_summary.csv            # where missed fraud sits in the ranking
│       └── feature_ablation.csv                # metric movement attributed per feature group
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
│   ├── threshold.py            # capacity-based K, decision threshold, capacity sweep
│   ├── explain.py              # SHAP global importance + per-alert reason codes
│   ├── error_analysis.py       # review-queue TP/FP/FN profiling, missed-fraud ranking
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
stays around 1.4GB even with 3 CV folds, 6 models, and 20 Optuna trials in a
single run. Each Optuna trial fits across all 3 CV folds (its objective is
mean PR-AUC across folds, not a single fold — see Results), which is the
dominant cost — `optuna.n_trials` was cut from 25 to 10 per model
specifically to keep this reasonable on this project's 8GB/dual-core dev
machine, once the raw CSV is cached. A full Sprint 2 run takes **~28
minutes** end to end (measured at 28.0 min on the dev machine described
above; per-run timings and memory land in `reports/train_<run_id>.log`,
which is generated rather than committed — see `.gitignore`),
up from ~15 in Sprint 1 — the feature set nearly doubled, 11 → 20, so every
fit in every Optuna trial costs more, and the SHAP/error-analysis/ablation
stage is new. Peak RSS is 1,367 MB, reached while fitting the fold
baselines rather than during feature loading. The
one-time feature rebuild after a `FEATURE_VERSION` bump adds ~2.5 minutes
and is cached thereafter; DuckDB is held to an explicit 2GB
`memory_limit` (`config.yaml`) rather than its default ~80%-of-RAM, which
is what previously hung this machine. Each run persists the final CV fold's
trained models plus a `metadata.json` (feature list, git commit, Optuna best
params, per-fold metrics, and the deployable `decision_threshold`) to
`models/<run_id>/`, a log to
`reports/train_<run_id>.log`, and every (model, fold) result to the local
MLflow store (`mlflow.db`) — browse with:

```
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## Design notes

- **Leakage-safe features**: account-level aggregates (prior transaction
  count, prior average amount, destination fan-in and 24h velocity) are
  computed from each account's history strictly *before* the current
  transaction, via SQL window frames ending at `1 PRECEDING` — never from
  same-timestamp or future rows. The velocity window uses a `RANGE` frame
  over `step`, so "last 24 hours" means 24 simulated hours rather than the
  previous 24 rows, and same-hour peers are excluded too (intra-hour
  ordering in PaySim is a row-order artifact, so treating concurrent
  transactions as unobserved is the defensible reading). Verified against
  an independent self-join restatement — see Results.
- **Features follow the data, not the plan**: the roadmap's per-origin
  velocity and fan-out features were dropped after profiling showed
  `nameOrig` is 99.85% unique, rather than shipped as columns that would be
  zero for 99.85% of rows. See Results.
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
- **The operating point is set from capacity, not from labels**: K is
  `reviews_per_day x window length`, so the resulting score threshold is
  computable before any label exists and ships as part of the model
  artifact. Precision at that K is always reported against its ceiling
  (`total_fraud / K`), because once recall saturates that ceiling — not the
  model — is what caps the number.
- **Explainability as an artifact, not a notebook cell**: SHAP global
  importance plus per-alert reason codes (feature, raw value, signed
  contribution) are written to `data/processed/` and logged to MLflow on
  every run, which is the form an AML audit trail needs.
- **Optuna tuning + MLflow tracking**: XGBoost/LightGBM hyperparameters are
  tuned via Optuna (sequential trials — this dev machine has no spare
  cores), and every (model, fold) result plus tuning outcome is logged to
  a local MLflow store instead of overwriting a single CSV run to run.

## Roadmap

See `ROADMAP.md` for the full sprint-by-sprint plan. Sprints 0–2 are done
(engineering hygiene; experiment rigor with CV/Optuna/MLflow; and the
capacity-based operating point, destination velocity/graph features, SHAP,
and error analysis described above). Next up is Sprint 3 — a FastAPI
service wrapping the model and its `decision_threshold`, plus a Dockerfile
and a batch-scoring path — followed by CI/CD and testing, drift monitoring,
and portfolio polish.
