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
distinction is the point of Sprint 2. Sprint 3 removed two zero-SHAP-attribution
features (`orig_prior_txn_count`, `orig_prior_avg_amount` — see below), so the
model here is retrained on **18 features**, not 20; the numbers moved by
noise, not by a real effect, which is the expected result of dropping inputs
the model already wasn't using.

| Model | PR-AUC (mean ± std) | Weighted BCE | Precision@capacity (mean ± std) | Recall@capacity |
|---|---|---|---|---|
| LightGBM (tuned) — `best_model` | 0.9974 ± 0.0045 | 0.025 | 0.5293 ± 0.0318 | 0.997 |
| LightGBM (default params) | 0.9967 ± 0.0058 | 0.040 | 0.5291 ± 0.0315 | 0.997 |
| XGBoost (tuned) | 0.9968 ± 0.0054 | **0.810** | 0.5278 ± 0.0316 | 0.997 |
| XGBoost (default params) | 0.9967 ± 0.0056 | 0.082 | 0.5258 ± 0.0299 | 0.997 |
| HistGradientBoosting (class_weight=balanced) | 0.9935 ± 0.0052 | 0.025 | 0.5254 ± 0.0364 | 0.996 |
| Logistic Regression (class_weight=balanced) | 0.512 ± 0.433 | 0.095 | 0.3848 ± 0.2697 | 0.713 |
| Lasso-penalized Logistic Regression | 0.528 ± 0.442 | 0.094 | 0.3796 ± 0.2770 | 0.698 |
| Ridge-penalized Logistic Regression | 0.507 ± 0.443 | 0.087 | 0.3777 ± 0.2723 | 0.700 |

**Read the tree rows as a tie.** The five tree models span 0.5254–0.5293 on
the selection metric while the fold-to-fold std is ±0.03 — a spread ~10x
smaller than the noise. `best_model` is LightGBM (tuned), but nothing here
supports a claim that it is genuinely better than the other four; the metric
separates trees from linear baselines decisively and does not separate trees
from each other at all. Note also that the std is *not* mostly model
variance — see "The capacity metric is not comparable across folds" below
for why the mean of these three folds hides three very different operating
points. XGBoost (tuned) carries a weighted-BCE ~10x worse than its own
untuned default and ~32x worse than the best-calibrated model
(HistGradientBoosting, 0.025), which is the
Sprint 1 calibration finding reproduced and amplified on the larger feature
set — see "Optuna improved ranking but hurt calibration" below for why.

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
| 100 | 1,617 | 1.0000 | 0.380 | 1,617 | 0 |
| 250 | 4,042 | 1.0000 | 0.951 | 4,042 | 0 |
| 500 (configured) | 8,083 | 0.5257 | 1.000 | 4,250 | 3,835 |
| 1,000 | 16,167 | 0.2629 | 1.000 | 4,250 | 11,917 |
| 2,000 | 32,333 | 0.1314 | 1.000 | 4,250 | 28,086 |

**The model is at the precision ceiling at every capacity level.** Once
recall saturates, precision at K is bounded above by `total_fraud / n_flagged`
(the queue actually worked, which can exceed K on a tie — see `threshold.py`),
so every review slot beyond the fraud volume can only add a false positive.
At 500/day the reported 0.5257 precision is exactly 4250/8085 — the
arithmetic maximum, not a model shortfall. The 3,835 false positives exist
because the queue is sized ~1.9x actual fraud volume.

The operationally interesting line is 250/day: **zero false positives at
95.1% recall**. Going from 250 to 500 reviews/day buys the last 208 frauds
at a cost of 3,835 false positives — roughly **18 false positives per
additional fraud caught**. That is a staffing decision with an explicit
exchange rate attached; Sprint 3's `src/economics.py` turns it into an actual
net-value question (see below) rather than the pipeline picking a point on
this curve by itself.

Two honest caveats. First, sitting on the ceiling at *every* level means the
ranking is essentially perfect on this fold — that reflects how separable
PaySim is (see the fold-2 finding below), and is not a claim that a
real-world model would behave this way. It is also specific to this fold:
on fold 1 the same model is *not* at the ceiling, missing 8 of 887 frauds
(recall 0.991, precision 0.5627 against a ceiling of 887/1562 = 0.5678).
Second, choosing 250/day because this table says so is itself a
label-informed decision; it is legitimate capacity planning on historical
data rather than model tuning, but it will drift as fraud volume changes,
which is a Sprint 8 monitoring concern.

### The exchange rate has a price now: `src/economics.py` (Sprint 3)

The 18-false-positives-per-fraud figure above is an exchange rate, not a
decision — it says nothing about whether staffing up is worth it in money.
`src/economics.py` computes

```
net_value(K) = frauds_caught(K) x avg_fraud_amount x recovery_rate
             - alerts_reviewed(K) x cost_per_review
             - frauds_missed(K)  x avg_fraud_amount x liability_rate
```

against the real final-fold average fraud amount (**1,572,443**, measured,
not assumed) and reports three things:

**The naive "find the optimum" version is degenerate.** Solving for the
`cost_per_review` at which staffing up from 250 to 500 reviews/day stops
paying off gives a break-even range of **84,942–161,795** across
`recovery_rate` 0.05–1.00 (`liability_rate=1.0`, i.e. a missed fraud is a
full loss). No real alert review costs that, so full recall wins under every
plausible assumption and the "optimum" never moves — reporting it as a
finding would be presenting arithmetic as insight, so the module reports the
degeneracy explicitly instead (`degeneracy_check`).

**Cost of a capacity constraint** (`capacity_constraint_cost`): staffing only
250/day instead of 500/day misses 208 frauds, i.e. **~327,068,118 in
exposure**, or **~1,308,272 per marginal review-seat-day**. This is the
number a manager can take into a headcount conversation — linear and
honest, not an optimum.

**Ticket-size crossover** (`ticket_size_crossover`): sweeping `avg_fraud_amount`
at fixed business rates (`cost_per_review=200`, `recovery_rate=0.5`,
`liability_rate=1.0`, `config.yaml`) finds where the recommended staffing
level actually changes — at this fold's capacity sweep, recommended staffing
moves from 100→250 reviews/day around a **500** average-fraud-amount ticket,
and 250→500 around **5,000**. PaySim's own 1,572,443 average sits nowhere
near either crossover, which is the point: this sweep is where the exercise
becomes relevant to a UPI-scale (hundreds-to-low-thousands ticket) fraud
profile rather than PaySim's. Full curve in
`data/processed/capacity_economics.csv`.

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
scored PR-AUC 0.06–0.08 on fold 2 — a genuine collapse. Adding the
transaction-type indicators recovered a large part of it: fraud in PaySim
occurs only in TRANSFER and CASH_OUT, and "type is one of two values" *is*
expressible as a hyperplane constraint, unlike the value-band rule below, so
giving a linear model an axis-aligned handle on the same pattern helped. The
Sprint 2 writeup reported this recovery at PR-AUC 0.24–0.41 (Lasso 0.239,
Ridge 0.304, Logistic Regression 0.410).

**The Sprint 3 retrain reproduces the direction of that recovery but not its
size, and that gap is itself worth recording rather than quietly overwriting
the old numbers.** On the current run, the same three models on the same
fold score PR-AUC **0.113 (Lasso), 0.085 (Ridge), 0.088 (Logistic
Regression)** — still well above the pre-Sprint-2 0.06–0.08 floor, but far
below the 0.24–0.41 previously published.

While investigating the gap, a real bug surfaced and was fixed: `config.yaml`
claimed `random_state` was "injected separately... so it stays consistent
across all models," but `train_pipeline.py` never actually passed it to any
of the three `LogisticRegression` constructions. Harmless for the plain and
Ridge-penalized models (`lbfgs`, their default solver, is deterministic
regardless of the seed), but Lasso's config sets `solver: saga` — a
stochastic method — so its coefficients, and every metric derived from them,
were not reproducible run to run. Fixed by passing `random_state` explicitly
to all three (`tests/test_train_pipeline_determinism.py` pins it). Re-running
after the fix moved Lasso's fold-2 PR-AUC by only 0.0003 (0.1128 → 0.1132),
and left Logistic Regression and Ridge — already deterministic — bit-identical
across repeated runs of this codebase. That rules out solver randomness as
the explanation for the large gap versus the originally-published 0.24–0.41:
whatever moved those numbers, it was not run-to-run noise. The two remaining
candidates, not distinguished here, are the two-feature removal itself
changing the optimization landscape on this near-perfectly-separable fold, or
this retrain running in a different Python/scikit-learn/scipy environment
than whichever machine produced the original Sprint 2 numbers — MLE logistic
regression's fragility under near-separability makes either plausible.
Isolating them would need a controlled 20-feature rerun on this exact
environment, which is out of scope here. The structural point is unaffected
either way: none of these figures approach 1.0000, and the value-band rule
below remains inexpressible to a linear model regardless of which exact
number the solver lands on. Full breakdown in `model_comparison_by_fold.csv`.

The linear baselines' 3-fold PR-AUC means are listed in the results table,
but read the ±0.43–0.44 standard deviations as the real content: each is one
near-perfect window (fold 3, PR-AUC 0.95–0.99) averaged with two poor ones
(folds 1–2), not a performance estimate. Their precision@capacity figures
(0.38–0.38 ± 0.27–0.28) are the more stable comparison, and still sit
clearly below every tree model.

### Optuna improved ranking but hurt calibration — and that's informative too

10-trial Optuna tuning per model (search spaces in `train_pipeline.py`;
cut from 25 to keep full-run time reasonable — see Running It) scores
each trial's PR-AUC averaged across all 3 CV folds, not a single fold (an
earlier version scored against one fold that turned out to be trivially
separable — see the finding above — which gave Optuna no real signal to
discriminate between trials). The natural read of "did tuning help" is
tuned-vs-default *within* the same model family, since that is the actual
delta Optuna's objective was optimizing against.

For LightGBM, tuning helped on both axes: mean PR-AUC 0.9974 vs. 0.9967
untuned, and weighted BCE (`custom_metrics.py`, which *is*
calibration-sensitive, unlike PR-AUC) *improved* too — 0.025 vs. 0.040
untuned. For XGBoost, tuning bought almost nothing on PR-AUC (0.9968 vs.
0.9967 untuned — within noise) while wrecking calibration: weighted BCE
**0.810 versus 0.082** untuned, ~10x worse. (On the Sprint 1 feature set
this gap was 0.332 vs 0.127; widening the feature set widened the gap
rather than closing it, which is what an unconstrained PR-AUC search should
be expected to do given more room to trade calibration for ranking.) Two
separate cost-sensitivity mechanisms are in tension here: `scale_pos_weight`
during training now correctly reflects the resampled training ratio
(~50:1, matching what `class_weight='balanced'` uses for the other
models — see the fix below), while the weighted-BCE *evaluation* metric
uses the true deployment-time ratio (~300–1700:1). Optuna's hyperparameter
search, chasing PR-AUC only, found an XGBoost configuration that ranks
fraud negligibly better while drifting far from well-calibrated
probabilities under that evaluation-time weighting — LightGBM's tuning
landing on a configuration that improved both is not guaranteed by the
objective, just a nicer outcome this run. Concretely: **Optuna's per-family
"tuned" configuration is not automatically the safer choice**, which is why
`train_pipeline.py` does not select `best_model` by mean PR-AUC alone.

Sprint 1 moved that selection to Precision@K at K=1x fraud count. Sprint 2
moved it once more, to precision at the review-capacity operating point,
for the reason given above: K=1x-fraud-count is still defined by the
labels, so "the model that wins at K=1x" is a criterion you could not
actually evaluate on the day you had to deploy. Capacity-based K has the
same operational shape and *is* computable in advance. Under that criterion
`best_model` is **LightGBM (tuned)**, which also happens to top mean PR-AUC
this run — though see the tie caveat under the results table: that is not a
claim it is genuinely better than the other four tree models, only that a
tree model should be selected. It disagrees with the model that tops mean
Precision@K=1x-fraud-count (LightGBM, default params) — a reminder that
which specific tree model nominally leads a given metric is itself close to
noise at this spread, which is exactly the point the tie caveat makes.
Optuna's tuning still runs and its result is still reported for comparison
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

Feature count went 11 → 20 in Sprint 2, then 20 → **18** in Sprint 3 (see
below). All new aggregates are strictly prior-only; the window-frame
implementation was verified against an independent correlated-self-join
restatement over 8,471 rows across 300 busy destinations, with zero
disagreements, zero first-transactions showing prior history, and same-step
peers correctly excluded from the velocity window.

### What the new features were actually worth: not much

The ablation (`data/processed/feature_ablation.csv`, LightGBM defaults, final
fold, K=8,083; re-run on the current 18-feature set) attributes the movement
rather than asserting it:

| Feature set | n | PR-AUC | Weighted BCE | False positives | Fraud missed |
|---|---|---|---|---|---|
| Sprint 1 features only | 9 | 0.99954 | 0.0078 | 3,978 | 2 |
| + transaction type | 13 | 0.99989 | 0.0026 | 3,836 | 0 |
| + destination aggregates | 14 | 0.99993 | 0.0035 | 3,833 | 0 |
| all Sprint 2 features | 18 | 0.99996 | 0.0022 | 3,834 | 0 |

Net effect of the nine Sprint 2 features: **144 fewer false positives out of
~3,978 (a 3.6% reduction) and 2 fewer missed frauds**. The "Sprint 1 features
only" baseline arm now has 9 features rather than 11 — it excludes the two
features Sprint 3 removed (see below) — and its false-positive count moved
up accordingly (was 3,870 with them present). That is itself informative: the
two removed features carried exactly zero SHAP attribution in the *full*
18-feature model, where the destination-side aggregates already cover the
relevant history, but they were not fully redundant in this narrower
9-feature arm, which has no destination-side signal at all to fall back on.
"Dead weight in the full model" and "worthless in isolation" are different
claims, and only the first one motivated removing them. The two feature
groups still do different jobs: transaction type is what eliminates the
missed frauds and improves calibration ~3x, while the destination aggregates
cut false positives but *worsen* calibration on their own; only combined do
both metrics land at their best.

### SHAP confirmed two features were dead weight — and Sprint 3 removed them

Global attribution (`shap_global_importance.csv`, mean |SHAP| over 20k
sampled test rows, current 18-feature model) is led by the
balance-reconciliation features: `orig_balance_mismatch` (25.2% of total
attribution), `orig_emptied` (17.4%), `orig_balance_delta` (8.6%),
`amount_to_balance_ratio` (7.6%). The nine Sprint 2 features together carry
~23.3%, led by `is_cash_out` (7.0%) and `dest_prior_avg_amount` (4.3%).

Sprint 2 had kept two Sprint-1-era features, `orig_prior_txn_count` and
`orig_prior_avg_amount`, deliberately in the feature set specifically to see
whether SHAP would flag them: profiling had already shown both to be
near-dead (nonzero on just 0.15% of rows, since `nameOrig` is 99.85%
unique — see ACCOUNT STRUCTURE above). SHAP confirmed it —
**both had exactly zero attribution** — so Sprint 3 removed them
(`FEATURE_VERSION` 2 → 3, feature count 20 → 18) and retrained. The
retrained numbers throughout this document reflect that removal; they move
by noise, not by a real effect, which is the expected result of dropping
two provably inert inputs.

Per-alert reason codes for the 50 highest-scoring alerts are in
`shap_alert_reasons.csv` — each alert's top contributing features with their
raw (un-scaled) values, which is the form an analyst or an AML audit needs,
rather than a bare score.

### Error analysis: the queue has no misses, and the false positives are small transfers

At 500 reviews/day on the final fold the queue is 4,250 true fraud, 3,835
false positives, **0 fraud missed**. The false positives separate from true
fraud most sharply on transaction size and account-draining behaviour
(`error_analysis_profile.csv`, gaps in pooled-std units): `orig_balance_delta`
−7.57 sd (FP mean 11,839 vs TP mean 1,555,748), `dest_amount_to_prior_avg_ratio`
−4.22 sd, `amount` −2.94 sd, `orig_balance_mismatch` +2.22 sd, `orig_emptied`
−2.07 sd (FP 0.076 vs TP 0.971).

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
│       ├── feature_ablation.csv                # metric movement attributed per feature group
│       └── capacity_economics.csv              # net value per capacity level (src/economics.py)
├── models/                     # trained model artifacts per run (not committed)
│   └── <run_id>/                 scaler + one .joblib per model (final CV fold) + metadata.json
├── mlflow.db                   # local MLflow tracking store (not committed)
├── reports/                    # per-run training logs (generated)
├── src/
│   ├── config.py               # loads config.yaml
│   ├── generate_sample_data.py # builds a schema-accurate synthetic sample for local dev
│   ├── schema.py                # pandera schema, validated against a raw-table sample at ingest
│   ├── features.py             # feature engineering as a DuckDB SQL query
│   ├── cv.py                   # expanding-window time-based CV fold generator
│   ├── custom_metrics.py       # weighted BCE loss + Precision@K / Recall@K
│   ├── threshold.py            # capacity-based K, decision threshold, capacity sweep
│   ├── explain.py              # SHAP global importance + per-alert reason codes
│   ├── error_analysis.py       # review-queue TP/FP/FN profiling, missed-fraud ranking
│   ├── economics.py             # capacity-constraint cost + fraud-ticket-size crossover
│   ├── export_bundle.py         # emits the versioned model_bundle/ serving artifact
│   ├── build_dest_state.py      # emits dest_state.parquet, the serving-time feature snapshot
│   ├── train_pipeline.py       # main training + CV + Optuna tuning + MLflow pipeline
│   ├── inference/               # Sprint 4: the only code path from raw txn to score
│   │   ├── bundle.py             # load + sha256-verify a model_bundle/vN/
│   │   ├── state.py              # destination-state lookup (hashed searchsorted)
│   │   ├── features.py           # raw transaction dict -> 18-feature vector
│   │   ├── score.py              # vector -> probability + live TreeSHAP reason codes
│   │   └── rules.py              # hard-block rule layer, evaluated before the model
│   └── api/                     # Sprint 4: FastAPI service wrapping src/inference/
│       ├── main.py               # routes + create_app() factory
│       ├── schemas.py            # Pydantic v2 request/response models
│       ├── batch.py              # within-batch destination-state accumulation
│       ├── errors.py             # RFC 7807 problem+json handlers
│       ├── rate_limit.py         # per-IP rate limiting
│       ├── limits.py             # request body size cap
│       ├── metrics.py            # in-memory counters backing /metrics
│       └── audit.py              # structured JSON prediction audit log
├── tests/                      # pytest suite, 166 tests: features/metrics/threshold/cv/schema/
│                                #   economics/bundle/golden-file/train-determinism/error-analysis/
│                                #   explain (Sprints 0-3), inference unit + skew (plumbing & state)
│                                #   + API contract (Sprint 4)
├── model_bundle/v1/             # committed, versioned serving artifact (see ARCHITECTURE.md §3)
├── config.yaml                 # paths, CV, model, Optuna, and MLflow config
├── requirements-train.txt       # laptop / CI training environment (also runs the API's tests)
├── requirements-serve.txt       # serving image only -- no sklearn/shap/duckdb/mlflow/pandas;
│                                #   verified against a real uvicorn process in an isolated venv
├── dashboard/requirements.txt   # Streamlit Community Cloud (Sprint 5)
├── ROADMAP.md                  # production-readiness plan, sprint by sprint
└── README.md
```

## Running it

```
pip install -r requirements-train.txt
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

### Serving image (Sprint 6)

```
docker build -t fraud-api .
docker run -p 8000:8080 fraud-api
# or: docker compose up
```

Built and integration-tested exclusively in CI (`.github/workflows/ci.yml`)
— this dev machine has no Docker (ARCHITECTURE.md §8, GIT_WORKFLOW.md).

Measured in CI (run `30788266390`, the first green run — not a local
benchmark, and not an estimate):

| Metric | Measured |
|---|---|
| Image size | **510.6 MB** uncompressed |
| Cold start (`docker run` → first `200` from `/ready`) | **3,307 ms** |
| `/score` latency, in-container | **5.3 ms** |
| `/score` latency, end-to-end over the Docker port mapping | **10.3 ms** |

**The image misses this sprint's own <400MB target by 28%.** Recorded as a
deviation rather than quietly restated: the prime suspect is `pyarrow`,
carried solely to read `dest_state.parquet`, on top of the `scipy` that
`lightgbm` pulls in. Confirming that and deciding whether to change the
bundle's storage format is open work, tracked in ROADMAP Sprint 6.

Getting the first green run required three real defects to be fixed, none
of which any local test could have caught — see ROADMAP Sprint 6.

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
- **The exchange rate has a price, and the naive optimum is named as
  degenerate rather than hidden**: `src/economics.py` (Sprint 3) turns the
  capacity sweep into net value, tested that the obvious "find the optimum"
  framing is degenerate against this project's own data (full recall always
  wins), and reports the two questions that aren't — the cost of a capacity
  constraint, and the fraud-ticket-size crossover — instead of asserting an
  optimum that doesn't exist. See Results.
- **The serving artifact is version-portable by construction**: `model.txt`
  is LightGBM's native text format, not a pickle, and the scaler ships as
  two JSON arrays applied in pure numpy — no scikit-learn, duckdb, mlflow,
  shap, or pandas in `model_bundle/v1/`, verified by reproducing a 200-row
  golden file to floating-point equality in a venv containing only
  `requirements-serve.txt`. See `src/export_bundle.py`, `src/build_dest_state.py`,
  and `ARCHITECTURE.md` §3.

## Roadmap

See `ROADMAP.md` for the full sprint-by-sprint plan and `ARCHITECTURE.md`
for the target system design. Sprints 0–3 are done: engineering hygiene;
experiment rigor with CV/Optuna/MLflow; the capacity-based operating point,
destination velocity/graph features, SHAP, and error analysis described
above; and, in Sprint 3, a `pytest` suite, a pandera ingest schema, the
`src/economics.py` business-decision layer, the two zero-SHAP features'
removal, and the versioned `model_bundle/v1/` serving artifact (native
LightGBM text format + a pure-numpy scaler + the per-destination state
snapshot — no scikit-learn/duckdb/mlflow in the serving path, verified by
reproducing the golden file in a venv containing only
`requirements-serve.txt`).

**Sprint 4 is also done**: `src/inference/` (the single code path from raw
transaction to score — bundle loading, destination-state lookup, feature
construction, scoring, and an illustrative hard-block rule layer) and a
FastAPI service (`/health`, `/ready`, `/model-info`, `/score`,
`/score/batch`, `/metrics`) wrapping it. Both training/serving skew tests
pass — feature-construction and model/scaler plumbing each verified against
independent ground truth to floating-point precision — and the full API was
verified end to end (a real `uvicorn` process handling real HTTP requests,
not just an import check) in a `requirements-serve.txt`-only venv, with
`pandas`/`scikit-learn`/`duckdb`/`shap`/`mlflow`/`optuna` absent. `/score`
measured p95 = 4.9ms locally against a 100ms target. Two real
bundle-integrity bugs were also found and fixed in the process (a Windows
line-ending checkout issue plus a stale checksum, both in
`model_bundle/v1/`) — see `ARCHITECTURE.md` §12 for detail. A follow-up
audit of Sprints 0-4 together (still 2026-08-02) found no functional bugs
in the untouched Sprint 0-2 modules, closed the one real gap it did find
(`error_analysis.py`/`explain.py` were the only two non-trivial `src/`
modules without dedicated tests), and fixed a few small drift risks
(a hardcoded value duplicated instead of shared with its own constant, a
swallowed exception with no log line, three CSVs generated but never
logged to MLflow). The `pytest` suite now stands at 166 tests, all
passing. Next up is the Streamlit dashboard (Sprint 5), followed by CI/CD,
cloud deployment, drift monitoring, and portfolio polish.
