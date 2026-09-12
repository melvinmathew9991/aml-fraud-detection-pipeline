# Model Card — AML Fraud Detection

The live service is the source of truth for everything in this card:
`GET /model-info` on <https://fraud-api-amj2cl4jhq-uc.a.run.app> returns the
model name, bundle version, feature list, threshold, expected precision/recall,
the precision **ceiling**, the snapshot step and the limitations list. Every
figure below was verified against that endpoint, the committed CSVs under
`data/processed/`, or the DuckDB store — on 2026-09-12, not carried forward
from an earlier draft.

---

## 1. Model details

| Field | Value |
|---|---|
| Model | **LightGBM (tuned)** |
| Selected on | mean **Precision@capacity** across 3 folds — *not* PR-AUC (see §6) |
| Bundle | `model_bundle/v2/` — 8.27 MB measured on disk |
| Bundle contents | `model.txt` (LightGBM native text) · `scaler.json` (pure-numpy `StandardScaler`) · `threshold.json` · `dest_state.npz` · `bundle_meta.json` (sha256 per file) |
| `feature_version` | 3 (18 features) |
| `run_id` / `trained_at` | `20260801T130131Z` |
| `git_commit` | `b874804-dirty` — see §9, this is a known provenance gap |
| Decision threshold | `8.077004481140283e-06` |
| Framework | LightGBM 4.7.0; serving uses `fastapi, uvicorn, pydantic, numpy, lightgbm` only |

v2 is a **packaging** bump of v1, not a retrain: the arrays were converted from
v1's parquet rather than rebuilt, so every value is v1's and the model identity
(`run_id`, `trained_at`, `git_commit`) is unchanged.

---

## 2. Intended use

**Rank incoming payment transactions into a capacity-bounded review queue for
human analysts.** The deployable artifact is a score *plus the threshold that
produced it*, so an alert is auditable after the fact.

The operating point is derived from **analyst headcount, not from the labels**:
K = `reviews_per_day × days in the window`, and the score cutoff is read off at
that K. The true fraud count for today's transactions is only knowable in
hindsight, so tuning an operating point against it is a quiet form of using the
labels at decision time.

### Out of scope

- **Not an autonomous blocker.** The hard-block rule layer in `src/inference/rules.py`
  is explicitly illustrative, not tuned: measured precision on training data is
  **~10% at a ~0.25% block rate**. The model score is the primary signal.
- **Not an online feature store.** Destination state is a frozen snapshot (§7).
- **Not validated outside PaySim.** See §10 — the headline accuracy is a
  property of the data generator.
- **Not a regulatory filing decision.** Nothing here produces or approves an STR/SAR.

---

## 3. Training data

| Property | Value | Source |
|---|---|---|
| Dataset | PaySim (Lopez-Rojas, Elmir & Axelsson, 2016) | Kaggle `ealaxi/paysim1` |
| Rows | **6,362,620** | DuckDB store, verified 2026-09-12 |
| Fraud rows | **8,213** | same |
| Fraud rate | **0.1291%** | same |
| `step` range | 1–743 (one simulated hour per step) | same |
| Validation | 3 **expanding-window time-based** CV folds, boundaries 0.4/0.6/0.8/1.0 | `config.yaml`, `src/cv.py` |
| Class handling | train-side undersampling to 50:1; **test left at the natural rate** | `config.yaml` |

Undersampling is train-side only and deliberately so — evaluating on a rebalanced
test set would report a precision the deployment would never see.

---

## 4. Features (18)

`feature_version = 3`. Order is load-bearing: the serving vector must match this
exactly, which is enforced by a golden-file test, not by inspection.

**Stateless — computed from the request payload alone (13)**
`amount` · `hour_of_day` · `is_night` · `orig_balance_delta` ·
`dest_balance_delta` · `orig_balance_mismatch` · `orig_emptied` ·
`amount_to_balance_ratio` · `dest_is_merchant` · `is_transfer` · `is_cash_out` ·
`is_cash_in` · `is_debit`

**Stateful — require the destination snapshot (5)**
`dest_prior_txn_count` · `dest_prior_avg_amount` · `dest_amount_to_prior_avg_ratio` ·
`dest_txn_count_24h` · `dest_amount_sum_24h`

### Two features were removed for having zero attribution

`orig_prior_txn_count` and `orig_prior_avg_amount` carried **exactly zero** SHAP
attribution and were dropped in Sprint 3 (20 → 18 features). They were dead for a
structural reason found by profiling first: `nameOrig` is 99.85% unique
(6,344,009 of 6,353,307 origin accounts appear exactly once), so per-origin
history is empty for almost every row. Velocity and graph aggregates were built
on the **destination** side instead, which is where the repeat structure lives
and is the more AML-relevant direction.

---

## 5. Performance, per fold

`LightGBM (tuned)`, from `data/processed/model_comparison_by_fold.csv`:

| Fold | PR-AUC | Capacity K | Precision@capacity | Recall@capacity |
|---|---|---|---|---|
| 1 | 0.9921 | 1,562 | 0.5627 | 0.9910 |
| 2 | 1.0000 | 1,542 | 0.4994 | 1.0000 |
| 3 (deployed) | 1.0000 | 8,083 | 0.5257 | 1.0000 |
| **mean ± std** | **0.9974 ± 0.0045** | — | **0.5293 ± 0.0318** | 0.997 |

Fold 2's PR-AUC is 0.9999966 and fold 3's 0.9999599 — displayed as 1.0000 at
four decimals, which is itself the point of §10.

---

## 6. The precision ceiling — read this before the precision number

**The model sits at the arithmetic precision ceiling at every capacity level on
the deployed fold.** Once recall saturates, precision at K is bounded above by
`total_fraud / n_flagged`. At 500 reviews/day that is 4,250/8,085 = **0.5257** —
identical to the reported precision. The 3,835 false positives exist because the
queue is sized ~1.9× actual fraud volume, not because the ranking is weak.

Capacity sweep on fold 3 (4,250 fraud in 16.2 days), from
`data/processed/capacity_sweep.csv`:

| Reviews/day | K | Precision | Recall | Fraud caught | False positives |
|---|---|---|---|---|---|
| 100 | 1,617 | 1.0000 | 0.380 | 1,617 | 0 |
| 250 | 4,042 | 1.0000 | 0.951 | 4,042 | 0 |
| **500 (configured)** | 8,083 | 0.5257 | 1.000 | 4,250 | 3,835 |
| 1,000 | 16,167 | 0.2629 | 1.000 | 4,250 | 11,917 |
| 2,000 | 32,333 | 0.1314 | 1.000 | 4,250 | 28,086 |
| 5,000 | 80,833 | 0.0526 | 1.000 | 4,250 | 76,583 |

This is also why `best_model` is selected on Precision@capacity rather than
PR-AUC: Optuna found XGBoost hyperparameters that rank marginally better on
PR-AUC while calibrating substantially worse, and "best mean PR-AUC" and "best
model" are not the same model here.

---

## 7. The threshold is fold-3-specific, and that is a real limitation

`decision_threshold = 8.077004481140283e-06` was derived on **fold 3 only**, at
500 reviews/day. The folds are not measuring the same thing: transaction volume
across the three windows varies roughly 5× (~401k, ~419k, ~77k per day), so the
same 500/day yields alert rates of **0.125%, 0.119% and 0.647%** respectively.

A score threshold fixes a *score*, not a *queue length*. Measured across all 31
drift windows, the queue runs **272–4,594 alerts/day against a capacity of 500,
exceeding it in 14 of the 30 full windows**. Recall at capacity stayed **1.000 in
all 31 windows**, and precision at capacity sat exactly on its own ceiling in all
31.

Across the **30 windows with a sufficient sample**, precision at capacity runs
**0.4320–0.6400, mean 0.5294** — against the bundle's shipped
`expected_precision` of 0.5257, measured on fold 3 alone, which therefore
generalised across the whole dataset. The 31st window is excluded deliberately
and the exclusion is load-bearing: window 30 is 272 rows that are *all* fraud, so
its precision is 1.0 and including it moves the mean to 0.5446. It is reported
and plotted but never triggers anything, because 272 rows cannot support a
distributional claim.

---

## 8. The destination-state snapshot

Serving carries a read-only, point-in-time table of **571,961 non-merchant
destinations**, frozen at **step 743**, shipped as `dest_state.npz` keyed by
64-bit hash. Merchant (`M%`) destinations average 1.0005 transactions each, so
their prior history is ~always empty; they are not stored and resolve to the
cold-start default.

Every `/score` response carries **`state_hit: true|false`** so a caller can tell
whether the five stateful features came from the snapshot or from cold-start
defaults. Silently defaulting them is the most dangerous failure mode in the
system — a model receiving zeros for five features returns plausible-looking
scores and nothing errors.

### Limitations as served (verbatim from `/model-info`)

1. The destination-state snapshot is frozen at a single training-time step; it is a point-in-time approximation, not an online feature store.
2. Scoring the same transaction twice via `/score` returns the same answer — state does not accumulate. `/score/batch` accumulates within one batch only.
3. Serving state is as-of one instant; training features were as-of each row's own timestamp. The two agree only for transactions arriving after the snapshot step.
4. The hard-block rule layer is illustrative, not tuned: measured precision on the training data is low (~10%) at a ~0.25% block rate. The model score is the primary signal.

---

## 9. Provenance gap — stated, not hidden

**The deployed model cannot be rebuilt byte-for-byte from source.** The bundle
records `run_id: 20260801T130131Z` and `git_commit: b874804-dirty`, but neither
`models/20260801T130131Z/` nor its training log still exists (the newest
surviving local run is `20260728T172950Z`, confirmed 2026-09-12). The committed
bundle is the only copy, which is why it is tracked in git rather than treated
as a build output, and why its sha256 checksums are verified at every startup.

Rebuilding from source would also **not** be byte-identical for an unrelated,
pre-existing reason: 39 of 571,961 destination averages differ at float32
epsilon because DuckDB sums its parallel `AVG` in a nondeterministic order.

---

## 10. The honest caveat about accuracy

**PaySim fraud is close to trivially separable, and the near-perfect numbers
above are a property of the data generator, not a modelling achievement.** Fraud
is synthetically produced by near-deterministic rules — the source account is
drained to the cent, confined to `TRANSFER` and `CASH_OUT`. In one fold's test
window (simulated hours 282–355), **763 of 770 fraud transactions — 99.1% —
have `amount_to_balance_ratio` exactly 1.00**, and **100% have
`dest_is_merchant = 0`**: a narrow, nonlinear value band that trees split out
trivially and a single linear hyperplane structurally cannot express. Re-derived
from the raw transaction table on 2026-09-12; four different readings of "drained
to the cent" (exact equality, ratio rounded to 2dp, a ±0.005 tolerance, and
`newbalanceOrig = 0`) all return the same 763 rows.

A PR-AUC of 0.9974 on this dataset should be read as "the pipeline is wired up
correctly", not "this model would catch 99.7% of real fraud". **The system
around the model — leakage-safe features, capacity-derived operating point,
train/serve skew closed to floating-point equality, monitoring, rollback — is
the deliverable. The score is not.**

### Fairness and data governance

PaySim contains **no protected attributes** (no age, gender, geography,
ethnicity or income), so no subgroup fairness analysis is possible on it — and
none is claimed. That is a property of the dataset, not evidence of an unbiased
model. A real deployment would require subgroup performance analysis before
anything reached a customer, and the account identifiers here are synthetic, so
nothing in this system has processed real personal data.

---

*Maintained as part of Sprint 9. Cross-references: `ARCHITECTURE.md` §0 and §2,
`MONITORING.md` for drift and retraining criteria, `AUDIT.md` for the defect
register, `README.md` for the business-impact analysis.*
