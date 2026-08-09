# GCP Deployment & Free-Tier Analysis

Everything this project needs to know about Google Cloud: what it will use, what
that costs, what the free tier actually covers, and the specific settings that
separate a $0 deployment from a recurring bill.

**Researched 2026-08-03 against official Google documentation.** Every figure
here is sourced (§11). Cloud pricing changes — re-verify before signing up
rather than trusting this file's age.

Scope: Sprints 7–12 (`ROADMAP.md`). Deployment target and rationale are in
`ARCHITECTURE.md` §8/§9; this document is the cost and free-tier layer.

---

## 1. Verdict

**The project completes at $0**, with 99%+ headroom on every Cloud Run
dimension. The free tier is not the binding constraint — configuration
discipline is.

Two things can cost money, and neither is about scale:

1. A single CLI flag (`--min-instances`), worth ~$61/month if wrong.
2. Letting the $300 trial lapse instead of upgrading — which *deletes the
   deployment* rather than billing for it.

---

## 2. Free-tier allowances (verified 2026-08-03)

| Service | Always-free allowance |
|---|---|
| Cloud Run — requests | 2,000,000 / month |
| Cloud Run — memory | 360,000 GB-seconds / month |
| Cloud Run — compute | 180,000 vCPU-seconds / month |
| Cloud Run — egress | 1 GB / month from North America |
| Artifact Registry | 0.5 GB storage / month |
| Cloud Logging | First 50 GiB per project / month |
| Cloud Scheduler | 3 jobs / month |
| Cloud Build | 2,500 build-minutes / month |
| Cloud Monitoring | Cloud Run native metrics **non-chargeable**; 150 MiB/month custom metrics |

**Unresolved discrepancy.** Google's free-tier documentation states 180,000
vCPU-seconds / 360,000 GB-seconds. A Cloud Run pricing source states 240,000 /
450,000. The two could not be reconciled during research. **This document uses
the lower pair throughout.** At this project's consumption (~0.5% of either),
the difference is immaterial — but it is recorded rather than papered over.

**Two structural facts that are easy to miss:**

- The free tier is **per billing account, not per project**. Splitting work
  across projects does not multiply the allowance.
- **A billing account is required to access the free tier at all.** "Free tier"
  does not mean "no payment method."

---

## 3. Services used, by sprint

| Sprint | Adds | Billing surface |
|---|---|---|
| 7 — Deployment | Cloud Run, Artifact Registry, Workload Identity Federation | GCP, free tier |
| 8 — Monitoring & drift | Cloud Scheduler (1 of 3 free jobs), Cloud Logging, Cloud Monitoring | GCP, free tier |
| 9 — Portfolio polish | Documentation only | none |
| 10 — Persistence & auth | Neon Postgres, API-key auth | **not GCP** |
| 11 — Graph & statistics | Local compute | none |
| 12 — GenAI narratives | Gemini API free tier | **not GCP billing** |

**Half the remaining roadmap never touches GCP billing.** The GCP surface is
essentially Sprints 7 and 8. Sprint 10's database is Neon (chosen over Supabase,
whose free projects pause after 7 days idle — disqualifying for a demo link),
Sprint 12's LLM is the Gemini free tier, Sprint 11 runs locally, Sprint 9 is prose.

Also not GCP: **Streamlit Community Cloud** hosts the dashboard, and **GitHub
Actions** builds the image. The latter matters — because CI builds and pushes,
Cloud Build's 2,500-minute allowance is never touched.

---

## 4. This project's measured consumption

Measured, not estimated. Image figures come from CI run `30795258811`; latency
from runs `30788266390` / `30792154470`.

| Metric | Measured |
|---|---|
| Image, uncompressed | 510.6 MB |
| Image, compressed (registry storage) | 169.8 MB (33%) |
| Cold start (`docker run` → first 200 from `/ready`) | 3,298–3,307 ms |
| `/score` latency, in-container | 5.3 ms |
| `/score` response size | ~1.2 KB |

Against the free tier, at a realistic portfolio load of 300 visits × 5 requests:

| Resource | Use | Allowance | Consumed |
|---|---|---|---|
| vCPU-seconds | 998 | 180,000 | 0.55% |
| GB-seconds | 499 | 360,000 | 0.14% |
| Requests | 1,500 | 2,000,000 | 0.07% |
| Egress | 0.002 GB | 1 GB | 0.17% |
| Artifact Registry | 0.17–0.34 GB | 0.5 GB | **35–66%** |

**Artifact Registry is the tightest resource by an order of magnitude**, and the
only one that grows as the project does.

Scaling headroom: **~50,000 visits/month** is where this stops being free, and
**egress breaks first** (114% of allowance) while compute is still at 94%.

---

## 5. The four conditions for $0

### ① Upgrade to a Paid billing account

Counterintuitive, and the one most likely to be got wrong. Staying on the $300
trial to "avoid paying" is exactly what destroys the deployment:

> Your Free Trial billing account auto-closes if you spend the $300 credit or
> 90 days pass and you don't upgrade. **All resources you created during the
> trial are stopped**, and are permanently deleted after a 30-day grace period.

Upgrading costs nothing by itself:

> you continue to have access to the Free Tier … When you stay within the Free
> Tier limits, these resources are not charged against your Free Trial credits
> or to your Cloud Billing account's payment method.

**Upgrade early, not at day 89.** The credit also *masks* misconfiguration for
90 days — a `min-instances=1` mistake looks free right up until it isn't.

### ② `--min-instances=0`

The single highest-consequence setting in the project.

| Setting | vCPU-seconds/month | % of free tier | Cost |
|---|---|---|---|
| `min-instances=0` | ~998 | 0.55% | $0 |
| `min-instances=1` | 2,628,000 | **1,460%** | ~$61/month |

Confirmed behaviour: *"if min instances is set to 0, you are not billed when
instances are idle."* Everything else in this document is rounding error beside
this flag. The console UI nudges toward keeping an instance warm to avoid cold
starts; do not accept it. A 3.3 s cold start is the correct trade for a
portfolio demo.

### ③ Artifact Registry retention = 2

See §6. Enforced at repository creation, not retrofitted, and watched by CI.

### ④ Keep the drift job within Cloud Scheduler's 3 free jobs

Sprint 8 needs one. Alternatively move it to a GitHub Actions cron schedule,
which is free for public repositories and removes the GCP dependency entirely.

---

## 6. Artifact Registry — the tight one

The retention policy was originally planned as "3 versions" against an assumed
~185 MB image. The real image is **510.6 MB uncompressed / 169.8 MB
compressed**, which invalidated the arithmetic.

Google documents the free tier as "0.5 GB", which is ambiguous, and the two
readings disagree at exactly this size:

| Retention | 0.5 GiB (512 MiB) | 0.5 GB decimal (500 MB) |
|---|---|---|
| 2 versions | fits, +172 MiB | fits, +144 MB |
| 3 versions | fits, **+2.6 MiB** | **breaches, −34 MB** |

**Retention is set to 2**, which is safe under both readings. A policy that only
holds under the favourable interpretation of a billing unit is not a policy, and
0.5% headroom is erased by one dependency bump.

**Layer deduplication is upside, not licence.** Artifact Registry shares layers
within a repository — *"images with common layers share those layers"* — so two
versions cost less than 2 × 169.8 MB. The `python:3.12-slim` base (~50 MB) is
identical across builds and stored once. Whether the large venv layer dedupes
depends on `pip install` producing a byte-identical layer in CI, which is not
guaranteed without a layer cache. **The retention decision assumes no
deduplication**; anything better is margin.

**This cannot go stale silently.** The container job in `.github/workflows/ci.yml`
measures compressed size on every build, computes how many versions fit against
the conservative bound, and raises a CI warning if the image outgrows the
policy. The 185 MB figure went unnoticed for a day; this one is re-measured on
every run.

---

## 7. Cost failure modes

| Failure | Consequence | Guard |
|---|---|---|
| `min-instances` > 0 | ~$61/month, indefinite | Verify in the **deployed revision**, not the deploy command |
| Trial lapses without upgrade | Deployment stopped, then deleted | Upgrade to Paid early |
| Image grows past retention budget | Artifact Registry overage | CI warning on the build that causes it |
| Traffic > ~50k visits/month | Egress overage first | Budget alert |
| Secret Manager used for credentials | ~$0.06/secret version/month — small but **not zero** | Use Cloud Run env vars from GitHub Actions secrets instead |
| Tier 2 region chosen | Allowance consumed faster for identical work | Deploy to `us-central1` (Tier 1) |

**On Secret Manager:** its pricing page could not be read during research
(repeated truncation), so its free-tier terms are **unconfirmed**. Historically
it charges per active secret version per month. The recommended path avoids it
entirely — inject Sprint 10's API key and Neon connection string as Cloud Run
environment variables from GitHub Actions secrets at deploy time. That is $0 and
keeps credentials out of the repository just as effectively.

**On billing mechanics:** cold starts *are* billed. *"Billable instance time is
rounded up to the nearest 100 milliseconds and includes when the instance is
starting."* Request-based billing charges *"during request processing, container
startup, and container shutdown."* At portfolio traffic this is 0.55% of the
allowance, but it means scale-from-zero is not free — it is cheap.

---

## 8. Region

Deploy to **`us-central1`**. The free tier is applied as a spending-based
discount at Tier 1 pricing and aggregated per billing account. A Tier 2 region
costs more per vCPU-second, so identical work consumes the allowance faster.

---

## 9. Pre-deploy checklist

Ordered so every control exists before anything is publicly reachable. This is
deliberately not the ROADMAP's build order — the controls come first.

- [ ] Create GCP project (**user action** — involves payment details)
- [ ] Enable billing
- [ ] **Upgrade to a Paid billing account** (§5①)
- [ ] **Budget alert at $1/month — before the first deploy, not after**
- [ ] Enable Artifact Registry + Cloud Run APIs
- [ ] Create Artifact Registry repo in `us-central1` **with the 2-version cleanup policy applied at creation**
- [ ] Configure Workload Identity Federation (no service-account JSON in repo secrets)
- [ ] Deploy: `--min-instances=0 --max-instances=2 --memory=512Mi --cpu=1 --concurrency=80 --timeout=30s`
- [ ] **Verify `min-instances=0` in the deployed revision**, not just the command
- [ ] Confirm the budget alert is active
- [ ] Deploy the dashboard to Streamlit Community Cloud, pointed at the Cloud Run URL

## 10. Post-deploy monitoring

- Cloud Run native metrics are **non-chargeable** — use them freely.
- Do not push custom metrics without checking the 150 MiB/month ceiling.
- The `/metrics` endpoint is in-memory and served by the app; it does not write
  to Cloud Monitoring and therefore costs nothing.
- Check billing in week 1 even if everything looks right. The trial credit
  absorbs mistakes silently.

---

## 11. Sources

All accessed 2026-08-03:

- [Free Google Cloud features and trial offer](https://docs.cloud.google.com/free/docs/free-cloud-features) — allowances, billing-account requirement, per-account scope, trial expiry
- [Google Cloud Free Trial FAQs](https://cloud.google.com/signup-faqs) — trial auto-close, grace period, upgrade path
- [Cloud Run pricing](https://cloud.google.com/run/pricing) — billing models, region tiers
- [Billing settings for services | Cloud Run](https://docs.cloud.google.com/run/docs/configuring/billing-settings) — request-based vs instance-based
- [Set minimum instances for services | Cloud Run](https://docs.cloud.google.com/run/docs/configuring/min-instances) — idle billing
- [Best practices for cost-optimized Cloud Run services](https://docs.cloud.google.com/run/docs/tips/services-cost-optimization)
- [Cloud Scheduler pricing](https://cloud.google.com/scheduler/pricing) — 3 free jobs per billing account
- [Google Cloud Observability pricing](https://cloud.google.com/products/observability/pricing) — non-chargeable metrics, 150 MiB custom
- [Container concepts | Artifact Registry](https://docs.cloud.google.com/artifact-registry/docs/container-concepts) — layer deduplication
- [Secret Manager pricing](https://cloud.google.com/secret-manager/pricing) — **could not be read; terms unconfirmed**

---

## 12. What this document does not cover

Stated so its conclusions are not read as broader than they are:

- **No GCP account exists yet.** Nothing here has been validated against a live
  project, a real bill, or an actual deploy. Every figure is documentation plus
  arithmetic on measured local/CI numbers.
- **No load testing.** The ~50,000 visits/month break-even is modelled from a
  measured cold start and request latency, not observed under concurrency.
- **Secret Manager pricing is unconfirmed** (§7).
- **The Cloud Run allowance discrepancy is unresolved** (§2).
- **Pricing changes.** These terms were true on 2026-08-03 to the best of the
  research; re-verify at signup.
