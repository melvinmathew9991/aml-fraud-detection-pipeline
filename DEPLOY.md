# Deployment Runbook (Sprint 7)

The executable half of `GCP.md`. That document decides *what* the deployment
costs and *why* each control exists; this one is the ordered sequence of
commands that puts it in place, with the output of each step recorded as it is
actually run.

Read alongside:

- `GCP.md` — free-tier analysis, cost model, the four conditions for $0
- `ARCHITECTURE.md` §8 (CI/CD) and §9 (cost controls)
- `ROADMAP.md` Sprint 7 — scope and Definition of Done

**Every command here is run manually by the operator**, not by an agent. Flag
names were verified against **Google Cloud SDK 581.0.0** (core `2026.08.14`);
gcloud's surface changes, so re-check `--help` if a flag is rejected.

---

## 0. Progress

Nothing is marked done here until its command has actually been run and its
output recorded in the phase below. A phase written but not yet executed says
so.

| Phase | What | Status |
|---|---|---|
| 0 | Organization preflight (org policies) | **Done** 2026-09-09 |
| 1 | CLI auth, project, billing link | **Done** 2026-09-09 |
| 2 | Enable APIs | **Done** 2026-09-09 |
| 3 | Artifact Registry + cleanup policy | **Done** 2026-09-09 |
| 4 | Service account + Workload Identity Federation | **Done** 2026-09-09 |
| 5 | GitHub repository variables | **Partial** 2026-09-09 -- `GCP_PROJECT_ID` withheld |
| 6 | First deploy via CI | **Done** 2026-09-09 |
| 7 | Verification, cold start, rollback drill | **Done** 2026-09-09 |
| 7.5 | Streamlit Community Cloud dashboard | **Done** 2026-09-09 |

## 0.1 Established identifiers

Filled in from Phase 0/1 as actually run, not planned. Every `<placeholder>`
below this line has been substituted with these values.

| | |
|---|---|
| Organization | `meriatmelvin-org` |
| Organization ID | `1059247637719` |
| Project ID | `aml-fraud-detection` |
| Project number | `893810819766` |
| Abandoned | `aml-fraud-detection-9991` / `68907718223` -- deleted 2026-09-09 |
| Billing account | `019A9C-7B7A66-F1618C` (`My Billing Account`, OPEN) |
| Region | `us-central1` |
| Artifact Registry repo | `fraud-api` |
| Cloud Run service | `fraud-api` |
| GitHub repo | `melvinmathew9991/aml-fraud-detection-pipeline` |
| Cloud Run URL | `https://fraud-api-amj2cl4jhq-uc.a.run.app` |
| Dashboard URL | `https://aml-fraud-detection-pipeline.streamlit.app/` |

The project number and project ID are both used below and are **not**
interchangeable: WIF resource paths take the number, everything else the ID.

---

**Already done before Phase 1** (browser, operator):

- [x] Google Cloud account created, trial taken -- **INR 28,694 / 90 days**,
      the India-locale form of the $300 in `GCP.md`. Untouched as of 2026-09-09,
      with **67 days left (expires ~2026-11-15)**.
- [x] `gcloud` installed -- SDK 581.0.0, core 2026.08.14. Verified on PATH
      2026-09-09. `ROADMAP.md` Sprint 7 still lists this as a to-do.
- [x] The account has an **organization** (`meriatmelvin-org`). This runbook was
      written for a bare personal account; Phase 0 exists because of it, and it
      changes the Phase 1.3 command.
- [ ] **Upgraded to a Paid billing account** (`GCP.md` §5① — the trial is not a
      resting place: it *stops and then deletes* resources at day 90)
- [x] Budget alert at ~₹100 (the roadmap's $1), thresholds 50 / 90 / 100%,
      scoped to the project. Created 2026-09-09, **confirmed by the operator,
      not verified from the CLI** -- `gcloud billing budgets list` needs
      `billingbudgets.googleapis.com` enabled, and enabling an extra API purely
      to read back a setting was not worth it. Sprint 7's DoD asks for "budget
      alert confirmed active", so re-confirm it in the Console at §7 rather than
      inheriting this line.

---

## 0.5 Organization preflight

_Written, not yet run._

The account is not a bare personal one -- it sits under an organization,
`meriatmelvin-org`. That introduces exactly one way this sprint's Definition of
Done can fail outright, and it is cheaper to find out here than at deploy time.

### 0.5.1 Authenticate and find the org

```
gcloud auth login
gcloud organizations list
```

Use the same Google account the Cloud account was created under. Record the
`ID` column as the org ID -- Phase 1.3 needs it.

If `organizations list` returns nothing, the likely cause is a missing
`resourcemanager.organizations.get` permission rather than an absent org; the
Console's project picker shows the org either way.

### 0.5.2 The check that matters: domain restricted sharing

```
gcloud org-policies describe constraints/iam.allowedPolicyMemberDomains   --organization=1059247637719 --effective
```

Legacy surface, if that form is rejected:

```
gcloud resource-manager org-policies describe   iam.allowedPolicyMemberDomains --organization=1059247637719 --effective
```

**Why this one.** Organizations created after May 2024 receive Google's
"secure by default" policy set, and this constraint is in it. It forbids binding
`allUsers` to an IAM role -- which is precisely what `--allow-unauthenticated`
does in the `deploy` job of `.github/workflows/ci.yml`. If it is enforced:

- the deploy job fails at the IAM step, not the build step
- the error names neither Cloud Run nor the org policy; it reads roughly
  `One or more users named in the policy do not belong to a permitted customer`
- Sprint 7's DoD, "public URL serves `/score`", cannot be met as written

**Result: NOT enforced.** Run 2026-09-09:

```
$ gcloud org-policies describe constraints/iam.allowedPolicyMemberDomains \
    --organization=1059247637719 --effective
name: organizations/1059247637719/policies/iam.allowedPolicyMemberDomains
spec:
  rules:
  - allowAll: true
```

`allowAll: true` means public IAM bindings are permitted, so
`--allow-unauthenticated` will succeed and Sprint 7's DoD is achievable as
written. **§1.7 is therefore not needed** -- skip it.

Worth keeping the section rather than deleting it: the check is one command, the
failure it guards against is silent until deploy time, and a re-created project
under a different org would need it again.

### 0.5.3 The other default policies, and why they do not bite

| Constraint | Effect here |
|---|---|
| `iam.disableServiceAccountKeyCreation` | **None.** Phase 4 is keyless WIF by design (`ARCHITECTURE.md` §8) -- policy and architecture already agree. |
| `iam.automaticIamGrantsForDefaultServiceAccounts` | **None.** The Cloud Run runtime SA needs no project permissions; the container calls no Google API. |
| `compute.skipDefaultNetworkCreation` | **None.** Cloud Run fully managed needs no VPC. |
| `compute.vmExternalIpAccess`, `sql.restrictPublicIp` | **None.** No Compute Engine or Cloud SQL in Sprints 7-12. |

Only the constraint in §0.5.2 is load-bearing.

### 0.5.4 Org-level roles held

Checked 2026-09-09 -- `resourcemanager.projectCreator` is the one Phase 1.3
requires, and it is present:

```
roles/resourcemanager.organizationAdmin
roles/resourcemanager.projectCreator
roles/resourcemanager.projectMover
roles/billing.admin
roles/billing.creator
roles/iam.workforcePoolAdmin
roles/serviceusage.serviceUsageAdmin
```

`billing.admin` also covers the budget alert in §1.6 without extra grants.

---

## 1. Authenticate, create the project, link billing

### 1.1 Authenticate the CLI

```
gcloud auth login
```

Browser flow. Use the same Google account the Cloud account was created under.
Signed into several Google accounts? This is the step that quietly goes wrong --
the wrong choice logs in successfully to an account with no billing and no org.

**Windows: expect this to fail on the first try.** PowerShell's default
execution policy refuses to run `gcloud.ps1`:

```
gcloud : File ...\google-cloud-sdkin\gcloud.ps1 cannot be loaded because
running scripts is disabled on this system.
```

Encountered 2026-09-09. Two fixes, and the second is the right one for a runbook
that is ~40 gcloud invocations long:

```
gcloud.cmd auth login    # the batch wrapper, unaffected by execution policy
```

```
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

`CurrentUser` scope needs no administrator prompt, and `RemoteSigned` is the
workstation default Microsoft recommends -- local scripts run, downloaded ones
must be signed. Every later `gcloud` in this document assumes it has been set.

This affects the operator's PowerShell only. Git Bash and CI invoke the shell
wrapper rather than the `.ps1` and never hit it.

### 1.2 Find the billing account

```
gcloud billing accounts list
```

Record the `ACCOUNT_ID` (form `01A2B3-C4D5E6-F7G8H9`). `OPEN: True` is required
— a closed billing account is what a lapsed trial leaves behind.

**Recorded 2026-09-09:**

```
ACCOUNT_ID            NAME                OPEN  MASTER_ACCOUNT_ID
019A9C-7B7A66-F1618C  My Billing Account  True
```

### 1.3 Create the project

Project IDs are **globally unique across all of Google Cloud**. This section
originally predicted that a bare `aml-fraud-detection` would "very likely be
taken" and pre-emptively suffixed it; on 2026-09-09 the bare ID turned out to be
free and is what the project uses. Constraints are 6-30 characters, lowercase
letters, digits and hyphens.

**Project IDs are never released.** A deleted project's ID cannot be reused --
not after the 30-day purge, not ever. So this is a one-shot choice, and it is
worth spending a moment on: the ID appears in every Artifact Registry image
path (`us-central1-docker.pkg.dev/<project>/fraud-api/fraud-api:<sha>`) and in
the WIF principal set.

```
gcloud projects create aml-fraud-detection   --name="AML Fraud Detection"   --organization=1059247637719
```

**`--organization` is new**, and required now that the account has one -- without
it the project is created unparented, which org policy may refuse. It also needs
`roles/resourcemanager.projectCreator` at the org; the account that created the
organization holds that by default.

Do **not** reuse the auto-created "My First Project". §8's teardown story
("delete the *project*") only holds if the project contains nothing else.

### 1.4 Make it the default for every later command

```
gcloud config set project aml-fraud-detection
```

Everything downstream assumes this is set. Skipping it means silently operating
against whatever project gcloud last had configured — a genuinely dangerous
default when billing is involved.

### 1.5 Link billing

```
gcloud billing projects link aml-fraud-detection \
  --billing-account=YOUR_ACCOUNT_ID
```

Linking billing is what makes Cloud Run possible at all. It does not itself
start charging anything; free-tier usage stays at zero cost either way
(`GCP.md` §5①).

### 1.6 Re-scope the budget alert to this project

Console → **Billing → Budgets & alerts** → edit the existing budget → scope it
to the new project.

The budget was created account-wide because no project existed yet. An
account-wide budget still works as a backstop, but a project-scoped one tells
you *which* resource is spending. Keep the thresholds at 50 / 90 / 100%.

**Recorded 2026-09-09.** Phase 1 was run twice, and the record should say so
rather than read as if it went straight through.

A first project, `aml-fraud-detection-9991` (number `68907718223`), was created
from the CLI and had billing linked. Its suffix existed only because §1.3
assumed the bare ID would be taken. The operator then created
`aml-fraud-detection` in the Console -- **the bare ID was available**,
falsifying that assumption -- and that project is the one this runbook now
targets.

The `-9991` project holds no resources. Shutting it down and re-pointing the CLI
default are the two steps that close this phase:

```
gcloud config set project aml-fraud-detection
gcloud projects delete aml-fraud-detection-9991
```

`config set project` matters more than it looks: gcloud's default was still the
abandoned project, and every unqualified command in Phases 2-4 inherits it.
Deletion is a 30-day pending state, reversible with `gcloud projects undelete`,
after which the ID is gone permanently.

**Done 2026-09-09:**

```
$ gcloud config set project aml-fraud-detection
Updated property [core/project].

$ gcloud projects delete aml-fraud-detection-9991
Deleted [.../projects/aml-fraud-detection-9991].
You can undo this operation for a limited period by running:
    $ gcloud projects undelete aml-fraud-detection-9991

$ gcloud config get-value project
aml-fraud-detection
```

Phase 1 is closed: the budget alert was created in the Console on 2026-09-09.

The surviving project, as verified:

```
$ gcloud projects describe aml-fraud-detection \
    --format="value(projectNumber,parent.type,parent.id)"
893810819766  organization  1059247637719

$ gcloud billing projects describe aml-fraud-detection \
    --format="value(billingEnabled)"
True
```

`parent.type: organization` with `parent.id` matching the org ID is what
confirms the project is parented rather than orphaned -- the check §1.3 warns
about. Console-created projects under an org get this automatically; the
`--organization` flag is what buys it on the CLI path.

Both projects came up with the same ~22 default APIs (BigQuery, Storage,
Logging, Monitoring and friends). None of them bill anything while idle, and
none of them are the four Phase 2 enables -- Cloud Run and Artifact Registry are
not in the default set.

**Phase 2 done 2026-09-09.** `gcloud services enable` returned
`Operation "operations/acf.p2-893810819766-..." finished successfully`, and all
four verify as enabled:

```
artifactregistry.googleapis.com
iamcredentials.googleapis.com
run.googleapis.com
sts.googleapis.com
```

The operation id embeds project number `893810819766`, which is the confirmation
that it landed on the right project rather than an inherited default.

**Still outstanding in §1.6: the budget alert.** Billing is now enabled and no
budget exists. Nothing can spend yet -- no APIs, no registry, no service -- but
this must be closed before Phase 6, per `GCP.md` §5: *before* the first deploy,
not after.

---

## 1.7 Org policy exception -- only if §0.5.2 said "enforced"

> **NOT NEEDED for this deployment.** §0.5.2 came back `allowAll: true` on
> 2026-09-09 -- the constraint is not enforced, so there is nothing to except.
> Retained for the case where the project is rebuilt under a different
> organization. Skip to Phase 2.

This is here rather than in Phase 0 for an ordering reason: the exception is
**scoped to the project**, so the project has to exist first.

Grant yourself org-policy admin once:

```
gcloud organizations add-iam-policy-binding 1059247637719   --member="user:<YOUR_GOOGLE_ACCOUNT_EMAIL>"   --role="roles/orgpolicy.policyAdmin"
```

Then relax the constraint **for this project only**:

```
cat > /tmp/allow-public-invoker.yaml <<'EOF'
constraint: constraints/iam.allowedPolicyMemberDomains
listPolicy:
  allValues: ALLOW
EOF

gcloud resource-manager org-policies set-policy /tmp/allow-public-invoker.yaml   --project=aml-fraud-detection
```

Confirm it took, from the project's point of view:

```
gcloud org-policies describe constraints/iam.allowedPolicyMemberDomains   --project=aml-fraud-detection --effective
```

**Project-scoped, not org-wide, on purpose.** Turning the constraint off at the
org node would allow public IAM bindings in every project the account will ever
have, to fix one service that needs a public URL. The blast radius of the
exception should match the blast radius of the requirement.

**Recorded output:** _(pending -- fill in when run)_

---

## 2. Enable the APIs

_Written, not yet run._

```
gcloud services enable \
  run.googleapis.com \
  artifactregistry.googleapis.com \
  iamcredentials.googleapis.com \
  sts.googleapis.com
```

Four, and each is load-bearing: Cloud Run to serve, Artifact Registry to store
the image, and `iamcredentials` + `sts` for the Workload Identity Federation
token exchange in Phase 4. Enabling can take a minute or two to propagate.

**Not enabled: `compute.googleapis.com`.** Cloud Run fully managed does not need
it, and it is not free of consequence -- enabling Compute Engine is what creates
the `893810819766-compute@developer.gserviceaccount.com` default service
account. Cloud Run uses that account as its *runtime* identity unless told
otherwise, which is the one snag Phase 4 has to decide about. See §4.0.

---

## 3. Artifact Registry and the cleanup policy

_Written, not yet run._

**This is two commands, not one.** `gcloud artifacts repositories create` has no
cleanup-policy flag (verified, SDK 581.0.0); policies are set by a separate
`set-cleanup-policies` call. `GCP.md` §9 previously read "applied at creation",
which is not achievable — the achievable and equivalent guarantee is **the
policy is in place before any image is pushed**, which is what this ordering
gives.

Retention is **2 versions**, sized in `GCP.md` §6 against a measured
510.6 MB uncompressed / 169.8 MB compressed image (CI run `30795258811`). Three
versions breach the decimal reading of the "0.5 GB" free tier by 34 MB.

*[2026-09-12: the image has since been re-measured twice — 518.9/172.7 MB (run
`34344122710`), then **367.5/125.0 MB** after PR #21 (`cea59f4`) dropped
`pyarrow`. At 125.0 MB three versions would fit; the deployed policy is
deliberately still 2. The commands below are unchanged and still correct.]*

### 3.1 Create the repository

The name is **not** free choice: `.github/workflows/ci.yml` hardcodes
`AR_REPO: fraud-api` and builds the image path as
`us-central1-docker.pkg.dev/<project>/fraud-api/fraud-api:<sha>`. A different
name here means a push to a repository that does not exist.

```
gcloud artifacts repositories create fraud-api   --repository-format=docker   --location=us-central1   --description="Serving image for the fraud-detection API"
```

`us-central1` matches `GCP_REGION` in the workflow, and is a Tier 1 region --
`GCP.md` §8 explains why that is a cost decision rather than a default.

### 3.2 Write the cleanup policy

```
mkdir -p deploy
cat > deploy/artifact-registry-cleanup.json <<'EOF'
[
  {
    "name": "keep-2-most-recent",
    "action": { "type": "Keep" },
    "mostRecentVersions": { "keepCount": 2 }
  },
  {
    "name": "delete-everything-else",
    "action": { "type": "Delete" },
    "condition": { "tagState": "ANY", "olderThan": "0s" }
  }
]
EOF
```

Two rules, and the order they are written in does not matter: **Keep rules take
precedence over Delete rules**, so the delete rule cannot evict the two versions
the keep rule protects. Writing only the delete rule would empty the repository;
writing only the keep rule would never reclaim anything.

Commit this file. A retention policy that exists solely as a past CLI invocation
is not reproducible, and `GCP.md` §6's whole argument rests on the number `2`.

### 3.3 Apply it -- dry run first

```
gcloud artifacts repositories set-cleanup-policies fraud-api   --location=us-central1   --policy=deploy/artifact-registry-cleanup.json   --dry-run
```

Then, once the repository description confirms the policy is attached:

```
gcloud artifacts repositories describe fraud-api --location=us-central1

gcloud artifacts repositories set-cleanup-policies fraud-api   --location=us-central1   --policy=deploy/artifact-registry-cleanup.json   --no-dry-run
```

**One caveat that the free-tier maths in `GCP.md` §6 does not mention:** cleanup
policies are evaluated **asynchronously**, on Google's schedule rather than on
push. Storage can therefore sit above 2 versions for some hours after a build.
At 169.8 MB compressed, a third version transiently present is 509 MB against a
500 MB decimal allowance -- a real, if brief, overshoot. It is bounded by the
policy and by `max 2` builds mattering at a time, not by anything this runbook
does, and it is recorded rather than papered over.

*[At the post-PR-#21 125.0 MB, the same transient third version is 375 MB and
no longer overshoots; a transient fourth would be exactly 500 MB. The overshoot
described here was real at the size it was measured at, and the mechanism is
unchanged.]*

**Understated, as it turned out.** That paragraph models one transient third
version for "some hours". What actually happened on 2026-09-09 was **seven
versions and 1,018 MB after nine hours** — twice the allowance, not 2% over it —
because the sweep had not run at all, and the day carried seven deploys rather
than one. The policy was attached and correct throughout; nothing was
misconfigured. See `GCP.md` §6 for the full record and the manual cleanup.

**Done 2026-09-09.** Repository created, then the policy applied on a second
command exactly as §3.3 warns. The dry run was skipped deliberately: the
repository was empty, so a simulated sweep had nothing to report.

```
$ gcloud artifacts repositories describe fraud-api --location=us-central1
Repository Size: 0.000MB
cleanupPolicies:
  delete-everything-else:
    action: DELETE
    condition:
      tagState: ANY
  keep-2-most-recent:
    action: KEEP
    mostRecentVersions:
      keepCount: 2
registryUri: us-central1-docker.pkg.dev/aml-fraud-detection/fraud-api
```

Two things to read off that. `registryUri` matches the path `ci.yml` composes
from `GCP_REGION`, `vars.GCP_PROJECT_ID` and `AR_REPO` -- the names line up end
to end. And the absence of a `cleanupPolicyDryRun` field is what confirms the
policy is live rather than simulating.

`olderThan: 0s` does not survive into the stored condition. `tagState: ANY`
alone matches every version, and the KEEP rule is what spares the newest two.

Policy file committed at `deploy/artifact-registry-cleanup.json`.

**Not enabled: `containerscanning.googleapis.com`**, hence the repository's
`SCANNING_DISABLED`. That is Google's paid scanner; the pipeline runs Trivy in
the `container` job instead, free, and it gates the build before any push.

---

## 4. Service account and Workload Identity Federation

_Written, not yet run._

Keyless: no service-account JSON key is created, downloaded, or stored in the
repository (`ARCHITECTURE.md` §8). GitHub Actions presents a short-lived OIDC
token; GCP exchanges it for a scoped access token.

**The attribute condition is a security control, not boilerplate.** A provider
without one will accept tokens from *any* GitHub repository on the internet.

Commands below use **literal values, not shell variables**, because the operator
shell on this machine is PowerShell and `$VAR` / `$(...)` do not carry over from
the bash idiom. Substitute `893810819766` by hand from §4.1.

Two identities are involved, and conflating them is the usual way this phase
goes wrong:

| Identity | Who it is | What it does |
|---|---|---|
| **deployer SA** | `deployer@aml-fraud-detection.iam.gserviceaccount.com` | What GitHub Actions impersonates. Pushes images, deploys revisions, shifts traffic. |
| **runtime SA** | see §4.0 | What the *container* runs as in production. Needs nothing. |

### 4.0 Decide the runtime identity

Cloud Run runs every service as some service account. With no `--service-account`
flag it uses the Compute Engine default SA,
`893810819766-compute@developer.gserviceaccount.com`.

**Correction, 2026-09-09.** This section originally claimed that account exists
only once `compute.googleapis.com` is enabled, and that Phase 2's not enabling
it therefore forced the issue. That is wrong: the account was present after
Phase 5 with Compute Engine still disabled -- enabling `run.googleapis.com` is
enough to provision it. The recommendation below is unchanged, but it rests on
least privilege rather than on the account being absent.

Two facts observed rather than assumed, both from the Phase 5 output:

- the default SA holds **no project roles at all**. The org's
  `automaticIamGrantsForDefaultServiceAccounts` constraint is doing exactly what
  it is for -- older projects hand that account `roles/editor` automatically.
  §0.5.3 listed this constraint as having no effect here; it in fact has a
  useful one.
- it is nonetheless a **shared** identity. Every future Cloud Run service, job
  or Compute instance in this project defaults to it, so anything granted to it
  later is silently granted to this API too.

So there is a decision here, and it should be made before the first deploy
rather than discovered by one:

- **Recommended: a dedicated runtime SA with no roles at all.** The container
  scores transactions and calls no Google API, so zero permissions is the
  correct grant. It also avoids enabling Compute Engine purely to obtain a
  default account. Cost: one line added to the `deploy` job's flags.
- **Alternative: enable `compute.googleapis.com`** and accept the default SA.
  No workflow change, but it provisions an account broader than this service
  needs, in a project whose whole thesis is minimal surface.

Taking the recommendation:

```
gcloud iam service-accounts create fraud-api-runtime \
  --display-name="Cloud Run runtime identity (no permissions by design)"
```

and add to the `COMMON` variable in the `deploy` job of
`.github/workflows/ci.yml`:

```
--service-account=fraud-api-runtime@aml-fraud-detection.iam.gserviceaccount.com
```

**Decision taken 2026-09-09: the dedicated runtime SA.** The edit is made -- the
`deploy` job's `COMMON` block now carries
`--service-account=fraud-api-runtime@${{ vars.GCP_PROJECT_ID }}.iam.gserviceaccount.com`,
written against the repository variable rather than a hardcoded project id so it
follows the project the rest of the workflow already targets. It is uncommitted,
and belongs in the same commit as the rest of Phase 4.

### 4.1 Create the deployer SA, and get the project number

```
gcloud iam service-accounts create deployer \
  --display-name="GitHub Actions deployer"

gcloud projects describe aml-fraud-detection --format="value(projectNumber)"
```

Record the number. The WIF resource names in §4.4-4.6 use the project *number*,
not the project ID. They are not interchangeable, and a provider string built
with the ID fails authentication with an unhelpful message.

### 4.2 Grant the deployer exactly what the pipeline uses

Read the `deploy` job before granting anything: it deploys, describes revisions,
reads two autoscaling annotations, runs `update-traffic`, and on failure calls
`gcloud run services logs read`. That is the whole permission requirement.

```
gcloud projects add-iam-policy-binding aml-fraud-detection \
  --member="serviceAccount:deployer@aml-fraud-detection.iam.gserviceaccount.com" \
  --role="roles/run.admin"

gcloud projects add-iam-policy-binding aml-fraud-detection \
  --member="serviceAccount:deployer@aml-fraud-detection.iam.gserviceaccount.com" \
  --role="roles/artifactregistry.writer"
```

`roles/run.admin` is what lets the deploy job set the public invoker binding that
`--allow-unauthenticated` requests -- which is why §0.5.2's org policy, not this
role, is the thing that can block a public URL.

`artifactregistry.writer`, not `admin`: the pipeline pushes images and never
administers the repository. §3 created the repo and its cleanup policy by hand
precisely so CI would not need the rights to.

**Impersonation of the runtime SA.** Deploying a service that *runs as* an
account requires `iam.serviceAccountUser` on that account:

```
gcloud iam service-accounts add-iam-policy-binding \
  fraud-api-runtime@aml-fraud-detection.iam.gserviceaccount.com \
  --member="serviceAccount:deployer@aml-fraud-detection.iam.gserviceaccount.com" \
  --role="roles/iam.serviceAccountUser"
```

(If §4.0 took the Compute Engine alternative instead, the target is
`893810819766-compute@developer.gserviceaccount.com`.)

Optional, and only to keep one failure path useful: the log read on a failed
smoke test needs `roles/logging.viewer`. Without it that step prints an error and
continues, because the workflow guards it with `|| true`. Grant it if you want
failed deploys to be diagnosable from the Actions log:

```
gcloud projects add-iam-policy-binding aml-fraud-detection \
  --member="serviceAccount:deployer@aml-fraud-detection.iam.gserviceaccount.com" \
  --role="roles/logging.viewer"
```

### 4.3 Create the Workload Identity Pool

```
gcloud iam workload-identity-pools create github \
  --location=global \
  --display-name="GitHub Actions"
```

### 4.4 Create the OIDC provider -- with the attribute condition

```
gcloud iam workload-identity-pools providers create-oidc github \
  --location=global \
  --workload-identity-pool=github \
  --display-name="GitHub OIDC" \
  --issuer-uri="https://token.actions.githubusercontent.com" \
  --attribute-mapping="google.subject=assertion.sub,attribute.repository=assertion.repository,attribute.repository_owner=assertion.repository_owner" \
  --attribute-condition="assertion.repository == 'melvinmathew9991/aml-fraud-detection-pipeline'"
```

**`--attribute-condition` is the security control.** GitHub's OIDC issuer is
shared by every repository on GitHub. A provider that maps `attribute.repository`
but does not *condition* on it will exchange a token minted by anyone's workflow
for credentials to this project. The condition is what makes "keyless" mean "only
this repository" rather than "no key, and no lock either".

Recent gcloud refuses to create a provider that maps restricted attributes
without a condition. That is a guardrail, not a substitute for reading the flag.

### 4.5 Let the repository impersonate the deployer

```
gcloud iam service-accounts add-iam-policy-binding \
  deployer@aml-fraud-detection.iam.gserviceaccount.com \
  --role="roles/iam.workloadIdentityUser" \
  --member="principalSet://iam.googleapis.com/projects/893810819766/locations/global/workloadIdentityPools/github/attribute.repository/melvinmathew9991/aml-fraud-detection-pipeline"
```

`principalSet://` with `attribute.repository/<owner>/<repo>` scopes the grant to
the repository, so §4.4's condition and this binding restrict independently --
belt and braces on the only path into the project.

### 4.6 The three values Phase 5 needs

| Variable | Value |
|---|---|
| `GCP_PROJECT_ID` | `aml-fraud-detection` |
| `GCP_WIF_PROVIDER` | `projects/893810819766/locations/global/workloadIdentityPools/github/providers/github` |
| `GCP_DEPLOY_SA` | `deployer@aml-fraud-detection.iam.gserviceaccount.com` |

Note `providers/github` -- the *provider* resource path, not the pool path.
A pool path here is the single most common WIF misconfiguration.

**Done 2026-09-09.** Both service accounts created, all four deployer grants
applied, pool and provider created, repository bound. Verified rather than
assumed -- the two bindings that do not appear in the project IAM policy were
checked on the service accounts themselves:

```
$ gcloud iam service-accounts list
deployer@aml-fraud-detection.iam.gserviceaccount.com           GitHub Actions deployer
fraud-api-runtime@aml-fraud-detection.iam.gserviceaccount.com  Cloud Run runtime identity (no permissions by design)
893810819766-compute@developer.gserviceaccount.com             Default compute service account

$ gcloud iam service-accounts get-iam-policy fraud-api-runtime@...
roles/iam.serviceAccountUser  ['serviceAccount:deployer@aml-fraud-detection.iam.gserviceaccount.com']

$ gcloud iam service-accounts get-iam-policy deployer@...
roles/iam.workloadIdentityUser  ['principalSet://iam.googleapis.com/projects/893810819766/locations/global/workloadIdentityPools/github/attribute.repository/melvinmathew9991/aml-fraud-detection-pipeline']

$ gcloud iam workload-identity-pools providers describe github     --location=global --workload-identity-pool=github
projects/893810819766/.../providers/github
assertion.repository == 'melvinmathew9991/aml-fraud-detection-pipeline'
```

`attributeCondition` coming back non-empty is the check that matters. An empty
one means the provider accepts tokens from any repository on GitHub, and nothing
downstream would fail visibly -- it is a security hole, not a broken build.

Project-level roles on `deployer`, from the Phase 5 policy dump:
`roles/run.admin`, `roles/artifactregistry.writer`, `roles/logging.viewer`.
`fraud-api-runtime` appears in no project binding at all, which is the intended
end state.

---

## 5. GitHub repository variables

_Written, not yet run._

Three repository **variables** — not secrets, because with WIF none of these
values are credentials:

| Variable | Value |
|---|---|
| `GCP_PROJECT_ID` | the project ID from §1.3 |
| `GCP_WIF_PROVIDER` | `projects/<number>/locations/global/workloadIdentityPools/<pool>/providers/<provider>` |
| `GCP_DEPLOY_SA` | `<sa-name>@<project>.iam.gserviceaccount.com` |

Setting `GCP_PROJECT_ID` is what switches deployment on: every GCP step in
`.github/workflows/ci.yml`, and the `deploy` job itself, is gated on it being
non-empty. Until then the pipeline runs its PR gate and skips deployment, which
is why `main` stayed green while this was being built.

### 5.1 Set them

`gh` is installed (v2.97.0) and on PATH. It needs its own browser login once --
`gh auth login`, device flow, run by the operator.

```
gh variable set GCP_WIF_PROVIDER --repo melvinmathew9991/aml-fraud-detection-pipeline \
  --body "projects/893810819766/locations/global/workloadIdentityPools/github/providers/github"

gh variable set GCP_DEPLOY_SA --repo melvinmathew9991/aml-fraud-detection-pipeline \
  --body "deployer@aml-fraud-detection.iam.gserviceaccount.com"
```

### 5.2 Set `GCP_PROJECT_ID` last, and deliberately

```
gh variable set GCP_PROJECT_ID --repo melvinmathew9991/aml-fraud-detection-pipeline \
  --body "aml-fraud-detection"

gh variable list --repo melvinmathew9991/aml-fraud-detection-pipeline
```

**This variable is the arming switch, and the order above is not cosmetic.**
Every GCP step in the workflow is gated on `vars.GCP_PROJECT_ID != ''`. The
moment it is non-empty, the next push to `main` builds, pushes an image and
deploys a live public service. Set it while Phases 1-4 are half-done and the
first deploy fails somewhere in the middle -- with an image already consuming
registry storage.

Set the other two first, verify with `gh variable list`, then set this one.

**Partially done 2026-09-09.** The two non-arming variables are set; the arming
one is deliberately withheld until the PR gate is green:

```
$ gh variable list --repo melvinmathew9991/aml-fraud-detection-pipeline
GCP_DEPLOY_SA     deployer@aml-fraud-detection.iam.gserviceaccount.com
GCP_WIF_PROVIDER  projects/893810819766/locations/global/workloadIdentityPools/github/providers/github
```

`GCP_PROJECT_ID` remains **unset**. The sequencing is the point: with it unset,
opening a PR exercises the full build -- lint, tests, `docker build`, the 512 MiB
memory check and Trivy -- with every GCP step skipped. That proves the image is
sound before any credential is exchanged or any image reaches the registry. Set
it only after that gate is green, then merge.

Note `gh variable list` needs `--repo` when run from outside a clone; without it
`gh` tries to infer the repository from the working directory and fails with
`not a git repository`.

---

## 6. First deploy

### 6.0 PR gate, run 34344122710 (2026-09-09)

Green on all three jobs, with `deploy` skipped as designed -- `GCP_PROJECT_ID`
was still unset, so the pipeline proved the build without any credential
exchange or any image reaching the registry.

```
Lint, type-check, test, smoke-train    success
Serving dependency isolation           success
Build image, integration test, trivy   success
Deploy to Cloud Run                    skipped
```

**Measured, and the figures have drifted.** The sizing argument in `GCP.md` §6
and `ARCHITECTURE.md` rests on CI run `30795258811`; this run reports:

| Metric | Run 30795258811 | Run 34344122710 | Change |
|---|---|---|---|
| Uncompressed | 510.6 MB | **518.9 MB** | +8.3 MB |
| Compressed (registry estimate) | 169.8 MB | **172.7 MB** | +2.9 MB |
| Versions fitting free tier | 2 | **2** | unchanged |
| Memory after a scored request | -- | **180.9 MiB / 512 MiB (35%)** | -- |

Retention of 2 still holds and no `::warning::` fired, so the policy needs no
change. But the older numbers are now stale wherever they are quoted as current,
and the drift is exactly what the re-measuring step exists to surface. The
compressed figure has roughly 65 MB of headroom before `FITS` drops to 1 and the
warning does fire.

The memory reading is a **floor**, not a peak: it follows a single-transaction
score, not the 10,000-row batch `/score/batch` accepts.

### 6.1 The deploy itself

**Done 2026-09-09.** Live at **https://fraud-api-amj2cl4jhq-uc.a.run.app**,
revision `fraud-api-00001-jr7`, image tag `08fbf53`.

It took two attempts, and the reason is worth recording. The PR was merged with
`GCP_PROJECT_ID` still unset -- deliberately, per §5.2 -- so the post-merge run
built and tested everything and then **skipped** `deploy`, leaving `main` green
and nothing deployed. Setting the variable does not retrigger anything by
itself; the pipeline needs a workflow run, and `gh run rerun` on the merge
commit supplied one. The disarmed state is not a failure mode, but it does mean
"merged and green" is not the same as "deployed", and only the `deploy` job's
own conclusion distinguishes them.

Verified independently of CI's own smoke test:

```
$ curl https://fraud-api-amj2cl4jhq-uc.a.run.app/ready
{"status":"ready","detail":null,"bundle_version":"v1"}

$ curl -X POST .../score -d '{"step":743,"type":"TRANSFER","amount":250000.0,...}'
decision REVIEW | probability 0.9999998584 | latency_ms 4.78
reasons: amount_to_balance_ratio 12.69, orig_balance_mismatch 10.65,
         orig_emptied 8.43, is_transfer 1.14
```

Cost controls read off the **deployed revision**, which is what `GCP.md` §7 asks
for rather than trusting the deploy flags:

| Setting | Value |
|---|---|
| `minScale` | unset, meaning 0 |
| `maxScale` | 2 |
| memory | 512Mi |
| service account | `fraud-api-runtime@aml-fraud-detection.iam.gserviceaccount.com` |

The service-account line confirms the §4.0 decision reached production: the
public endpoint runs as an identity holding no roles on anything.

There is no `/` route, so the service's root returns `{"detail":"Not Found"}`.
The endpoints are `/health`, `/ready`, `/model-info`, `/score`, `/score/batch`,
`/metrics`, with Swagger UI at `/docs`.

Merging to `main` triggers the full pipeline: `lint-test` → `serving-isolation`
→ `container` (build, integration test, trivy, push) → `deploy`.

The `container` job runs the image under `--memory=512m --memory-swap=512m`,
the same ceiling `deploy` passes to Cloud Run, and reports memory in use after a
scored request. So the memory budget is proven in CI before it is enforced by
Cloud Run — an OOM shows up as a red build with an explicit OOM-kill message,
not as a 503 on a public URL. The reported figure is a floor: it comes from a
single-transaction score, not the 10,000-row batch `/score/batch` allows.

The first deploy is a deliberate special case. A service's first revision takes
100% of traffic regardless of `--no-traffic`, because there is no other revision
to serve — so unlike every later deploy, it goes live before the smoke test
rather than after it. The job says so in a `::notice::` rather than leaving it
to surprise someone.

---

## 7. Verification, cold start, rollback drill

_Written, not yet run._

Sprint 7's DoD (`ROADMAP.md`):

- [x] **public URL serves `/score`** -- verified 2026-09-09, §6.1
- [x] **cold-start latency measured and documented** -- 5.31 s round trip, §7.2
- [x] **budget alert confirmed active** -- §7.1
- [x] **a deliberately broken deploy rolls back** -- §7.3

### 7.1 Budget alert, verified rather than asserted

`gcloud billing budgets list` needs `billingbudgets.googleapis.com`, which §1.6
declined to enable purely to read a setting back. The DoD asks for *confirmed*,
so it was enabled and the budget read:

```
?100 Monthly Budget Alert   100  INR   projects/893810819766
thresholds: 0.5, 0.9, 1.0 (CURRENT_SPEND)
```

Amount, currency, project scope and all three thresholds match §1.6. The
operator's earlier confirmation is now evidence.

### 7.3 Rollback drill

The drill was run at the Cloud Run layer rather than by merging a deliberately
broken commit -- it exercises the same mechanism without putting a known-bad
commit in `main`'s history. The working image was redeployed with its startup
command replaced by `sh -c "exit 1"`, using the same `--no-traffic --tag=candidate`
flags the `deploy` job uses:

```
$ gcloud run deploy fraud-api --image=<same image> --no-traffic --tag=candidate \
    --command=sh --args="-c,exit 1" ...
Deployment failed
ERROR: The user-provided container failed to start and listen on the port
defined provided by the PORT=8080 environment variable...
```

**The failure is the drill passing.** The traffic split afterwards:

```
{'percent': 100, 'revisionName': 'fraud-api-00001-jr7'}
{'revisionName': 'fraud-api-00002-dec', 'tag': 'candidate',
 'url': 'https://candidate---fraud-api-amj2cl4jhq-uc.a.run.app'}
```

The broken revision's traffic entry carries **no `percent` field at all**. It
was never given a share to lose. `ARCHITECTURE.md` §8's "traffic stays on the
previous revision" is demonstrated, and the property is stronger than rollback:
there was nothing to undo, because the healthy revision never stopped serving.

What this drill does **not** cover: the `deploy` job's own `/ready` +
`bundle_version` gate, which guards the case where a container starts
successfully but serves the wrong bundle. That path remains untested.

#### The drill has a mandatory cleanup step, and skipping it broke the next deploy

**`--command` and `--args` apply to the service template, not to one revision,
and `gcloud run deploy` merges into existing service config rather than
replacing it.** So `sh -c "exit 1"` persisted on the service after the drill,
and the next CI deploy -- a docs-only change, with a correctly built and
Trivy-clean image -- produced revision `fraud-api-00004-qaf`, which failed to
start for exactly the same reason the drill revision did.

The reset must be run before any further deploy:

```
gcloud run services update fraud-api --region=us-central1 --command="" --args=""
```

An empty string is the documented way to reset either field to the image
default (`gcloud run services update --help`). Deleting the drill revision does
**not** do this -- the poisoned setting lives on the service, not the revision,
which is why the earlier `FAILED_PRECONDITION` on deleting `00002-dec` was a
red herring rather than the loose end it looked like.

**The failure was contained, and by the mechanism this section exists to
verify.** Traffic stayed at 100% on `fraud-api-00001-jr7` throughout, so an
accidental broken deploy -- reaching production through CI, not through a
deliberate drill -- did not take the service down. That is a stronger
demonstration of the rollback property than the scripted drill above, because
nobody arranged it.

### 7.2 Cold start

**Measured 2026-09-09 12:42:27Z, after 21 minutes of verified zero traffic.**

```
COLD    http=200  tls=0.131s  ttfb=5.311s  total=5.311s
WARM1   http=200             ttfb=0.323s
WARM2   http=200             ttfb=0.342s
```

| | Round trip | Attributable to startup |
|---|---|---|
| Cold | **5.311 s** | ~4.98 s |
| Warm | 0.323-0.342 s | -- |

**Reported as round-trip from India to `us-central1`, not as container startup
time.** The warm figure of ~0.33 s is almost entirely network transit -- the API
self-reports `latency_ms` of 4.8 for the same work -- so roughly 0.32 s of the
cold figure is transit too, leaving ~4.98 s of actual cold start. A measurement
taken from inside `us-central1` would report a materially smaller number, and
neither figure is wrong; they answer different questions. This one answers
"what does a first-time visitor experience", which is the question a portfolio
link raises.

~5 s sits inside the 3-5 s band `ARCHITECTURE.md` §6 predicted, at the top of
it. It is the cost of `min-instances=0`, and it is the right trade: the
alternative, `min-instances=1`, is ~USD 61/month indefinitely (`GCP.md` §5②)
to save five seconds on an idle portfolio endpoint.

**The first attempt is recorded because it was wrong.** An earlier measurement
returned 368 ms against 330/319 ms warm -- a 38 ms spread. That is a warm
instance, and had it been written down as the cold-start figure it would have
understated the real number by 14x. Cloud Run had not yet scaled to zero. The
only reliable protocol is a verified idle window with no traffic from any
source, dashboard included.

Two things this exposed:

- Cloud Run had not yet scaled to zero. A genuine measurement needs a verified
  idle period with **no** traffic, including from the dashboard.
- Warm round-trip is ~320 ms while the API self-reports `latency_ms` of ~4.8.
  So roughly 315 ms of the observed figure is network transit from India to
  `us-central1`, not compute. Any cold-start number from this location carries
  that transit cost and should be reported as round-trip, not startup time.

**On cold start:** CI cannot measure this honestly. The smoke test warms the
very revision that would be measured, and a genuine cold start requires the
service to have scaled to zero first (~15 minutes idle). The deploy job
therefore does **not** report a cold-start number. It is measured separately,
after an idle period, rather than logging a warm number under a cold label.

**On the rollback drill:** the pipeline's rollback is structural — a new
revision is deployed with `--no-traffic --tag=candidate` and only receives
traffic after `/ready` returns 200 with a matching `bundle_version`. A failed
deploy therefore requires no undo; the previous revision never stopped serving.
The drill is to prove that, by deploying something deliberately broken and
confirming the live URL is unaffected.

---

## 7.5 The dashboard: Streamlit Community Cloud

> **Done 2026-09-09.** Live at **https://aml-fraud-detection-pipeline.streamlit.app/**,
> pointed at the Cloud Run service. Deferred earlier the same day and then
> delivered, so Sprint 7 closes with no outstanding scope.

Three things the deploy surfaced that the procedure below did not anticipate.

**Secrets can be set after the first deploy.** §7.5.2 implies they go in during
creation, under *Advanced settings*. They were missed, the app deployed anyway,
and adding `API_BASE_URL` afterwards from **Settings -> Secrets** restarted the
app on its own in under a minute -- no redeploy, no rebuild. The landing page
was unaffected throughout, because it makes no API call by design. Worth knowing:
missing this at creation costs nothing.

**It first ran on Python 3.14.7, and is now pinned to 3.12.** Skipping *Advanced
settings* at creation let Streamlit choose its own default, and it chose an
interpreter nothing else in this project is tested against -- CI runs 3.11 and
3.12, the container is `python:3.12-slim`, `pyproject.toml` targets `py311`. All
four pinned dependencies installed and the app ran, so it was a latent
inconsistency rather than a break: the dashboard was the only component on an
untested Python. Pinned to **3.12** in the app's settings on 2026-09-09, which
matches the serving container.

The general lesson is the one §7.5.1 should have stated outright: **an unset
version is a choice, made by the platform, that you inherit.**

**Streamlit found two candidate requirements files** and chose by its own
resolution order:

```
WARN: More than one requirements file detected in the repository.
Available options: uv .../dashboard/requirements.txt, poetry .../pyproject.toml
Used: uv with .../dashboard/requirements.txt
```

It picked correctly. But the choice is not ours to make, and if it ever resolves
the other way the dashboard installs the full training stack -- `shap`, `numba`,
`llvmlite`, `mlflow`, `optuna` -- which is the exact separation
`ARCHITECTURE.md` §3 exists to maintain. Recorded there as well.

`ROADMAP.md` Sprint 7 includes "Deploy Streamlit Community Cloud from the public
repo, pointed at the API". This runbook did not cover it and §8 did not list it
as out of scope, so it was simply missing. It goes last because it needs the
Cloud Run URL from §7 to point at.

### 7.5.1 Create the app

At `share.streamlit.io`, sign in with GitHub, then **New app**:

| Field | Value |
|---|---|
| Repository | `melvinmathew9991/aml-fraud-detection-pipeline` |
| Branch | `main` |
| Main file path | `dashboard/Home.py` |

Dependencies need no configuration: `dashboard/requirements.txt` is resolved
relative to the app directory, which is why it lives beside `Home.py` rather
than at the repo root (`ARCHITECTURE.md` §3).

### 7.5.2 Point it at the API

`dashboard/common.py:40` reads the base URL from the environment, with a
localhost fallback:

```
API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000").rstrip("/")
```

In the app's **Settings -> Secrets**, add:

```
API_BASE_URL = "https://<the Cloud Run URL from section 7>"
```

Streamlit Community Cloud exposes top-level secrets as environment variables as
well as through `st.secrets`, which is what makes `os.environ.get` find it.
**Verify rather than assume:** `common.py:134` renders the resolved base URL in
the sidebar. If it still shows `http://localhost:8000`, the secret did not reach
the environment, and the fix is to read it via `st.secrets` instead.

### 7.5.3 What this costs

$0, and it is outside GCP entirely. It does consume GCP budget in one indirect
way: every dashboard page that calls the API is a Cloud Run request against the
2,000,000/month free allowance (`GCP.md` §2). At portfolio traffic this is
noise.

Cold start is the visible cost, not the billed one. `min-instances=0` means an
idle service takes seconds to answer, which is why `common.py:41-42` pairs a 5s
normal timeout with a 60s cold-start retry -- a scaled-to-zero service should
look slow, not broken.

**The API is public and unauthenticated** until Sprint 10 adds an API key. The
dashboard is not what protects it; `--max-instances=2` is the ceiling on what
abuse can cost (`ARCHITECTURE.md` §9).

**Recorded output:** _(pending -- fill in when run)_

---

## 8. What this runbook does not cover

- **Sprint 8+** (monitoring, drift, Cloud Scheduler) — separate sprints.
- **Custom domains, TLS beyond the default `run.app` certificate, CDN.** The
  generated Cloud Run URL is the deliverable.
- **Secret Manager.** Deliberately unused. Sprint 10's API key and Neon
  connection string are injected as Cloud Run environment variables from GitHub
  Actions secrets instead: $0, and equally effective at keeping credentials out
  of the repository (`GCP.md` §7).
- **Teardown.** If the deployment is retired, deleting the *project* is the
  reliable way to stop all billing — deleting individual resources leaves
  registry storage behind.
