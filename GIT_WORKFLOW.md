# Git Workflow & Migration Plan

How this repository is branched, committed, tagged and protected — plus the
one-time migration needed to get from the current state to that model.

Companion to `ROADMAP.md` (sprint plan) and `ARCHITECTURE.md` (system design).

---

## 0. Attribution policy

**All commits are authored solely by the repository owner.** No AI co-author
trailers, no tool footers in commit messages or PR bodies.

Concretely, none of the following appear in any commit message or PR body:

```
Co-Authored-By: Claude ...
Claude-Session: ...
🤖 Generated with [Claude Code](...)
```

Enforcement is mechanical, not manual — see §5 (`commit-msg` hook). The `.claude/`
directory is already gitignored and has never been tracked (verified
2026-08-01 against full history).

### Identity check

GitHub attributes a commit to an account by matching the **author email** to a
verified email on that account. Current state, verified 2026-08-01:

| Commit | Author | Status |
|---|---|---|
| `0772bb9` | `melvinmathew9991@gmail.com` | OK |
| `e8eb7ff` | `102222281+melvinmathew9991@users.noreply.github.com` | OK (GitHub web-merge identity) |

Both map to the same account, so the contribution graph is correct. Keep
`git config user.email` matching a verified GitHub email:

```bash
git config user.name  "Melvin Mathew"
git config user.email "melvinmathew9991@gmail.com"
```

---

## 1. One-time history migration

### What needs fixing

Exactly one commit carries AI trailers:

```
0772bb9  Sprint 1: XGBoost/LightGBM, time-based CV, Optuna tuning, MLflow tracking
         Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
         Claude-Session: https://claude.ai/code/session_...
```

It is reachable from `main` and already pushed to `origin`. Rewriting it also
rewrites the merge commit above it (`e8eb7ff`), because a commit's hash covers
its ancestry.

**Blast radius:** 2 commits change SHA. This is a solo repository with no other
clones and no open PRs, which is the only reason a `main` rewrite is acceptable
here. Do not do this on a shared branch.

### Procedure

Run these in order. **Steps 1-4 are safe and reversible; step 5 is not.**

```bash
# 1. Backup. If anything goes wrong, this branch is the way back.
git branch backup/pre-rewrite-2026-08-01
git push origin backup/pre-rewrite-2026-08-01

# 2. Install the tool (git-filter-branch is deprecated and mangles merge commits)
pip install git-filter-repo

# 3. Strip the trailers from every commit message in the repo
git filter-repo --force --message-callback '
import re
msg = message.decode("utf-8", "replace")
msg = re.sub(r"^Co-Authored-By: Claude.*$\n?", "", msg, flags=re.M|re.I)
msg = re.sub(r"^Claude-Session:.*$\n?", "", msg, flags=re.M|re.I)
msg = re.sub(r"^.*Generated with \[?Claude Code.*$\n?", "", msg, flags=re.M|re.I)
msg = re.sub(r"^https://claude\.ai/code/session_.*$\n?", "", msg, flags=re.M)
return re.sub(r"\n{3,}", "\n\n", msg).rstrip().encode("utf-8") + b"\n"
'

# 4. VERIFY before pushing. Expect: 0 matches, and history otherwise identical.
git log --all --format='%B' | grep -ciE 'claude|anthropic'   # must print 0
git log --oneline -6
git diff backup/pre-rewrite-2026-08-01 main --stat           # must be EMPTY (content unchanged)

# 5. Point of no return. git-filter-repo removes 'origin', so re-add it.
git remote add origin https://github.com/melvinmathew9991/aml-fraud-detection-pipeline.git
git push --force-with-lease origin main
```

**The callback in step 3 was dry-run against the real `0772bb9` message on
2026-08-01**: 43 lines -> 39, zero `claude`/`anthropic` references remaining, body
otherwise byte-identical. The four trailer lines are the only thing removed.

### After the rewrite

- The stale `sprint-1-cv-optuna-mlflow` and `data-layer-hardening` branches
  (local and remote) still point at old SHAs. Delete them — they are merged and
  serve no purpose:
  ```bash
  git branch -D sprint-1-cv-optuna-mlflow data-layer-hardening
  git push origin --delete sprint-1-cv-optuna-mlflow data-layer-hardening
  ```
- PR #2 on GitHub will show as merged with commits that no longer exist on
  `main`. Cosmetic only; nothing to fix.
- Keep `backup/pre-rewrite-*` for a week, then delete it. **Deleting it is what
  finally removes the trailers from GitHub's reflog reachability.**

### Alternative if the rewrite feels risky

Leave history alone and apply the policy from the next commit forward. One
historical trailer on a merged Sprint 1 commit is a small cost. The rewrite is
recommended but genuinely optional — decide before step 5, not after.

---

## 2. Migrating the current working tree

The working tree currently holds **three logically distinct bodies of work**
that must not become one commit:

| # | Work | Files |
|---|---|---|
| 1 | Sprint 2 implementation | `src/threshold.py`, `src/explain.py`, `src/error_analysis.py`, `src/features.py`, `src/train_pipeline.py`, `config.yaml`, `requirements.txt`, 6 new + 4 regenerated CSVs |
| 2 | Sprint 2 audit corrections | `README.md`, `ROADMAP.md` (Status), plus fixes inside the files above |
| 3 | Architecture & plan for Sprints 3-12 | `ARCHITECTURE.md` (new), `ROADMAP.md` (sections 1-4), `GIT_WORKFLOW.md` (this file) |

Work 1 and 2 belong together — the project's own rule is that a sprint is not
staged until implementation *and* its audit are complete, so they are one
iteration. Work 3 is independent and gets its own branch.

### Branch A — `sprint-2-capacity-shap-error-analysis`

```bash
git switch -c sprint-2-capacity-shap-error-analysis

# Commit 1: code, config, generated evidence
git add src/threshold.py src/explain.py src/error_analysis.py \
        src/features.py src/train_pipeline.py config.yaml requirements.txt \
        data/processed/*.csv
git commit          # message: yours

# Commit 2: the written results + audit corrections
git add README.md
git add -p ROADMAP.md     # <- stage ONLY the Sprint 2 / audit Status hunks
git commit          # message: yours
```

**The one fiddly step** is `git add -p ROADMAP.md`: that file contains both the
Sprint 2 Status entries (branch A) and the new sections 1-4 plan (branch B).
Accept the Status hunks, reject the sections 1-4 hunks. Verify before
committing:

```bash
git diff --cached ROADMAP.md   # should show ONLY Status-section changes
git diff ROADMAP.md            # remainder stays for branch B
```

Then open a PR into `main`, review the diff yourself, and merge.

### Branch B — `docs/architecture-and-plan-sprints-3-12`

Created **after** branch A merges, so it starts from an up-to-date `main` and
cannot conflict on `ROADMAP.md`:

```bash
git switch main && git pull
git switch -c docs/architecture-and-plan-sprints-3-12
git add ARCHITECTURE.md GIT_WORKFLOW.md ROADMAP.md
git commit          # message: yours
```

### Retroactive tags once both are merged

```bash
git tag -a v0.2.0 -m "Sprint 2: capacity-based operating point, SHAP, error analysis"
git push origin v0.2.0
```

---

## 3. Branch model (going forward)

`main` is always deployable. Nothing commits directly to it after §1.

| Prefix | Use | Merge style |
|---|---|---|
| `sprint-N-<slug>` | One per sprint from `ROADMAP.md` | **Merge commit** — preserves the implementation-then-audit story, which is a selling point of this repo |
| `fix/<slug>` | Bug fix outside a sprint | Squash |
| `docs/<slug>` | Documentation only | Squash |
| `chore/<slug>` | Tooling, deps, gitignore | Squash |
| `ci/<slug>` | Workflow changes | Squash |
| `backup/<slug>` | Pre-rewrite safety net, deleted after | never merged |

Planned sprint branches:

```
sprint-3-tests-economics-bundle        sprint-8-monitoring-drift
sprint-4-inference-core-api            sprint-9-portfolio-polish
sprint-5-streamlit-dashboard           sprint-10-persistence-auth-feedback
sprint-6-containerization-ci           sprint-11-graph-analytics-stats
sprint-7-cloud-deployment              sprint-12-genai-str-narratives
```

One PR per sprint. A sprint branch is not opened until the previous sprint's PR
is merged — the plan is sequential and the dependencies are real (Sprint 11's
graph output feeds Sprint 12).

---

## 4. Commit message convention

[Conventional Commits](https://www.conventionalcommits.org/). Enables changelog
generation later and reads as deliberate in a portfolio.

```
<type>(<scope>): <subject>

<body: WHY, not what — the diff already says what>

<footer: Refs #N / BREAKING CHANGE:>
```

**Types:** `feat` `fix` `docs` `test` `refactor` `perf` `chore` `ci` `build`

**Scopes** (match the project layout): `features` `pipeline` `cv` `threshold`
`economics` `explain` `inference` `api` `dashboard` `monitoring` `bundle`
`graph` `genai` `ci` `deps`

Rules:
- Subject: imperative, lower case, no trailing period, <= 72 chars
- Body wrapped at 72; explain the reasoning, tradeoff, or the bug's cause
- One logical change per commit — if the body needs "and", split it
- **No AI attribution trailers** (§0)

> **Message authorship:** commit message text is written by the repository
> owner, not drafted by tooling. Staging is prepared; wording is yours.

---

## 5. Safety hooks

`.githooks/` committed to the repo, activated once per clone:

```bash
git config core.hooksPath .githooks
```

**`commit-msg`** — rejects AI attribution, mechanically enforcing §0:

```bash
#!/bin/sh
if grep -qiE 'co-authored-by:.*(claude|anthropic|copilot)|claude-session|generated with \[?claude' "$1"; then
  echo "commit-msg: AI attribution trailer found. See GIT_WORKFLOW.md §0." >&2
  exit 1
fi
```

**`pre-commit`** — blocks the two mistakes that are painful to undo:

```bash
#!/bin/sh
# 1. Large files (the 493MB raw CSV, the 684MB DuckDB store)
for f in $(git diff --cached --name-only --diff-filter=A); do
  [ -f "$f" ] || continue
  sz=$(wc -c < "$f")
  if [ "$sz" -gt 20971520 ]; then
    echo "pre-commit: $f is $((sz/1048576))MB (>20MB). Intended? See .gitignore." >&2
    exit 1
  fi
done
# 2. Secrets (Gemini key from Sprint 12, Neon connection string from Sprint 10)
if git diff --cached | grep -qE 'AIza[0-9A-Za-z_-]{35}|postgres(ql)?://[^ ]+:[^ ]+@'; then
  echo "pre-commit: possible secret in staged diff." >&2
  exit 1
fi
```

Both are advisory guards against accident, not security controls.

---

## 6. Branch protection on `main`

Configure on GitHub **after** the §1 rewrite (protection blocks force-push):

- Require a pull request before merging
- Require status checks to pass — enable once Sprint 6 lands CI
- Require branches to be up to date before merging
- Block force pushes and deletions
- Do **not** require approvals (solo repo; self-review on the PR page is the
  point, and requiring approval would deadlock)

---

## 7. Tags & releases

Semantic versioning, one minor version per sprint:

| Tag | Milestone |
|---|---|
| `v0.2.0` | Sprint 2 — capacity operating point, SHAP, error analysis |
| `v0.3.0` | Sprint 3 — tests, economics, model bundle (18 features) |
| `v0.4.0` | Sprint 4 — inference core + API |
| ... | one per sprint |
| **`v1.0.0`** | **Sprint 7 — first live deployment** |
| `v1.1.0`+ | Sprints 8-12 |

Model bundles are tagged independently, since they version on a different axis:
`bundle-v1` (18 features, Sprint 3), `bundle-v2` (+ graph features, Sprint 11).
`bundle_meta.json` records the git commit that produced it, so an artifact is
always traceable to source.

GitHub Releases at `v1.0.0` and after, with the live demo URL in the notes.

---

## 8. What never gets committed

Already in `.gitignore`, restated because two of these have bitten this project:

- `data/raw/*.csv` — 493MB source data
- `data/processed/*.duckdb` — 684MB regenerable store
- `models/**` — run artifacts (**but** `model_bundle/` is negated and *is*
  committed, ~7.3MB — it is a release artifact, see ARCHITECTURE §3)
- `mlruns/`, `mlflow.db`, `reports/*.log`, `__pycache__/`, `.venv/`
- `.claude/`, `.env`

**The recurring failure mode in this repo is the opposite one**: files cited by
name in `README.md`/`ROADMAP.md` as evidence that were never actually tracked.
The Sprint 1 audit caught it once; the Sprint 2 review caught a related case.

Run this before merging any sprint PR:

```bash
git status --porcelain --untracked-files=all -- src/ data/processed/ | grep '^??'
```

Every line is a file in a source or evidence directory that is neither tracked
nor deliberately ignored. For each one, decide explicitly: stage it, or add it
to `.gitignore`. Never leave it in limbo — that limbo is exactly how a README
ends up citing evidence the repo does not contain.

*(Verified 2026-08-01: prints the 9 outstanding Sprint 2 files. An earlier
version of this check grepped filenames out of the docs and missed files listed
in README's directory tree — this form checks the actual invariant instead.)*
