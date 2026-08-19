# Protecting `main` (manual GitHub setup)

Branch protection is a GitHub *repository setting*, not a file in the repository, so it cannot be
created by code. The repository owner applies it once, by hand, after the first push. PRD §42, §54.

## What we require

| Rule | Value | Why |
|---|---|---|
| Pull request before merging | required | every change is reviewed (PRD §42) |
| Required approvals | **1** | at least one reviewer per PR |
| Required status checks | **`lint-test`, `pipeline-no-llm`, `failure-path`, `determinism`** | the milestone cannot end without a green CI run (PRD §54) |
| Branches up to date before merging | required | the checks run against the code that will land |
| Force pushes | blocked | history stays auditable |
| Deletions | blocked | `main` cannot be removed |

### The four required checks (US-35)

Each is one job in `.github/workflows/ci.yml`, and the check name GitHub shows is the job's name.
They are separate checks on purpose: when `main` goes red, the name alone says what broke.

| Check | What it proves | Typical time |
|---|---|---|
| `lint-test` | `ruff` is clean, dependencies are consistent (`pip check`), and the unit suite passes (`-m "not slow"`) | ~8 min |
| `pipeline-no-llm` | the real pipeline still runs end to end on the committed sample and produces all eight required artifacts | ~5 min |
| `failure-path` | a deliberately broken input stops **gracefully**: exit code 2, `FLOW STOPPED: …`, and the published artifacts untouched (§39) | ~3 min |
| `determinism` | the same pipeline run twice produces byte-identical numbers (§40), plus US-34's reproducibility suites | ~10 min |

The three heavy checks `needs: lint-test`, so they start together once the fast gate is green; the
whole workflow finishes well inside the 40-minute budget the issue sets.

**A status check only becomes selectable in the GitHub UI after GitHub has seen it report at
least once.** So merge (or push) a branch running this workflow before trying to add the four
names below — they will not appear in the search box until then. Renaming a job renames its check
and silently drops the protection: `tests/test_ci_workflow.py` asserts the four names in the
workflow match the four listed here, so a rename fails the build instead of quietly weakening
`main`.

## Steps (GitHub web UI)

1. Push the repository to GitHub and merge or push a commit so that the workflow
   `.github/workflows/ci.yml` has run **at least once** with all four jobs.
2. Open the repository → **Settings** → **Branches** (left sidebar) → **Add branch ruleset**
   (or **Add classic branch protection rule**).
3. Name the ruleset `protect-main`, set **Enforcement status** to **Active**.
4. **Target branches** → *Add target* → **Include default branch** (`main`).
5. Enable **Require a pull request before merging**, and set **Required approvals = 1**.
6. Enable **Require status checks to pass**, tick **Require branches to be up to date before
   merging**, then search for and add all four checks:
   **`lint-test`**, **`pipeline-no-llm`**, **`failure-path`**, **`determinism`**.
7. Enable **Block force pushes** and **Restrict deletions**.
8. Leave *Allow specified actors to bypass* empty — including administrators. If the team needs an
   emergency escape hatch, add the repository owner only, and note the reason in the PR.
9. Click **Create**.

## Verifying it works

```bash
git checkout main
git commit --allow-empty -m "should be rejected"
git push origin main
# expected: "protected branch hook declined" — pushing directly to main is refused
```

Then open a normal PR from a `feature/US-NN-short-name` branch and confirm that GitHub shows
"Review required" and **4 expected checks**, and that **Merge** stays disabled until all of them
are satisfied.

To confirm the setting is actually live (and to capture it for the record — PRD §49 asks for the
settings export in `docs/`):

```bash
gh api repos/:owner/:repo/branches/main/protection \
  --jq '{checks: .required_status_checks.contexts,
         strict: .required_status_checks.strict,
         reviews: .required_pull_request_reviews.required_approving_review_count,
         admins: .enforce_admins.enabled}'
```

A `404 Branch not protected` means the ruleset was never applied — the merge button will happily
merge an unreviewed, untested branch until it is.

## Same rules via the GitHub CLI (optional)

```bash
gh api -X PUT repos/:owner/:repo/branches/main/protection \
  -H "Accept: application/vnd.github+json" \
  -F "required_status_checks[strict]=true" \
  -F "required_status_checks[contexts][]=lint-test" \
  -F "required_status_checks[contexts][]=pipeline-no-llm" \
  -F "required_status_checks[contexts][]=failure-path" \
  -F "required_status_checks[contexts][]=determinism" \
  -F "enforce_admins=true" \
  -F "required_pull_request_reviews[required_approving_review_count]=1" \
  -F "restrictions=null" \
  -F "allow_force_pushes=false" \
  -F "allow_deletions=false"
```
