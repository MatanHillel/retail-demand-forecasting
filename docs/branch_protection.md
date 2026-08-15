# Protecting `main` (manual GitHub setup)

Branch protection is a GitHub *repository setting*, not a file in the repository, so it cannot be
created by code. The repository owner applies it once, by hand, after the first push. PRD §42, §54.

## What we require

| Rule | Value | Why |
|---|---|---|
| Pull request before merging | required | every change is reviewed (PRD §42) |
| Required approvals | **1** | at least one reviewer per PR |
| Required status check | **`ci`** | the milestone cannot end without a green CI run (PRD §54) |
| Branches up to date before merging | required | the check runs against the code that will land |
| Force pushes | blocked | history stays auditable |
| Deletions | blocked | `main` cannot be removed |

## Steps (GitHub web UI)

1. Push the repository to GitHub and merge or push this bootstrap commit so that the workflow
   `.github/workflows/ci.yml` has run **at least once** — a status check only becomes selectable
   after GitHub has seen it report.
2. Open the repository → **Settings** → **Branches** (left sidebar) → **Add branch ruleset**
   (or **Add classic branch protection rule**).
3. Name the ruleset `protect-main`, set **Enforcement status** to **Active**.
4. **Target branches** → *Add target* → **Include default branch** (`main`).
5. Enable **Require a pull request before merging**, and set **Required approvals = 1**.
6. Enable **Require status checks to pass**, tick **Require branches to be up to date before
   merging**, then search for and add the check named **`ci`** (the `jobs.ci` job in
   `.github/workflows/ci.yml`).
7. Enable **Block force pushes** and **Restrict deletions**.
8. Leave *Allow specified actors to bypass* empty — including administrators. If the team needs an
   emergency escape hatch, add the repository owner only, and note the reason in the PR.
9. Click **Create**.

## Verifying it works

```bash
git checkout main
git commit --allow-empty -m "should be rejected"
git push origin main
# expected: "protected branch hook declined" — push directly to main is refused
```

Then open a normal PR from a `feature/US-NN-short-name` branch and confirm that GitHub shows
"Review required" and "1 expected check: ci", and that **Merge** stays disabled until both are
satisfied.

## Same rules via the GitHub CLI (optional)

```bash
gh api -X PUT repos/:owner/:repo/branches/main/protection \
  -H "Accept: application/vnd.github+json" \
  -F "required_status_checks[strict]=true" \
  -F "required_status_checks[contexts][]=ci" \
  -F "enforce_admins=true" \
  -F "required_pull_request_reviews[required_approving_review_count]=1" \
  -F "restrictions=null" \
  -F "allow_force_pushes=false" \
  -F "allow_deletions=false"
```
