# What

<!-- What does this PR change? Reference the Linear issue, e.g. AI-5 / US-00. -->

# Why

<!-- Why is this needed? Reference the PRD section(s) it implements, e.g. PRD §41, §42. -->

# How tested

<!-- Commands run and their result, e.g. `pytest -q`, `ruff check src tests`,
     `python -m pipeline --no-llm --sample --skip-tuning`. Paste the relevant output. -->

# Checklist

- [ ] `pytest -q` passes locally
- [ ] `ruff check src tests` is clean
- [ ] CI (`ci`) is green on this PR
- [ ] No raw data committed (`git ls-files data/raw` prints only `data/raw/.gitkeep`) — PRD §42
- [ ] No new hard-coded numbers; thresholds/dates/seeds live in `config/*.yaml` — PRD §14, §40
- [ ] PRD section(s) referenced above; `CLAUDE.md` updated if it disagrees with the PRD
- [ ] "Report for review" (plain-language explanation) written for the issue
- [ ] At least one reviewer requested
