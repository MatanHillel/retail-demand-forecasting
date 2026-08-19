# Documentation index

Start at the repository [`README.md`](../README.md). This directory holds everything longer than a
README section.

| Document | What it covers |
|---|---|
| [`PRD.md`](PRD.md) | The specification of record (PRD v1.3) — every `§NN` reference elsewhere in this repository points here. If any other document disagrees with it, the PRD wins. |
| [`interfaces.md`](interfaces.md) | The generated, authoritative surface of `pipeline.paths`, `pipeline.config`, `pipeline.run_context`, `pipeline.validation` and every foundational module built since — signatures, published file schemas, and the usage rules (`ctx.out`, staging, `run_log.json`) every step must obey. |
| [`flow.md`](flow.md) | The ten-step CrewAI Flow end to end: the step diagram, `FlowState`, the staging lifecycle (§39), the failure-handling path, and LLM mode (the two crew kickoffs, the numeric guard, cost/caching). |
| [`reproducibility.md`](reproducibility.md) | What determinism means here, how to reproduce a run, exactly what is (and is not) guaranteed byte-identical between two runs, artifact checksums, and running two copies side by side with `--out-root`. |
| [`contributing.md`](contributing.md) | Branch naming, the PR workflow, `make ci-local`, what CI checks and why, and things that fail review. |
| [`branch_protection.md`](branch_protection.md) | The GitHub branch-protection settings applied to `main` (a manual, one-time repository setting, not a file CI can enforce). |

## Not yet part of this repository

Two deliverables named in the PRD's course-deliverables section (§53) are produced by later user
stories and are not linked above because the files do not exist yet:

* `docs/demo_script.md` — the ≤ 5-minute demo recording script (US-39).
* `docs/presentation.pptx` — the 10–12 slide business presentation (US-38).

This index will link them once those issues land.
