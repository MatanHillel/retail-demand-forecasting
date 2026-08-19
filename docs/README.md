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
| [`presentation.pptx`](presentation.pptx) | The 10–12 slide business presentation (§53.2). **Generated, never hand-edited** — `python scripts/build_presentation.py` rebuilds it from the artifacts, so an edit made in PowerPoint is lost on the next run and, worse, is a number nothing computed. |
| [`presentation_notes.md`](presentation_notes.md) | The speaker notes for that deck, one section per slide, plus which run the numbers came from and which input tables were carried forward from an earlier run. |

Both presentation files are written by `scripts/build_presentation.py`, which reads `artifacts/`,
`config/` and this directory's `PRD.md`; the screenshots on slide 10 are captured from the running
Streamlit app by `scripts/capture_screens.py` into `img/screens/`, and the four figures the deck
draws for itself land in `img/deck/`.

## Not yet part of this repository

One deliverable named in the PRD's course-deliverables section (§53) is produced by a later user
story and is not linked above because the file does not exist yet:

* `docs/demo_script.md` — the ≤ 5-minute demo recording script (US-39).

This index will link it once that issue lands.
