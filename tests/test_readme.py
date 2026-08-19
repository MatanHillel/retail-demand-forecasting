"""The README is what a new team member, reviewer or course grader reads first (US-36, PRD §41-42,
§48-49). This file proves the parts of it that would otherwise silently rot:

* the eight required artifact names are present — read from `pipeline.paths.REQUIRED_ARTIFACTS`
  itself, never re-typed here, so the two can never drift apart (docs/interfaces.md, "Interface
  corrections");
* the two commands a reader will actually copy-paste (`python -m pipeline --no-llm`,
  `streamlit run src/app/Home.py`) are present verbatim;
* the dataset attribution (`CC BY 4.0`, `Chen`) and the §7 wording (`Recommended Target Inventory`)
  are present;
* no relative link in README.md or docs/README.md points at a file that does not exist;
* no leftover `TODO` marker.
"""

from __future__ import annotations

import re

import pytest

from pipeline import paths

README = paths.PROJECT_ROOT / "README.md"
DOCS_README = paths.DOCS_DIR / "README.md"

#: [text](target) — captures the target. Markdown link syntax only; not images (handled the same).
LINK_PATTERN = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")


@pytest.fixture(scope="module")
def readme_text() -> str:
    return README.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def docs_readme_text() -> str:
    return DOCS_README.read_text(encoding="utf-8")


def _relative_links(markdown: str) -> list[str]:
    """Every link target's file part — not an external URL, an anchor-only link or a mailto:.

    A same-file heading anchor (`file.md#heading`) is stripped down to `file.md`: only the file's
    existence is asserted here, not that the heading anchor itself resolves.
    """
    links = []
    for target in LINK_PATTERN.findall(markdown):
        target = target.split(" ", 1)[0].strip()  # drop an optional "title" after the URL
        if target.startswith(("http://", "https://", "mailto:", "#")):
            continue
        links.append(target.split("#", 1)[0])
    return links


# --------------------------------------------------------------------------
# required artifact names — sourced from pipeline.paths, never re-typed
# --------------------------------------------------------------------------
def test_required_artifacts_are_all_named(readme_text) -> None:
    missing = [p.name for p in paths.REQUIRED_ARTIFACTS if p.name not in readme_text]
    assert not missing, f"README is missing the required artifact name(s): {missing}"


# --------------------------------------------------------------------------
# the commands a reader will copy-paste
# --------------------------------------------------------------------------
def test_the_no_llm_pipeline_command_is_present(readme_text) -> None:
    assert "python -m pipeline --no-llm" in readme_text


def test_the_streamlit_command_is_present(readme_text) -> None:
    assert "streamlit run src/app/Home.py" in readme_text


# --------------------------------------------------------------------------
# attribution and §7 wording
# --------------------------------------------------------------------------
def test_the_dataset_licence_is_stated(readme_text) -> None:
    assert "CC BY 4.0" in readme_text
    assert "Chen" in readme_text


def test_the_output_is_called_recommended_target_inventory(readme_text) -> None:
    assert "Recommended Target Inventory" in readme_text


def test_the_results_block_states_a_run_id(readme_text) -> None:
    assert "run `" in readme_text or "run id" in readme_text.lower()


def test_no_todo_marker_is_left(readme_text) -> None:
    assert "TODO" not in readme_text


# --------------------------------------------------------------------------
# supporting files this issue must create
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "relative",
    [
        "docs/README.md",
        "LICENSE",
        "DATA_LICENSE.md",
        "CITATION.cff",
        "scripts/readme_numbers.py",
    ],
)
def test_required_documentation_files_exist(relative) -> None:
    assert (paths.PROJECT_ROOT / relative).is_file(), f"{relative} is missing"


# --------------------------------------------------------------------------
# every relative link resolves to a real file
# --------------------------------------------------------------------------
def test_every_relative_link_in_readme_resolves(readme_text) -> None:
    broken = [
        link
        for link in _relative_links(readme_text)
        if not (README.parent / link).resolve().exists()
    ]
    assert not broken, f"README.md links to file(s) that do not exist: {broken}"


def test_every_relative_link_in_docs_readme_resolves(docs_readme_text) -> None:
    broken = [
        link
        for link in _relative_links(docs_readme_text)
        if not (DOCS_README.parent / link).resolve().exists()
    ]
    assert not broken, f"docs/README.md links to file(s) that do not exist: {broken}"
