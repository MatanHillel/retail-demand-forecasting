"""Fixtures shared by the whole test suite.

Credential hygiene is the only thing here. ``load_env_file`` (US-33 / :mod:`crews.environment`)
reads a real ``.env`` from the working directory, and the developer machines that run this suite
have one. Without the guard below, a test that clears ``OPENAI_API_KEY`` to prove the
no-credential path still exits cleanly would silently get the developer's real key back and
assert nothing — it would pass locally for the wrong reason and fail only in CI, which has no
``.env``. Tests must not depend on what is on the machine running them.
"""

from __future__ import annotations

import pytest

from crews.environment import ENV_FILE_DISABLE_VARIABLE


@pytest.fixture(autouse=True)
def _never_read_a_real_dotenv(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stop :func:`crews.environment.load_env_file` reading a real ``.env`` during tests."""
    monkeypatch.setenv(ENV_FILE_DISABLE_VARIABLE, "1")
