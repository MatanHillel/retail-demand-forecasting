"""Run context, structured logging and the staging/promotion mechanism (PRD §37, §39, §40).

Every pipeline step runs inside a :class:`RunContext`. The context records what happened — run id,
dataset hash, configuration snapshot, seed, per-step row counts, durations, warnings, metrics,
champion, errors and artifact paths — and writes it to ``artifacts/run_log.json`` on success *and*
on failure. The Streamlit app reads only that file (plus ``artifacts/validation_report.json`` when
the status is ``failed``), so the schema below is a published interface: extend it, never rename it.

Schema of ``run_log.json``::

    run_id           "20260815T190523Z-3f9a1c"    unique name of one execution
    started_at       ISO-8601 UTC
    finished_at      ISO-8601 UTC | null
    mode             "no-llm" | "llm"
    status           "running" | "success" | "failed"
    seed             int                          from model_config.yaml
    data             {file, sha256, rows, columns}
    config_snapshot  {...}                        from pipeline.config.config_snapshot()
    versions         {python, pandas, numpy, sklearn, crewai, streamlit}
    steps            [{name, status, started_at, duration_s, inputs, outputs,
                       row_counts, warnings}]
    warnings         [str]
    metrics          {...}
    champion         {...} | null
    errors           [{step, type, message, traceback}]
    artifacts        {key: repo-relative path}

Failure handling (PRD §39): artifacts are written through :meth:`RunContext.out`, which redirects
them into ``artifacts/_staging/<run_id>/`` when the context was started with ``staging=True``.
They reach their real locations only when :meth:`RunContext.promote` is called, and promotion is
refused once the run has failed — that is the mechanical guarantee that a failed run never
overwrites good forecasts. ``run_log.json`` and ``validation_report.json`` deliberately bypass
staging: the app must be able to read them precisely *because* the run failed.

This module imports no CrewAI class — library versions are read from package metadata, which does
not import the package — so ``--no-llm`` runs stay free of any LLM import.
"""

from __future__ import annotations

import contextlib
import importlib.metadata
import json
import logging
import os
import platform
import random
import re
import shutil
import time
import traceback
import uuid
from collections.abc import Iterator
from contextvars import ContextVar
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from pipeline import paths
from pipeline.config import config_snapshot, load_model_config

RunMode = Literal["no-llm", "llm"]
RunStatus = Literal["running", "success", "failed"]

#: ``YYYYMMDDTHHMMSSZ`` followed by six hex characters, e.g. ``20260815T190523Z-3f9a1c``.
RUN_ID_PATTERN = re.compile(r"^\d{8}T\d{6}Z-[0-9a-f]{6}$")

LOGGER_NAME = "pipeline"
REDACTED = "***REDACTED***"

# Environment variables whose *values* must never reach a log line or an artifact.
_SECRET_ENV_PATTERN = re.compile(r"(KEY|TOKEN|SECRET|PASSWORD)", re.IGNORECASE)
# Anthropic / OpenAI style keys, in case one is pasted into a message directly.
_SECRET_VALUE_PATTERN = re.compile(r"sk-[A-Za-z0-9_\-]{16,}")
_MIN_SECRET_LENGTH = 8

# Name of the step currently executing, so every log line can be attributed to it.
_CURRENT_STEP: ContextVar[str | None] = ContextVar("current_step", default=None)


def _utcnow() -> str:
    """Current time as an ISO-8601 UTC string — the project's only timestamp format."""
    return datetime.now(UTC).isoformat()


def new_run_id() -> str:
    """Return a fresh run id: a UTC timestamp plus six random hex characters.

    The timestamp makes runs sortable and human-readable; the suffix keeps two runs started in
    the same second apart.
    """
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{uuid.uuid4().hex[:6]}"


# --------------------------------------------------------------------------
# seeding
# --------------------------------------------------------------------------
def set_global_seed(seed: int | None = None) -> int:
    """Seed every source of randomness from the single global seed and return it.

    The value is read from ``model_config.yaml`` when not given — it is never written into code
    (PRD §40). ``PYTHONHASHSEED`` is exported for *child* processes; it cannot change hashing in
    the interpreter that is already running, so nothing in this project may depend on the
    iteration order of a set or a dict built from hashes.
    """
    resolved = load_model_config().seed if seed is None else seed
    random.seed(resolved)
    np.random.seed(resolved)
    os.environ["PYTHONHASHSEED"] = str(resolved)
    return resolved


# --------------------------------------------------------------------------
# logging
# --------------------------------------------------------------------------
def redact(text: str) -> str:
    """Replace anything that looks like a credential with ``***REDACTED***``.

    Two sources are covered: the *values* of environment variables whose name mentions a key,
    token, secret or password, and API keys pasted literally into a message.
    """
    cleaned = _SECRET_VALUE_PATTERN.sub(REDACTED, text)
    for name, value in os.environ.items():
        if len(value) >= _MIN_SECRET_LENGTH and _SECRET_ENV_PATTERN.search(name):
            cleaned = cleaned.replace(value, REDACTED)
    return cleaned


class _RedactingFilter(logging.Filter):
    """Scrubs credentials out of a record before any handler formats it."""

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        cleaned = redact(message)
        if cleaned != message:
            record.msg = cleaned
            record.args = ()
        return True


class _JsonLinesFormatter(logging.Formatter):
    """One JSON object per line: ``time``, ``level``, ``step``, ``message``."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "time": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "step": getattr(record, "step", None) or _CURRENT_STEP.get(),
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def _console_handler() -> logging.Handler:
    handler = logging.StreamHandler()
    handler.set_name(f"{LOGGER_NAME}-console")
    handler.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-7s %(message)s"))
    return handler


def get_logger(run_id: str | None = None, base_dir: Path | None = None) -> logging.Logger:
    """Return the shared ``pipeline`` logger, writing to the console and to a per-run file.

    Safe to call repeatedly: handlers are named and added at most once, so a second call does not
    duplicate output. When ``run_id`` is given, JSON lines are also appended to
    ``logs/run_<run_id>.log``.
    """
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if not any(isinstance(existing, _RedactingFilter) for existing in logger.filters):
        logger.addFilter(_RedactingFilter())

    handler_names = {handler.get_name() for handler in logger.handlers}
    if f"{LOGGER_NAME}-console" not in handler_names:
        logger.addHandler(_console_handler())

    if run_id is not None:
        name = f"{LOGGER_NAME}-run-{run_id}"
        if name not in handler_names:
            root = paths.PROJECT_ROOT if base_dir is None else Path(base_dir)
            log_dir = root / paths.LOGS_DIR.relative_to(paths.PROJECT_ROOT)
            log_dir.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_dir / f"run_{run_id}.log", encoding="utf-8")
            file_handler.set_name(name)
            file_handler.setFormatter(_JsonLinesFormatter())
            logger.addHandler(file_handler)

    return logger


def close_log_handlers(run_id: str) -> None:
    """Detach and close this run's file handler — used by tests and by long-lived processes."""
    logger = logging.getLogger(LOGGER_NAME)
    name = f"{LOGGER_NAME}-run-{run_id}"
    for handler in list(logger.handlers):
        if handler.get_name() == name:
            logger.removeHandler(handler)
            handler.close()


# --------------------------------------------------------------------------
# run records
# --------------------------------------------------------------------------
class RunData(BaseModel):
    """Identity of the input dataset for this run (PRD §40)."""

    model_config = ConfigDict(extra="forbid")

    file: str | None = None
    sha256: str | None = None
    rows: int | None = None
    columns: int | None = None


class StepRecord(BaseModel):
    """What one pipeline step did: how long it took, what it read and wrote, what it dropped."""

    model_config = ConfigDict(extra="forbid")

    name: str
    status: RunStatus = "running"
    started_at: str
    duration_s: float | None = None
    inputs: dict[str, Any] = Field(default_factory=dict)
    outputs: dict[str, Any] = Field(default_factory=dict)
    row_counts: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class RunContext(BaseModel):
    """One execution of the pipeline, serialisable to ``run_log.json`` at any moment."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    run_id: str
    started_at: str
    finished_at: str | None = None
    mode: RunMode
    status: RunStatus = "running"
    seed: int
    data: RunData = Field(default_factory=RunData)
    config_snapshot: dict[str, Any] = Field(default_factory=dict)
    versions: dict[str, str | None] = Field(default_factory=dict)
    steps: list[StepRecord] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)
    champion: dict[str, Any] | None = None
    errors: list[dict[str, Any]] = Field(default_factory=list)
    artifacts: dict[str, str] = Field(default_factory=dict)

    # Everything below is bookkeeping that must never reach run_log.json.
    _base_dir: Path = PrivateAttr(default=paths.PROJECT_ROOT)
    _staging: bool = PrivateAttr(default=False)
    _staged_paths: dict[Path, Path] = PrivateAttr(default_factory=dict)
    _logger: logging.Logger | None = PrivateAttr(default=None)

    # -- construction -------------------------------------------------------
    @classmethod
    def start(
        cls,
        mode: RunMode = "no-llm",
        *,
        staging: bool = False,
        seed: int | None = None,
        base_dir: Path | None = None,
    ) -> RunContext:
        """Begin a run: allocate a run id, seed everything, snapshot the configuration.

        ``base_dir`` relocates ``artifacts/`` and ``logs/`` and exists so tests can run against a
        temporary directory; production callers leave it alone.
        """
        run_id = new_run_id()
        context = cls(
            run_id=run_id,
            started_at=_utcnow(),
            mode=mode,
            seed=set_global_seed(seed),
            config_snapshot=config_snapshot(),
            versions=_package_versions(),
        )
        context._base_dir = paths.PROJECT_ROOT if base_dir is None else Path(base_dir)
        context._staging = staging
        context._logger = get_logger(run_id, context._base_dir)
        context.logger.info(f"run {run_id} started (mode={mode}, staging={staging})")
        return context

    # -- accessors ----------------------------------------------------------
    @property
    def logger(self) -> logging.Logger:
        if self._logger is None:
            self._logger = get_logger(self.run_id, self._base_dir)
        return self._logger

    @property
    def base_dir(self) -> Path:
        return self._base_dir

    @property
    def staging_dir(self) -> Path:
        """Where this run's artifacts are written before they are promoted."""
        return self._rebase(paths.ARTIFACTS_DIR) / "_staging" / self.run_id

    def _rebase(self, path: Path) -> Path:
        """Translate a canonical path from :mod:`pipeline.paths` into this run's base directory."""
        if self._base_dir == paths.PROJECT_ROOT:
            return path
        return self._base_dir / Path(path).relative_to(paths.PROJECT_ROOT)

    # -- steps --------------------------------------------------------------
    @contextlib.contextmanager
    def step(self, name: str, inputs: dict[str, Any] | None = None) -> Iterator[StepRecord]:
        """Run one named step, timing it and capturing any exception into ``errors``.

        The exception is re-raised after being recorded — this context manager observes failures,
        it never swallows them.
        """
        record = StepRecord(name=name, started_at=_utcnow(), inputs=dict(inputs or {}))
        self.steps.append(record)
        token = _CURRENT_STEP.set(name)
        started = time.perf_counter()
        self.logger.info(f"step '{name}' started")
        try:
            yield record
        except Exception as exception:
            record.status = "failed"
            record.duration_s = round(time.perf_counter() - started, 3)
            self.status = "failed"
            self.errors.append(
                {
                    "step": name,
                    "type": type(exception).__name__,
                    "message": redact(str(exception)),
                    "traceback": redact(traceback.format_exc()),
                }
            )
            self.logger.error(f"step '{name}' failed: {exception}")
            raise
        else:
            record.status = "success"
            record.duration_s = round(time.perf_counter() - started, 3)
            self.logger.info(f"step '{name}' finished in {record.duration_s}s")
        finally:
            _CURRENT_STEP.reset(token)

    @property
    def current_step(self) -> StepRecord | None:
        name = _CURRENT_STEP.get()
        if name is None:
            return None
        for record in reversed(self.steps):
            if record.name == name:
                return record
        return None

    def log_rows(self, name: str, before: int, removed: int, after: int) -> None:
        """Record one line of a row-count waterfall (e.g. a cleaning rule) on the active step."""
        record = self.current_step
        if record is None:
            raise RuntimeError("log_rows() must be called inside a ctx.step(...) block")
        record.row_counts[name] = {"before": before, "removed": removed, "after": after}
        self.logger.info(f"{name}: {before} → {after} rows ({removed} removed)")

    def warn(self, message: str) -> None:
        """Record a non-fatal problem on the run and, when inside one, on the current step."""
        cleaned = redact(message)
        self.warnings.append(cleaned)
        record = self.current_step
        if record is not None:
            record.warnings.append(cleaned)
        self.logger.warning(cleaned)

    def record_data(
        self,
        file: str | Path | None = None,
        sha256: str | None = None,
        rows: int | None = None,
        columns: int | None = None,
    ) -> None:
        """Record the identity of the input dataset (PRD §40)."""
        self.data = RunData(
            file=None if file is None else str(file),
            sha256=sha256,
            rows=rows,
            columns=columns,
        )

    def record_metrics(self, metrics: dict[str, Any]) -> None:
        """Merge computed metrics into the run log. Values are computed elsewhere, never here."""
        self.metrics = {**self.metrics, **metrics}

    def record_artifact(self, key: str, path: Path | str) -> None:
        """Record where an artifact will live once promoted, as a repo-relative path."""
        self.artifacts = {**self.artifacts, key: self._relative(Path(path))}

    def _relative(self, path: Path) -> str:
        """Repo-relative POSIX form, so run logs are comparable across machines."""
        try:
            return Path(path).relative_to(self._base_dir).as_posix()
        except ValueError:
            return Path(path).as_posix()

    # -- staging & promotion (PRD §39) --------------------------------------
    def out(self, path: Path | str) -> Path:
        """Return the path an artifact should be *written* to.

        Without staging this is the path itself. With staging it is the same relative location
        under ``artifacts/_staging/<run_id>/``; the real location is only written by
        :meth:`promote`. Either way the parent directory exists when this returns.
        """
        final = Path(path)
        if not final.is_absolute():
            final = self._base_dir / final
        if not self._staging:
            final.parent.mkdir(parents=True, exist_ok=True)
            return final
        try:
            relative = final.relative_to(self._base_dir)
        except ValueError as error:
            raise ValueError(
                f"cannot stage {final}: artifact paths must live under {self._base_dir}"
            ) from error
        staged = self.staging_dir / relative
        staged.parent.mkdir(parents=True, exist_ok=True)
        self._staged_paths[staged] = final
        return staged

    def promote(self) -> list[Path]:
        """Move every staged file to its final location and return the promoted paths.

        Per file: copy into a temporary name beside the destination, then ``os.replace`` onto the
        destination — an atomic rename, so a reader never sees a half-written artifact. Refused
        once the run has failed (PRD §39: failed runs never overwrite good results).
        """
        if self.status == "failed":
            raise RuntimeError("refusing to promote artifacts from a failed run (PRD §39)")
        if not self._staging:
            return []
        promoted: list[Path] = []
        for staged, final in sorted(self._staged_paths.items()):
            if not staged.exists():
                self.warn(f"staged artifact was never written: {self._relative(final)}")
                continue
            final.parent.mkdir(parents=True, exist_ok=True)
            temporary = final.with_name(f"{final.name}.tmp-{self.run_id}")
            shutil.copy2(staged, temporary)
            os.replace(temporary, final)
            staged.unlink()
            promoted.append(final)
        self.logger.info(f"promoted {len(promoted)} artifact(s) from staging")
        return promoted

    def discard_staging(self) -> None:
        """Delete this run's staging directory. Optional — staged files are git-ignored."""
        shutil.rmtree(self.staging_dir, ignore_errors=True)
        self._staged_paths.clear()

    # -- finishing ----------------------------------------------------------
    def finish(self, status: RunStatus = "success") -> Path:
        """Stamp the end of the run and write the run log. A failed run stays failed."""
        if self.status != "failed":
            self.status = status
        self.finished_at = _utcnow()
        return self.write_run_log()

    def write_run_log(self, path: Path | None = None, archive_dir: Path | None = None) -> Path:
        """Write ``artifacts/run_log.json`` plus an archive copy under ``logs/``.

        Called on success *and* on failure — the app decides what to show from ``status``. This
        file bypasses staging on purpose: a failed run must still report that it failed.
        """
        destination = self._rebase(paths.RUN_LOG) if path is None else Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(
            self.model_dump(mode="json"), indent=2, sort_keys=True, ensure_ascii=False
        )
        destination.write_text(payload + "\n", encoding="utf-8")

        archive_root = self._rebase(paths.LOGS_DIR) if archive_dir is None else Path(archive_dir)
        archive_root.mkdir(parents=True, exist_ok=True)
        (archive_root / f"run_{self.run_id}.json").write_text(payload + "\n", encoding="utf-8")
        return destination


def _package_versions() -> dict[str, str | None]:
    """Versions of Python and the key libraries (PRD §40).

    Read from installed package *metadata*, which does not import the package — so a ``--no-llm``
    run records the CrewAI version without ever importing CrewAI.
    """
    distributions = {
        "pandas": "pandas",
        "numpy": "numpy",
        "sklearn": "scikit-learn",
        "crewai": "crewai",
        "streamlit": "streamlit",
    }
    versions: dict[str, str | None] = {"python": platform.python_version()}
    for label, distribution in distributions.items():
        try:
            versions[label] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[label] = None
    return versions
