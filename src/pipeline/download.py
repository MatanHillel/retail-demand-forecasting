"""Raw-data acquisition: download, SHA-256 integrity, loading and the CI sample (US-03).

Dataset: *Online Retail II*, UCI Machine Learning Repository (Chen, 2019, CC BY 4.0) — the
canonical citation string is the mandatory ``citation`` key of ``config/data_sources.yaml``
(PRD §8, §48) and is printed by the CLI at intake. Raw data is never committed (PRD §42):
``data/raw/*`` is git-ignored, this script downloads the file and verifies its SHA-256 hash.

CLI (PRD §37 step 1, "Dataset intake")::

    python -m pipeline.download [--source uci|kaggle] [--record-hash] [--make-sample] [--force]

Exit codes: 0 success, 2 validation failure (graceful stop), 1 unexpected exception.

Deterministic-step contract (``docs/interfaces.md``): the check functions here
(:func:`verify_hash`, :func:`validate_raw_schema`) are pure — they return a
:class:`~pipeline.validation.ValidationResult` and never raise or touch disk; the CLI writes
``validation_report.json`` and raises :class:`~pipeline.validation.FlowValidationError`. The
raw file, its parquet cache and the sample fixture are *inputs*, not run artifacts, so they are
deliberately written to their real locations rather than through ``ctx.out()`` staging.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from pipeline import paths
from pipeline.config import (
    CleaningConfig,
    DataSources,
    clear_config_cache,
    load_cleaning_config,
    load_data_sources,
    load_model_config,
    load_non_inventory_codes,
)
from pipeline.run_context import RunContext, get_logger
from pipeline.validation import (
    FlowValidationError,
    ValidationResult,
    Violation,
    write_validation_report,
)

# Construction targets of the CI sample fixture (issue US-03 §2). These describe the fixture,
# not any pipeline rule, and have no home in config/*.yaml — the US-01 models reject extra keys
# (see the drift-check comment on AI-8). Tests import these instead of repeating the numbers.
# The issue's indicative 300 codes produce ~6 MB on the real data, breaching the ≤ 5 MB
# acceptance budget, which wins — reported on AI-8.
SAMPLE_STOCK_CODES = 210
SAMPLE_MIN_CANCELLATION_ROWS = 200
SAMPLE_MIN_BAD_PRICE_ROWS = 100
SAMPLE_MIN_BAD_QUANTITY_ROWS = 100
SAMPLE_MIN_DUPLICATE_ROWS = 100
SAMPLE_MAX_BYTES = 5_000_000  # "target size ≤ 5 MB" (strict SI reading, safe either way)

#: Cached fast copy of the slow-to-read xlsx, keyed by the xlsx SHA-256 (git-ignored).
PARQUET_CACHE: Path = paths.RAW_DIR / "online_retail_II.parquet"

_CHUNK_BYTES = 1 << 20
_EXAMPLE_LIMIT = 5
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

#: PRD §39 graceful-stop message templates for this module's two checks — re-exported by
#: ``flow.failure`` so every §39 wording lives in exactly one place.
MISSING_COLUMN = "Missing required column {column}"
RAW_HASH_MISMATCH = "raw data hash mismatch (expected {expected}, got {actual})"

# Kaggle mirrors occasionally rename columns; normalise them back to the UCI names.
_KAGGLE_RENAMES = {
    "InvoiceNo": "Invoice",
    "CustomerID": "Customer ID",
    "Customer_ID": "Customer ID",
    "UnitPrice": "Price",
}


# --------------------------------------------------------------------------
# hashing
# --------------------------------------------------------------------------
def compute_sha256(path: Path | str) -> str:
    """Return the SHA-256 hex digest of a file, reading it in chunks."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def verify_hash(path: Path | str, expected: str | None = None) -> ValidationResult:
    """Compare a file's SHA-256 with the expected digest — pure, never raises.

    ``expected`` defaults to ``data_sources.yaml → expected_sha256``. When no digest has been
    recorded yet the check passes with a note; the CLI warns and suggests ``--record-hash``.
    """
    if expected is None:
        expected = load_data_sources().expected_sha256
    actual = compute_sha256(path)
    extra = {"file": str(path), "expected": expected, "actual": actual}
    if expected is None:
        extra["note"] = "no expected_sha256 recorded yet"
        return ValidationResult(step="raw_hash", passed=True, extra=extra)
    if actual == expected:
        return ValidationResult(step="raw_hash", passed=True, extra=extra)
    violation = Violation(
        step="raw_hash",
        rule="expected_sha256",
        message=RAW_HASH_MISMATCH.format(expected=expected, actual=actual),
    )
    return ValidationResult(step="raw_hash", passed=False, violations=[violation], extra=extra)


def record_expected_hash(digest: str, path: Path | None = None) -> None:
    """Write the computed digest into ``data_sources.yaml`` — the only allowed write to config.

    The YAML text is rewritten line-wise (the loaded model is frozen) and the config cache is
    cleared so a re-read in the same process sees the new value. Note the current run's
    ``config_snapshot`` was taken at start and still shows the old value — expected behaviour.
    """
    target = paths.DATA_SOURCES if path is None else Path(path)
    text = target.read_text(encoding="utf-8")
    updated, count = re.subn(
        r"(?m)^expected_sha256:.*$", f'expected_sha256: "{digest}"', text
    )
    if count != 1:
        raise ValueError(f"expected exactly one expected_sha256 line in {target}, found {count}")
    target.write_text(updated, encoding="utf-8")
    clear_config_cache()


# --------------------------------------------------------------------------
# download
# --------------------------------------------------------------------------
def download_raw(
    source: str | None = None, *, force: bool = False, ctx: RunContext | None = None
) -> Path:
    """Ensure the raw file exists under ``data/raw/`` and return its path.

    An existing file is never overwritten unless ``force`` is set. On any download problem a
    ``RuntimeError`` with manual-download instructions (URL, filename, expected hash) is raised.
    """
    cfg = load_data_sources()
    resolved = cfg.preferred if source is None else source
    logger = ctx.logger if ctx is not None else get_logger()
    if resolved == "uci":
        destination = paths.RAW_DIR / cfg.uci.archive_member
    elif resolved == "kaggle":
        destination = paths.RAW_DIR / cfg.kaggle.file
    else:
        raise ValueError(f"unknown source {resolved!r}; expected 'uci' or 'kaggle'")
    if destination.is_file() and not force:
        logger.info(f"raw file already present, not re-downloading: {destination}")
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    if resolved == "uci":
        _download_uci(cfg, destination, logger)
    else:
        _download_kaggle(cfg, destination, logger)
    return destination


def _manual_instructions(cfg: DataSources, destination: Path, reason: object) -> str:
    expected = cfg.expected_sha256 or "not recorded yet (run --record-hash after downloading)"
    return (
        f"automatic download failed ({reason}). Download the archive manually from "
        f"{cfg.uci.url} (or the Kaggle dataset {cfg.kaggle.dataset}), extract "
        f"{cfg.uci.archive_member!r} and place it at {destination}. "
        f"Expected SHA-256: {expected}"
    )


def _download_uci(cfg: DataSources, destination: Path, logger: logging.Logger) -> None:
    archive = destination.with_name(destination.name + ".zip.tmp")
    logger.info(f"downloading {cfg.uci.url}")
    try:
        with requests.get(cfg.uci.url, stream=True, timeout=120) as response:
            response.raise_for_status()
            with archive.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=_CHUNK_BYTES):
                    handle.write(chunk)
        with zipfile.ZipFile(archive) as bundle:
            with bundle.open(cfg.uci.archive_member) as member, destination.open("wb") as out:
                shutil.copyfileobj(member, out)
    except Exception as error:
        destination.unlink(missing_ok=True)
        raise RuntimeError(_manual_instructions(cfg, destination, error)) from error
    finally:
        archive.unlink(missing_ok=True)
    logger.info(f"downloaded and extracted {destination.name}")


def _download_kaggle(cfg: DataSources, destination: Path, logger: logging.Logger) -> None:
    has_credentials = (Path.home() / ".kaggle" / "kaggle.json").is_file() or (
        os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")
    )
    instructions = (
        f"Kaggle credentials not found or the Kaggle CLI is unavailable. Either configure an "
        f"API token (kaggle.com → Settings → Create New Token) and re-run, or download "
        f"{cfg.kaggle.file!r} manually from https://www.kaggle.com/datasets/{cfg.kaggle.dataset} "
        f"and place it at {destination}."
    )
    if not has_credentials:
        raise RuntimeError(instructions)
    command = [
        "kaggle",
        "datasets",
        "download",
        "-d",
        cfg.kaggle.dataset,
        "-p",
        str(destination.parent),
        "--unzip",
    ]
    logger.info(f"running: {' '.join(command)}")
    try:
        subprocess.run(command, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        raise RuntimeError(instructions) from error
    if not destination.is_file():
        raise RuntimeError(instructions)
    logger.info(f"downloaded {destination.name}")


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------
def load_raw(
    path: Path | str | None = None, *, ctx: RunContext | None = None
) -> tuple[pd.DataFrame, dict]:
    """Load the raw extract (UCI xlsx sheets or a single CSV) into one DataFrame.

    Returns ``(df, meta)`` with ``meta = {file, sha256, rows, columns, rows_per_sheet}``
    (``columns`` is the column list — §10 step 1). When a run context is given, the dataset
    identity is recorded via ``ctx.record_data`` and the shape details go to the step outputs.
    """
    cfg = load_data_sources()
    logger = ctx.logger if ctx is not None else get_logger()
    resolved = _resolve_raw_path(cfg) if path is None else Path(path)
    if not resolved.is_file():
        raise FileNotFoundError(
            f"raw data file not found: {resolved}. Run `python -m pipeline.download` first."
        )
    digest = compute_sha256(resolved)
    if resolved.suffix.lower() == ".xlsx":
        frame = _load_xlsx(resolved, digest, cfg, logger)
    else:
        frame = _load_csv(resolved, logger)
    frame = _canonical_column_order(frame)
    rows_per_sheet = {
        str(sheet): int(count)
        for sheet, count in frame["source_sheet"].value_counts().sort_index().items()
    }
    meta = {
        "file": str(resolved),
        "sha256": digest,
        "rows": int(len(frame)),
        "columns": list(frame.columns),
        "rows_per_sheet": rows_per_sheet,
    }
    logger.info(
        f"raw data loaded: {meta['rows']} rows × {len(meta['columns'])} columns "
        f"from {resolved.name}; rows per sheet: {rows_per_sheet}"
    )
    if ctx is not None:
        ctx.record_data(file=resolved, sha256=digest, rows=len(frame), columns=frame.shape[1])
        step = ctx.current_step
        if step is not None:
            step.outputs["raw_columns"] = list(frame.columns)
            step.outputs["raw_rows_per_sheet"] = rows_per_sheet
    return frame, meta


def _resolve_raw_path(cfg: DataSources) -> Path:
    uci_path = paths.RAW_DIR / cfg.uci.archive_member
    kaggle_path = paths.RAW_DIR / cfg.kaggle.file
    if uci_path.is_file():
        return uci_path
    if kaggle_path.is_file():
        return kaggle_path
    return paths.RAW_DIR / cfg.uci.archive_member if cfg.preferred == "uci" else kaggle_path


def _load_xlsx(
    path: Path, digest: str, cfg: DataSources, logger: logging.Logger
) -> pd.DataFrame:
    marker = PARQUET_CACHE.with_name(PARQUET_CACHE.name + ".sha256")
    if (
        PARQUET_CACHE.is_file()
        and marker.is_file()
        and marker.read_text(encoding="utf-8").strip() == digest
    ):
        logger.info(f"reading cached parquet copy {PARQUET_CACHE.name}")
        return pd.read_parquet(PARQUET_CACHE)
    sheets = []
    for sheet in cfg.uci.sheets:
        logger.info(f"reading sheet {sheet!r} (slow — a parquet copy is cached afterwards)")
        # Invoice/StockCode carry letter suffixes and Description holds occasional numeric
        # cells; all three must be text or the parquet conversion rejects the mixed column.
        part = pd.read_excel(
            path,
            sheet_name=sheet,
            dtype={"Invoice": str, "StockCode": str, "Description": str},
        )
        part["source_sheet"] = sheet
        sheets.append(part)
    frame = pd.concat(sheets, ignore_index=True)
    frame.to_parquet(PARQUET_CACHE, index=False)
    marker.write_text(digest + "\n", encoding="utf-8")
    logger.info(f"cached parquet copy at {PARQUET_CACHE.name} (keyed by sha256)")
    return frame


def _load_csv(path: Path, logger: logging.Logger) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"Invoice": str, "StockCode": str})
    frame = frame.drop(columns=[c for c in frame.columns if c.startswith("Unnamed:")])
    renames = {old: new for old, new in _KAGGLE_RENAMES.items() if old in frame.columns}
    if renames:
        logger.info(f"renaming CSV columns to the UCI names: {renames}")
        frame = frame.rename(columns=renames)
    if "InvoiceDate" in frame.columns:
        # Explicit ISO format, never day-first inference (issue §3); failures become NaT + log.
        parsed = pd.to_datetime(frame["InvoiceDate"], format="ISO8601", errors="coerce")
        failures = int(parsed.isna().sum()) - int(frame["InvoiceDate"].isna().sum())
        if failures:
            logger.warning(f"{failures} InvoiceDate value(s) failed to parse and became NaT")
        frame["InvoiceDate"] = parsed
    if "source_sheet" not in frame.columns:
        frame["source_sheet"] = path.name
    return frame


def _canonical_column_order(frame: pd.DataFrame) -> pd.DataFrame:
    required = load_cleaning_config().raw.required_columns
    ordered = [column for column in required if column in frame.columns]
    extras = [
        column
        for column in frame.columns
        if column not in ordered and column != "source_sheet"
    ]
    return frame[[*ordered, *extras, "source_sheet"]]


# --------------------------------------------------------------------------
# schema validation (PRD §10 step 2)
# --------------------------------------------------------------------------
def validate_raw_schema(df: pd.DataFrame, cfg: CleaningConfig | None = None) -> ValidationResult:
    """Check the raw schema — pure; the caller writes the report and raises on failure."""
    resolved = load_cleaning_config() if cfg is None else cfg
    step = "raw_schema"
    violations: list[Violation] = []

    for column in resolved.raw.required_columns:
        if column not in df.columns:
            violations.append(
                Violation(
                    step=step,
                    rule="required_columns",
                    message=MISSING_COLUMN.format(column=column),
                )
            )

    for column in ("Quantity", "Price"):
        if column not in df.columns or pd.api.types.is_numeric_dtype(df[column]):
            continue
        coerced = pd.to_numeric(df[column], errors="coerce")
        bad = df[column].notna() & coerced.isna()
        if bad.any():
            violations.append(
                Violation(
                    step=step,
                    rule="numeric",
                    message=f"Column {column} contains non-numeric values",
                    count=int(bad.sum()),
                    examples=df.loc[bad, column].astype(str).head(_EXAMPLE_LIMIT).tolist(),
                )
            )

    if "InvoiceDate" in df.columns and not pd.api.types.is_datetime64_any_dtype(
        df["InvoiceDate"]
    ):
        coerced = pd.to_datetime(df["InvoiceDate"], errors="coerce")
        bad = df["InvoiceDate"].notna() & coerced.isna()
        if bad.any():
            violations.append(
                Violation(
                    step=step,
                    rule="datetime",
                    message="Column InvoiceDate contains values that do not parse as dates",
                    count=int(bad.sum()),
                    examples=df.loc[bad, "InvoiceDate"].astype(str).head(_EXAMPLE_LIMIT).tolist(),
                )
            )

    for column in ("Invoice", "StockCode"):
        if column not in df.columns:
            continue
        nulls = int(df[column].isna().sum())
        if nulls:
            violations.append(
                Violation(
                    step=step,
                    rule="non_null",
                    message=f"Column {column} contains {nulls} null value(s)",
                    count=nulls,
                )
            )

    return ValidationResult(
        step=step, passed=not violations, violations=violations, checked_rows=int(len(df))
    )


# --------------------------------------------------------------------------
# CI sample fixture
# --------------------------------------------------------------------------
def make_sample(
    df: pd.DataFrame,
    ctx: RunContext | None = None,
    out: Path | str | None = None,
    seed: int | None = None,
    *,
    n_stock_codes: int = SAMPLE_STOCK_CODES,
    min_cancellation_rows: int = SAMPLE_MIN_CANCELLATION_ROWS,
    min_bad_price_rows: int = SAMPLE_MIN_BAD_PRICE_ROWS,
    min_bad_quantity_rows: int = SAMPLE_MIN_BAD_QUANTITY_ROWS,
    min_duplicate_rows: int = SAMPLE_MIN_DUPLICATE_ROWS,
    max_bytes: int = SAMPLE_MAX_BYTES,
) -> Path:
    """Write a deterministic raw subsample for CI so every cleaning step removes something.

    Selection: all rows of ``n_stock_codes`` codes drawn with the seeded RNG from a *sorted*
    code list (never from a set — in-process hashing is not seedable), every non-inventory-code
    row, every bad-debt adjustment row, topped up to the minimum counts of cancellations,
    non-positive prices/quantities and exact duplicates, plus at least one row from every month
    present in the input. The output is sorted deterministically before writing.
    """
    cleaning = load_cleaning_config()
    rules = cleaning.rules
    resolved_seed = load_model_config().seed if seed is None else seed
    destination = paths.FIXTURES_DIR / "raw_sample.csv" if out is None else Path(out)
    logger = ctx.logger if ctx is not None else get_logger()

    def warn(message: str) -> None:
        if ctx is not None:
            ctx.warn(message)
        else:
            logger.warning(message)

    frame = df.reset_index(drop=True)
    invoices = frame["Invoice"].astype(str)
    stock_codes = frame["StockCode"].astype(str)
    quantity = pd.to_numeric(frame["Quantity"], errors="coerce")
    price = pd.to_numeric(frame["Price"], errors="coerce")
    selected = pd.Series(False, index=frame.index)

    codes = sorted(stock_codes.dropna().unique())
    rng = np.random.default_rng(resolved_seed)
    if len(codes) > n_stock_codes:
        positions = rng.choice(len(codes), size=n_stock_codes, replace=False)
        chosen = {codes[position] for position in sorted(positions)}
    else:
        chosen = set(codes)
    selected |= stock_codes.isin(chosen)

    non_inventory = set(load_non_inventory_codes()["stock_code"])
    selected |= stock_codes.isin(non_inventory)

    cancel_mask = invoices.str.startswith(tuple(rules.cancellation_prefixes))
    adjust_mask = invoices.str.startswith(tuple(rules.adjustment_prefixes))
    selected |= adjust_mask  # every bad-debt adjustment row (their count is indicative only)

    def top_up(candidates: pd.Series, minimum: int, label: str) -> None:
        nonlocal selected
        have = int((selected & candidates).sum())
        needed = minimum - have
        if needed <= 0:
            return
        pool = frame.index[candidates & ~selected]
        if len(pool) < needed:
            warn(
                f"raw data offers only {have + len(pool)} {label} row(s); "
                f"sample target is {minimum}"
            )
        selected[pool[:needed]] = True

    special = cancel_mask | adjust_mask
    top_up(cancel_mask, min_cancellation_rows, "cancellation")
    # Rows each later cleaning step must remove: non-positive quantities on regular invoices,
    # and non-positive prices that survive the quantity rule (§10 steps 6-7).
    top_up(
        ~special & (quantity <= rules.drop_quantity_at_or_below),
        min_bad_quantity_rows,
        "non-positive-quantity",
    )
    top_up(
        ~special
        & (quantity > rules.drop_quantity_at_or_below)
        & (price <= rules.drop_price_at_or_below),
        min_bad_price_rows,
        "non-positive-price",
    )

    duplicate_columns = [c for c in cleaning.raw.required_columns if c in frame.columns]

    def duplicates_in_sample() -> int:
        return int(frame[selected].duplicated(subset=duplicate_columns, keep="first").sum())

    for _ in range(2):  # second pass corrects the estimate when groups were partially selected
        needed = min_duplicate_rows - duplicates_in_sample()
        if needed <= 0:
            break
        duplicate_mask = frame.duplicated(subset=duplicate_columns, keep=False)
        groups = frame[duplicate_mask].groupby(duplicate_columns, sort=False, dropna=False)
        for indices in groups.groups.values():
            already = int(selected[indices].sum())
            if already == len(indices):
                continue
            selected[indices] = True
            needed -= len(indices) - max(already, 1)
            if needed <= 0:
                break
    if duplicates_in_sample() < min_duplicate_rows:
        warn(
            f"raw data offers only {duplicates_in_sample()} exact-duplicate row(s); "
            f"sample target is {min_duplicate_rows}"
        )

    months = pd.to_datetime(frame["InvoiceDate"], errors="coerce").dt.to_period("M")
    for month in months.dropna().sort_values().unique():
        in_month = months == month
        if not (selected & in_month).any():
            selected[frame.index[in_month][:1]] = True

    sample = frame[selected].copy()
    sort_columns = [c for c in [*cleaning.raw.required_columns, "source_sheet"] if c in sample]
    sample = sample.sort_values(
        by=sort_columns, kind="mergesort", na_position="last"
    ).reset_index(drop=True)

    output = sample.copy()
    if pd.api.types.is_datetime64_any_dtype(output["InvoiceDate"]):
        output["InvoiceDate"] = output["InvoiceDate"].dt.strftime(_DATE_FORMAT)
    destination.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(
        destination, index=False, float_format="%.10g", lineterminator="\n", encoding="utf-8"
    )

    size = destination.stat().st_size
    if size > max_bytes:
        warn(f"sample size {size} bytes exceeds the target of {max_bytes} bytes")
    logger.info(
        f"sample written: {len(sample)} rows, {sample['StockCode'].nunique()} stock codes, "
        f"{size} bytes → {destination}"
    )
    if ctx is not None:
        ctx.record_metrics(
            {
                "sample_rows": int(len(sample)),
                "sample_bytes": int(size),
                "sample_stock_codes": int(sample["StockCode"].nunique()),
            }
        )
    return destination


# --------------------------------------------------------------------------
# CLI (PRD §37 step 1 — dataset intake)
# --------------------------------------------------------------------------
def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m pipeline.download",
        description="Download the Online Retail II raw data, verify its SHA-256 hash, "
        "validate the raw schema and optionally build the CI sample fixture.",
    )
    parser.add_argument("--source", choices=("uci", "kaggle"), default=None)
    parser.add_argument(
        "--record-hash",
        action="store_true",
        help="record the computed SHA-256 in config/data_sources.yaml when it is null",
    )
    parser.add_argument(
        "--make-sample",
        action="store_true",
        help="write the deterministic CI sample to tests/fixtures/raw_sample.csv",
    )
    parser.add_argument(
        "--force", action="store_true", help="re-download even if the raw file exists"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    ctx = RunContext.start(mode="no-llm")
    ctx.write_run_log()  # the download can take minutes; leave an honest status on disk early
    try:
        with ctx.step(
            "dataset_intake",
            inputs={
                "source": args.source,
                "record_hash": args.record_hash,
                "make_sample": args.make_sample,
                "force": args.force,
            },
        ):
            sources = load_data_sources()
            ctx.logger.info(f"dataset: {sources.citation}")
            raw_path = download_raw(args.source, force=args.force, ctx=ctx)
            digest = compute_sha256(raw_path)
            if sources.expected_sha256 is None:
                print(f"SHA-256: {digest}")
                if args.record_hash:
                    record_expected_hash(digest)
                    print(f"recorded expected_sha256 in {paths.DATA_SOURCES.name}")
                else:
                    ctx.warn(
                        "expected_sha256 is null; re-run with --record-hash to record it"
                    )
            else:
                hash_result = verify_hash(raw_path, expected=sources.expected_sha256)
                if not hash_result.passed:
                    write_validation_report(hash_result, run_id=ctx.run_id)
                    raise FlowValidationError(hash_result)
                print(f"hash verified: {digest}")

            frame, meta = load_raw(raw_path, ctx=ctx)
            schema_result = validate_raw_schema(frame)
            write_validation_report(schema_result, run_id=ctx.run_id)
            if not schema_result.passed:
                raise FlowValidationError(schema_result)
            print(f"raw data OK: {meta['rows']} rows, {len(meta['columns'])} columns")

            if args.make_sample:
                sample_path = make_sample(frame, ctx)
                print(
                    f"sample written: {sample_path} "
                    f"({ctx.metrics['sample_rows']} rows, {ctx.metrics['sample_bytes']} bytes)"
                )
    except FlowValidationError as error:
        ctx.finish(status="failed")
        print(str(error), file=sys.stderr)
        return 2
    except Exception:
        ctx.finish(status="failed")
        raise
    ctx.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
