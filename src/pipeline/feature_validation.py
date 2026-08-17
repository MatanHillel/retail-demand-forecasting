"""Feature validation & leakage tests — proof by code, not by assertion (PRD §16, §17, §37 step 5).

Two deterministic checks stand between ``features.csv`` and the rest of the pipeline:

* :func:`validate_features` — schema and structural rules: the target is present, every configured
  feature exists with no ``NaN``, ``forecast_origin`` is exactly ``target_month - 1``, the active
  rule (§14) is satisfied both ways (no inactive row slipped in, no active row is missing), every
  target lies inside the modelled range and never on the never-scored month, and the primary key is
  unique.
* :func:`leakage_check` — the no-leakage boundary of §16 itself: ``lag_1, lag_2, lag_3,
  rolling_mean_3`` are recomputed directly from ``clean_data.csv`` on a sampled set of rows and
  compared byte-for-byte with the feature values; a sample of products has its future months
  perturbed and :func:`pipeline.features.build_features` — the real production function, not a
  re-derived expectation — is called again, and every feature at the perturbed target month and
  earlier must be unchanged while the target's own outcome (``y``) must move, proving the
  perturbation actually reached the future; and the two calendar columns are recomputed from
  ``target_month`` to confirm they are the only values a feature is allowed to read from month ``t``
  itself.

Both functions are **pure** — they return a :class:`~pipeline.validation.ValidationResult` and touch
neither disk nor a run context (``docs/interfaces.md`` §6 rule 5); :func:`write_feature_validation`
is the separate ``ctx``-taking writer that merges both results into
``artifacts/reports/feature_validation.json``.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pipeline import paths
from pipeline.active import active_mask
from pipeline.config import ModelConfig, load_cleaning_config, load_model_config
from pipeline.contract import read_panel
from pipeline.features import FEATURE_COLUMNS, build_features
from pipeline.run_context import RunContext
from pipeline.validation import (
    FlowValidationError,
    ValidationResult,
    Violation,
    write_validation_report,
)

#: ``step`` recorded on every violation and on the validation report (§37 step 5).
STEP = "feature_validation"

#: Leakage failures use this exact wording (§39); structural failures use ``feature validation
#: failed (<n> violations)`` — built where the message is composed, never hard-coded twice.
LEAKAGE_FAILURE_MESSAGE = "feature uses information after forecast origin"

#: How many offending rows a violation carries as illustration.
_EXAMPLE_LIMIT = 5
#: Lag recomputation carries up to ten examples (§2), more than the other checks.
_LAG_EXAMPLE_LIMIT = 10

#: Every third target month is perturbed per product — part of the check's own definition (like the
#: window lengths in ``pipeline.features``), not a tunable sample size.
_PERMUTATION_TARGET_STRIDE = 3
#: Magnitude of the random perturbation applied to a tampered month — large enough that a genuine
#: change can never coincide with the original value by chance.
_TAMPER_UNITS_RANGE = (1_000, 5_000)
_TAMPER_INVOICE_RANGE = (10, 50)
_TAMPER_PRICE_RANGE = (5.0, 50.0)


def _repo_relative(path: Path) -> Path:
    """Repo-relative form of a canonical ``paths.*`` constant (interfaces.md §6 rule 12)."""
    return path.relative_to(paths.PROJECT_ROOT)


def _as_str_frame(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Copy ``frame`` with ``columns`` forced to plain strings — keys must compare like keys."""
    frame = frame.copy()
    for column in columns:
        frame[column] = frame[column].astype(str)
    return frame


class _CheckBuilder:
    """Accumulates named checks and their violations for one :class:`ValidationResult`."""

    def __init__(self, step: str) -> None:
        self.step = step
        self.checks: list[dict[str, Any]] = []
        self.violations: list[Violation] = []

    def add(
        self,
        name: str,
        passed: bool,
        count: int | None = None,
        examples: list[Any] | None = None,
        message: str | None = None,
    ) -> None:
        examples = examples or []
        self.checks.append(
            {"name": name, "passed": bool(passed), "count": int(count or 0), "examples": examples}
        )
        if not passed:
            self.violations.append(
                Violation(
                    step=self.step,
                    rule=name,
                    message=message or f"{name} failed",
                    count=count,
                    examples=examples or None,
                )
            )

    def result(self, checked_rows: int, extra: dict[str, Any] | None = None) -> ValidationResult:
        extra = dict(extra or {})
        extra["checks"] = self.checks
        return ValidationResult(
            step=self.step,
            passed=all(check["passed"] for check in self.checks),
            violations=self.violations,
            checked_rows=checked_rows,
            extra=extra,
        )


# --------------------------------------------------------------------------
# validate_features — structural rules (§2 items 1-7)
# --------------------------------------------------------------------------
def validate_features(
    features_df: pd.DataFrame, panel_df: pd.DataFrame, cfg: ModelConfig, k: int
) -> ValidationResult:
    """Structural validation of ``features.csv`` against the panel and the configuration.

    Pure — no ``ctx``, no disk, never raises. One check per §2 rule 1-7, each recorded whether it
    passed or failed so the published ``feature_validation.json`` always lists all ten check names.
    """
    cleaning_cfg = load_cleaning_config()
    builder = _CheckBuilder(STEP)
    rows = int(len(features_df))

    # 1. target_present
    if "y" not in features_df.columns:
        builder.add("target_present", False, message="target column y is missing")
    else:
        nan_count = int(features_df["y"].isna().sum())
        builder.add(
            "target_present",
            nan_count == 0,
            count=nan_count,
            message=f"target column y has {nan_count} NaN value(s)" if nan_count else None,
        )

    # 2. required_features_present
    missing_columns = [name for name in cfg.features if name not in features_df.columns]
    builder.add(
        "required_features_present",
        not missing_columns,
        count=len(missing_columns),
        examples=missing_columns,
        message=(
            f"missing required feature column(s): {', '.join(missing_columns)}"
            if missing_columns
            else None
        ),
    )

    # 3. no_nan_in_required
    present_features = [name for name in cfg.features if name in features_df.columns]
    nan_counts = {name: int(features_df[name].isna().sum()) for name in present_features}
    offending = {name: count for name, count in nan_counts.items() if count > 0}
    builder.add(
        "no_nan_in_required",
        not offending,
        count=sum(offending.values()),
        examples=list(offending),
        message=(
            f"NaN found in feature column(s): {', '.join(offending)}" if offending else None
        ),
    )

    # 4. origin_before_target
    if {"forecast_origin", "target_month"} <= set(features_df.columns):
        origins = pd.PeriodIndex(features_df["forecast_origin"].astype(str), freq="M")
        targets = pd.PeriodIndex(features_df["target_month"].astype(str), freq="M")
        offset = np.array(
            [(target - origin).n for target, origin in zip(targets, origins, strict=True)]
        )
        bad_mask = offset != 1
        bad_count = int(bad_mask.sum())
        examples = (
            features_df.loc[bad_mask, ["stock_code", "forecast_origin", "target_month"]]
            .head(_EXAMPLE_LIMIT)
            .to_dict("records")
        )
        builder.add(
            "origin_before_target",
            bad_count == 0,
            count=bad_count,
            examples=examples,
            message=(
                f"{bad_count} row(s) have forecast_origin not equal to target_month - 1"
                if bad_count
                else None
            ),
        )
    else:
        builder.add(
            "origin_before_target", False, message="forecast_origin/target_month column missing"
        )

    # 5. active_rule — recomputed mismatch (§14) and coverage (no missing active pair)
    activity = _as_str_frame(active_mask(panel_df, k), ["stock_code", "month"]).rename(
        columns={"month": "target_month"}
    )
    keyed_features = _as_str_frame(features_df, ["stock_code", "target_month"])
    merged = keyed_features[["stock_code", "target_month"]].merge(
        activity, on=["stock_code", "target_month"], how="left", indicator=True
    )
    is_active = merged["is_active"].astype("boolean").fillna(False).astype(bool)
    mismatch_mask = (merged["_merge"] != "both") | (~is_active)
    mismatch_count = int(mismatch_mask.sum())

    first_target = cfg.split.first_target_month
    last_target = cleaning_cfg.raw.last_full_month
    in_range_active = activity[
        (activity["target_month"] >= first_target)
        & (activity["target_month"] <= last_target)
        & activity["is_active"]
    ]
    key_columns = ["stock_code", "target_month"]
    covered_pairs = set(
        map(tuple, keyed_features[key_columns].itertuples(index=False, name=None))
    )
    missing_pairs = [
        pair
        for pair in map(tuple, in_range_active[key_columns].itertuples(index=False, name=None))
        if pair not in covered_pairs
    ]
    active_bad = mismatch_count + len(missing_pairs)
    active_examples = merged.loc[mismatch_mask, ["stock_code", "target_month"]].head(
        _EXAMPLE_LIMIT
    ).to_dict("records") + [
        {"stock_code": pair[0], "target_month": pair[1]} for pair in missing_pairs[:_EXAMPLE_LIMIT]
    ]
    builder.add(
        "active_rule",
        active_bad == 0,
        count=active_bad,
        examples=active_examples[:_EXAMPLE_LIMIT],
        message=(
            f"{mismatch_count} row(s) fail the active rule and {len(missing_pairs)} active "
            "pair(s) are missing from features"
            if active_bad
            else None
        ),
    )

    # 6. target_range
    never_score = set(cfg.split.never_score)
    target_months = keyed_features["target_month"]
    out_of_range = keyed_features[
        (target_months < first_target)
        | (target_months > last_target)
        | target_months.isin(never_score)
    ]
    builder.add(
        "target_range",
        out_of_range.empty,
        count=int(len(out_of_range)),
        examples=sorted(out_of_range["target_month"].unique())[:_EXAMPLE_LIMIT],
        message=(
            f"{len(out_of_range)} row(s) fall outside {first_target}..{last_target} or in a "
            "never-scored month"
            if not out_of_range.empty
            else None
        ),
    )

    # 7. key_unique
    duplicate_mask = keyed_features.duplicated(subset=["stock_code", "target_month"])
    duplicate_count = int(duplicate_mask.sum())
    builder.add(
        "key_unique",
        duplicate_count == 0,
        count=duplicate_count,
        examples=keyed_features.loc[duplicate_mask, ["stock_code", "target_month"]]
        .head(_EXAMPLE_LIMIT)
        .to_dict("records"),
        message=(
            f"{duplicate_count} duplicate (stock_code, target_month) key(s)"
            if duplicate_count
            else None
        ),
    )

    return builder.result(rows)


# --------------------------------------------------------------------------
# leakage_check — the no-leakage boundary itself (§2 leakage_check)
# --------------------------------------------------------------------------
def _month_value_lookup(panel_df: pd.DataFrame) -> tuple[pd.Series, str | None]:
    """A ``(stock_code, month) -> units_sold`` lookup and the dataset's first observed month.

    Months before that first month were never observed and must be excluded from a window rather
    than treated as zero — the same truncation rule :mod:`pipeline.features` applies (§17, §47).
    """
    if panel_df.empty:
        return pd.Series(dtype="float64"), None
    keyed = _as_str_frame(panel_df, ["stock_code", "month"])
    lookup = keyed.set_index(["stock_code", "month"])["units_sold"]
    dataset_min_month = str(keyed["month"].min())
    return lookup, dataset_min_month


def _recomputed_value(
    lookup: pd.Series, dataset_min_month: str | None, stock_code: str, month: str
) -> float | None:
    """Units at ``month`` for ``stock_code``, or ``None`` if ``month`` predates the dataset."""
    if dataset_min_month is None or month < dataset_min_month:
        return None
    try:
        return float(lookup.loc[(stock_code, month)])
    except KeyError:
        return 0.0  # observed month, product not yet launched — a zero-demand month (§17)


def _lag_recomputation_check(
    builder: _CheckBuilder,
    features_df: pd.DataFrame,
    panel_df: pd.DataFrame,
    seed: int,
    sample_rows: int,
) -> int:
    """Recompute ``lag_1, lag_2, lag_3, rolling_mean_3`` from the panel on a sampled set of rows."""
    sample_size = min(sample_rows, len(features_df))
    sample = (
        features_df.sample(n=sample_size, random_state=seed)
        if sample_size
        else features_df.iloc[:0]
    )
    lookup, dataset_min_month = _month_value_lookup(panel_df)

    mismatches: list[dict[str, Any]] = []
    for row in sample.itertuples():
        stock_code = str(row.stock_code)
        target = str(row.target_month)
        window_values: list[float] = []
        row_mismatches: dict[str, list[float]] = {}
        for offset, column in ((1, "lag_1"), (2, "lag_2"), (3, "lag_3")):
            month = str(pd.Period(target, freq="M") - offset)
            expected = _recomputed_value(lookup, dataset_min_month, stock_code, month)
            if expected is None:
                continue
            window_values.append(expected)
            actual = float(getattr(row, column))
            if round(actual) != round(expected):
                row_mismatches[column] = [expected, actual]
        if window_values:
            expected_mean3 = sum(window_values) / len(window_values)
            actual_mean3 = float(row.rolling_mean_3)
            if abs(actual_mean3 - expected_mean3) > 1e-6:
                row_mismatches["rolling_mean_3"] = [expected_mean3, actual_mean3]
        if row_mismatches:
            mismatches.append(
                {"stock_code": stock_code, "target_month": target, "mismatches": row_mismatches}
            )

    bad = len(mismatches)
    builder.add(
        "lag_recomputation",
        bad == 0,
        count=bad,
        examples=mismatches[:_LAG_EXAMPLE_LIMIT],
        message=(
            f"{bad} sampled row(s) disagree with lags/rolling_mean_3 recomputed from clean_data"
            if bad
            else None
        ),
    )
    return sample_size


def _calendar_only_future_info_check(builder: _CheckBuilder, features_df: pd.DataFrame) -> None:
    """The only columns allowed to read month ``t`` itself: recompute them and compare (§17)."""
    periods = pd.PeriodIndex(features_df["target_month"].astype(str), freq="M")
    bad_mask = (features_df["target_month_of_year"].to_numpy() != periods.month) | (
        features_df["target_quarter"].to_numpy() != periods.quarter
    )
    bad_count = int(bad_mask.sum())
    builder.add(
        "calendar_only_future_info",
        bad_count == 0,
        count=bad_count,
        examples=features_df.loc[bad_mask, ["stock_code", "target_month"]]
        .head(_EXAMPLE_LIMIT)
        .to_dict("records"),
        message=(
            f"{bad_count} row(s) have target_month_of_year/target_quarter that do not match "
            "target_month"
            if bad_count
            else None
        ),
    )


def _permutation_future_months_check(
    builder: _CheckBuilder,
    features_df: pd.DataFrame,
    panel_df: pd.DataFrame,
    cfg: ModelConfig,
    k: int,
    seed: int,
    permutation_products: int,
) -> int:
    """Perturb future months for sampled products and rebuild with the real ``build_features``.

    Every feature at the perturbed target and earlier must be unchanged (§16); ``y`` at the
    perturbed target must change, proving the perturbation actually reached the future. Rebuilds
    against the *whole* tampered panel (not a per-product subset) so the truncated-window arithmetic
    (§17, §47) sees the same month grid the original ``features_df`` was built from.
    """
    keyed_features = _as_str_frame(features_df, ["stock_code", "target_month"])
    products = sorted(keyed_features["stock_code"].unique())
    sample_count = min(permutation_products, len(products))
    rng = random.Random(seed)
    sampled_products = rng.sample(products, sample_count) if sample_count else []

    if not sampled_products:
        builder.add("permutation_future_months", True, count=0)
        return 0

    cleaning_cfg = load_cleaning_config()
    first_target = cfg.split.first_target_month
    last_target = cleaning_cfg.raw.last_full_month

    tampered_panel = _as_str_frame(panel_df, ["stock_code", "month"])
    assignments: dict[str, str] = {}
    targets_by_product: dict[str, list[str]] = {}

    for product in sampled_products:
        product_targets = sorted(
            keyed_features.loc[keyed_features["stock_code"] == product, "target_month"]
        )
        if not product_targets:
            continue
        targets_by_product[product] = product_targets
        candidates = product_targets[::_PERMUTATION_TARGET_STRIDE] or product_targets
        target_t = candidates[len(candidates) // 2]
        assignments[product] = target_t

        mask = (tampered_panel["stock_code"] == product) & (tampered_panel["month"] >= target_t)
        touched = int(mask.sum())
        if touched == 0:
            continue
        units_offset = np.array(
            [rng.randint(*_TAMPER_UNITS_RANGE) for _ in range(touched)], dtype="float64"
        )
        invoice_offset = np.array(
            [rng.randint(*_TAMPER_INVOICE_RANGE) for _ in range(touched)], dtype="float64"
        )
        new_price = np.array(
            [rng.uniform(*_TAMPER_PRICE_RANGE) for _ in range(touched)], dtype="float64"
        )
        tampered_panel.loc[mask, "units_sold"] = (
            tampered_panel.loc[mask, "units_sold"].to_numpy() + units_offset
        )
        tampered_panel.loc[mask, "invoice_count"] = (
            tampered_panel.loc[mask, "invoice_count"].to_numpy() + invoice_offset
        )
        tampered_panel.loc[mask, "avg_unit_price"] = new_price

    rebuilt = build_features(tampered_panel, k, first_target, last_target, cfg, include_target=True)
    rebuilt_indexed = _as_str_frame(rebuilt, ["stock_code", "target_month"]).set_index(
        ["stock_code", "target_month"]
    )
    original_indexed = keyed_features.set_index(["stock_code", "target_month"])

    mismatch_examples: list[dict[str, Any]] = []
    mismatch_count = 0
    sanity_checked = 0
    sanity_failed = 0

    for product, target_t in assignments.items():
        for target in targets_by_product[product]:
            key = (product, target)
            if key not in rebuilt_indexed.index or key not in original_indexed.index:
                continue
            if target <= target_t:
                for column in FEATURE_COLUMNS:
                    expected = original_indexed.loc[key, column]
                    actual = rebuilt_indexed.loc[key, column]
                    if abs(float(expected) - float(actual)) > 1e-6:
                        mismatch_count += 1
                        if len(mismatch_examples) < _LAG_EXAMPLE_LIMIT:
                            mismatch_examples.append(
                                {
                                    "stock_code": product,
                                    "target_month": target,
                                    "feature": column,
                                    "expected": float(expected),
                                    "actual": float(actual),
                                }
                            )
            if target == target_t:
                sanity_checked += 1
                if int(original_indexed.loc[key, "y"]) == int(rebuilt_indexed.loc[key, "y"]):
                    sanity_failed += 1

    bad = mismatch_count + sanity_failed
    builder.add(
        "permutation_future_months",
        bad == 0,
        count=bad,
        examples=mismatch_examples,
        message=(
            f"{mismatch_count} feature value(s) changed after permuting future months and "
            f"{sanity_failed} of {sanity_checked} perturbed target(s) failed the sanity check "
            "(y did not change)"
            if bad
            else None
        ),
    )
    return len(sampled_products)


def leakage_check(
    features_df: pd.DataFrame, panel_df: pd.DataFrame, cfg: ModelConfig, seed: int
) -> ValidationResult:
    """The no-leakage boundary itself: lag recomputation, permutation, calendar-only future info.

    Pure — no ``ctx``, no disk, never raises, never modifies ``features_df``. Sample sizes come from
    ``cfg.validation`` and the seed is the caller's, so a re-run is byte-identical (§40).
    """
    k = cfg.active_rule.k
    builder = _CheckBuilder(STEP)
    rows = int(len(features_df))

    lag_sample_size = _lag_recomputation_check(
        builder, features_df, panel_df, seed, cfg.validation.lag_sample_rows
    )
    permutation_sample_size = _permutation_future_months_check(
        builder, features_df, panel_df, cfg, k, seed, cfg.validation.permutation_products
    )
    _calendar_only_future_info_check(builder, features_df)

    return builder.result(
        rows,
        extra={
            "sample_sizes": {
                "lag_sample_rows": lag_sample_size,
                "permutation_products": permutation_sample_size,
            },
            "seed": seed,
        },
    )


# --------------------------------------------------------------------------
# writer (``docs/interfaces.md`` §6 rule 5)
# --------------------------------------------------------------------------
def write_feature_validation(results: list[ValidationResult], ctx: RunContext) -> Path:
    """Merge both results into ``artifacts/reports/feature_validation.json`` and write it.

    Goes through ``ctx.out()`` — an ordinary run artifact, unlike ``validation_report.json``
    (``docs/interfaces.md`` §8 sweep of this issue).
    """
    destination = ctx.out(_repo_relative(paths.FEATURE_VALIDATION))
    checks: list[dict[str, Any]] = []
    sample_sizes: dict[str, Any] = {}
    seed: int | None = None
    for result in results:
        checks.extend(result.extra.get("checks", []))
        sample_sizes.update(result.extra.get("sample_sizes", {}))
        if "seed" in result.extra:
            seed = result.extra["seed"]

    payload = {
        "passed": all(result.passed for result in results),
        "run_id": ctx.run_id,
        "timestamp": datetime.now(UTC).isoformat(),
        "checks": checks,
        "sample_sizes": sample_sizes,
        "seed": seed,
    }
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    ctx.record_artifact("feature_validation", _repo_relative(paths.FEATURE_VALIDATION))
    return destination


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def run(argv: list[str] | None = None) -> int:
    """Standalone entry point. Exit codes: 0 success, 2 validation stop, 1 crash."""
    parser = argparse.ArgumentParser(
        prog="python -m pipeline.feature_validation",
        description="Validate data/processed/features.csv for schema and leakage (PRD §16, §37)",
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=paths.FEATURES,
        help="features file to validate (default: the committed data/processed/features.csv)",
    )
    parser.add_argument(
        "--clean-data",
        type=Path,
        default=paths.CLEAN_DATA,
        help="panel to recompute against (default: the committed data/processed/clean_data.csv)",
    )
    args = parser.parse_args(argv)

    ctx = RunContext.start(mode="no-llm")
    try:
        with ctx.step(STEP):
            for required in (args.clean_data, args.features):
                if not required.is_file():
                    raise FileNotFoundError(f"{required} not found")
            model_cfg = load_model_config()
            panel = read_panel(args.clean_data)
            features_df = pd.read_csv(
                args.features,
                dtype={
                    "stock_code": "string",
                    "forecast_origin": "string",
                    "target_month": "string",
                },
            )
            k = model_cfg.active_rule.k
            feature_result = validate_features(features_df, panel, model_cfg, k)
            leakage_result = leakage_check(features_df, panel, model_cfg, model_cfg.seed)
            write_feature_validation([feature_result, leakage_result], ctx)

            combined = ValidationResult(
                step=STEP,
                passed=feature_result.passed and leakage_result.passed,
                violations=feature_result.violations + leakage_result.violations,
                checked_rows=int(len(features_df)),
            )
            if not combined.passed:
                message = (
                    LEAKAGE_FAILURE_MESSAGE
                    if not leakage_result.passed
                    else f"feature validation failed ({len(combined.violations)} violations)"
                )
                write_validation_report(combined, run_id=ctx.run_id)
                raise FlowValidationError(combined, message)
    except FlowValidationError as error:
        ctx.finish(status="failed")
        print(str(error), file=sys.stderr)
        return 2
    except Exception:
        ctx.finish(status="failed")
        raise
    ctx.finish()
    print("feature validation: PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
