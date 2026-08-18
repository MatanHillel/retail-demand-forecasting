"""Latest operational forecast and the Recommended Target Inventory plan (US-23, PRD §4, §7,
§14-§16, §24-§28, §36, §37 step 8, §41).

This is the step the whole project exists for: for every **active** product, what the champion
expects to sell next month, how uncertain that number is, and how many units are worth having on
the shelf to cover it.

**One origin, one target.** The forecast origin is ``cleaning_config.yaml -> raw.last_full_month``
(2011-11) and the target is the month after it (2011-12). December 2011 is a *partial* month — the
extract stops on 9 December — so it is never scored (§8, §21) and its own sales must never reach a
feature. That boundary is not assumed here, it is *checked*:
:func:`validate_operational_inputs` rebuilds the operational features from a panel whose partial
month has been deliberately corrupted and requires the result to be byte-identical, which is only
possible if nothing from month ``t`` was read (issue §8: a ``ValidationResult``, never a bare
``assert`` — an assert vanishes under ``python -O`` and produces an ``AssertionError`` instead of a
``FLOW STOPPED: `` message).

**Three artifacts, three audiences.**

* ``artifacts/models/model.joblib`` — the champion *configuration* refit on every training row
  through 2011-11, which is what makes the operational forecast use the most recent history. The
  hold-out candidate models of US-17 stay at ``artifacts/models/<model_id>.joblib`` and are not
  touched; ``model_meta.json`` records that distinction (issue §3, design decision).
* ``artifacts/forecasts/latest_forecast.csv`` — one row per **active** product: the raw demand
  forecast and the two headline history numbers Screens 1-3 show beside it.
* ``artifacts/forecasts/inventory_plan.csv`` — one row per product **in the whole panel**, so the
  operations view is complete: active products carry a Recommended Target Inventory, inactive ones
  and products with no history through the origin carry an explicit ``status`` and no number
  (§15 — a cold-start product gets no model forecast and no invented history).

**The output is a Recommended Target Inventory, and only that** (§7). The dataset has no on-hand
and no on-order data, so a replenishment or re-order size cannot be computed — only a target level.
The phrase is written into ``model_meta.json`` as :data:`INVENTORY_OUTPUT_NAME` so the name travels
with the artifacts, and :func:`tests.test_latest_forecast` checks that the wrong name appears
nowhere under ``src/`` or in a written forecast.

The two formulas come from :mod:`pipeline.inventory` (US-21) and σ from :mod:`pipeline.sigma`
(US-20); nothing is re-derived here. ``sigma_table`` takes the *universe* of products to price from
the rows it finds at the evaluation month, and the back-test stops at target 2011-11 — so
:func:`operational_sigma` appends one residual-free placeholder row per active product at 2011-12,
which leaves every eligible residual (``target_month < 2011-12``, i.e. everything through 2011-11)
exactly as §26/§27 require.
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from pipeline import paths
from pipeline.baselines import B1, B2, B3
from pipeline.config import (
    CleaningConfig,
    InventoryPolicy,
    ModelConfig,
    load_cleaning_config,
    load_inventory_policy,
    load_model_config,
)
from pipeline.contract import read_panel
from pipeline.features import build_features_for_origin
from pipeline.inventory import safety_stock, target_inventory
from pipeline.models import make_model
from pipeline.run_context import RunContext
from pipeline.sigma import sigma_table
from pipeline.validation import (
    FlowValidationError,
    ValidationResult,
    Violation,
    write_validation_report,
)

#: Step name on every ``Violation``, on the validation report and on the CLI's ``ctx.step(...)``.
STEP_NAME = "latest_forecast"

#: The §7 name of this step's output — a stock *level* to aim for, never a re-order size. The
#: dataset carries no on-hand and no on-order position, so no order size can be computed at all.
INVENTORY_OUTPUT_NAME = "Recommended Target Inventory"

#: ``status`` values of ``inventory_plan.csv`` (§15). The inactive one is built from the configured
#: ``active_rule.k`` — the number of months is configuration, never a literal (CLAUDE.md §2 rule 4).
STATUS_FORECAST = "Forecast"
STATUS_NEW_PRODUCT = "Insufficient History / New Product"
INACTIVE_STATUS_TEMPLATE = "Inactive (no sales in last {k} months)"

#: Column order of ``artifacts/forecasts/latest_forecast.csv``.
LATEST_FORECAST_COLUMNS: list[str] = [
    "stock_code",
    "description",
    "forecast_origin",
    "target_month",
    "model",
    "prediction",
    "lag_1",
    "rolling_mean_3",
    "abc_class",
    "is_active",
    "status",
]

#: Column order of ``artifacts/forecasts/inventory_plan.csv``. ``uncertainty_ratio`` is the extra
#: column of issue §4 (Screen 2's "high uncertainty" filter); every other column and their relative
#: order are the issue's §2 list.
INVENTORY_PLAN_COLUMNS: list[str] = [
    "stock_code",
    "description",
    "forecast_origin",
    "target_month",
    "model",
    "forecast",
    "sigma",
    "sigma_source",
    "n_residuals_product",
    "z",
    "safety_stock",
    "target_inventory",
    "uncertainty_ratio",
    "abc_class",
    "last_month_units",
    "ma3_units",
    "months_since_last_sale",
    "product_age_months",
    "status",
    "run_id",
]

#: The feature each naive baseline reads when it is the champion. B3 is absent on purpose: it is
#: "reference only" (§19) and its rule needs the panel's month ``t-12``, not a feature column.
_BASELINE_FEATURE = {B1: "lag_1", B2: "rolling_mean_3"}

#: Multiplier and offset applied to the partial month in :func:`validate_operational_inputs`. Any
#: pair that changes every value would do; these are large enough that a leak could not cancel out.
_PERTURB_FACTOR = 1000
_PERTURB_OFFSET = 12345

#: Columns of the panel a feature could conceivably read (:mod:`pipeline.features` reads the first
#: three; the rest are perturbed too so the guard stays valid if the feature set ever widens).
_PERTURBABLE_COLUMNS = [
    "units_sold",
    "invoice_count",
    "avg_unit_price",
    "gross_revenue",
    "sale_line_count",
    "max_line_qty",
]

#: How many rows the CLI's sanity report lists.
_TOP_N = 10


def _repo_relative(path: Path) -> Path:
    """Repo-relative form of a canonical path constant (``docs/interfaces.md`` §6 rule 12)."""
    return path.relative_to(paths.PROJECT_ROOT)


def _resolve_read(ctx: RunContext, relative: Path) -> Path:
    """Locate an artifact *this run* wrote: the staged copy first, the final location second.

    ``ctx.out()`` must never be used to *find* a file — it registers the path for promotion, so
    reading through it would make ``promote()`` warn about an artifact that was never written; and
    the final path still holds the *previous* run's copy until promotion
    (``docs/interfaces.md`` §6 rule 7).
    """
    staged = ctx.staging_dir / relative
    if staged.is_file():
        return staged
    return ctx.base_dir / relative


def inactive_status(k: int) -> str:
    """``"Inactive (no sales in last 6 months)"`` for the configured ``k``, never a literal 6."""
    return INACTIVE_STATUS_TEMPLATE.format(k=k)


def _nullable_int(values: pd.Series) -> pd.Series:
    """A whole-number column that can also be empty — ``936`` and ``<NA>``, never ``936.000000``.

    ``Int64`` (nullable) rather than ``int64``: a product without a forecast has no target
    inventory at all, and ``to_csv`` writes that as an empty cell rather than a fabricated zero.
    """
    return values.astype("Float64").astype("Int64")


# --------------------------------------------------------------------------
# BaselineForecaster — a champion baseline still has to be a loadable model.joblib
# --------------------------------------------------------------------------
class BaselineForecaster:
    """A serialisable stand-in for a naive baseline, so ``model.joblib`` always exists (§41).

    A baseline winning the §20 gates is a legitimate, expected outcome, and ``model.joblib`` is one
    of the eight required artifacts — so the champion must be persistable even when it is a rule of
    thumb rather than a fitted estimator. ``predict`` reads one already-computed feature column
    (B1 = ``lag_1``, last month's units; B2 = ``rolling_mean_3``, the three-month average), which is
    exactly what :func:`pipeline.baselines.predict_baselines` does, so the operational number cannot
    drift from the back-tested one. Nothing is fitted, so there is no ``fit`` method.

    ``B3_seasonal_naive`` is deliberately unsupported: its rule reads the panel's month ``t-12``
    rather than a feature, and it is reference-only (§19) — it is never the model the champion gates
    compare against.
    """

    def __init__(self, model_id: str) -> None:
        if model_id not in _BASELINE_FEATURE:
            supported = sorted(_BASELINE_FEATURE)
            reason = (
                " (B3 is reference only (§19) and its rule reads the panel's month t-12, not a "
                "feature column)"
                if model_id == B3
                else ""
            )
            raise ValueError(
                f"{model_id}: BaselineForecaster supports {supported}{reason}"
            )
        self.model_id = model_id
        self.feature = _BASELINE_FEATURE[model_id]

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """The baseline's forecast for every row, as a float array — unclipped, like an
        estimator's ``predict``; the caller applies the §19 business clip."""
        return np.asarray(features[self.feature], dtype=float)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid only
        return f"BaselineForecaster(model_id={self.model_id!r}, feature={self.feature!r})"


# --------------------------------------------------------------------------
# champion resolution (issue §8: ctx.champion first, the JSON only as a fallback)
# --------------------------------------------------------------------------
def resolve_champion(ctx: RunContext) -> dict[str, Any]:
    """The §20 decision this run must act on: ``ctx.champion`` first, ``champion_decision.json``
    second.

    US-22 sets ``ctx.champion`` on the same context, so under the Flow the decision is already in
    memory — and re-reading the file would be wrong there, because under ``staging=True`` the final
    ``champion_decision.json`` still holds the *previous* run's decision until ``promote()``
    (``docs/interfaces.md`` §6 rule 7). The file is read only by the standalone CLI, where the run
    that produced it has already finished and promoted.

    There is no way to name a champion by hand: PRD §20 is executed by code, and CLAUDE.md forbids a
    ``--force-champion`` flag, so a missing decision is a hard stop rather than a prompt.
    """
    if ctx.champion:
        return dict(ctx.champion)
    if paths.CHAMPION_DECISION.is_file():
        return json.loads(paths.CHAMPION_DECISION.read_text(encoding="utf-8"))
    raise FileNotFoundError(
        f"{paths.CHAMPION_DECISION} not found and ctx.champion is unset. The champion is chosen by "
        "the §20 gates in US-22 (AI-27); run that step first — this module never picks one itself."
    )


def champion_id(decision: dict[str, Any]) -> str:
    """The champion model id inside a decision dict, whatever shape US-22 gave the rest of it."""
    value = decision.get("champion")
    if isinstance(value, dict):  # a nested {"champion": {"model": ...}} shape
        value = value.get("model") or value.get("model_id") or value.get("id")
    if not isinstance(value, str) or not value:
        raise ValueError(
            "the champion decision carries no champion model id under the key 'champion'; "
            f"got {decision.get('champion')!r}"
        )
    return value


def holdout_metrics_reference(
    decision: dict[str, Any], model_id: str, ctx: RunContext
) -> dict[str, float | None]:
    """``{wmape, bias}`` for the champion on the hold-out — carried into ``model_meta.json``.

    Taken from the decision dict when US-22 put it there, and otherwise read back from US-19's
    ``holdout_metrics_overall.csv``, which is the single source of every hold-out number in this
    project (:mod:`pipeline.evaluate`). Nothing is recomputed here — this module reports a metric,
    it never produces one.
    """
    carried = decision.get("holdout_metrics_reference") or decision.get("holdout_metrics")
    if isinstance(carried, dict) and {"wmape", "bias"} <= set(carried):
        return {"wmape": float(carried["wmape"]), "bias": float(carried["bias"])}

    relative = _repo_relative(paths.EVAL_TABLES_DIR / "holdout_metrics_overall.csv")
    source = _resolve_read(ctx, relative)
    if not source.is_file():
        ctx.warn(
            f"no hold-out metrics for {model_id}: the champion decision carries none and "
            f"{relative.as_posix()} does not exist"
        )
        return {"wmape": None, "bias": None}

    table = pd.read_csv(source)
    row = table.loc[table["model"] == model_id]
    if row.empty:
        ctx.warn(f"{relative.as_posix()} has no row for the champion {model_id}")
        return {"wmape": None, "bias": None}
    return {"wmape": float(row["wmape"].iloc[0]), "bias": float(row["bias"].iloc[0])}


# --------------------------------------------------------------------------
# operational features (§16) — one origin, one target, built once and reused
# --------------------------------------------------------------------------
def operational_origin(cleaning_cfg: CleaningConfig) -> str:
    """The forecast origin: the last month the data covers in full (§16) — 2011-11 here."""
    return cleaning_cfg.raw.last_full_month


def operational_features(
    panel_df: pd.DataFrame, cfg: ModelConfig, origin: str
) -> pd.DataFrame:
    """The §17 features for the single target ``origin + 1``, for every product active there (§14).

    A thin call into :func:`pipeline.features.build_features_for_origin` so the operational features
    take exactly the code path the training features took — the one thing that guarantees they
    cannot drift apart. The whole panel is passed in, partial month included: the panel's own month
    grid is what defines the target row's existence, and the shift-by-one construction is what keeps
    that month's *values* unreachable (:func:`validate_operational_inputs` proves it).
    """
    return build_features_for_origin(panel_df, origin, cfg.active_rule.k, cfg)


def _perturbed_panel(panel_df: pd.DataFrame, months: list[str]) -> pd.DataFrame:
    """A copy of the panel whose rows in ``months`` carry deliberately wrong measurements."""
    corrupted = panel_df.copy()
    mask = corrupted["month"].astype(str).isin(months)
    for column in _PERTURBABLE_COLUMNS:
        if column in corrupted.columns:
            corrupted[column] = corrupted[column].astype(float)
            corrupted.loc[mask, column] = (
                corrupted.loc[mask, column] * _PERTURB_FACTOR + _PERTURB_OFFSET
            )
    return corrupted


def validate_operational_inputs(
    operational_df: pd.DataFrame,
    train_features_df: pd.DataFrame,
    panel_df: pd.DataFrame,
    origin: str,
    cfg: ModelConfig,
    cleaning_cfg: CleaningConfig,
) -> ValidationResult:
    """Three invariants of the latest operational forecast — pure: no ``ctx``, no disk (issue §8).

    * ``forecast_origin`` / ``target_month`` — every operational row is made at ``origin`` for
      ``origin + 1``, the §16 definition.
    * ``partial_month_not_used`` — rebuilding the operational features from a panel whose partial
      months (``cleaning_config -> raw.partial_months``, i.e. December 2011) carry corrupted
      measurements produces the identical frame. If any feature read month ``t``, the two frames
      would differ; equality is therefore proof the §16 boundary held, not an assertion that it did.
    * ``refit_window`` — no row the champion is refit on has a target after the origin. Everything
      with ``month <= origin`` is allowed (§16); anything later is not.

    The caller writes the report with ``write_validation_report(result, run_id=ctx.run_id)`` and
    raises :class:`~pipeline.validation.FlowValidationError` — a deterministic step decides nothing
    about bad data itself (``docs/interfaces.md`` §4).
    """
    violations: list[Violation] = []
    target = str(pd.Period(origin, freq="M") + 1)

    for column, expected in (("forecast_origin", origin), ("target_month", target)):
        wrong = operational_df.loc[operational_df[column].astype(str) != expected, column]
        if len(wrong):
            violations.append(
                Violation(
                    step=STEP_NAME,
                    rule=column,
                    message=(
                        f"{len(wrong)} operational row(s) have {column} != {expected}; the latest "
                        f"forecast is made at origin {origin} for target {target} only (§16)"
                    ),
                    count=int(len(wrong)),
                    examples=sorted(wrong.astype(str).unique())[:5],
                )
            )

    partial_months = [month for month in cleaning_cfg.raw.partial_months if month > origin]
    if partial_months:
        rebuilt = operational_features(_perturbed_panel(panel_df, partial_months), cfg, origin)
        if list(rebuilt.columns) != list(operational_df.columns) or len(rebuilt) != len(
            operational_df
        ):
            violations.append(
                Violation(
                    step=STEP_NAME,
                    rule="partial_month_not_used",
                    message=(
                        "corrupting the partial month(s) "
                        f"{partial_months} changed the shape of the operational feature frame "
                        f"({len(operational_df)}x{len(operational_df.columns)} -> "
                        f"{len(rebuilt)}x{len(rebuilt.columns)})"
                    ),
                )
            )
        else:
            differing = [
                column
                for column in operational_df.columns
                if not operational_df[column].reset_index(drop=True).equals(
                    rebuilt[column].reset_index(drop=True)
                )
            ]
            if differing:
                violations.append(
                    Violation(
                        step=STEP_NAME,
                        rule="partial_month_not_used",
                        message=(
                            f"{len(differing)} operational feature(s) changed when the partial "
                            f"month(s) {partial_months} were corrupted, so they read month t "
                            "(§8, §16: December 2011 never enters a feature)"
                        ),
                        count=len(differing),
                        examples=differing[:5],
                    )
                )

    late = train_features_df.loc[train_features_df["target_month"].astype(str) > origin]
    if len(late):
        violations.append(
            Violation(
                step=STEP_NAME,
                rule="refit_window",
                message=(
                    f"{len(late)} refit row(s) have a target month after the forecast origin "
                    f"{origin}; the champion may only be refit on targets known at the origin (§16)"
                ),
                count=int(len(late)),
                examples=sorted(late["target_month"].astype(str).unique())[:5],
            )
        )

    return ValidationResult(
        step=STEP_NAME,
        passed=not violations,
        violations=violations,
        checked_rows=int(len(operational_df)),
        extra={"origin": origin, "target_month": target, "partial_months": partial_months},
    )


# --------------------------------------------------------------------------
# refit_champion — model.joblib + model_meta.json, both through ctx.out() (issue §8)
# --------------------------------------------------------------------------
def refit_champion(
    features_df: pd.DataFrame,
    champion: str,
    cfg: ModelConfig,
    ctx: RunContext,
    *,
    decision: dict[str, Any] | None = None,
    origin: str | None = None,
) -> Any:
    """Refit the champion on every training row and persist it as ``model.joblib`` (§41).

    "Refit" means training the chosen model once more on all the data available at the forecast
    origin, so the operational forecast benefits from the most recent months — the hold-out
    candidates of US-17 stop at 2011-05 and stay where they are, under
    ``artifacts/models/<model_id>.joblib``. A champion that is a naive baseline is persisted as a
    :class:`BaselineForecaster`, which carries the same ``predict`` interface, and
    ``model_meta.json`` records ``kind: baseline``.

    Both writes go through ``ctx.out()`` (issue §8): ``model.joblib`` is one of the eight required
    artifacts, and written straight to its final location a failed refit would destroy the last good
    champion — precisely what the §39 staging guarantee exists to prevent.
    """
    spec = cfg.models[champion]
    resolved_origin = (
        operational_origin(load_cleaning_config()) if origin is None else origin
    )
    is_baseline = spec.kind == "baseline"

    if is_baseline:
        estimator: Any = BaselineForecaster(champion)
        n_rows = 0
    else:
        estimator = make_model(champion, cfg, ctx.seed)
        estimator.fit(features_df[list(cfg.features)], features_df["y"].astype(float))
        n_rows = int(len(features_df))

    model_destination = ctx.out(_repo_relative(paths.MODEL))
    joblib.dump(estimator, model_destination)
    ctx.record_artifact("model", _repo_relative(paths.MODEL))

    targets = features_df["target_month"].astype(str)
    meta = {
        "champion": champion,
        "kind": "baseline" if is_baseline else spec.kind,
        "loss": spec.loss,
        "params": dict(spec.params),
        "train_targets": {
            "start": None if targets.empty else str(targets.min()),
            "end": None if targets.empty else str(targets.max()),
        },
        "n_rows": n_rows,
        "features": list(cfg.features),
        "seed": ctx.seed,
        "sklearn_version": ctx.versions.get("sklearn"),
        "refit_at": datetime.now(UTC).isoformat(),
        "run_id": ctx.run_id,
        "forecast_origin": resolved_origin,
        "target_month": str(pd.Period(resolved_origin, freq="M") + 1),
        "holdout_metrics_reference": holdout_metrics_reference(decision or {}, champion, ctx),
        "inventory_output_name": INVENTORY_OUTPUT_NAME,
        "note": (
            "model.joblib is the champion configuration refit on every training target through the "
            "forecast origin, for the operational forecast only. The hold-out candidate models "
            "(trained through the training window and scored on the hold-out) stay at "
            "artifacts/models/<model_id>.joblib and are unchanged by this step."
        ),
    }
    meta_destination = ctx.out(_repo_relative(paths.MODEL_META))
    meta_destination.write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    ctx.record_artifact("model_meta", _repo_relative(paths.MODEL_META))

    return estimator


# --------------------------------------------------------------------------
# build_latest_forecast — the active products' demand forecast for the operational month
# --------------------------------------------------------------------------
def _descriptions(panel_df: pd.DataFrame) -> pd.Series:
    """Each product's most recent non-empty description — display only, never a feature (§13.2)."""
    frame = panel_df[["stock_code", "month", "description"]].copy()
    frame["stock_code"] = frame["stock_code"].astype(str)
    frame = frame.sort_values(["stock_code", "month"], kind="mergesort")
    return frame.groupby("stock_code")["description"].last()


def build_latest_forecast(
    panel_df: pd.DataFrame,
    champion_model: Any,
    cfg: ModelConfig,
    ctx: RunContext,
    *,
    champion: str,
    abc_train_df: pd.DataFrame,
    origin: str | None = None,
    features_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """The champion's demand forecast for every active product at ``origin + 1``, written to disk.

    ``features_df`` lets the caller pass in the frame :func:`operational_features` already produced
    — building it costs a full pass over the panel and the CLI validates it before this point, so
    rebuilding it here would do the same work twice. Predictions are clipped at zero for business
    use, exactly as :func:`pipeline.models.fit_predict_one_origin` clips the back-test's.
    """
    resolved_origin = (
        operational_origin(load_cleaning_config()) if origin is None else origin
    )
    frame = (
        operational_features(panel_df, cfg, resolved_origin)
        if features_df is None
        else features_df
    )

    raw = np.asarray(champion_model.predict(frame[list(cfg.features)]), dtype=float)

    abc_lookup = abc_train_df[["stock_code", "abc_class"]].copy()
    abc_lookup["stock_code"] = abc_lookup["stock_code"].astype(str)
    descriptions = _descriptions(panel_df)

    latest = pd.DataFrame(
        {
            "stock_code": frame["stock_code"].astype(str),
            "forecast_origin": frame["forecast_origin"].astype(str),
            "target_month": frame["target_month"].astype(str),
            "model": champion,
            "prediction": np.clip(raw, 0, None),
            "lag_1": frame["lag_1"].astype("int64").astype("Int64"),
            "rolling_mean_3": frame["rolling_mean_3"].astype(float),
            "is_active": frame["is_active"].astype(bool),
            "status": STATUS_FORECAST,
        }
    )
    latest["description"] = latest["stock_code"].map(descriptions)
    latest = latest.merge(abc_lookup, on="stock_code", how="left")

    latest = (
        latest.sort_values("stock_code", kind="mergesort")
        .reset_index(drop=True)[LATEST_FORECAST_COLUMNS]
    )

    destination = ctx.out(_repo_relative(paths.LATEST_FORECAST))
    latest.to_csv(
        destination, index=False, float_format="%.6f", lineterminator="\n", encoding="utf-8"
    )
    ctx.record_artifact("latest_forecast", _repo_relative(paths.LATEST_FORECAST))
    return latest


# --------------------------------------------------------------------------
# operational_sigma — US-20's table, asked about a month the back-test does not reach
# --------------------------------------------------------------------------
def operational_sigma(
    backtest_df: pd.DataFrame,
    abc_train_df: pd.DataFrame,
    latest_df: pd.DataFrame,
    champion: str,
    policy_cfg: InventoryPolicy,
) -> pd.DataFrame:
    """Robust σ for the operational month, from every back-test residual before it (§26, §27).

    :func:`pipeline.sigma.sigma_table` reads the *universe* of products to price off the rows it
    finds at the evaluation month, and the rolling back-test's last target is 2011-11 — so this
    appends one placeholder row per active product at the operational month, carrying no residual.
    Eligibility is unchanged by that: a residual counts only when ``target_month < t`` **and** it is
    not NaN, so the placeholders price the products without ever pricing themselves, and every
    residual through 2011-11 is used.
    """
    target = str(latest_df["target_month"].iloc[0]) if len(latest_df) else None
    if target is None:
        return sigma_table(backtest_df, abc_train_df, [], champion, policy_cfg)

    universe = pd.DataFrame(
        {
            "stock_code": latest_df["stock_code"].astype(str),
            "target_month": target,
            "model": champion,
            "residual": np.nan,
        }
    )
    columns = ["stock_code", "target_month", "model", "residual"]
    history = backtest_df.loc[backtest_df["model"] == champion, columns].copy()
    history["stock_code"] = history["stock_code"].astype(str)
    history["target_month"] = history["target_month"].astype(str)

    combined = pd.concat([history, universe], ignore_index=True)
    return sigma_table(combined, abc_train_df, [target], champion, policy_cfg)


# --------------------------------------------------------------------------
# build_inventory_plan — the deterministic policy layer, over the whole panel universe
# --------------------------------------------------------------------------
def build_inventory_plan(
    latest_df: pd.DataFrame,
    sigma_df: pd.DataFrame,
    policy_cfg: InventoryPolicy,
    ctx: RunContext,
    *,
    panel_df: pd.DataFrame,
    abc_train_df: pd.DataFrame,
    features_df: pd.DataFrame,
    cfg: ModelConfig,
    origin: str | None = None,
) -> pd.DataFrame:
    """Turn the forecast into a Recommended Target Inventory and cover the whole product universe.

    Active products get ``target_inventory = ceil(max(0, forecast + z x sigma))`` — the §28 formula,
    taken from :mod:`pipeline.inventory` rather than restated, with ``safety_stock = z x sigma``
    (§25) from the same module. Every other product in the panel gets a row with no numbers and an
    explicit ``status``: ``Inactive (no sales in last k months)`` when it has history but has
    stopped selling, and ``Insufficient History / New Product`` when it has no observed month at or
    before the forecast origin at all (§15 — a cold-start product gets no model forecast and no
    invented history).

    ``last_month_units`` is read from the panel for *every* row, including the ones with no
    forecast, so the operations view always shows what actually happened last month; on a forecast
    row it is by construction the same number as the ``lag_1`` feature. ``months_since_last_sale``
    and
    ``product_age_months`` come from ``features_df`` — they are §17 features and are defined only on
    an active row, so a row without a forecast keeps ``<NA>`` rather than a number produced by a
    second implementation of the same rule (CLAUDE.md §11).
    """
    resolved_origin = (
        operational_origin(load_cleaning_config()) if origin is None else origin
    )
    target = str(pd.Period(resolved_origin, freq="M") + 1)
    z = policy_cfg.z

    panel = panel_df.copy()
    panel["stock_code"] = panel["stock_code"].astype(str)
    panel["month"] = panel["month"].astype(str)

    plan = pd.DataFrame({"stock_code": sorted(panel["stock_code"].unique())})
    observed = set(panel.loc[panel["month"] <= resolved_origin, "stock_code"])
    origin_units = panel.loc[panel["month"] == resolved_origin].set_index("stock_code")[
        "units_sold"
    ]

    forecast_part = latest_df[["stock_code", "prediction", "rolling_mean_3", "status"]].rename(
        columns={"prediction": "forecast", "rolling_mean_3": "ma3_units"}
    )
    plan = plan.merge(forecast_part, on="stock_code", how="left")
    plan["status"] = plan["status"].fillna(
        pd.Series(
            np.where(
                plan["stock_code"].isin(observed),
                inactive_status(cfg.active_rule.k),
                STATUS_NEW_PRODUCT,
            ),
            index=plan.index,
        )
    )

    sigma_part = sigma_df[["stock_code", "sigma", "sigma_source", "n_residuals_product"]].copy()
    sigma_part["stock_code"] = sigma_part["stock_code"].astype(str)
    plan = plan.merge(sigma_part, on="stock_code", how="left")

    abc_lookup = abc_train_df[["stock_code", "abc_class"]].copy()
    abc_lookup["stock_code"] = abc_lookup["stock_code"].astype(str)
    plan = plan.merge(abc_lookup, on="stock_code", how="left")

    history = features_df[["stock_code", "months_since_last_sale", "product_age_months"]].copy()
    history["stock_code"] = history["stock_code"].astype(str)
    plan = plan.merge(history, on="stock_code", how="left")

    is_forecast = (plan["status"] == STATUS_FORECAST).to_numpy()
    sigma_values = plan["sigma"].to_numpy(dtype=float)
    forecast_values = plan["forecast"].to_numpy(dtype=float)
    # np.where evaluates both branches, so the formulas are fed zero-filled copies and the result is
    # masked back to NaN — a row without a forecast has no policy, not a policy computed from NaN.
    filled_forecast = np.nan_to_num(forecast_values)
    filled_sigma = np.nan_to_num(sigma_values)

    plan["z"] = np.where(is_forecast, z, np.nan)
    plan["safety_stock"] = np.where(is_forecast, safety_stock(filled_sigma, z), np.nan)
    plan["target_inventory"] = pd.array(
        np.where(is_forecast, target_inventory(filled_forecast, filled_sigma, z), np.nan),
        dtype="Float64",
    ).astype("Int64")
    plan["uncertainty_ratio"] = np.where(
        is_forecast, sigma_values / np.maximum(forecast_values, 1.0), np.nan
    )

    plan["description"] = plan["stock_code"].map(_descriptions(panel))
    plan["forecast_origin"] = resolved_origin
    plan["target_month"] = target
    plan["model"] = str(latest_df["model"].iloc[0]) if len(latest_df) else None
    plan["run_id"] = ctx.run_id
    plan["last_month_units"] = _nullable_int(plan["stock_code"].map(origin_units))
    plan["n_residuals_product"] = _nullable_int(plan["n_residuals_product"])
    plan["months_since_last_sale"] = _nullable_int(plan["months_since_last_sale"])
    plan["product_age_months"] = _nullable_int(plan["product_age_months"])

    plan = (
        plan.sort_values("stock_code", kind="mergesort")
        .reset_index(drop=True)[INVENTORY_PLAN_COLUMNS]
    )

    destination = ctx.out(_repo_relative(paths.INVENTORY_PLAN))
    plan.to_csv(
        destination, index=False, float_format="%.6f", lineterminator="\n", encoding="utf-8"
    )
    ctx.record_artifact("inventory_plan", _repo_relative(paths.INVENTORY_PLAN))
    return plan


# --------------------------------------------------------------------------
# sanity report (issue §2)
# --------------------------------------------------------------------------
def sanity_report(plan: pd.DataFrame, latest: pd.DataFrame, policy_cfg: InventoryPolicy) -> str:
    """The printed summary of a run: how many products, how many units, how uncertain, and the
    biggest positions. Every number is read off the two frames — nothing is recomputed."""
    forecast_rows = plan.loc[plan["status"] == STATUS_FORECAST]
    lines = [
        f"forecast origin: {plan['forecast_origin'].iloc[0] if len(plan) else '-'}",
        f"target month:    {plan['target_month'].iloc[0] if len(plan) else '-'}",
        f"champion:        {plan['model'].iloc[0] if len(plan) else '-'}",
        f"z:               {policy_cfg.z}",
        "",
        f"active products (status={STATUS_FORECAST}): {len(forecast_rows)}",
    ]
    for status, count in plan["status"].value_counts().items():
        if status != STATUS_FORECAST:
            lines.append(f"  {status}: {count}")
    lines += [
        "",
        f"total forecast units:  {float(latest['prediction'].sum()):,.0f}",
        f"total {INVENTORY_OUTPUT_NAME}: "
        f"{float(forecast_rows['target_inventory'].astype('Float64').sum()):,.0f}",
        "",
        "share by sigma_source:",
    ]
    if len(forecast_rows):
        shares = forecast_rows["sigma_source"].value_counts(normalize=True)
        for level in policy_cfg.sigma.fallback_levels:
            lines.append(f"  {level}: {float(shares.get(level, 0.0)):.2%}")

    lines += ["", f"top {_TOP_N} by {INVENTORY_OUTPUT_NAME}:"]
    top = forecast_rows.sort_values("target_inventory", ascending=False).head(_TOP_N)
    columns = [
        "stock_code",
        "description",
        "forecast",
        "sigma",
        "sigma_source",
        "safety_stock",
        "target_inventory",
    ]
    lines.append(top[columns].to_string(index=False))
    return "\n".join(lines)


# --------------------------------------------------------------------------
# run_latest_forecast — the public entry point (Flow step 8 calls this)
# --------------------------------------------------------------------------
def run_latest_forecast(cfg: ModelConfig, ctx: RunContext) -> dict[str, Any]:
    """Refit the champion, forecast the operational month and write the inventory plan.

    Reads ``clean_data.csv``, ``features.csv``, ``backtest_predictions.csv`` and ``abc_train.csv``
    from their canonical locations — correct standalone, where ``staging=False`` and all four are
    final. A future Flow integration (US-33 step 8) must instead hand in the frames the producing
    steps returned: under ``staging=True`` those files sit in ``artifacts/_staging/<run_id>/`` while
    the final paths still hold the previous run's copies (``docs/interfaces.md`` §6 rule 7).
    """
    cleaning_cfg = load_cleaning_config()
    policy_cfg = load_inventory_policy()

    abc_path = paths.EVAL_TABLES_DIR / "abc_train.csv"
    required = (paths.CLEAN_DATA, paths.FEATURES, paths.BACKTEST_PREDICTIONS, abc_path)
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(
                f"{path} not found. Run `python -m pipeline.panel`, `python -m pipeline.features`, "
                "`python -m pipeline.backtest` and `python -m pipeline.split abc` first."
            )

    decision = resolve_champion(ctx)
    champion = champion_id(decision)

    panel = read_panel(paths.CLEAN_DATA)
    train_features = pd.read_csv(
        paths.FEATURES,
        dtype={"stock_code": "string", "forecast_origin": "string", "target_month": "string"},
    )
    backtest = pd.read_csv(
        paths.BACKTEST_PREDICTIONS,
        dtype={
            "stock_code": "string",
            "forecast_origin": "string",
            "target_month": "string",
            "model": "string",
        },
    )
    abc = pd.read_csv(abc_path, dtype={"stock_code": "string"})

    origin = operational_origin(cleaning_cfg)
    features = operational_features(panel, cfg, origin)

    validation = validate_operational_inputs(
        features, train_features, panel, origin, cfg, cleaning_cfg
    )
    write_validation_report(validation, run_id=ctx.run_id)
    if not validation.passed:
        raise FlowValidationError(validation)

    model = refit_champion(train_features, champion, cfg, ctx, decision=decision, origin=origin)
    latest = build_latest_forecast(
        panel,
        model,
        cfg,
        ctx,
        champion=champion,
        abc_train_df=abc,
        origin=origin,
        features_df=features,
    )
    sigma_df = operational_sigma(backtest, abc, latest, champion, policy_cfg)
    plan = build_inventory_plan(
        latest,
        sigma_df,
        policy_cfg,
        ctx,
        panel_df=panel,
        abc_train_df=abc,
        features_df=features,
        cfg=cfg,
        origin=origin,
    )

    forecast_rows = plan.loc[plan["status"] == STATUS_FORECAST]
    ctx.log_rows(
        "inventory_plan_status",
        before=int(len(plan)),
        removed=int(len(plan) - len(forecast_rows)),
        after=int(len(forecast_rows)),
    )
    shares = (
        forecast_rows["sigma_source"].value_counts(normalize=True)
        if len(forecast_rows)
        else pd.Series(dtype=float)
    )
    ctx.record_metrics(
        {
            "latest_forecast": {
                "champion": champion,
                "forecast_origin": origin,
                "target_month": str(pd.Period(origin, freq="M") + 1),
                "active_products": int(len(forecast_rows)),
                "panel_products": int(len(plan)),
                "total_forecast_units": float(latest["prediction"].sum()),
                "total_target_inventory": float(
                    forecast_rows["target_inventory"].astype("Float64").sum()
                ),
                "z": policy_cfg.z,
                "sigma_source_shares": {
                    level: float(shares.get(level, 0.0))
                    for level in policy_cfg.sigma.fallback_levels
                },
            }
        }
    )

    return {
        "champion": champion,
        "model": model,
        "latest_forecast": latest,
        "sigma_table": sigma_df,
        "inventory_plan": plan,
        "validation": validation,
    }


# --------------------------------------------------------------------------
# CLI: python -m pipeline.latest_forecast
# --------------------------------------------------------------------------
def run(argv: list[str] | None = None) -> int:
    """Standalone entry point: ``python -m pipeline.latest_forecast`` (AC 1)."""
    args = list(sys.argv[1:] if argv is None else argv)
    if args:
        print("usage: python -m pipeline.latest_forecast", file=sys.stderr)
        return 2

    ctx = RunContext.start(mode="no-llm")
    try:
        cfg = load_model_config()
        with ctx.step(STEP_NAME):
            result = run_latest_forecast(cfg, ctx)
    except Exception:
        ctx.finish(status="failed")
        raise
    ctx.finish()

    print(
        sanity_report(
            result["inventory_plan"], result["latest_forecast"], load_inventory_policy()
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
