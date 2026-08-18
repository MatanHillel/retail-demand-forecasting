"""Quarterly aggregation of one-step-ahead monthly forecasts (US-24, PRD §1, §6.1, §6.2, §31, §32,
§50).

**There is no quarterly model.** Every quarterly figure in this module is a *sum of three genuine
one-step-ahead monthly forecasts*, each produced at its own forecast origin (the month
immediately before the one it predicts) by the rolling back-test of :mod:`pipeline.backtest`
(US-18). A quarter is never forecast as a single unit, and no model here ever sees more than one
month ahead — that is the MVP limitation :data:`LIMITATION_TEXT` documents, and it is why this
module writes nothing under ``artifacts/models/``.

**Two kinds of quarter row.**

* A **back-tested** quarter (``stock_code, model, quarter``) sums the three monthly
  ``backtest_predictions.csv`` rows that fall in it. ``complete`` is ``True`` only when all three
  of the quarter's calendar months are present for that product and model — this single rule
  covers both a calendar-incomplete quarter (2011-Q4, which the back-test only reaches for October
  and November, §21) and a product that was not yet active for the whole quarter (§14): either way
  ``n_months < 3`` and the quarter is excluded from :func:`quarterly_metrics`.
* The **rolling operational estimate** (:func:`rolling_quarter_estimate`) is the one exception the
  PRD explicitly asks for (§32): the current partial quarter's total, built from the two already
  known months' actual sales plus the single genuine one-step-ahead forecast for the third
  (``latest_forecast.csv``, US-23). It is always ``complete = False`` — it is an operational
  estimate, never a scored quarter — and carries no ``actual_sum``, because the quarter itself has
  not happened yet.

**wMAPE and Bias** come from :func:`pipeline.metrics.metrics_table`, never re-derived, computed
only over ``complete`` quarters (§21: December 2011 must never enter a metric, and an incomplete
quarter is exactly as unscoreable). ``quarterly_metrics.csv`` mirrors the ``scope``/``group``
convention of :mod:`pipeline.inventory` (``_scoped_kpis``): one block of rows at
``scope="overall"`` (``quarter="all"``) and one at ``scope="quarter"`` (one row per real quarter).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from pipeline import paths
from pipeline.baselines import B2
from pipeline.config import CleaningConfig, ModelConfig, load_cleaning_config, load_model_config
from pipeline.contract import read_panel
from pipeline.latest_forecast import STATUS_FORECAST, champion_id, resolve_champion
from pipeline.metrics import format_pct, metrics_table
from pipeline.run_context import RunContext

#: Step name on ``ctx.step(...)`` and in log lines.
STEP_NAME = "quarterly_aggregation"

#: ``scope`` values of ``quarterly_metrics.csv`` (mirrors ``pipeline.inventory``'s scope/group
#: convention: the overall row's ``quarter`` is the literal ``"all"``, never an empty cell).
SCOPE_OVERALL = "overall"
SCOPE_QUARTER = "quarter"
_OVERALL_QUARTER_LABEL = "all"

#: ``estimate_type`` of the rolling operational row of ``quarterly_forecast.csv`` (issue §2) — the
#: only row where the "forecast" is a mix of already-known actuals and one genuine forecast.
ROLLING_ESTIMATE_TYPE = "actuals+next_month_forecast"

#: Column order of ``quarterly_forecast.csv`` — a published schema, extend but never rename.
QUARTERLY_FORECAST_COLUMNS: list[str] = [
    "stock_code",
    "quarter",
    "model",
    "forecast_sum",
    "actual_sum",
    "n_months",
    "complete",
    "months_included",
    "estimate_type",
]

#: Column order of ``quarterly_metrics.csv``.
QUARTERLY_METRICS_COLUMNS: list[str] = [
    "model",
    "scope",
    "quarter",
    "wmape",
    "bias",
    "mae",
    "rmse",
    "n_rows",
    "sum_actual",
    "sum_forecast",
    "negative_share",
]

#: A calendar quarter has exactly three months — a definition, not a tunable (like the 3/6-month
#: windows of :mod:`pipeline.features`).
_MONTHS_PER_QUARTER = 3

#: ``months_included`` is a list of ``YYYY-MM`` strings serialised into one CSV cell; ``;`` is
#: unambiguous because no month string ever contains it.
_MONTH_LIST_SEP = ";"

#: Deterministic text (CLAUDE.md §2 rule 4: no LLM ever writes a narrative number) stating the
#: MVP's quarterly-forecasting limitation, inserted into the evaluation report and model card
#: (US-25). The exact phrase "cannot forecast all three months of a quarter at the start of the
#: quarter" is a published contract of this module (issue §6 acceptance criterion).
LIMITATION_TEXT = """# Quarterly forecast — methodology limitation

This project trains and evaluates a single **one-step-ahead monthly model**: at any forecast
origin, it predicts only the single month that immediately follows. Every quarterly figure in
`quarterly_forecast.csv` is a **sum of three such one-step-ahead forecasts**, each produced at its
own origin (the month before the one it predicts) by the rolling back-test — never a separate
quarterly model, and never a single forecast covering all three months at once.

As a direct consequence, this MVP
**cannot forecast all three months of a quarter at the start of the quarter**.
The second and third months of any quarter can only be forecast once the preceding month's data
becomes available, one month at a time. A genuine start-of-quarter, three-month-ahead forecast
would require a separate multi-horizon or recursive forecasting approach; that is out of scope for
this MVP and is a candidate v2 extension (PRD §32, §50).

The rolling estimate for the current partial quarter is not an exception to this: it combines the
already-observed actual sales of the quarter's completed months with the single genuine
one-step-ahead forecast for the next month, and is therefore never treated as a scored, complete
quarter (`complete = False`, no `actual_sum`).
"""


def _repo_relative(path: Path) -> Path:
    """Repo-relative form of a canonical path constant (``docs/interfaces.md`` §6 rule 12)."""
    return path.relative_to(paths.PROJECT_ROOT)


# --------------------------------------------------------------------------
# calendar-quarter arithmetic — pure, no config, no disk
# --------------------------------------------------------------------------
def quarter_label(month: str) -> str:
    """``"2011-08"`` -> ``"2011-Q3"`` — calendar quarters (Q1 = Jan-Mar), not the dataset's own
    Dec-Nov "year" (issue §3: quarters must not be assumed to align with that)."""
    period = pd.Period(month, freq="M")
    quarter = (period.month - 1) // _MONTHS_PER_QUARTER + 1
    return f"{period.year}-Q{quarter}"


def quarter_months(quarter: str) -> list[str]:
    """The three ``YYYY-MM`` months of a ``"YYYY-Qn"`` label, in calendar order."""
    year_str, quarter_str = quarter.split("-Q")
    year = int(year_str)
    first_month = (int(quarter_str) - 1) * _MONTHS_PER_QUARTER + 1
    last_month = first_month + _MONTHS_PER_QUARTER
    return [f"{year}-{month:02d}" for month in range(first_month, last_month)]


def default_models(champion: str) -> list[str]:
    """The champion and the main baseline B2 (issue §2), deduplicated when the champion *is* B2."""
    return [champion] if champion == B2 else [champion, B2]


# --------------------------------------------------------------------------
# aggregate_quarterly — pure (issue §8: stays pure, the ctx-carrying caller writes)
# --------------------------------------------------------------------------
def aggregate_quarterly(
    backtest_df: pd.DataFrame, models: list[str], cfg: ModelConfig
) -> pd.DataFrame:
    """Sum the rolling back-test's monthly forecasts into one row per ``(stock_code, model,
    quarter)`` (issue §2, pure — no ``ctx``, no disk).

    Restricted to ``cfg.backtest.first_origin + 1 .. cfg.backtest.last_origin + 1`` — the genuine
    one-step-ahead range :mod:`pipeline.backtest` produced — so a caller that hands in extra rows
    cannot smuggle a later or earlier month into a quarterly sum. ``n_months`` counts *distinct*
    target months actually present for that product and model, and ``complete`` is ``True`` only
    at ``n_months == 3``: this single rule is what makes both a calendar-incomplete quarter
    (2011-Q4, capped at October and November, §21) and a product not yet active for the whole
    quarter (§14) fall out of the same check, rather than two special cases.
    """
    valid_start = str(pd.Period(cfg.backtest.first_origin, freq="M") + 1)
    valid_end = str(pd.Period(cfg.backtest.last_origin, freq="M") + 1)

    target_month = backtest_df["target_month"].astype(str)
    rows = backtest_df.loc[
        backtest_df["model"].isin(models) & target_month.between(valid_start, valid_end)
    ].copy()
    rows["stock_code"] = rows["stock_code"].astype(str)
    rows["target_month"] = rows["target_month"].astype(str)
    rows["quarter"] = rows["target_month"].map(quarter_label)

    grouped = (
        rows.groupby(["stock_code", "model", "quarter"], sort=False)
        .agg(
            forecast_sum=("prediction", "sum"),
            actual_sum=("actual", "sum"),
            n_months=("target_month", "nunique"),
            months_included=(
                "target_month",
                lambda months: _MONTH_LIST_SEP.join(sorted(months.astype(str).unique())),
            ),
        )
        .reset_index()
    )
    grouped["complete"] = grouped["n_months"] == _MONTHS_PER_QUARTER
    grouped["estimate_type"] = ""

    return (
        grouped.sort_values(["quarter", "stock_code", "model"], kind="mergesort")
        .reset_index(drop=True)[QUARTERLY_FORECAST_COLUMNS]
    )


# --------------------------------------------------------------------------
# rolling_quarter_estimate — the operational, partial-quarter row (§32)
# --------------------------------------------------------------------------
def rolling_quarter_estimate(
    latest_df: pd.DataFrame,
    panel_df: pd.DataFrame,
    champion: str,
    cleaning_cfg: CleaningConfig,
) -> pd.DataFrame:
    """The current partial quarter's operational total: known actuals + one genuine forecast.

    For every product ``latest_forecast.csv`` (US-23) has a December 2011 forecast for,
    ``forecast_sum = actual(Oct) + actual(Nov) + forecast(Dec)`` — actuals read straight from the
    panel, the forecast from ``latest_df["prediction"]``. There is no ``actual_sum``: the quarter
    has not happened yet, so there is nothing to compare against, and this row is always
    ``complete = False`` — it must never enter :func:`quarterly_metrics` (§21, §32).
    """
    origin = cleaning_cfg.raw.last_full_month
    target = str(pd.Period(origin, freq="M") + 1)
    quarter = quarter_label(target)
    months = quarter_months(quarter)
    prior_months = [month for month in months if month < target]

    panel = panel_df.copy()
    panel["stock_code"] = panel["stock_code"].astype(str)
    panel["month"] = panel["month"].astype(str)

    forecast_rows = latest_df.copy()
    if "status" in forecast_rows.columns:
        forecast_rows = forecast_rows.loc[forecast_rows["status"] == STATUS_FORECAST]
    forecast_rows["stock_code"] = forecast_rows["stock_code"].astype(str)
    codes = sorted(forecast_rows["stock_code"].unique())

    prior_actual_sum = pd.Series(0.0, index=codes)
    for month in prior_months:
        month_units = panel.loc[panel["month"] == month].set_index("stock_code")["units_sold"]
        prior_actual_sum = prior_actual_sum.add(
            month_units.reindex(codes).fillna(0.0).astype(float), fill_value=0.0
        )

    forecast_lookup = forecast_rows.set_index("stock_code")["prediction"].astype(float)
    dec_forecast = forecast_lookup.reindex(codes).fillna(0.0)

    return pd.DataFrame(
        {
            "stock_code": codes,
            "quarter": quarter,
            "model": champion,
            "forecast_sum": (prior_actual_sum + dec_forecast).to_numpy(),
            "actual_sum": float("nan"),
            "n_months": len(months),
            "complete": False,
            "months_included": _MONTH_LIST_SEP.join(months),
            "estimate_type": ROLLING_ESTIMATE_TYPE,
        }
    )[QUARTERLY_FORECAST_COLUMNS]


# --------------------------------------------------------------------------
# quarterly_metrics — pure (issue §8), complete quarters only
# --------------------------------------------------------------------------
def quarterly_metrics(qdf: pd.DataFrame) -> pd.DataFrame:
    """wMAPE and Bias (US-15) per model, overall and per quarter — complete quarters only (§21).

    The rolling operational row is excluded by construction: it is always ``complete = False``.
    Mirrors :mod:`pipeline.inventory`'s ``scope``/``group`` convention — the overall block's
    ``quarter`` is the literal ``"all"``, never an empty cell.
    """
    complete = qdf.loc[qdf["complete"]]
    if complete.empty:
        return pd.DataFrame(columns=QUARTERLY_METRICS_COLUMNS)

    overall = metrics_table(complete, "actual_sum", "forecast_sum", group_cols=["model"])
    overall.insert(1, "scope", SCOPE_OVERALL)
    overall.insert(2, "quarter", _OVERALL_QUARTER_LABEL)

    by_quarter = metrics_table(
        complete, "actual_sum", "forecast_sum", group_cols=["model", "quarter"]
    )
    by_quarter.insert(1, "scope", SCOPE_QUARTER)

    combined = pd.concat([overall, by_quarter], ignore_index=True)[QUARTERLY_METRICS_COLUMNS]
    return (
        combined.sort_values(["model", "scope", "quarter"], kind="mergesort")
        .reset_index(drop=True)
    )


# --------------------------------------------------------------------------
# writers (issue §8: every write through ctx.out())
# --------------------------------------------------------------------------
def _write_forecast(frame: pd.DataFrame, ctx: RunContext) -> Path:
    destination = ctx.out(_repo_relative(paths.QUARTERLY_FORECAST))
    frame.to_csv(
        destination, index=False, float_format="%.6f", lineterminator="\n", encoding="utf-8"
    )
    ctx.record_artifact("quarterly_forecast", _repo_relative(paths.QUARTERLY_FORECAST))
    return destination


def _write_metrics(frame: pd.DataFrame, ctx: RunContext) -> Path:
    destination = ctx.out(_repo_relative(paths.QUARTERLY_METRICS))
    frame.to_csv(
        destination, index=False, float_format="%.6f", lineterminator="\n", encoding="utf-8"
    )
    ctx.record_artifact("quarterly_metrics", _repo_relative(paths.QUARTERLY_METRICS))
    return destination


def _write_limitation(ctx: RunContext) -> Path:
    destination = ctx.out(_repo_relative(paths.QUARTERLY_LIMITATION))
    destination.write_text(LIMITATION_TEXT, encoding="utf-8")
    ctx.record_artifact("quarterly_limitation", _repo_relative(paths.QUARTERLY_LIMITATION))
    return destination


# --------------------------------------------------------------------------
# run_quarterly_aggregation — the public entry point (issue §2/§8)
# --------------------------------------------------------------------------
def run_quarterly_aggregation(
    cfg: ModelConfig,
    ctx: RunContext,
    *,
    backtest_df: pd.DataFrame,
    latest_df: pd.DataFrame,
    panel_df: pd.DataFrame,
    cleaning_cfg: CleaningConfig,
    champion: str,
    models: list[str] | None = None,
) -> dict[str, Any]:
    """Aggregate the back-test into quarters, add the rolling estimate, score and write.

    Never re-reads ``paths.*`` mid-run: every input is a DataFrame the caller already has, which is
    what keeps this function safe to call from inside the orchestrated Flow, where the upstream
    files are still sitting under ``artifacts/_staging/<run_id>/`` (``docs/interfaces.md`` §6
    rule 7, issue §8). ``champion`` must come from ``ctx.champion`` under the Flow, never re-read
    from ``champion_decision.json`` mid-run, for the same reason.
    """
    with ctx.step(STEP_NAME):
        resolved_models = models if models is not None else default_models(champion)

        backtested = aggregate_quarterly(backtest_df, resolved_models, cfg)
        rolling = rolling_quarter_estimate(latest_df, panel_df, champion, cleaning_cfg)
        combined = (
            pd.concat([backtested, rolling], ignore_index=True)[QUARTERLY_FORECAST_COLUMNS]
            .sort_values(["quarter", "stock_code", "model"], kind="mergesort")
            .reset_index(drop=True)
        )

        metrics = quarterly_metrics(backtested)

        n_before = int(len(backtested))
        n_complete = int(backtested["complete"].sum())
        ctx.log_rows(
            "quarterly_completeness",
            before=n_before,
            removed=n_before - n_complete,
            after=n_complete,
        )

        _write_forecast(combined, ctx)
        _write_metrics(metrics, ctx)
        _write_limitation(ctx)

        overall_metrics = metrics.loc[metrics["scope"] == SCOPE_OVERALL]
        ctx.record_metrics(
            {
                "quarterly": {
                    str(row["model"]): {"wmape": float(row["wmape"]), "bias": float(row["bias"])}
                    for row in overall_metrics.to_dict(orient="records")
                }
            }
        )

    return {
        "quarterly_forecast": combined,
        "quarterly_metrics": metrics,
    }


# --------------------------------------------------------------------------
# CLI: python -m pipeline.quarterly
# --------------------------------------------------------------------------
def _read_backtest(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        dtype={
            "stock_code": "string",
            "forecast_origin": "string",
            "target_month": "string",
            "model": "string",
        },
    )


def _read_latest_forecast(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        dtype={
            "stock_code": "string",
            "forecast_origin": "string",
            "target_month": "string",
            "model": "string",
            "status": "string",
        },
    )


def run() -> int:
    """Standalone entry point: ``python -m pipeline.quarterly`` (AC 1)."""
    ctx = RunContext.start(mode="no-llm")
    try:
        cfg = load_model_config()
        cleaning_cfg = load_cleaning_config()

        required = (paths.BACKTEST_PREDICTIONS, paths.LATEST_FORECAST, paths.CLEAN_DATA)
        for path in required:
            if not path.is_file():
                raise FileNotFoundError(
                    f"{path} not found. Run `python -m pipeline.backtest`, "
                    "`python -m pipeline.latest_forecast` and `python -m pipeline.panel` first."
                )

        decision = resolve_champion(ctx)
        champion = champion_id(decision)

        backtest_df = _read_backtest(paths.BACKTEST_PREDICTIONS)
        latest_df = _read_latest_forecast(paths.LATEST_FORECAST)
        panel_df = read_panel(paths.CLEAN_DATA)

        with_step = run_quarterly_aggregation(
            cfg,
            ctx,
            backtest_df=backtest_df,
            latest_df=latest_df,
            panel_df=panel_df,
            cleaning_cfg=cleaning_cfg,
            champion=champion,
        )
        metrics = with_step["quarterly_metrics"]
        display = metrics.copy()
        display["wmape"] = display["wmape"].map(format_pct)
        display["bias"] = display["bias"].map(format_pct)
        print(display.to_string(index=False))
    except Exception:
        ctx.finish(status="failed")
        raise
    ctx.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
