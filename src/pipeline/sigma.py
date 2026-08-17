"""Robust σ — out-of-sample forecast-error spread with a product -> ABC -> global fallback
(US-20, PRD §22, §25, §26, §27, §47).

``Robust σ = mad_scale × MAD`` of **out-of-sample back-test residuals available before the
evaluation month** (CLAUDE.md §2 rule 8: robust σ only, never a plain standard deviation, never
in-sample residuals).
This is the one place that estimate is computed; the inventory simulation (US-21) and the
operational forecast (US-23) both consume this module's output rather than recomputing it.

**Eligibility, precisely.** For evaluation month ``t``, a residual is eligible only if its
``target_month < t``. An origin ``o``'s target is always ``o + 1`` (``pipeline.backtest``'s own
invariant), so "the residual's origin is before ``t``" means ``o < t − 1``, i.e.
``target_month < t`` — the forecast at evaluation month ``t`` is made from origin ``t − 1``, and
only outcomes realized strictly before that origin are legitimate history. A residual with
``target_month == t`` is the row being forecast right now and must never enter its own σ. This
resolves the PRD's "origins before t" language exactly as documented in the issue: "implemented as
``target_month < t`` (residual known at origin t−1)."

**Fallback hierarchy (§27), pooled residuals — never a mean of per-product σs.** Level 1 uses a
product's own eligible residuals when there are at least ``sigma.min_residuals_product`` of them.
Level 2 **pools** (concatenates) the eligible residuals of every product sharing the same
training-window ``abc_class`` and computes one σ over that combined multiset — this is materially
different from averaging each product's own σ, because pooling captures the between-product spread
a shared estimate is supposed to represent, while a mean of σs would only average away each
product's own noise and silently discard how differently-scaled the group's products are. Level 3
pools every eligible residual regardless of class. The three ``sigma_source`` labels are never
string literals here — they come from ``policy_cfg.sigma.fallback_levels``, which
``pipeline.config.SigmaConfig`` already validates to be exactly
``["product", "abc_group", "global"]`` in that order.

**No active-rule recomputation.** ``sigma_table()`` takes no ``features_df``; the active-product
universe for evaluation month ``t`` is read directly off ``backtest_df`` (the ``stock_code``s with
``model == model AND target_month == t``) — that row exists for every active product by
:mod:`pipeline.backtest`'s own row-set guarantee, B3 included even where its prediction is NaN.

**zero_mad.** Because ``mad_scale > 0`` is a plain multiplier, ``sigma == 0`` if and only if the MAD
that produced it was itself exactly zero (e.g. constant residuals). No extra fallback is triggered
when this happens (PRD keeps the semantics simple) — the row is written with ``sigma = 0`` and
whichever ``sigma_source`` actually produced it, with ``zero_mad = True`` so the share can be
reported.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pipeline import paths
from pipeline.config import MODEL_IDS, InventoryPolicy, load_inventory_policy, load_model_config
from pipeline.run_context import RunContext

#: Column order of ``artifacts/forecasts/sigma_table.csv``.
SIGMA_TABLE_COLUMNS: list[str] = [
    "stock_code",
    "target_month",
    "model",
    "sigma",
    "sigma_source",
    "n_residuals_product",
    "abc_class",
    "zero_mad",
]

#: Column order of ``artifacts/reports/evaluation_tables/sigma_summary.csv``.
SIGMA_SUMMARY_COLUMNS: list[str] = [
    "model",
    "target_month",
    "n_products",
    "share_product",
    "share_abc_group",
    "share_global",
    "median_sigma",
    "share_zero_mad",
]


def _repo_relative(path: Path) -> Path:
    """Repo-relative form of a canonical path constant (``docs/interfaces.md`` §6 rule 12)."""
    return path.relative_to(paths.PROJECT_ROOT)


# --------------------------------------------------------------------------
# robust_sigma_from_residuals — the one formula, never a plain standard deviation (§26)
# --------------------------------------------------------------------------
def robust_sigma_from_residuals(e: np.ndarray, mad_scale: float) -> float:
    """``mad_scale x median(|e - median(e)|)``. NaN for fewer than 1 residual, not a standard
    deviation.

    A handful of extreme residuals (e.g. one wholesale order, §47) do not inflate this the way a
    standard deviation would, because the median is insensitive to how far an outlier actually is —
    only that it exists.
    """
    array = np.asarray(e, dtype=float)
    if array.size == 0:
        return float("nan")
    median = np.median(array)
    mad = np.median(np.abs(array - median))
    return float(mad_scale * mad)


# --------------------------------------------------------------------------
# sigma_table — pure, no ctx, no disk. One model, every eval month, vectorised per month.
# --------------------------------------------------------------------------
def _sigma_for_month(
    model_rows: pd.DataFrame,
    abc_lookup: pd.DataFrame,
    target_month: str,
    model: str,
    mad_scale: float,
    min_residuals: int,
    level_product: str,
    level_abc_group: str,
    level_global: str,
) -> pd.DataFrame:
    """One evaluation month's σ for every product active at ``target_month``, fully vectorised.

    ``model_rows`` is already filtered to one model; this only ever groups by ``stock_code`` (up to
    a few thousand groups) or by ``abc_class`` (three groups) — never a per-product Python loop.
    """
    active = model_rows.loc[
        model_rows["target_month"] == target_month, ["stock_code"]
    ].drop_duplicates()
    active = active.merge(abc_lookup, on="stock_code", how="left")

    eligible = model_rows.loc[
        (model_rows["target_month"] < target_month) & model_rows["residual"].notna(),
        ["stock_code", "residual"],
    ].merge(abc_lookup, on="stock_code", how="left")

    if len(eligible):
        med_p = eligible.groupby("stock_code")["residual"].transform("median")
        mad_p = (eligible["residual"] - med_p).abs().groupby(eligible["stock_code"]).transform(
            "median"
        )
        product_level = eligible.groupby("stock_code").agg(
            n_residuals_product=("residual", "size")
        )
        product_level["sigma_product"] = (
            mad_scale * mad_p.groupby(eligible["stock_code"]).first()
        )
        # groupby(...).agg(...) leaves the group key as the index, not a column — reset it so the
        # merges below (on="stock_code" / on="abc_class") have a real column to join on.
        product_level = product_level.reset_index()

        med_a = eligible.groupby("abc_class")["residual"].transform("median")
        mad_a = (eligible["residual"] - med_a).abs().groupby(eligible["abc_class"]).transform(
            "median"
        )
        abc_level = eligible.groupby("abc_class").agg(n_residuals_abc=("residual", "size"))
        abc_level["sigma_abc"] = mad_scale * mad_a.groupby(eligible["abc_class"]).first()
        abc_level = abc_level.reset_index()

        sigma_global = robust_sigma_from_residuals(eligible["residual"].to_numpy(), mad_scale)
    else:
        product_level = pd.DataFrame(
            columns=["stock_code", "n_residuals_product", "sigma_product"]
        )
        abc_level = pd.DataFrame(columns=["abc_class", "n_residuals_abc", "sigma_abc"])
        sigma_global = float("nan")

    result = active.merge(product_level, on="stock_code", how="left").merge(
        abc_level, on="abc_class", how="left"
    )
    result["n_residuals_product"] = result["n_residuals_product"].fillna(0).astype(int)
    result["n_residuals_abc"] = result["n_residuals_abc"].fillna(0).astype(int)

    use_product = result["n_residuals_product"] >= min_residuals
    use_abc = (~use_product) & (result["n_residuals_abc"] >= min_residuals)

    sigma = np.select(
        [use_product, use_abc],
        [result["sigma_product"], result["sigma_abc"]],
        default=sigma_global,
    )
    source = np.select(
        [use_product, use_abc],
        [level_product, level_abc_group],
        default=level_global,
    )
    zero_mad = np.where(np.isnan(sigma), False, sigma == 0.0)

    return pd.DataFrame(
        {
            "stock_code": result["stock_code"],
            "target_month": target_month,
            "model": model,
            "sigma": sigma,
            "sigma_source": source,
            "n_residuals_product": result["n_residuals_product"],
            "abc_class": result["abc_class"],
            "zero_mad": zero_mad,
        }
    )


def sigma_table(
    backtest_df: pd.DataFrame,
    abc_train_df: pd.DataFrame,
    eval_months: list[str],
    model: str,
    policy_cfg: InventoryPolicy,
) -> pd.DataFrame:
    """σ for one candidate model, every evaluation month in ``eval_months`` (issue §2, pure).

    ``eval_months`` need not be the hold-out months specifically — US-23 later calls this with the
    operational month added, using every residual with ``target_month <= 2011-11``.
    """
    level_product, level_abc_group, level_global = policy_cfg.sigma.fallback_levels
    mad_scale = policy_cfg.sigma.mad_scale
    min_residuals = policy_cfg.sigma.min_residuals_product

    abc_lookup = abc_train_df[["stock_code", "abc_class"]].copy()
    abc_lookup["stock_code"] = abc_lookup["stock_code"].astype(str)

    model_rows = backtest_df.loc[backtest_df["model"] == model, :].copy()
    model_rows["stock_code"] = model_rows["stock_code"].astype(str)
    model_rows["target_month"] = model_rows["target_month"].astype(str)

    frames = [
        _sigma_for_month(
            model_rows,
            abc_lookup,
            target_month,
            model,
            mad_scale,
            min_residuals,
            level_product,
            level_abc_group,
            level_global,
        )
        for target_month in eval_months
    ]
    table = pd.concat(frames, ignore_index=True)
    return (
        table.sort_values(["target_month", "stock_code"], kind="mergesort")
        .reset_index(drop=True)[SIGMA_TABLE_COLUMNS]
    )


# --------------------------------------------------------------------------
# sigma_summary — pure, no ctx, no disk. All models/months already concatenated.
# --------------------------------------------------------------------------
def sigma_summary(table: pd.DataFrame, policy_cfg: InventoryPolicy) -> pd.DataFrame:
    """Per (model, target_month): share of products at each ``sigma_source`` level, median σ,
    share flagged ``zero_mad``. Pure aggregation of :func:`sigma_table`'s output.
    """
    level_product, level_abc_group, level_global = policy_cfg.sigma.fallback_levels

    rows = []
    for (model_id, month), group in table.groupby(["model", "target_month"], sort=False):
        n = len(group)
        counts = group["sigma_source"].value_counts()
        rows.append(
            {
                "model": model_id,
                "target_month": month,
                "n_products": n,
                "share_product": float(counts.get(level_product, 0)) / n if n else float("nan"),
                "share_abc_group": (
                    float(counts.get(level_abc_group, 0)) / n if n else float("nan")
                ),
                "share_global": float(counts.get(level_global, 0)) / n if n else float("nan"),
                "median_sigma": float(group["sigma"].median()),
                "share_zero_mad": float(group["zero_mad"].mean()) if n else float("nan"),
            }
        )
    return pd.DataFrame(rows, columns=SIGMA_SUMMARY_COLUMNS)


# --------------------------------------------------------------------------
# writers — through ctx.out() (issue §8)
# --------------------------------------------------------------------------
def write_sigma_outputs(
    table: pd.DataFrame, summary: pd.DataFrame, ctx: RunContext
) -> tuple[Path, Path]:
    """Write ``sigma_table.csv`` and ``sigma_summary.csv`` through ``ctx.out()``."""
    table_destination = ctx.out(_repo_relative(paths.SIGMA_TABLE))
    table.to_csv(
        table_destination, index=False, float_format="%.6f", lineterminator="\n", encoding="utf-8"
    )
    ctx.record_artifact("sigma_table", _repo_relative(paths.SIGMA_TABLE))

    summary_canonical = paths.EVAL_TABLES_DIR / "sigma_summary.csv"
    summary_destination = ctx.out(_repo_relative(summary_canonical))
    summary.to_csv(
        summary_destination,
        index=False,
        float_format="%.6f",
        lineterminator="\n",
        encoding="utf-8",
    )
    ctx.record_artifact("sigma_summary", _repo_relative(summary_canonical))

    return table_destination, summary_destination


# --------------------------------------------------------------------------
# CLI: python -m pipeline.sigma
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


def run() -> int:
    """Standalone entry point: ``python -m pipeline.sigma`` (AC 1)."""
    ctx = RunContext.start(mode="no-llm")
    try:
        abc_path = paths.EVAL_TABLES_DIR / "abc_train.csv"
        for required in (paths.BACKTEST_PREDICTIONS, abc_path):
            if not required.is_file():
                raise FileNotFoundError(
                    f"{required} not found. Run `python -m pipeline.backtest` and "
                    "`python -m pipeline.split abc` first."
                )

        cfg = load_model_config()
        policy_cfg = load_inventory_policy()
        backtest_df = _read_backtest(paths.BACKTEST_PREDICTIONS)
        abc_train_df = pd.read_csv(abc_path, dtype={"stock_code": "string"})

        eval_months = [
            str(period)
            for period in pd.period_range(
                cfg.split.holdout_targets.start, cfg.split.holdout_targets.end, freq="M"
            )
        ]

        with ctx.step("sigma"):
            frames = [
                sigma_table(backtest_df, abc_train_df, eval_months, model_id, policy_cfg)
                for model_id in MODEL_IDS
            ]
            table = (
                pd.concat(frames, ignore_index=True)
                .sort_values(["model", "target_month", "stock_code"], kind="mergesort")
                .reset_index(drop=True)
            )
            summary = sigma_summary(table, policy_cfg)

            ctx.log_rows("sigma_table", before=len(table), removed=0, after=len(table))

            write_sigma_outputs(table, summary, ctx)

            zero_mad_count = int(table["zero_mad"].sum())
            if zero_mad_count:
                ctx.warn(
                    f"{zero_mad_count} row(s) have MAD == 0 (zero_mad=True); sigma recorded as 0 "
                    "with no extra fallback"
                )

            metrics = {}
            for model_id, group in summary.groupby("model", sort=False):
                metrics[model_id] = {
                    "share_product": float(group["share_product"].mean()),
                    "share_abc_group": float(group["share_abc_group"].mean()),
                    "share_global": float(group["share_global"].mean()),
                }
            ctx.record_metrics({"sigma": metrics})

        for model_id, group in summary.groupby("model", sort=False):
            print(
                f"{model_id}: product={group['share_product'].mean():.2%} "
                f"abc_group={group['share_abc_group'].mean():.2%} "
                f"global={group['share_global'].mean():.2%}"
            )
    except Exception:
        ctx.finish(status="failed")
        raise
    ctx.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
