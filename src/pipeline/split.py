"""Temporal split, rolling origins & training-window ABC (US-16, PRD §18.1, §18.2, §19, §21, §22,
§23, §27).

Scikit-learn's shuffled train/test splitter is never imported under ``src/`` — the split in this
project is **temporal** (§21): older months train, the newest months test, and nothing is ever
shuffled. This module is the single place that encodes the split, the rolling back-test origins
and the training-window ABC groups, so that every step and every report — baselines (this issue),
model training (US-17), the back-test (US-18), evaluation (US-19), σ (US-24) and inventory KPIs
(US-25) — score the identical rows against the identical groups.

``SplitSpec`` is a thin wrapper over :class:`pipeline.config.SplitConfig` and
:class:`pipeline.config.BacktestConfig` — every validation the split needs already runs there
(``docs/interfaces.md`` §2); this module must not re-parse ``model_config.yaml`` or restate a rule.

**The ABC product universe.** :func:`pipeline.abc.compute_abc` only returns products observed at or
before ``through_month`` — a product launched after the training window had not been seen yet, so
classifying it from revenue would itself be a leak. :func:`abc_train` still needs an answer for
every product evaluation might join on (a product first sold in 2011-08, say, is still active and
scored in the hold-out), so it takes the product universe from the **panel**
(``clean_data.csv``) — a superset of anything downstream will ever look up — and gives those
products class **C** with ``abc_source = "default_no_train_history"``. Products observed in the
training window get ``abc_source = "training_window"``; every row gets an explicit value, never an
empty cell. The added rows get ``revenue_share = 0.0`` and ``cum_share = 0.0`` (not ``NaN``) because
they earned no revenue in the window that was actually classified.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from pydantic import BaseModel, ConfigDict

from pipeline import paths
from pipeline.abc import compute_abc
from pipeline.config import (
    BacktestConfig,
    InventoryPolicy,
    SplitConfig,
    load_inventory_policy,
    load_model_config,
)
from pipeline.contract import read_panel
from pipeline.run_context import RunContext

#: ``abc_train.csv`` column order (issue §8: §6 of the issue wins over §2 — five columns).
ABC_TRAIN_COLUMNS: list[str] = [
    "stock_code",
    "abc_class",
    "revenue_share",
    "cum_share",
    "abc_source",
]

#: ``abc_source`` values written to ``abc_train.csv`` — every row gets one, never an empty cell.
ABC_SOURCE_TRAINING_WINDOW = "training_window"
ABC_SOURCE_DEFAULT_NO_HISTORY = "default_no_train_history"

#: The class assigned to a product absent from the training window (no revenue observed there).
_DEFAULT_ABC_CLASS = "C"


def _repo_relative(path: Path) -> Path:
    """Repo-relative form of a canonical path constant (``docs/interfaces.md`` §6 rule 12)."""
    return path.relative_to(paths.PROJECT_ROOT)


class SplitSpec(BaseModel):
    """One object carrying the §21/§22 temporal design — a thin wrapper, not a second parser.

    Every validation this class would need (train end < hold-out start, validation ⊂ train,
    ``never_score`` disjoint from hold-out, ``first_target_month == train_targets.start``, origin
    ordering) already runs inside :class:`pipeline.config.SplitConfig` and
    :class:`pipeline.config.BacktestConfig`. Restating any of it here would let the two copies
    drift, so :meth:`load` only reads ``load_model_config()`` and holds the two typed sections.
    """

    model_config = ConfigDict(frozen=True)

    split: SplitConfig
    backtest: BacktestConfig

    @classmethod
    def load(cls) -> SplitSpec:
        """Build the spec from ``model_config.yaml`` (via the cached, validated loader)."""
        cfg = load_model_config()
        return cls(split=cfg.split, backtest=cfg.backtest)


def _month_column(frame: pd.DataFrame) -> pd.Series:
    return frame["target_month"].astype(str)


def train_mask(features_df: pd.DataFrame, spec: SplitSpec) -> pd.Series:
    """Rows whose ``target_month`` falls inside the training targets (§21)."""
    months = _month_column(features_df)
    return months.between(spec.split.train_targets.start, spec.split.train_targets.end)


def holdout_mask(features_df: pd.DataFrame, spec: SplitSpec) -> pd.Series:
    """Rows whose ``target_month`` falls inside the hold-out (§21). December 2011 is never here."""
    months = _month_column(features_df)
    return months.between(spec.split.holdout_targets.start, spec.split.holdout_targets.end)


def validation_mask(features_df: pd.DataFrame, spec: SplitSpec) -> pd.Series:
    """Rows whose ``target_month`` falls inside the internal validation folds (§21)."""
    months = _month_column(features_df)
    return months.between(spec.split.validation_targets.start, spec.split.validation_targets.end)


def rolling_origins(spec: SplitSpec) -> list[str]:
    """Every back-test origin ``first_origin … last_origin`` as ``YYYY-MM`` strings (§22)."""
    return [
        str(period)
        for period in pd.period_range(
            spec.backtest.first_origin, spec.backtest.last_origin, freq="M"
        )
    ]


def validation_origins(spec: SplitSpec) -> list[str]:
    """The subset of :func:`rolling_origins` whose target month is a validation target (§21)."""
    val_start = spec.split.validation_targets.start
    val_end = spec.split.validation_targets.end
    origins = []
    for origin in rolling_origins(spec):
        target = str(pd.Period(origin, freq="M") + 1)
        if val_start <= target <= val_end:
            origins.append(origin)
    return origins


def training_window_end(spec: SplitSpec) -> str:
    """Last training target month — the ABC leakage boundary for modelling/evaluation/σ (§18.2)."""
    return spec.split.train_targets.end


def abc_train(panel_df: pd.DataFrame, spec: SplitSpec, policy_cfg: InventoryPolicy) -> pd.DataFrame:
    """Training-window ABC classes for every product the panel has ever seen (§18.2, §23, §27).

    Pure — returns a frame, touches no disk; :func:`write_abc_train` performs the write. This is
    the **only** ABC table used for evaluation-by-group, σ fallback and inventory KPIs; the
    full-period ABC of E6 is descriptive only and must stay labelled as such (module docstring).
    """
    through_month = training_window_end(spec)
    abc = compute_abc(
        panel_df,
        through_month=through_month,
        a_cum_share=policy_cfg.abc.a_cum_share,
        b_cum_share=policy_cfg.abc.b_cum_share,
    )
    in_window = abc.loc[:, ["stock_code", "abc_class", "revenue_share", "cum_share"]].copy()
    in_window["stock_code"] = in_window["stock_code"].astype(str)
    in_window["abc_source"] = ABC_SOURCE_TRAINING_WINDOW

    universe = pd.Index(sorted(panel_df["stock_code"].astype(str).unique()))
    missing = universe.difference(pd.Index(in_window["stock_code"]))
    if len(missing):
        added = pd.DataFrame(
            {
                "stock_code": sorted(missing),
                "abc_class": _DEFAULT_ABC_CLASS,
                "revenue_share": 0.0,
                "cum_share": 0.0,
                "abc_source": ABC_SOURCE_DEFAULT_NO_HISTORY,
            }
        )
        in_window = pd.concat([in_window, added], ignore_index=True)

    return (
        in_window.sort_values("stock_code", kind="mergesort")
        .reset_index(drop=True)[ABC_TRAIN_COLUMNS]
    )


def write_abc_train(frame: pd.DataFrame, ctx: RunContext) -> Path:
    """Write ``artifacts/reports/evaluation_tables/abc_train.csv`` through ``ctx.out()``."""
    canonical = paths.EVAL_TABLES_DIR / "abc_train.csv"
    destination = ctx.out(_repo_relative(canonical))
    frame.to_csv(
        destination,
        index=False,
        float_format="%.6f",
        lineterminator="\n",
        encoding="utf-8",
    )
    ctx.record_artifact("abc_train", _repo_relative(canonical))
    return destination


def run(argv: list[str] | None = None) -> int:
    """Standalone entry point: ``python -m pipeline.split abc`` (AC 1)."""
    args = list(sys.argv[1:] if argv is None else argv)
    if args != ["abc"]:
        print("usage: python -m pipeline.split abc", file=sys.stderr)
        return 2

    ctx = RunContext.start(mode="no-llm")
    try:
        with ctx.step("abc_train"):
            if not paths.CLEAN_DATA.is_file():
                raise FileNotFoundError(
                    f"{paths.CLEAN_DATA} not found. Run `python -m pipeline.panel` first."
                )
            panel = read_panel(paths.CLEAN_DATA)
            spec = SplitSpec.load()
            policy_cfg = load_inventory_policy()

            products = int(panel["stock_code"].nunique())
            frame = abc_train(panel, spec, policy_cfg)
            ctx.log_rows("abc_train", before=products, removed=0, after=int(len(frame)))

            counts = frame["abc_class"].value_counts()
            default_no_history = int((frame["abc_source"] == ABC_SOURCE_DEFAULT_NO_HISTORY).sum())
            write_abc_train(frame, ctx)
            ctx.record_metrics(
                {
                    "abc_train_products": int(len(frame)),
                    "abc_train_class_a": int(counts.get("A", 0)),
                    "abc_train_class_b": int(counts.get("B", 0)),
                    "abc_train_class_c": int(counts.get("C", 0)),
                    "abc_train_default_no_history": default_no_history,
                }
            )
        print(
            f"products: {ctx.metrics['abc_train_products']}\n"
            f"class A: {ctx.metrics['abc_train_class_a']}\n"
            f"class B: {ctx.metrics['abc_train_class_b']}\n"
            f"class C: {ctx.metrics['abc_train_class_c']}\n"
            f"default_no_train_history: {ctx.metrics['abc_train_default_no_history']}"
        )
    except Exception:
        ctx.finish(status="failed")
        raise
    ctx.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
