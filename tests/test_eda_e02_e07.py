"""Descriptive EDA tests, E2–E7 (US-09, PRD §35A.1, §35A.2, §8, §18.2, §40, §55).

Structure, arithmetic identities and the leakage/partial-month rules only. The PRD's indicative
findings (Sep–Nov ≈ 35 % of units, UK ≈ 82 %) are expectations to compare a run against, never
assertions — pinning them would fail the moment the dataset is re-downloaded.

Every write goes to a ``tmp_path`` base directory, so no test touches the real ``artifacts/``.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from pipeline import paths
from pipeline.abc import ABC_CLASSES
from pipeline.cleaning import clean_transactions
from pipeline.config import load_cleaning_config, load_inventory_policy
from pipeline.download import load_raw
from pipeline.eda import e03_calendar_patterns, e06_abc, e07_top_products
from pipeline.eda.periods import MONTHS_PER_YEAR, add_months, full_years
from pipeline.eda.run_analyses import ANALYSES, index_path, run_analyses
from pipeline.panel import build_panel
from pipeline.run_context import RunContext, close_log_handlers

RAW_SAMPLE = paths.FIXTURES_DIR / "raw_sample.csv"
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"

EXPECTED_TABLES = {
    "E2": ["E02_monthly_units_revenue", "E02_seasonal_index", "E02_yoy", "E02_sep_nov_share"],
    "E3": ["E03_weekday", "E03_hour"],
    "E4": ["E04_products_per_month", "E04_new_disappearing_by_quarter"],
    "E5": ["E05_lifecycle", "E05_lifecycle_summary"],
    "E6": ["E06_abc_table", "E06_pareto"],
    "E7": ["E07_top20_units", "E07_top20_revenue"],
}

EXPECTED_FIGURES = {
    "E2": ["E02_monthly_units", "E02_seasonal_index"],
    "E3": ["E03_weekday", "E03_hour"],
    "E4": ["E04_products_per_month"],
    "E5": ["E05_lifecycle_hist"],
    "E6": ["E06_pareto"],
    "E7": ["E07_top20_units", "E07_top20_revenue"],
}

ALL_IDS = ["E2", "E3", "E4", "E5", "E6", "E7"]


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def sample_run(tmp_path_factory: pytest.TempPathFactory) -> SimpleNamespace:
    """One E2–E7 run over the committed CI fixture, shared by every structural test."""
    base = tmp_path_factory.mktemp("eda_e02_e07")
    cfg = load_cleaning_config()
    raw, _ = load_raw(RAW_SAMPLE)
    ctx = RunContext.start(mode="no-llm", base_dir=base)
    try:
        with ctx.step("eda_e02_e07"):
            clean_df, _ = clean_transactions(raw, cfg, ctx)
            returns = pd.read_parquet(base / "data/processed/returns_lines.parquet")
            panel_df = build_panel(clean_df, returns, cfg, ctx)
            results = run_analyses(ALL_IDS, clean_df, panel_df, cfg, ctx)
        ctx.finish()
    finally:
        close_log_handlers(ctx.run_id)
    return SimpleNamespace(
        base=base, ctx=ctx, clean=clean_df, panel=panel_df, cfg=cfg, results=results
    )


@pytest.fixture
def ctx(tmp_path: Path):
    context = RunContext.start(mode="no-llm", base_dir=tmp_path)
    yield context
    close_log_handlers(context.run_id)


def _tables_dir(base: Path) -> Path:
    return base / paths.EDA_TABLES_DIR.relative_to(paths.PROJECT_ROOT)


def _figures_dir(base: Path) -> Path:
    return base / paths.FIGURES_DIR.relative_to(paths.PROJECT_ROOT)


# --------------------------------------------------------------------------
# every analysis returns and writes what it promises
# --------------------------------------------------------------------------
@pytest.mark.parametrize("analysis_id", ALL_IDS)
def test_run_returns_the_declared_tables_and_figures(
    sample_run: SimpleNamespace, analysis_id: str
) -> None:
    result = sample_run.results[analysis_id]
    assert sorted(result["tables"]) == sorted(EXPECTED_TABLES[analysis_id])
    assert sorted(result["figures"]) == sorted(EXPECTED_FIGURES[analysis_id])


@pytest.mark.parametrize("analysis_id", ALL_IDS)
def test_every_figure_file_exists_and_is_a_png(
    sample_run: SimpleNamespace, analysis_id: str
) -> None:
    for name in EXPECTED_FIGURES[analysis_id]:
        figure = _figures_dir(sample_run.base) / f"{name}.png"
        assert figure.is_file(), f"missing figure: {name}"
        assert figure.read_bytes().startswith(PNG_MAGIC)


@pytest.mark.parametrize("analysis_id", ALL_IDS)
def test_every_table_file_exists(sample_run: SimpleNamespace, analysis_id: str) -> None:
    tables = _tables_dir(sample_run.base)
    for name in EXPECTED_TABLES[analysis_id]:
        written = list(tables.glob(f"{name}.*"))
        assert written, f"missing table: {name}"


def test_no_artifact_uses_windows_line_endings(sample_run: SimpleNamespace) -> None:
    """Same bytes on Windows and in CI (§40)."""
    for path in sorted(_tables_dir(sample_run.base).glob("E0*.*")):
        assert b"\r\n" not in path.read_bytes(), f"CRLF line endings in {path.name}"


# --------------------------------------------------------------------------
# E2 — arithmetic identities
# --------------------------------------------------------------------------
def test_seasonal_index_averages_to_one(sample_run: SimpleNamespace) -> None:
    """Shares of a year sum to 1, so twelve indices scaled by 12 must average exactly 1.0."""
    seasonal = sample_run.results["E2"]["tables"]["E02_seasonal_index"]
    assert len(seasonal) == MONTHS_PER_YEAR
    assert seasonal["seasonal_index"].mean() == pytest.approx(1.0)
    assert seasonal["mean_share"].sum() == pytest.approx(1.0)


def test_monthly_table_covers_every_month_and_flags_only_the_partial_one(
    sample_run: SimpleNamespace,
) -> None:
    monthly = sample_run.results["E2"]["tables"]["E02_monthly_units_revenue"]
    cfg = sample_run.cfg
    assert list(monthly["month"]) == sorted(sample_run.clean["month"].astype(str).unique())
    assert monthly["month"].is_monotonic_increasing
    flagged = set(monthly.loc[monthly["is_partial"], "month"])
    assert flagged == set(cfg.raw.partial_months) & set(monthly["month"])


def test_yoy_covers_the_december_to_november_windows(sample_run: SimpleNamespace) -> None:
    """The two full years are Dec→Nov, not calendar years (§35A E2)."""
    yoy = sample_run.results["E2"]["tables"]["E02_yoy"]
    windows = full_years(sample_run.cfg)
    assert len(yoy) == len(windows)
    for (start, end), row in zip(windows, yoy.itertuples(), strict=True):
        assert row.first_month == start
        assert row.last_month == end
        assert add_months(start, MONTHS_PER_YEAR - 1) == end
    assert pd.isna(yoy["units_growth"].iloc[0])  # nothing to compare the first year against


def test_yoy_and_peak_share_exclude_the_partial_month(sample_run: SimpleNamespace) -> None:
    """A partial month may never enter a total (§8)."""
    monthly = sample_run.results["E2"]["tables"]["E02_monthly_units_revenue"]
    yoy = sample_run.results["E2"]["tables"]["E02_yoy"]
    partial_units = int(monthly.loc[monthly["is_partial"], "units"].sum())
    assert partial_units > 0, "fixture should contain the partial month"
    assert int(yoy["units"].sum()) == int(monthly["units"].sum()) - partial_units


def test_peak_share_is_a_fraction_and_has_a_pooled_row(sample_run: SimpleNamespace) -> None:
    peak = sample_run.results["E2"]["tables"]["E02_sep_nov_share"]
    assert "pooled" in set(peak["year"])
    assert ((peak["units_share"] >= 0) & (peak["units_share"] <= 1)).all()
    assert ((peak["revenue_share"] >= 0) & (peak["revenue_share"] <= 1)).all()


def test_sep_nov_share_json_is_a_records_array(sample_run: SimpleNamespace) -> None:
    written = _tables_dir(sample_run.base) / "E02_sep_nov_share.json"
    payload = json.loads(written.read_text(encoding="utf-8"))
    assert isinstance(payload, list)
    assert {"year", "units_share", "revenue_share"} <= set(payload[0])


# --------------------------------------------------------------------------
# E3
# --------------------------------------------------------------------------
def test_weekday_and_hour_totals_match_the_transactions(sample_run: SimpleNamespace) -> None:
    weekday = sample_run.results["E3"]["tables"]["E03_weekday"]
    hour = sample_run.results["E3"]["tables"]["E03_hour"]
    total_units = int(sample_run.clean["quantity"].sum())
    assert int(weekday["units"].sum()) == total_units
    assert int(hour["units"].sum()) == total_units
    assert int(weekday["lines"].sum()) == len(sample_run.clean)
    assert weekday["weekday"].between(0, len(e03_calendar_patterns.WEEKDAY_NAMES) - 1).all()
    assert hour["hour"].between(0, 23).all()


# --------------------------------------------------------------------------
# E4
# --------------------------------------------------------------------------
def test_products_per_month_counts_only_selling_products(sample_run: SimpleNamespace) -> None:
    per_month = sample_run.results["E4"]["tables"]["E04_products_per_month"]
    panel = sample_run.panel
    expected = (
        panel.loc[panel["units_sold"] > 0]
        .groupby(panel.loc[panel["units_sold"] > 0, "month"].astype(str))["stock_code"]
        .nunique()
    )
    assert list(per_month["products"]) == [int(expected[month]) for month in per_month["month"]]


def test_final_quarter_reports_no_disappearing_count(sample_run: SimpleNamespace) -> None:
    """The last quarter is right-censored, so its disappearing cell stays empty (§35A E4)."""
    churn = sample_run.results["E4"]["tables"]["E04_new_disappearing_by_quarter"]
    censored = churn.loc[churn["is_right_censored"]]
    assert len(censored) == 1
    assert censored["disappearing_products"].isna().all()
    assert churn.loc[~churn["is_right_censored"], "disappearing_products"].notna().all()


def test_first_quarter_is_flagged_left_censored(sample_run: SimpleNamespace) -> None:
    """Products already selling in the first quarter are not launches (glossary, §18.1)."""
    churn = sample_run.results["E4"]["tables"]["E04_new_disappearing_by_quarter"]
    left = churn.loc[churn["is_left_censored"]]
    assert len(left) == 1
    assert left.index[0] == churn.index[0]
    # By construction the censored quarter absorbs the whole opening catalogue, so it dwarfs
    # every later quarter — which is exactly why the figure clips it.
    assert int(left["new_products"].iloc[0]) > int(
        churn.loc[~churn["is_left_censored"], "new_products"].max()
    )


def test_new_products_sum_to_the_catalogue(sample_run: SimpleNamespace) -> None:
    """Every product is new exactly once — in the quarter it first sold."""
    churn = sample_run.results["E4"]["tables"]["E04_new_disappearing_by_quarter"]
    panel = sample_run.panel
    products = panel.loc[panel["units_sold"] > 0, "stock_code"].nunique()
    assert int(churn["new_products"].sum()) == products


# --------------------------------------------------------------------------
# E5
# --------------------------------------------------------------------------
def test_lifecycle_spans_are_consistent(sample_run: SimpleNamespace) -> None:
    lifecycle = sample_run.results["E5"]["tables"]["E05_lifecycle"]
    assert (lifecycle["selling_months"] >= 1).all()
    # A product cannot sell in more months than its span covers.
    assert (lifecycle["selling_months"] <= lifecycle["span_months"]).all()
    assert (lifecycle["first_month"] <= lifecycle["last_month"]).all()


def test_lifecycle_summary_shares_are_fractions(sample_run: SimpleNamespace) -> None:
    summary = sample_run.results["E5"]["tables"]["E05_lifecycle_summary"]
    assert len(summary) == 1
    row = summary.iloc[0]
    assert row["products"] == len(sample_run.results["E5"]["tables"]["E05_lifecycle"])
    assert 0.0 <= row["short_lifecycle_share"] <= 1.0
    assert 0.0 <= row["long_lifecycle_share"] <= 1.0


# --------------------------------------------------------------------------
# E6
# --------------------------------------------------------------------------
def test_abc_table_has_three_classes_covering_every_product(
    sample_run: SimpleNamespace,
) -> None:
    table = sample_run.results["E6"]["tables"]["E06_abc_table"]
    pareto = sample_run.results["E6"]["tables"]["E06_pareto"]
    assert list(table["abc_class"]) == list(ABC_CLASSES)
    assert int(table["products"].sum()) == len(pareto)
    assert table["revenue_share"].sum() == pytest.approx(1.0)
    assert table["product_share"].sum() == pytest.approx(1.0)


def test_abc_respects_the_configured_cut_offs(sample_run: SimpleNamespace) -> None:
    """Class A is the head up to ``a_cum_share``; the product crossing it falls to B."""
    pareto = sample_run.results["E6"]["tables"]["E06_pareto"]
    thresholds = load_inventory_policy().abc
    a_rows = pareto.loc[pareto["abc_class"] == "A"]
    if len(a_rows):
        assert a_rows["cum_revenue_share"].max() <= thresholds.a_cum_share + 1e-6
    b_rows = pareto.loc[pareto["abc_class"] == "B"]
    if len(b_rows):
        assert b_rows["cum_revenue_share"].max() <= thresholds.b_cum_share + 1e-6


def test_abc_excludes_the_partial_month(sample_run: SimpleNamespace) -> None:
    """E6 cuts at the last full month, so December 2011 revenue is not classified (§8)."""
    pareto = sample_run.results["E6"]["tables"]["E06_pareto"]
    panel = sample_run.panel
    full = panel.loc[panel["month"].astype(str) <= sample_run.cfg.raw.last_full_month]
    assert pareto["revenue"].sum() == pytest.approx(float(full["gross_revenue"].sum()))


def test_abc_table_is_labelled_full_period(sample_run: SimpleNamespace) -> None:
    """The descriptive classes must never be mistaken for the training-window ones (§18.2)."""
    table = sample_run.results["E6"]["tables"]["E06_abc_table"]
    assert set(table["window"]) == {e06_abc.WINDOW_LABEL}


# --------------------------------------------------------------------------
# E7
# --------------------------------------------------------------------------
def test_top20_tables_are_ranked_and_capped(sample_run: SimpleNamespace) -> None:
    for name, column in (("E07_top20_units", "units"), ("E07_top20_revenue", "revenue")):
        ranked = sample_run.results["E7"]["tables"][name]
        assert len(ranked) == e07_top_products.TOP_N
        assert list(ranked["rank"]) == list(range(1, e07_top_products.TOP_N + 1))
        assert ranked[column].is_monotonic_decreasing
        assert ((ranked["share"] > 0) & (ranked["share"] <= 1)).all()


def test_top20_excludes_the_partial_month(sample_run: SimpleNamespace) -> None:
    ranked = sample_run.results["E7"]["tables"]["E07_top20_units"]
    panel = sample_run.panel
    full = panel.loc[~panel["month"].astype(str).isin(sample_run.cfg.raw.partial_months)]
    top = ranked.iloc[0]
    rows = full.loc[full["stock_code"].astype(str) == top["stock_code"], "units_sold"]
    assert int(top["units"]) == int(rows.sum())


# --------------------------------------------------------------------------
# the index
# --------------------------------------------------------------------------
def test_index_lists_every_analysis_that_ran(sample_run: SimpleNamespace) -> None:
    written = sample_run.base / index_path()
    entries = json.loads(written.read_text(encoding="utf-8"))
    assert [entry["id"] for entry in entries] == ALL_IDS
    for entry in entries:
        assert entry["table_names"] == sorted(EXPECTED_TABLES[entry["id"]])
        assert entry["figure_names"] == sorted(EXPECTED_FIGURES[entry["id"]])
        assert entry["one_line_summary_placeholder"] == ""


def test_index_merges_instead_of_overwriting(sample_run: SimpleNamespace, ctx: RunContext) -> None:
    """Running one analysis must not delete the entries the others wrote (US-11 needs all 14)."""
    with ctx.step("eda_partial"):
        run_analyses(["E6"], sample_run.clean, sample_run.panel, sample_run.cfg, ctx)
        first = json.loads((ctx.base_dir / index_path()).read_text(encoding="utf-8"))
        assert [entry["id"] for entry in first] == ["E6"]

        run_analyses(["E3"], sample_run.clean, sample_run.panel, sample_run.cfg, ctx)
        second = json.loads((ctx.base_dir / index_path()).read_text(encoding="utf-8"))
    assert [entry["id"] for entry in second] == ["E3", "E6"]


def test_unknown_analysis_id_is_rejected(sample_run: SimpleNamespace, ctx: RunContext) -> None:
    with pytest.raises(ValueError, match="unknown analysis id"):
        with ctx.step("eda_bad_id"):
            run_analyses(["E99"], sample_run.clean, sample_run.panel, sample_run.cfg, ctx)


def test_registry_covers_e2_to_e7() -> None:
    """E2-E7 are registered, in order. Later issues append to the registry, so this is a
    containment check rather than an equality one — US-10 adds E8-E14 to the same tuple."""
    registered = [analysis.id for analysis in ANALYSES]
    assert registered[: len(ALL_IDS)] == ALL_IDS
    assert len(set(registered)) == len(registered), "no analysis may be registered twice"


# --------------------------------------------------------------------------
# staging (§39)
# --------------------------------------------------------------------------
def test_writes_are_staged_until_promotion(tmp_path: Path) -> None:
    cfg = load_cleaning_config()
    context = RunContext.start(mode="no-llm", staging=True, base_dir=tmp_path)
    try:
        with context.step("eda_e02_e07"):
            raw, _ = load_raw(RAW_SAMPLE)
            clean_df, _ = clean_transactions(raw, cfg, context)
            returns = pd.read_parquet(
                context.staging_dir / "data/processed/returns_lines.parquet"
            )
            panel_df = build_panel(clean_df, returns, cfg, context)
            run_analyses(["E6"], clean_df, panel_df, cfg, context)

        staged = context.staging_dir / paths.EDA_TABLES_DIR.relative_to(paths.PROJECT_ROOT)
        assert (staged / "E06_abc_table.csv").is_file()
        assert not (_tables_dir(tmp_path) / "E06_abc_table.csv").exists()

        context.promote()
        assert (_tables_dir(tmp_path) / "E06_abc_table.csv").is_file()
    finally:
        close_log_handlers(context.run_id)
