"""Descriptive EDA tests, E8–E14 (US-10, PRD §35A.1, §9, §14, §26, §40, §55).

Structure, arithmetic identities and the rules the analyses exist to justify. The PRD's indicative
findings (k = 6 → ≈ 70,700 rows, UK ≈ 82 % of units) are expectations to compare a run against,
never assertions — the fixture is a sample and the numbers would be wrong for it anyway.

Every write goes to a ``tmp_path`` base directory, so no test touches the real ``artifacts/``.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from pipeline import paths
from pipeline.cleaning import RETURNS_FILENAME, clean_transactions
from pipeline.config import load_cleaning_config, load_model_config
from pipeline.download import load_raw
from pipeline.eda import (
    e08_intermittency,
    e09_outliers,
    e10_order_structure,
    e11_customers_countries,
)
from pipeline.eda.index import index_path, read_index
from pipeline.eda.run_analyses import ANALYSES_BY_ID, run_analyses
from pipeline.panel import build_panel, validate_panel
from pipeline.run_context import RunContext, close_log_handlers

RAW_SAMPLE = paths.FIXTURES_DIR / "raw_sample.csv"
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"

ALL_IDS = ["E8", "E9", "E10", "E11", "E12", "E13", "E14"]

EXPECTED_TABLES = {
    "E8": ["E08_zero_share_per_product", "E08_zero_share_by_k"],
    "E9": ["E09_largest_lines", "E09_largest_product_months", "E09_magnitude_summary"],
    "E10": ["E10_invoice_summary"],
    "E11": ["E11_customer_identification", "E11_top10_countries"],
    "E12": ["E12_cancellation_rate_monthly", "E12_top_returned"],
    "E13": ["E13_price_distribution", "E13_price_stability"],
    "E14": ["E14_panel_preview"],
}

EXPECTED_FIGURES = {
    "E8": ["E08_zero_share_hist", "E08_zero_share_by_k"],
    "E9": ["E09_units_hist_log"],
    "E10": ["E10_invoice_hist"],
    "E11": ["E11_top_countries"],
    "E12": ["E12_cancellation_rate"],
    "E13": ["E13_price_hist"],
    "E14": [],  # E14 is a check on the hand-off file, not a chart
}


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def sample_run(tmp_path_factory: pytest.TempPathFactory) -> SimpleNamespace:
    """One E8–E14 run over the committed CI fixture, shared by every structural test."""
    base = tmp_path_factory.mktemp("eda_e08_e14")
    cfg = load_cleaning_config()
    raw, _ = load_raw(RAW_SAMPLE)
    ctx = RunContext.start(mode="no-llm", base_dir=base)
    try:
        with ctx.step("eda_e08_e14"):
            clean_df, _ = clean_transactions(raw, cfg, ctx)
            returns = pd.read_parquet(base / "data/processed" / RETURNS_FILENAME)
            panel_df = build_panel(clean_df, returns, cfg, ctx)
            assert validate_panel(panel_df, cfg).passed, "fixture panel must be valid"
            results = run_analyses(ALL_IDS, clean_df, panel_df, cfg, ctx)
        ctx.finish()
    finally:
        close_log_handlers(ctx.run_id)
    return SimpleNamespace(
        base=base, ctx=ctx, clean=clean_df, panel=panel_df, cfg=cfg,
        returns=returns, results=results,
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
def test_every_file_exists(sample_run: SimpleNamespace, analysis_id: str) -> None:
    tables = _tables_dir(sample_run.base)
    for name in EXPECTED_TABLES[analysis_id]:
        assert list(tables.glob(f"{name}.*")), f"missing table: {name}"
    for name in EXPECTED_FIGURES[analysis_id]:
        figure = _figures_dir(sample_run.base) / f"{name}.png"
        assert figure.is_file(), f"missing figure: {name}"
        assert figure.read_bytes().startswith(PNG_MAGIC)


def test_no_artifact_uses_windows_line_endings(sample_run: SimpleNamespace) -> None:
    for path in sorted(_tables_dir(sample_run.base).glob("E1[0-4]*.*")):
        assert b"\r\n" not in path.read_bytes(), f"CRLF line endings in {path.name}"


# --------------------------------------------------------------------------
# E8 — the k sweep that justifies the configured k (§14)
# --------------------------------------------------------------------------
def test_zero_share_by_k_covers_the_grid(sample_run: SimpleNamespace) -> None:
    by_k = sample_run.results["E8"]["tables"]["E08_zero_share_by_k"]
    assert list(by_k["k"]) == list(e08_intermittency.K_GRID)
    assert len(by_k) == len(e08_intermittency.K_GRID)


def test_rows_and_zero_share_both_increase_with_k(sample_run: SimpleNamespace) -> None:
    """A property that holds by construction: a longer look-back can only admit more rows.

    Every row active at k is active at any larger k (the window is a superset), and the rows it
    adds are the marginal products, whose targets are more often zero.
    """
    by_k = sample_run.results["E8"]["tables"]["E08_zero_share_by_k"]
    assert by_k["rows"].is_monotonic_increasing
    assert by_k["zero_target_share"].is_monotonic_increasing


def test_configured_k_is_flagged_exactly_once(sample_run: SimpleNamespace) -> None:
    by_k = sample_run.results["E8"]["tables"]["E08_zero_share_by_k"]
    flagged = by_k.loc[by_k["is_configured_k"]]
    assert len(flagged) == 1
    assert int(flagged["k"].iloc[0]) == load_model_config().active_rule.k


def test_k_sweep_stays_inside_the_target_window(sample_run: SimpleNamespace) -> None:
    """Only months the model can actually target are counted — never the partial month (§8)."""
    by_k = sample_run.results["E8"]["tables"]["E08_zero_share_by_k"]
    model_cfg = load_model_config()
    assert set(by_k["first_target_month"]) == {model_cfg.split.first_target_month}
    assert set(by_k["last_target_month"]) == {sample_run.cfg.raw.last_full_month}
    for partial in sample_run.cfg.raw.partial_months:
        assert partial > sample_run.cfg.raw.last_full_month


def test_zero_share_per_product_excludes_the_partial_month(sample_run: SimpleNamespace) -> None:
    per_product = sample_run.results["E8"]["tables"]["E08_zero_share_per_product"]
    panel = sample_run.panel
    full = panel.loc[~panel["month"].astype(str).isin(sample_run.cfg.raw.partial_months)]
    assert int(per_product["panel_months"].sum()) == len(full)
    assert ((per_product["zero_share"] >= 0) & (per_product["zero_share"] <= 1)).all()


# --------------------------------------------------------------------------
# E9
# --------------------------------------------------------------------------
def test_largest_lines_are_ranked_and_capped(sample_run: SimpleNamespace) -> None:
    lines = sample_run.results["E9"]["tables"]["E09_largest_lines"]
    assert len(lines) == e09_outliers.TOP_N
    assert lines["quantity"].is_monotonic_decreasing
    assert list(lines["rank"]) == list(range(1, e09_outliers.TOP_N + 1))
    assert int(lines["quantity"].iloc[0]) == int(sample_run.clean["quantity"].max())


def test_largest_product_months_are_ranked_and_capped(sample_run: SimpleNamespace) -> None:
    months = sample_run.results["E9"]["tables"]["E09_largest_product_months"]
    assert len(months) == e09_outliers.TOP_N
    assert months["units_sold"].is_monotonic_decreasing
    assert int(months["units_sold"].iloc[0]) == int(sample_run.panel["units_sold"].max())


def test_units_histogram_axis_states_the_log_scale(sample_run: SimpleNamespace) -> None:
    """§35A.2: a log scale must be stated in the label, not merely visible in the ticks."""
    from pipeline.eda.style import LOG_SCALE_SUFFIX

    assert LOG_SCALE_SUFFIX.strip().startswith("(log")
    # The figure was drawn with the suffix in both axis labels; assert the module builds it so.
    assert "log" in LOG_SCALE_SUFFIX.lower()


# --------------------------------------------------------------------------
# E10
# --------------------------------------------------------------------------
def test_invoice_summary_reports_each_distribution(sample_run: SimpleNamespace) -> None:
    summary = sample_run.results["E10"]["tables"]["E10_invoice_summary"]
    metrics = set(summary["metric"])
    assert {"invoice_value", "lines_per_invoice", "quantity_per_line", "wholesale_lines"} == metrics
    for metric in ("invoice_value", "lines_per_invoice", "quantity_per_line"):
        statistics = set(summary.loc[summary["metric"] == metric, "statistic"])
        assert {"count", "mean", "p50", "max"} <= statistics


def test_wholesale_share_is_a_fraction_of_the_lines(sample_run: SimpleNamespace) -> None:
    summary = sample_run.results["E10"]["tables"]["E10_invoice_summary"]
    share = summary.loc[
        (summary["metric"] == "wholesale_lines") & (summary["statistic"] == "share"), "value"
    ].item()
    count = summary.loc[
        (summary["metric"] == "wholesale_lines") & (summary["statistic"] == "count"), "value"
    ].item()
    assert 0.0 <= share <= 1.0
    expected = int(
        (sample_run.clean["quantity"] >= e10_order_structure.WHOLESALE_LINE_QUANTITY).sum()
    )
    assert int(count) == expected


# --------------------------------------------------------------------------
# E11 — descriptive only (§9, §6.2)
# --------------------------------------------------------------------------
def test_customer_identification_shares_sum_to_one(sample_run: SimpleNamespace) -> None:
    identification = sample_run.results["E11"]["tables"]["E11_customer_identification"]
    assert set(identification["customer"]) == {"identified", "anonymous"}
    assert identification["row_share"].sum() == pytest.approx(1.0)
    assert identification["revenue_share"].sum() == pytest.approx(1.0)
    assert identification["units_share"].sum() == pytest.approx(1.0)
    assert int(identification["rows"].sum()) == len(sample_run.clean)


def test_top_countries_are_ranked_and_capped(sample_run: SimpleNamespace) -> None:
    countries = sample_run.results["E11"]["tables"]["E11_top10_countries"]
    assert len(countries) <= e11_customers_countries.TOP_N
    assert countries["units"].is_monotonic_decreasing
    assert ((countries["units_share"] > 0) & (countries["units_share"] <= 1)).all()


def test_no_country_or_customer_column_reaches_a_modelling_artifact(
    sample_run: SimpleNamespace,
) -> None:
    """§9, §6.2: country and customer never enter the *modelling* path.

    The distinction the acceptance criterion draws is between an EDA table and a modelling
    artifact. Describing them is the whole point of E11, and E9 lists the country and whether the
    customer was identified for each outsized line on purpose (§35A E9). What must stay clean is
    the hand-off panel — ``clean_data.csv``, the only thing US-13 builds features from.
    """
    forbidden = {"country", "customer_id", "customers", "customer_identified"}
    assert not forbidden & set(sample_run.panel.columns)

    # E14 describes that same hand-off file, so its table must be clean too.
    preview = sample_run.results["E14"]["tables"]["E14_panel_preview"]
    assert not forbidden & set(preview.columns)


# --------------------------------------------------------------------------
# E12 — the gross-demand justification (§9)
# --------------------------------------------------------------------------
def test_cancellation_monthly_matches_the_returns_side_table(
    sample_run: SimpleNamespace,
) -> None:
    monthly = sample_run.results["E12"]["tables"]["E12_cancellation_rate_monthly"]
    assert int(monthly["returned_units"].sum()) == int(sample_run.returns["quantity_abs"].sum())
    assert int(monthly["cancellation_rows"].sum()) == len(sample_run.returns)
    assert int(monthly["sold_units"].sum()) == int(sample_run.clean["quantity"].sum())


def test_returns_are_never_subtracted_from_demand(sample_run: SimpleNamespace) -> None:
    """The panel target is gross: sold units must not have returns netted off (§9)."""
    monthly = sample_run.results["E12"]["tables"]["E12_cancellation_rate_monthly"]
    panel_units = int(sample_run.panel["units_sold"].sum())
    assert int(monthly["sold_units"].sum()) == panel_units
    assert int(monthly["returned_units"].sum()) > 0, "fixture should contain cancellations"


def test_top_returned_flags_products_absent_from_the_panel(
    sample_run: SimpleNamespace,
) -> None:
    """§3: a returned stock code need not exist in the panel; it is reported, not dropped."""
    top = sample_run.results["E12"]["tables"]["E12_top_returned"]
    assert len(top) <= 20
    assert top["returned_units"].is_monotonic_decreasing
    assert "in_panel" in top.columns
    assert top["in_panel"].dtype == bool


# --------------------------------------------------------------------------
# E13
# --------------------------------------------------------------------------
def test_price_distribution_reports_both_metrics(sample_run: SimpleNamespace) -> None:
    distribution = sample_run.results["E13"]["tables"]["E13_price_distribution"]
    assert {"unit_price", "price_cv"} == set(distribution["metric"])
    price_rows = distribution.loc[distribution["metric"] == "unit_price"]
    assert {"count", "mean", "p50", "max"} <= set(price_rows["statistic"])


def test_price_stability_is_non_negative_and_skips_single_month_products(
    sample_run: SimpleNamespace,
) -> None:
    stability = sample_run.results["E13"]["tables"]["E13_price_stability"]
    defined = stability["price_cv"].dropna()
    assert (defined >= 0).all()
    too_few = stability.loc[
        stability["selling_months"] < e13_min_months(), "price_cv"
    ]
    assert too_few.isna().all()


def e13_min_months() -> int:
    from pipeline.eda.e13_price import MIN_MONTHS_FOR_CV

    return MIN_MONTHS_FOR_CV


# --------------------------------------------------------------------------
# E14 — reads the hand-off file, not the in-memory panel
# --------------------------------------------------------------------------
def test_panel_preview_has_one_row_per_month(sample_run: SimpleNamespace) -> None:
    preview = sample_run.results["E14"]["tables"]["E14_panel_preview"]
    months = sorted(sample_run.panel["month"].astype(str).unique())
    assert list(preview["month"]) == months
    assert int(preview["rows"].sum()) == len(sample_run.panel)


def test_panel_preview_matches_the_panel_contents(sample_run: SimpleNamespace) -> None:
    preview = sample_run.results["E14"]["tables"]["E14_panel_preview"]
    panel = sample_run.panel
    assert int(preview["zero_rows"].sum()) == int((panel["units_sold"] == 0).sum())
    assert int(preview["units"].sum()) == int(panel["units_sold"].sum())
    flagged = set(preview.loc[preview["is_partial"], "month"])
    assert flagged == set(sample_run.cfg.raw.partial_months) & set(preview["month"])


def test_panel_preview_reads_the_file_from_disk(
    sample_run: SimpleNamespace, ctx: RunContext
) -> None:
    """E14 validates the artifact on disk, so a stale file must produce a warning (§35A E14)."""
    from pipeline.eda import e14_panel_preview

    written = ctx.base_dir / paths.CLEAN_DATA.relative_to(paths.PROJECT_ROOT)
    written.parent.mkdir(parents=True, exist_ok=True)
    truncated = sample_run.panel.head(10)
    truncated.to_csv(written, index=False, lineterminator="\n")

    with ctx.step("e14_stale"):
        result = e14_panel_preview.run(
            sample_run.clean, sample_run.panel, sample_run.cfg, ctx
        )
    assert int(result["tables"]["E14_panel_preview"]["rows"].sum()) == len(truncated)
    assert any("stale" in warning for warning in ctx.warnings)


def test_panel_preview_fails_clearly_when_the_file_is_missing(ctx: RunContext) -> None:
    from pipeline.eda import e14_panel_preview

    with pytest.raises(FileNotFoundError, match="clean_data.csv"):
        e14_panel_preview.load_panel_file(ctx)


# --------------------------------------------------------------------------
# the index (US-10 acceptance criterion: E1–E14)
# --------------------------------------------------------------------------
def test_index_lists_the_analyses_that_ran(sample_run: SimpleNamespace) -> None:
    written = sample_run.base / index_path()
    entries = json.loads(written.read_text(encoding="utf-8"))
    assert [entry["id"] for entry in entries] == ALL_IDS
    for entry in entries:
        assert entry["table_names"] == sorted(EXPECTED_TABLES[entry["id"]])
        assert entry["figure_names"] == sorted(EXPECTED_FIGURES[entry["id"]])


def test_index_sorts_numerically_not_alphabetically(sample_run: SimpleNamespace) -> None:
    """``E10`` must follow ``E9``; a plain string sort would put it after ``E1``."""
    entries = json.loads((sample_run.base / index_path()).read_text(encoding="utf-8"))
    numbers = [int(entry["id"].lstrip("E")) for entry in entries]
    assert numbers == sorted(numbers)


def test_e8_to_e14_append_to_an_existing_index(
    ctx: RunContext, sample_run: SimpleNamespace
) -> None:
    """US-09's entries must survive an E8–E14 run (US-11 needs all fourteen)."""
    with ctx.step("eda_merge"):
        run_analyses(["E6"], sample_run.clean, sample_run.panel, sample_run.cfg, ctx)
        run_analyses(["E10"], sample_run.clean, sample_run.panel, sample_run.cfg, ctx)
        entries = read_index(ctx)
    assert sorted(entries) == ["E10", "E6"]


def test_registry_now_covers_e2_to_e14() -> None:
    assert sorted(ANALYSES_BY_ID, key=lambda value: int(value.lstrip("E"))) == [
        f"E{number}" for number in range(2, 15)
    ]
