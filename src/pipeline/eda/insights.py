"""``insights.md`` — the deterministic narrative (US-11, PRD §35, §35A.2, §38, §41).

Eight to twelve findings, each one sentence of fact plus one sentence of "so what" tied to a
decision the project actually made. Two rules shape everything here:

* **Every number is read out of a computed table.** Nothing is typed from the PRD and nothing is
  recalculated in prose. The sentence templates take table values and format them; the finished
  markdown is then run through :func:`pipeline.narrative.numbers_in_tables`, and generation
  **fails** if a single number cannot be traced back (§35A.2, §38).
* **The wording is fixed.** In LLM mode the Business & EDA Analyst agent (US-12) rewrites the
  prose, and the same guard validates its output. Keeping stable insight ids (``I01``…) is what
  lets the two versions be diffed line for line.

This module imports no LLM class: the deterministic report has to exist in ``--no-llm`` mode,
which is what CI runs (§37).
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from pydantic import BaseModel, ConfigDict

from pipeline import paths
from pipeline.config import CleaningConfig, load_model_config
from pipeline.narrative import numbers_in_tables
from pipeline.run_context import RunContext

#: PRD §35 asks for 8-12 insights. More than twelve stops being a summary.
MIN_INSIGHTS = 8
MAX_INSIGHTS = 12


class Insight(BaseModel):
    """One finding: what the data says, which table says it, and why it matters."""

    model_config = ConfigDict(extra="forbid")

    id: str
    e_ref: str
    text: str
    so_what: str
    numbers: dict[str, float]
    table_ref: str
    figure_ref: str | None = None


def _value(frame: pd.DataFrame, column: str, row: int = 0) -> float:
    return float(pd.to_numeric(frame[column]).iloc[row])


def _pct(value: float) -> str:
    """A share formatted as a percentage with one decimal — matched back by the guard."""
    return f"{value * 100:.1f} %"


def _count(value: float) -> str:
    return f"{int(round(value)):,}"


# --------------------------------------------------------------------------
# the insight builders, in report order
# --------------------------------------------------------------------------
def _seasonality(tables: dict[str, pd.DataFrame], cfg: CleaningConfig) -> Insight | None:
    frame = tables.get("E02_sep_nov_share")
    if frame is None or frame.empty:
        return None
    pooled = frame.loc[frame["year"] == "pooled"]
    row = pooled if len(pooled) else frame.tail(1)
    units = float(pd.to_numeric(row["units_share"]).iloc[0])
    return Insight(
        id="I01",
        e_ref="E2",
        text=f"September to November carries {_pct(units)} of the year's units.",
        so_what=(
            "so the hold-out window deliberately covers the peak — a model validated only on "
            "quiet months would be scored on data unlike the months that matter (§21)."
        ),
        numbers={"sep_nov_units_share": units},
        table_ref="E02_sep_nov_share",
        figure_ref="E02_monthly_units",
    )


def _growth(tables: dict[str, pd.DataFrame], cfg: CleaningConfig) -> Insight | None:
    frame = tables.get("E02_yoy")
    if frame is None or len(frame) < 2:
        return None
    units = _value(frame, "units_growth", -1)
    revenue = _value(frame, "revenue_growth", -1)
    return Insight(
        id="I02",
        e_ref="E2",
        text=(
            f"Between the two complete December-to-November years, units moved "
            f"{_pct(units)} while revenue moved {_pct(revenue)}."
        ),
        so_what=(
            "so volume and value are drifting apart, and a forecast of units cannot be read as a "
            "forecast of money."
        ),
        numbers={"units_growth": units, "revenue_growth": revenue},
        table_ref="E02_yoy",
        figure_ref="E02_monthly_units",
    )


def _weekday(tables: dict[str, pd.DataFrame], cfg: CleaningConfig) -> Insight | None:
    frame = tables.get("E03_weekday")
    if frame is None or frame.empty:
        return None
    quietest = frame.loc[pd.to_numeric(frame["units"]).idxmin()]
    busiest = frame.loc[pd.to_numeric(frame["units"]).idxmax()]
    return Insight(
        id="I03",
        e_ref="E3",
        text=(
            f"{quietest['weekday_name']} sees {_count(quietest['units'])} units against "
            f"{_count(busiest['units'])} on {busiest['weekday_name']}."
        ),
        so_what=(
            "so trading is a weekday business; the pattern is context for the reader and is never "
            "a model input, because the forecast is monthly (§35A E3)."
        ),
        numbers={
            "quietest_units": float(quietest["units"]),
            "busiest_units": float(busiest["units"]),
        },
        table_ref="E03_weekday",
        figure_ref="E03_weekday",
    )


def _portfolio(tables: dict[str, pd.DataFrame], cfg: CleaningConfig) -> Insight | None:
    frame = tables.get("E04_products_per_month")
    if frame is None or frame.empty:
        return None
    full = frame.loc[~frame["is_partial"].astype(bool)] if "is_partial" in frame else frame
    lowest = float(pd.to_numeric(full["products"]).min())
    highest = float(pd.to_numeric(full["products"]).max())
    return Insight(
        id="I04",
        e_ref="E4",
        text=(
            f"The number of products selling in a month ranges from {_count(lowest)} to "
            f"{_count(highest)}."
        ),
        so_what=(
            "so the catalogue is not stable, and forecasting every code every month would spend "
            "the model on products that are not selling — which is what the active-product rule "
            "prevents (§14)."
        ),
        numbers={"min_products": lowest, "max_products": highest},
        table_ref="E04_products_per_month",
        figure_ref="E04_products_per_month",
    )


def _lifecycle(tables: dict[str, pd.DataFrame], cfg: CleaningConfig) -> Insight | None:
    frame = tables.get("E05_lifecycle_summary")
    if frame is None or frame.empty:
        return None
    share = _value(frame, "short_lifecycle_share")
    months = _value(frame, "short_lifecycle_months")
    median = _value(frame, "median_selling_months")
    return Insight(
        id="I05",
        e_ref="E5",
        text=(
            f"{_pct(share)} of products sell in {int(months)} months or fewer, and the median "
            f"product sells in {int(median)} months."
        ),
        so_what=(
            "so long history is the exception; the feature set leans on short lags rather than "
            "assuming two years of it (§18.1)."
        ),
        numbers={"short_lifecycle_share": share, "median_selling_months": median},
        table_ref="E05_lifecycle_summary",
        figure_ref="E05_lifecycle_hist",
    )


def _concentration(tables: dict[str, pd.DataFrame], cfg: CleaningConfig) -> Insight | None:
    frame = tables.get("E06_abc_table")
    if frame is None or frame.empty:
        return None
    class_a = frame.loc[frame["abc_class"] == "A"]
    if class_a.empty:
        return None
    products = float(pd.to_numeric(class_a["product_share"]).iloc[0])
    revenue = float(pd.to_numeric(class_a["revenue_share"]).iloc[0])
    return Insight(
        id="I06",
        e_ref="E6",
        text=f"Class A is {_pct(products)} of the catalogue and {_pct(revenue)} of revenue.",
        so_what=(
            "so accuracy on a small head of products decides the result, and the safety-stock "
            "fallback groups products by that same class when a product has too few residuals "
            "(§27)."
        ),
        numbers={"a_product_share": products, "a_revenue_share": revenue},
        table_ref="E06_abc_table",
        figure_ref="E06_pareto",
    )


def _top_product(tables: dict[str, pd.DataFrame], cfg: CleaningConfig) -> Insight | None:
    frame = tables.get("E07_top20_units")
    if frame is None or frame.empty:
        return None
    top = frame.iloc[0]
    units = float(pd.to_numeric(frame["units"]).iloc[0])
    share = float(pd.to_numeric(frame["share"]).iloc[0])
    return Insight(
        id="I07",
        e_ref="E7",
        text=(
            f"The best-selling product, {top['stock_code']} {top['description']}, moved "
            f"{_count(units)} units — {_pct(share)} of all units."
        ),
        so_what=(
            "so no single product dominates volume; the ranking by revenue is a different list, "
            "which is why both are reported."
        ),
        numbers={"top_units": units, "top_share": share},
        table_ref="E07_top20_units",
        figure_ref="E07_top20_units",
    )


def _intermittency(tables: dict[str, pd.DataFrame], cfg: CleaningConfig) -> Insight | None:
    frame = tables.get("E08_zero_share_by_k")
    if frame is None or frame.empty:
        return None
    configured = frame.loc[frame["is_configured_k"].astype(bool)]
    if configured.empty:
        return None
    k = int(pd.to_numeric(configured["k"]).iloc[0])
    rows = float(pd.to_numeric(configured["rows"]).iloc[0])
    share = float(pd.to_numeric(configured["zero_target_share"]).iloc[0])
    widest = frame.iloc[-1]
    widest_k = int(widest["k"])
    widest_share = float(widest["zero_target_share"])
    return Insight(
        id="I08",
        e_ref="E8",
        text=(
            f"At the configured look-back of {k} months the model sees {_count(rows)} "
            f"product-months, {_pct(share)} of which sold nothing; widening the window to "
            f"{widest_k} months raises that to {_pct(widest_share)}."
        ),
        so_what=(
            "so the active-product window is a trade between coverage and empty targets, and "
            "this table is the justification for the configured value (§14)."
        ),
        numbers={"active_rows": rows, "zero_target_share": share},
        table_ref="E08_zero_share_by_k",
        figure_ref="E08_zero_share_by_k",
    )


def _outliers(tables: dict[str, pd.DataFrame], cfg: CleaningConfig) -> Insight | None:
    frame = tables.get("E09_largest_lines")
    if frame is None or frame.empty:
        return None
    largest = float(pd.to_numeric(frame["quantity"]).iloc[0])
    summary = tables.get("E09_magnitude_summary")
    median = None
    if summary is not None and not summary.empty:
        rows = summary.loc[
            (summary["metric"] == "units_sold_nonzero") & (summary["statistic"] == "p50")
        ]
        if len(rows):
            median = float(pd.to_numeric(rows["value"]).iloc[0])
    if median is None:
        return None
    return Insight(
        id="I09",
        e_ref="E9",
        text=(
            f"The largest single order line is {_count(largest)} units, against a median "
            f"product-month of {_count(median)} units."
        ),
        so_what=(
            "so errors are weighted by volume (wMAPE) and the safety stock uses a robust spread "
            "measure — a standard deviation would let a few wholesale orders inflate the stock "
            "recommendation for every similar product (§23, §26)."
        ),
        numbers={"largest_line_units": largest, "median_product_month_units": median},
        table_ref="E09_largest_lines",
        figure_ref="E09_units_hist_log",
    )


def _order_structure(tables: dict[str, pd.DataFrame], cfg: CleaningConfig) -> Insight | None:
    frame = tables.get("E10_invoice_summary")
    if frame is None or frame.empty:
        return None
    share_rows = frame.loc[
        (frame["metric"] == "wholesale_lines") & (frame["statistic"] == "share")
    ]
    median_rows = frame.loc[
        (frame["metric"] == "lines_per_invoice") & (frame["statistic"] == "p50")
    ]
    if share_rows.empty or median_rows.empty:
        return None
    share = float(pd.to_numeric(share_rows["value"]).iloc[0])
    median_lines = float(pd.to_numeric(median_rows["value"]).iloc[0])
    return Insight(
        id="I10",
        e_ref="E10",
        text=(
            f"Half of all invoices carry {_count(median_lines)} product lines or fewer, and "
            f"{_pct(share)} of lines are wholesale-sized."
        ),
        so_what=(
            "so this is a wholesaler rather than a shop, which is the context for the outsized "
            "lines above."
        ),
        numbers={"median_lines_per_invoice": median_lines, "wholesale_line_share": share},
        table_ref="E10_invoice_summary",
        figure_ref="E10_invoice_hist",
    )


def _customers(tables: dict[str, pd.DataFrame], cfg: CleaningConfig) -> Insight | None:
    identification = tables.get("E11_customer_identification")
    countries = tables.get("E11_top10_countries")
    if identification is None or identification.empty or countries is None or countries.empty:
        return None
    anonymous = identification.loc[identification["customer"] == "anonymous"]
    if anonymous.empty:
        return None
    anon_share = float(pd.to_numeric(anonymous["row_share"]).iloc[0])
    top = countries.iloc[0]
    country_share = float(pd.to_numeric(countries["units_share"]).iloc[0])
    return Insight(
        id="I11",
        e_ref="E11",
        text=(
            f"{_pct(anon_share)} of sales rows carry no customer id, and {top['country']} alone "
            f"accounts for {_pct(country_share)} of units."
        ),
        so_what=(
            "so anonymous rows are kept — the goods were still sold and still had to be in stock "
            "— and the model has no country dimension, because there is not enough of anywhere "
            "else to learn from (§9, §6.2)."
        ),
        numbers={"anonymous_row_share": anon_share, "top_country_units_share": country_share},
        table_ref="E11_customer_identification",
        figure_ref="E11_top_countries",
    )


def _returns(tables: dict[str, pd.DataFrame], cfg: CleaningConfig) -> Insight | None:
    frame = tables.get("E12_cancellation_rate_monthly")
    if frame is None or frame.empty:
        return None
    # Both totals are whole-column sums of the same table, so the share between them is a
    # derived percentage of two table values — exactly what the guard permits. Filtering the
    # partial month out of one side would make the ratio untraceable for no gain: it appears in
    # numerator and denominator alike.
    sold = float(pd.to_numeric(frame["sold_units"]).sum())
    returned = float(pd.to_numeric(frame["returned_units"]).sum())
    if not sold:
        return None
    return Insight(
        id="I12",
        e_ref="E12",
        text=(
            f"{_count(returned)} units came back against {_count(sold)} sold — "
            f"{_pct(returned / sold)} of the total."
        ),
        so_what=(
            "so returns are visible but never subtracted from the target: the stock had to be on "
            "the shelf to fill the original order, so netting them off would under-forecast "
            "exactly the products that get returned (§9)."
        ),
        numbers={"returned_units": returned, "sold_units": sold},
        table_ref="E12_cancellation_rate_monthly",
        figure_ref="E12_cancellation_rate",
    )


def _price(tables: dict[str, pd.DataFrame], cfg: CleaningConfig) -> Insight | None:
    frame = tables.get("E13_price_distribution")
    if frame is None or frame.empty:
        return None
    price_rows = frame.loc[(frame["metric"] == "unit_price") & (frame["statistic"] == "p50")]
    cv_rows = frame.loc[(frame["metric"] == "price_cv") & (frame["statistic"] == "p50")]
    if price_rows.empty or cv_rows.empty:
        return None
    price = float(pd.to_numeric(price_rows["value"]).iloc[0])
    cv = float(pd.to_numeric(cv_rows["value"]).iloc[0])
    return Insight(
        id="I13",
        e_ref="E13",
        text=(
            f"The median unit price is {price:.2f} and the typical product's price varies by "
            f"{_pct(cv)} of its own average across months."
        ),
        so_what=(
            "so price moves enough to be worth carrying as a lagged input, and steadily enough "
            "that last month's price is informative."
        ),
        numbers={"median_unit_price": price, "median_price_cv": cv},
        table_ref="E13_price_distribution",
        figure_ref="E13_price_hist",
    )


#: Candidate builders in report order. The first :data:`MAX_INSIGHTS` that produce something are
#: kept, so a run missing an analysis degrades to a shorter list instead of failing.
BUILDERS: tuple[Callable[[dict[str, pd.DataFrame], CleaningConfig], Insight | None], ...] = (
    _seasonality,
    _growth,
    _intermittency,
    _concentration,
    _lifecycle,
    _outliers,
    _customers,
    _returns,
    _top_product,
    _portfolio,
    _order_structure,
    _weekday,
    _price,
)


def build_insights(tables: dict[str, pd.DataFrame], cfg: CleaningConfig) -> list[Insight]:
    """Build the insight list from the computed tables. Pure — no disk, no ``ctx``."""
    insights = []
    for builder in BUILDERS:
        insight = builder(tables, cfg)
        if insight is not None:
            insights.append(insight)
    return insights[:MAX_INSIGHTS]


def render_markdown(
    insights: list[Insight], cfg: CleaningConfig, ctx: RunContext, figures: list[str]
) -> str:
    """Render ``insights.md``: header, the numbered findings, then the figure list."""
    model_cfg = load_model_config()
    lines = [
        "# What the data says",
        "",
        f"*Dataset:* Online Retail II (UCI), {cfg.raw.first_month} to "
        f"{cfg.raw.last_full_month} complete, plus the partial month "
        f"{', '.join(cfg.raw.partial_months)} (shown, never scored).",
        "",
        f"*Run:* `{ctx.run_id}` · *Generated:* {datetime.now(UTC).isoformat()}",
        "",
        "*Provenance:* every number below was computed by the pipeline and written to a table "
        "under `artifacts/reports/eda_tables/`. The narrative may be rewritten by the "
        "Business & EDA Analyst agent in LLM mode, but the numbers may only ever come from "
        "those tables, and the same guard checks both versions (PRD §38).",
        "",
        f"*Active-product window:* {model_cfg.active_rule.k} months. "
        f"*First target month:* {model_cfg.split.first_target_month}.",
        "",
        "## Insights",
        "",
    ]
    for number, insight in enumerate(insights, start=1):
        # The fact goes in bold without its full stop, so the "so what" reads as one sentence
        # rather than starting lower-case after a period.
        fact = insight.text.rstrip(".")
        lines.append(
            f"{number}. **{fact}** — {insight.so_what} "
            f"({insight.e_ref}, table `{insight.table_ref}`)"
        )
    lines += ["", "## Figures", ""]
    for name in figures:
        lines.append(f"- `figures/{name}.png`")
    lines.append("")
    return "\n".join(lines)


def generate_insights(
    tables: dict[str, pd.DataFrame],
    cfg: CleaningConfig,
    ctx: RunContext,
    *,
    figures: list[str] | None = None,
    out: Path | None = None,
) -> list[Insight]:
    """Build, validate and write ``insights.md``; return the insights.

    Raises:
        ValueError: if fewer than :data:`MIN_INSIGHTS` could be built, or if the finished
            markdown contains a number that is not in any of ``tables``. Both are failures of
            this generator, not of the data — publishing an unverifiable number is exactly what
            §38 forbids.
    """
    insights = build_insights(tables, cfg)
    if len(insights) < MIN_INSIGHTS:
        raise ValueError(
            f"only {len(insights)} insight(s) could be built from the supplied tables; "
            f"PRD §35 requires at least {MIN_INSIGHTS}. Missing tables: "
            f"{sorted(set(_required_tables()) - set(tables))}"
        )

    markdown = render_markdown(insights, cfg, ctx, figures or [])
    check = numbers_in_tables(markdown, tables)
    if not check.passed:
        raise ValueError(
            f"insights.md would state a number that is in no computed table — {check.summary()}. "
            "Every narrative number must be traceable to a table (PRD §35A.2, §38)."
        )

    relative = (paths.INSIGHTS if out is None else Path(out)).relative_to(paths.PROJECT_ROOT)
    destination = ctx.out(relative)
    destination.write_text(markdown, encoding="utf-8", newline="\n")
    ctx.record_artifact("insights", relative)
    ctx.logger.info(
        f"insights.md written: {relative.as_posix()} ({len(insights)} insights, "
        f"{check.checked} numbers all backed by tables)"
    )
    return insights


def _required_tables() -> list[str]:
    """Tables the builders read, for the error message when too few insights could be built."""
    return [
        "E02_sep_nov_share", "E02_yoy", "E03_weekday", "E04_products_per_month",
        "E05_lifecycle_summary", "E06_abc_table", "E07_top20_units", "E08_zero_share_by_k",
        "E09_largest_lines", "E09_magnitude_summary", "E10_invoice_summary",
        "E11_customer_identification", "E11_top10_countries",
        "E12_cancellation_rate_monthly", "E13_price_distribution",
    ]
