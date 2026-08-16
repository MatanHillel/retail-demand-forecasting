"""ABC classification — one definition for the whole project (US-06, PRD §18.2, §23, §27, §35A E6).

ABC sorts products by how much revenue they bring: **A** is the small head of the catalogue that
produces the first 80 % of revenue, **B** the next 15 %, **C** the long tail. Four different parts
of the system need it — the E6 Pareto figure, evaluation broken down by class (§23), the σ
fallback when a product has too few residuals (§27) and the inventory KPIs (§30) — so it lives
here, above :mod:`pipeline.eda`, and is computed exactly once per window.

**The leakage rule lives in the** ``through_month`` **argument.** Revenue is summed over months
``≤ through_month`` and nothing later, because a class derived from revenue the model has not seen
yet would leak the future into training, evaluation and σ:

* modelling, evaluation and σ fallback pass ``through_month = the last training target month``
  (§18.2, §23, §27) — never the end of the data;
* descriptive EDA (E6) may pass the last month of the panel to describe the whole period, and the
  figure must then be **labelled** as full-period ABC so it is never mistaken for the training
  classes.

Thresholds are never written into this file: they come from ``inventory_policy.yaml → abc`` via
:func:`pipeline.config.load_inventory_policy`, or from the caller's arguments (PRD §40).
"""

from __future__ import annotations

import pandas as pd

from pipeline.config import load_inventory_policy

#: Columns of the frame returned by :func:`compute_abc`, in order.
ABC_COLUMNS: list[str] = ["stock_code", "revenue", "revenue_share", "cum_share", "abc_class"]

#: The three classes, best to worst.
ABC_CLASSES: tuple[str, str, str] = ("A", "B", "C")

#: Slack allowed when comparing a cumulative share against a boundary. A running sum of floating
#: point shares lands on 0.95 as 0.9500000000000001, which without this would push the product
#: that sits exactly on the boundary into the next class — a class that would then depend on the
#: order of the additions rather than on the revenue.
_SHARE_TOLERANCE = 1e-9


def compute_abc(
    panel: pd.DataFrame,
    through_month: str,
    a_cum_share: float | None = None,
    b_cum_share: float | None = None,
) -> pd.DataFrame:
    """Classify every product in ``panel`` as A, B or C from revenue up to ``through_month``.

    Args:
        panel: the ``clean_data.csv`` panel — needs ``stock_code``, ``month`` (``YYYY-MM``) and
            ``gross_revenue``. Extra columns are ignored.
        through_month: inclusive cut-off, ``YYYY-MM``. Months after it are dropped before anything
            is summed; see the module docstring for which caller passes what.
        a_cum_share: cumulative-revenue share up to which a product is class A. ``None`` reads
            ``inventory_policy.yaml → abc.a_cum_share``.
        b_cum_share: cumulative-revenue share up to which a product is class B. ``None`` reads
            ``inventory_policy.yaml → abc.b_cum_share``.

    Returns:
        One row per product observed in the window, columns ``stock_code, revenue,
        revenue_share, cum_share, abc_class``, sorted by revenue descending with ``stock_code``
        ascending as the tie-break — so the result is byte-identical across runs (§40).

    Notes:
        * A product is class A while ``cum_share ≤ a_cum_share``, B while ``≤ b_cum_share``,
          otherwise C. The product that *crosses* a boundary therefore falls in the lower class.
        * A product with zero (or negative) revenue in the window is always C, whatever the
          cumulative shares say — which also keeps the degenerate all-zero window from classifying
          the whole catalogue as A.
        * Products with no row at or before ``through_month`` are absent from the result: at the
          forecast origin they had not been observed yet, so classifying them would itself be a
          use of future information.
    """
    thresholds = load_inventory_policy().abc
    a_share = thresholds.a_cum_share if a_cum_share is None else a_cum_share
    b_share = thresholds.b_cum_share if b_cum_share is None else b_cum_share
    if not 0 < a_share < b_share < 1:
        raise ValueError(
            f"ABC thresholds must satisfy 0 < a_cum_share ({a_share}) < b_cum_share ({b_share}) < 1"
        )

    window = panel.loc[panel["month"].astype(str) <= str(through_month), :]
    revenue = (
        window.groupby(window["stock_code"].astype(str), sort=False)["gross_revenue"]
        .sum()
        .rename("revenue")
        .reset_index()
    )
    revenue = revenue.sort_values(
        ["revenue", "stock_code"], ascending=[False, True], kind="mergesort"
    ).reset_index(drop=True)

    total = float(revenue["revenue"].sum())
    revenue["revenue_share"] = revenue["revenue"] / total if total > 0 else 0.0
    revenue["cum_share"] = revenue["revenue_share"].cumsum()

    positive = revenue["revenue"] > 0
    revenue["abc_class"] = "C"
    revenue.loc[positive & (revenue["cum_share"] <= b_share + _SHARE_TOLERANCE), "abc_class"] = "B"
    revenue.loc[positive & (revenue["cum_share"] <= a_share + _SHARE_TOLERANCE), "abc_class"] = "A"

    return revenue.loc[:, ABC_COLUMNS]
