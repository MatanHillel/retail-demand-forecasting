"""The active-product rule (PRD §14, US-05).

A product is **active** for a target month ``t`` when it sold at least one unit in the ``k``
months *before* ``t``::

    is_active(t) = any(units_sold > 0 in months t−k … t−1)

Month ``t`` itself is never inspected — that is the same no-leakage boundary the forecast origin
uses (§16), so this helper can be used both for EDA zero-share curves (US-10, several values of
``k``) and for feature engineering (US-13, ``k`` from ``model_config.yaml``) without the two
definitions drifting apart.

The computation is vectorised (shift + rolling within each product) and assumes the caller passes
the **zero-filled** panel of US-05, where each product's months are contiguous — the rolling
window counts rows, and rows equal months only under that guarantee, which
:func:`pipeline.panel.validate_panel` enforces.
"""

from __future__ import annotations

import pandas as pd

from pipeline.config import load_model_config

ACTIVE_COLUMNS: list[str] = ["stock_code", "month", "is_active"]


def active_mask(panel: pd.DataFrame, k: int | None = None) -> pd.DataFrame:
    """Return ``stock_code, month, is_active`` for every row of the panel.

    ``k`` defaults to ``model_config.yaml → active_rule.k`` (§14) so the window length is defined
    in exactly one place; passing it explicitly is equivalent and is what the EDA does when it
    sweeps several values.
    """
    window = load_model_config().active_rule.k if k is None else int(k)
    if window <= 0:
        raise ValueError(f"k must be a positive number of months, got {window}")

    frame = (
        panel[["stock_code", "month", "units_sold"]]
        .sort_values(["stock_code", "month"], kind="mergesort")
        .reset_index(drop=True)
    )
    units = pd.to_numeric(frame["units_sold"], errors="coerce").fillna(0)

    # Shift by one month first: month t must never see its own sales. Months before a product's
    # first row do not exist, which the fill treats as "no sales" (§14).
    prior = units.groupby(frame["stock_code"], sort=False).shift(1).fillna(0)
    sold_in_window = (
        prior.groupby(frame["stock_code"], sort=False)
        .rolling(window, min_periods=1)
        .sum()
        .reset_index(level=0, drop=True)
        .sort_index()
    )

    frame["is_active"] = (sold_in_window > 0).to_numpy()
    return frame[ACTIVE_COLUMNS]
