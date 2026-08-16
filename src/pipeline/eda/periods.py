"""Calendar helpers shared by the EDA analyses (US-09, PRD §8, §35A E2).

Two facts about this dataset make plain calendar arithmetic wrong, and both are handled here so
no analysis re-derives them:

* **A "year" runs December → November.** The extract starts in Dec 2009 and the last full month
  is Nov 2011, so the two complete years are Dec 2009–Nov 2010 and Dec 2010–Nov 2011 — not 2010
  and 2011 (§35A E2). Year-on-year growth and the seasonal index are computed over those windows.
* **December 2011 is partial** (it stops on the 9th, §8). It is excluded from every period total
  and drawn hatched wherever it appears in a time series, never silently dropped.

Nothing here reads a month from code: the boundaries come from ``cleaning_config.yaml → raw``.
"""

from __future__ import annotations

import pandas as pd

from pipeline.config import CleaningConfig

#: Months in a year. Structural, not a tunable threshold — the seasonal index is a share of the
#: annual total scaled so that an average month reads 1.0.
MONTHS_PER_YEAR = 12


def month_to_period(months: pd.Series) -> pd.PeriodIndex:
    """Parse a ``YYYY-MM`` string column into monthly periods, for arithmetic and sorting."""
    return pd.PeriodIndex(months.astype(str), freq="M")


def add_months(month: str, count: int) -> str:
    """``add_months("2009-12", 11)`` -> ``"2010-11"``."""
    return str(pd.Period(month, freq="M") + count)


def full_years(cfg: CleaningConfig) -> list[tuple[str, str]]:
    """The complete December→November windows the data covers, oldest first.

    Derived from ``raw.first_month`` and ``raw.last_full_month``: a window is included only when
    all twelve of its months are inside that range, so a partial year at either end is dropped
    rather than compared against a full one.
    """
    windows: list[tuple[str, str]] = []
    start = pd.Period(cfg.raw.first_month, freq="M")
    last_full = pd.Period(cfg.raw.last_full_month, freq="M")
    while start + (MONTHS_PER_YEAR - 1) <= last_full:
        end = start + (MONTHS_PER_YEAR - 1)
        windows.append((str(start), str(end)))
        start = end + 1
    return windows


def year_label(window: tuple[str, str]) -> str:
    """Human-readable name of a Dec→Nov window, e.g. ``"2009-12..2010-11"``."""
    return f"{window[0]}..{window[1]}"


def in_window(months: pd.Series, window: tuple[str, str]) -> pd.Series:
    """Boolean mask selecting the rows whose month falls inside ``window`` (inclusive)."""
    text = months.astype(str)
    return (text >= window[0]) & (text <= window[1])


def partial_positions(months: list[str], cfg: CleaningConfig) -> list[int]:
    """Index positions of the partial months inside an ordered list of months.

    Returned in the form :func:`pipeline.eda.style.hatch_partial` expects for a categorical axis.
    """
    return [index for index, month in enumerate(months) if month in cfg.raw.partial_months]


def full_months_only(frame: pd.DataFrame, cfg: CleaningConfig) -> pd.DataFrame:
    """Drop the partial month(s) — used for every *total*, share and ranking (§8)."""
    return frame.loc[~frame["month"].astype(str).isin(cfg.raw.partial_months), :]
