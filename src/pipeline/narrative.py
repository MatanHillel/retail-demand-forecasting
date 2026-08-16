"""The numbers-in-tables guard (US-11, PRD §35A.2, §38).

**No agent, and no generator, may state a number that is not in a computed table.** That is the
mechanism behind PRD §38: an LLM never calculates anything in this project, it interprets tables
and writes prose. This module is what enforces it — every narrative artifact
(``insights.md`` here, and later the crew notes of US-12, the evaluation report of US-25, the
model card of US-26 and the Flow narratives of US-33) is passed through
:func:`numbers_in_tables` before it is accepted. On failure the caller restores the deterministic
version rather than publishing an unverifiable claim.

Matching is deliberately forgiving about *presentation* and strict about *provenance*:

* ``35.5 %`` matches a table value of ``0.35512`` — a share written as a percentage, rounded to
  the decimals the sentence actually shows.
* ``1,024,951`` matches ``1024951`` — thousands separators are formatting.
* ``£20.05 M`` matches ``20050000`` — a magnitude suffix is formatting too.
* ``4,999`` does **not** match ``4998`` — that is a different number, and the point of the guard.

Years and small integers are whitelisted: a sentence may say "Dec 2009 to Nov 2011" or "k = 6"
without those being claims about the data. So are the analysis ids (``E08``), which are never
read as numbers because a digit preceded by a letter is not a token.
"""

from __future__ import annotations

import re
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

#: Multipliers for a magnitude suffix written after a number.
MAGNITUDES: dict[str, float] = {"k": 1e3, "m": 1e6, "bn": 1e9, "b": 1e9}

#: Small integers a sentence may use as counts, ranks or month numbers without citing a table.
#: Covers the twelve calendar months and the handful of "the top 20 products" style references.
MAX_WHITELISTED_SMALL_INTEGER = 20

#: Four-digit integers in this range read as calendar years, not as measurements.
YEAR_RANGE = (1900, 2100)

#: Default relative tolerance — a tenth of a percent, enough for float noise and rounding.
DEFAULT_TOLERANCE_REL = 0.005

#: Spans that are identifiers rather than claims about the data, removed before tokenising.
#: Without this the guard reads the generation timestamp, the run id and the PRD section
#: references as measurements and demands a table for each of them.
_IDENTIFIER_SPANS: tuple[re.Pattern[str], ...] = (
    # Markdown code spans: run ids, table names, file paths — `E02_yoy`, `20260816T125229Z-ab12`.
    re.compile(r"`[^`]*`"),
    # ISO-8601 timestamps, whose seconds and microseconds look like a precise measurement.
    re.compile(
        r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})?"
    ),
    # PRD references: §21, §18.1, §35A. Pointers to the specification, not numbers.
    re.compile(r"§\s?\d+(?:\.\d+)?[A-Za-z]?"),
)

#: A number token: optional currency, digits with optional thousands separators, optional
#: decimals, optional percent sign, optional magnitude suffix. The leading ``(?<![\w.])`` is what
#: keeps ``E08`` and ``I01`` from being read as numbers, and stops ``0.5`` matching twice.
_NUMBER = re.compile(
    r"(?<![\w.])"
    r"(?P<currency>[£$€])?\s?"
    r"(?P<value>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
    # A magnitude suffix has to stand alone: "20.05 M" is millions, "12 months" is not.
    r"(?:\s?(?P<suffix>bn|[kKmMbB])(?![A-Za-z]))?"
    r"\s?(?P<percent>%)?"
    r"(?![\w.]*\d)"
)


class NumberToken(BaseModel):
    """One number as it was written, with everything needed to judge what it could mean."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    raw: str
    value: float
    decimals: int
    is_percent: bool = False
    magnitude: float = 1.0

    @property
    def candidates(self) -> list[float]:
        """Every value this token could legitimately be citing.

        A percentage may be stored either way round — ``35.5 %`` against ``0.355`` or ``35.5`` —
        and a plain fraction may have been written out as a percentage, so both readings count.
        """
        values = [self.value * self.magnitude]
        if self.is_percent:
            values.append(self.value / 100.0)
        elif 0.0 < self.value < 1.0:
            values.append(self.value * 100.0)
        return values


class NarrativeCheck(BaseModel):
    """The outcome of checking one narrative against the tables that must back it."""

    model_config = ConfigDict(extra="forbid")

    passed: bool
    unmatched: list[str] = Field(default_factory=list)
    checked: int = 0
    table_values: int = 0

    def summary(self) -> str:
        if self.passed:
            return f"all {self.checked} narrative number(s) are backed by a computed table"
        return (
            f"{len(self.unmatched)} of {self.checked} narrative number(s) are not in any table: "
            f"{', '.join(self.unmatched)}"
        )


def strip_identifiers(text: str) -> str:
    """Blank out the spans that are names, not measurements, keeping the offsets intact."""
    for pattern in _IDENTIFIER_SPANS:
        text = pattern.sub(lambda match: " " * len(match.group(0)), text)
    return text


def _tokens(text: str) -> list[NumberToken]:
    """Every number in ``text``, with its formatting preserved."""
    tokens: list[NumberToken] = []
    for match in _NUMBER.finditer(strip_identifiers(text)):
        digits = match.group("value").replace(",", "")
        try:
            value = float(digits)
        except ValueError:  # pragma: no cover - the pattern only matches parseable numbers
            continue
        suffix = (match.group("suffix") or "").lower()
        decimals = len(digits.split(".")[1]) if "." in digits else 0
        tokens.append(
            NumberToken(
                raw=match.group(0).strip(),
                value=value,
                decimals=decimals,
                is_percent=match.group("percent") is not None,
                magnitude=MAGNITUDES.get(suffix, 1.0),
            )
        )
    return tokens


def extract_numbers(text: str) -> list[float]:
    """Return the numbers in ``text`` as written, in order of appearance.

    ``"Sep-Nov ~ 35 % of units; 1,024,951 rows"`` gives ``[35.0, 1024951.0]``: the percent sign and
    the thousands separators are formatting, not part of the value. A magnitude suffix *is* part
    of the value, so ``"£20.05 M"`` gives ``[20050000.0]``.
    """
    return [token.value * token.magnitude for token in _tokens(text)]


def _is_whitelisted(token: NumberToken) -> bool:
    """True for numbers a sentence may use without citing a table."""
    if token.decimals or token.is_percent or token.magnitude != 1.0:
        return False
    if not float(token.value).is_integer():
        return False
    value = int(token.value)
    if 0 <= value <= MAX_WHITELISTED_SMALL_INTEGER:
        return True
    return YEAR_RANGE[0] <= value <= YEAR_RANGE[1]


def _column_aggregates(frame: pd.DataFrame) -> list[float]:
    """Column totals, and the share one total is of another.

    §2 of the issue allows a narrative number to be "a derived percentage of two table values" —
    "returns are 5.7 % of units sold" is a fair claim even though no single cell holds 5.7. This
    is that rule, kept deliberately narrow: **column totals only**, and only ratios that land in
    ``(0, 1]``. Allowing every pair of cells would make almost any number matchable and the guard
    would stop guarding anything.
    """
    totals: list[float] = []
    for column in frame.columns:
        numeric = pd.to_numeric(frame[column], errors="coerce").dropna()
        if len(numeric):
            totals.append(float(numeric.sum()))

    shares = [
        numerator / denominator
        for numerator in totals
        for denominator in totals
        if denominator > 0 and 0 < numerator / denominator <= 1
    ]
    return totals + shares


def collect_table_values(tables: dict[str, Any]) -> list[float]:
    """Every numeric value in the supplied tables, flattened.

    Accepts DataFrames and plain dicts/lists — the analyses write both, and a JSON table read back
    with :func:`pipeline.eda.io.load_table` is a DataFrame while ``data_quality_findings.json`` is
    a nested dict. Column totals and the shares between them are included too, so a narrative may
    quote an aggregate it derived from the table rather than a single cell.
    """
    values: list[float] = []

    def walk(item: Any) -> None:
        if isinstance(item, pd.DataFrame):
            for column in item.columns:
                numeric = pd.to_numeric(item[column], errors="coerce").dropna()
                values.extend(float(value) for value in numeric)
            values.extend(_column_aggregates(item))
        elif isinstance(item, pd.Series):
            numeric = pd.to_numeric(item, errors="coerce").dropna()
            values.extend(float(value) for value in numeric)
        elif isinstance(item, dict):
            for entry in item.values():
                walk(entry)
        elif isinstance(item, list | tuple):
            for entry in item:
                walk(entry)
        elif isinstance(item, bool):
            return  # True is not the number 1
        elif isinstance(item, int | float):
            values.append(float(item))

    walk(tables)
    return values


def _close(written: float, actual: float, decimals: int, tolerance_rel: float) -> bool:
    """Does ``actual`` round to the number the sentence actually shows?

    The slack is the *smaller* of the relative tolerance and half a unit in the last displayed
    decimal place. Both bounds matter:

    * the relative tolerance absorbs float noise and the rounding a share picks up on its way to
      a percentage — ``0.35512`` displayed as ``35.5 %``;
    * the display precision stops that tolerance becoming a licence at scale. Without the cap,
      a tenth of a percent of 4,999 is 25, so ``4,999`` would happily "match" a table value of
      4,998 — a different number, which is exactly what the guard exists to catch.
    """
    quantum = 0.5 * (10.0**-decimals)
    scale = max(abs(written), abs(actual), 1e-12)
    allowed = min(tolerance_rel * scale, quantum)
    return abs(written - actual) <= allowed + 1e-12


def _backs(token: NumberToken, table_values: list[float], tolerance_rel: float) -> bool:
    """Is this token citing one of the table values?"""
    for candidate in token.candidates:
        # `candidates` may rescale the written value (a percentage read as a share), so compare
        # each candidate at the precision the sentence displayed it with.
        ratio = candidate / token.value if token.value else 1.0
        for value in table_values:
            if _close(candidate, value, token.decimals, tolerance_rel):
                return True
            # The sentence may show a rounded form of the table value: 0.35512 -> "35.5 %".
            if token.is_percent and _close(token.value, value * 100.0, token.decimals,
                                           tolerance_rel):
                return True
            if ratio and ratio != 1.0 and _close(
                token.value, value / ratio, token.decimals, tolerance_rel
            ):
                return True
    return False


def numbers_in_tables(
    text: str,
    tables: dict[str, Any],
    tolerance_rel: float = DEFAULT_TOLERANCE_REL,
) -> NarrativeCheck:
    """Check that every number in ``text`` appears in one of ``tables``.

    Args:
        text: the narrative — a markdown document, a paragraph, an agent's answer.
        tables: name -> DataFrame (or dict/list) of everything the narrative is allowed to cite.
        tolerance_rel: relative slack, defaulting to a tenth of a percent.

    Returns:
        A :class:`NarrativeCheck`. ``passed`` is false when any number is unaccounted for, and
        ``unmatched`` lists those numbers **as they were written**, so the message a human reads
        points at the text rather than at a float.
    """
    table_values = collect_table_values(tables)
    unmatched: list[str] = []
    checked = 0
    for token in _tokens(text):
        if _is_whitelisted(token):
            continue
        checked += 1
        if not _backs(token, table_values, tolerance_rel):
            unmatched.append(token.raw)
    return NarrativeCheck(
        passed=not unmatched,
        unmatched=unmatched,
        checked=checked,
        table_values=len(table_values),
    )
