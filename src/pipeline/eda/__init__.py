"""Shared EDA foundations (US-06, PRD §35A.2).

This package holds the cross-cutting plumbing every exploratory analysis reuses — the one
plotting style (:mod:`pipeline.eda.style`) and the one set of artifact readers/writers
(:mod:`pipeline.eda.io`). It deliberately contains **no analysis**: E1 lives in US-07, E2–E7 in
US-09 and E8–E14 in US-10. The ABC classification they all share is :mod:`pipeline.abc`, which
sits one level up because evaluation (§23), σ fallback (§27) and inventory KPIs (§30) use it too
and must not import an EDA package to get it.
"""

from __future__ import annotations
