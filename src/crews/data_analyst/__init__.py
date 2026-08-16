"""Data Analyst Crew — the three agents of PRD §35 (US-12).

Data Quality Analyst · Data Preparation Analyst · Business & EDA Analyst. Between them they
produce ``clean_data.csv``, ``eda_report.html``, ``insights.md`` and ``dataset_contract.json``,
using only the approved deterministic tools of :mod:`crews.data_analyst.tools` — the agents
interpret and write, they never calculate (PRD §38).
"""

from crews.data_analyst.crew import (
    AGENT_ORDER,
    TASK_ORDER,
    DataAnalystCrew,
    build_crew,
    run_data_analyst_crew,
    verify_outputs,
)
from crews.data_analyst.tools import DataAnalystToolset, make_tools

__all__ = [
    "AGENT_ORDER",
    "TASK_ORDER",
    "DataAnalystCrew",
    "DataAnalystToolset",
    "build_crew",
    "make_tools",
    "run_data_analyst_crew",
    "verify_outputs",
]
