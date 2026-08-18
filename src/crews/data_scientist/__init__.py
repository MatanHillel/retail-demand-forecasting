"""Data Scientist Crew — the three agents of PRD §36 (US-26).

Feature Engineering Specialist · Forecasting Model Scientist · Model Evaluation & Inventory
Scientist. Between them they produce ``features.csv``, ``model.joblib``, the back-test and
inventory forecasts, ``evaluation_report.md`` and ``model_card.md``, using only the approved
deterministic tools of :mod:`crews.data_scientist.tools` — the agents interpret and write, they
never calculate (PRD §38).
"""

from crews.data_scientist.crew import (
    AGENT_ORDER,
    TASK_ORDER,
    DataScientistCrew,
    build_crew,
    run_data_scientist_crew,
    verify_outputs,
)
from crews.data_scientist.tools import DataScientistToolset, make_tools

__all__ = [
    "AGENT_ORDER",
    "TASK_ORDER",
    "DataScientistCrew",
    "DataScientistToolset",
    "build_crew",
    "make_tools",
    "run_data_scientist_crew",
    "verify_outputs",
]
