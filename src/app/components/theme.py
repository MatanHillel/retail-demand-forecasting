"""Light CSS theming (US-27). Reuses the pipeline's colour palette, never redefines it."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from pipeline.eda.style import ABC_COLORS

_CSS_PATH = Path(__file__).resolve().parent.parent / "assets" / "style.css"


def apply_theme() -> None:
    """Inject the ABC palette as CSS custom properties, then the bundled stylesheet."""
    variables = "\n".join(f"  --abc-{key.lower()}: {value};" for key, value in ABC_COLORS.items())
    st.markdown(f"<style>:root {{\n{variables}\n}}</style>", unsafe_allow_html=True)
    if _CSS_PATH.exists():
        css = _CSS_PATH.read_text(encoding="utf-8")
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
