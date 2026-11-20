"""Embed the self-contained synthetic Beta Testing visual proposal."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import re

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BETA_ASSET_ROOT = PROJECT_ROOT / "beta_testing"


def _read_asset(name: str) -> str:
    path = BETA_ASSET_ROOT / name
    if not path.is_file():
        raise FileNotFoundError(f"Required Beta Testing asset is missing: {path}")
    return path.read_text(encoding="utf-8")


@lru_cache(maxsize=1)
def build_beta_testing_html() -> str:
    """Inline local assets so the component has no localhost or external dependency."""
    document = _read_asset("index.html")
    stylesheet = _read_asset("styles.css")
    synthetic_data = _read_asset("synthetic-data.js")
    application = _read_asset("app.js")

    document, css_count = re.subn(
        r'<link rel="stylesheet" href="styles\.css[^\"]*">',
        lambda _: f"<style>{stylesheet}</style>",
        document,
        count=1,
    )
    document, data_count = re.subn(
        r'<script src="synthetic-data\.js[^\"]*"></script>',
        lambda _: f"<script>{synthetic_data}</script>",
        document,
        count=1,
    )
    document, app_count = re.subn(
        r'<script src="app\.js[^\"]*"></script>',
        lambda _: f"<script>{application}</script>",
        document,
        count=1,
    )
    if (css_count, data_count, app_count) != (1, 1, 1):
        raise ValueError("Beta Testing index.html does not contain the expected local assets.")

    return document


def render_beta_testing() -> None:
    """Render the three-page proposal in one isolated, responsive component."""
    try:
        document = build_beta_testing_html()
    except (OSError, UnicodeError, ValueError) as exc:
        st.error("The Beta Testing module could not load its packaged assets.")
        st.caption(str(exc))
        return

    with st.container(key="beta_testing_module"):
        st.iframe(document, width="stretch", height="content")
