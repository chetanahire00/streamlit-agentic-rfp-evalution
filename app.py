"""Agentic RFP Evaluation and Supplier Ranking: Streamlit entry point.

Run locally with ``streamlit run app.py``.
"""

from __future__ import annotations

import logging

import streamlit as st

from config.settings import PROMPT_VERSION, Settings, get_settings
from src.db.repository import Repository
from src.llm.catalog import ProviderOption, load_catalog
from src.ui import state
from src.ui.components import footer_html, header_html, inject_css, render_html
from src.ui.criteria_tab import render_criteria_tab
from src.ui.input_tab import render_input_tab
from src.ui.leaderboard_tab import render_leaderboard_tab
from src.ui.run_details_tab import render_run_details_tab
from src.ui.scorecard_tab import render_scorecard_tab
from src.ui.sidebar import render_sidebar
from src.utils.logging_setup import configure_logging, get_logger

log = get_logger("src.app")

TABS = {
    "criteria": ":material/tune: Criteria",
    "input": ":material/upload_file: Supplier Input",
    "leaderboard": ":material/leaderboard: Leaderboard",
    "scorecards": ":material/fact_check: Scorecards",
    "details": ":material/receipt_long: Run Details",
}
TAB_KEY = "main_tabs"


@st.cache_resource(show_spinner=False)
def get_repository(db_path: str) -> Repository:
    """Create the repository and initialise the schema once per process (step 1: Setup)."""
    repo = Repository(db_path)
    repo.initialize()
    return repo


@st.cache_resource(show_spinner=False)
def get_catalog(path: str) -> list[ProviderOption]:
    """Provider/model catalogue from config/providers.yaml."""
    from pathlib import Path

    return load_catalog(Path(path))


def main(settings: Settings | None = None) -> None:
    """Render the whole app."""
    settings = settings or get_settings()
    configure_logging(logging.INFO)
    st.set_page_config(page_title="Agentic RFP Evaluator", page_icon=":material/analytics:",
                       layout="wide", initial_sidebar_state="expanded")
    inject_css()

    try:
        repo = get_repository(str(settings.db_path))
        catalog = get_catalog(str(settings.providers_file))
    except Exception:
        log.exception("Startup failed")
        st.error("The app could not start (database or configuration problem). Check the logs.")
        st.stop()

    run = state.get_run()
    render_html(header_html(run))
    sidebar = render_sidebar(catalog, settings)

    message = state.pop_flash()
    if message:
        kind, text = message
        {"success": st.success, "warning": st.warning, "info": st.info}.get(kind, st.info)(text)

    requested = st.session_state.pop(state.ACTIVE_TAB, None)
    if requested in TABS:
        st.session_state[TAB_KEY] = TABS[requested]

    tabs = st.tabs(list(TABS.values()), key=TAB_KEY, on_change="rerun")
    renderers = [
        lambda: render_criteria_tab(repo, settings),
        lambda: render_input_tab(repo, settings, sidebar),
        lambda: render_leaderboard_tab(run),
        lambda: render_scorecard_tab(run),
        lambda: render_run_details_tab(run, repo),
    ]
    for tab, render in zip(tabs, renderers, strict=True):
        with tab:
            try:
                render()
            except Exception:
                log.exception("Rendering a tab failed")
                st.error("This section could not be displayed. The error was logged; try reloading the page.")

    render_html(footer_html(PROMPT_VERSION))


if __name__ == "__main__":
    main()
