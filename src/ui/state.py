"""Session-state keys and accessors shared by the UI modules."""

from __future__ import annotations

from typing import Any

import streamlit as st

from src.models.schemas import RunResult

CURRENT_RUN = "current_run"
CONN_STATUS = "connection_status"
SAMPLES_LOADED = "samples_loaded"
FLASH = "flash_message"
ACTIVE_TAB = "active_tab"


def get_run() -> RunResult | None:
    """The run currently shown in the Leaderboard/Scorecards/Run Details tabs."""
    value = st.session_state.get(CURRENT_RUN)
    return value if isinstance(value, RunResult) else None


def set_run(run: RunResult | None) -> None:
    """Make ``run`` the active run."""
    st.session_state[CURRENT_RUN] = run


def flash(kind: str, message: str) -> None:
    """Queue a one-off message to show after the next rerun."""
    st.session_state[FLASH] = (kind, message)


def pop_flash() -> tuple[str, str] | None:
    """Return and clear the queued message."""
    value: Any = st.session_state.pop(FLASH, None)
    return value if isinstance(value, tuple) else None
