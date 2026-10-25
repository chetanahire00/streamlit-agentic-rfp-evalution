"""Criteria tab: edit criteria in a data editor with live weight total and validated save."""

from __future__ import annotations

import math
from typing import Any

import pandas as pd
import streamlit as st

from config.settings import Settings
from src.db.repository import CriterionDraft, Repository
from src.tools.input_validation import validate_active_criteria, validate_criteria_drafts, weight_total
from src.ui.components import render_html, weight_total_html
from src.ui.state import flash
from src.utils.errors import AppError
from src.utils.logging_setup import get_logger

log = get_logger(__name__)

COLUMNS = ["criterion_id", "is_active", "name", "description", "weight", "max_score"]


def _num(value: Any) -> float:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return math.nan
    return f


def drafts_from_frame(df: pd.DataFrame) -> list[CriterionDraft]:
    """Convert edited rows to drafts (blank trailing rows are ignored)."""
    drafts: list[CriterionDraft] = []
    for row in df.to_dict("records"):
        name = "" if pd.isna(row.get("name")) else str(row.get("name"))
        desc = "" if pd.isna(row.get("description")) else str(row.get("description"))
        weight, max_score = _num(row.get("weight")), _num(row.get("max_score"))
        if not name.strip() and not desc.strip() and math.isnan(weight) and math.isnan(max_score):
            continue
        cid = row.get("criterion_id")
        drafts.append(CriterionDraft(
            criterion_id=None if cid is None or pd.isna(cid) else int(cid),
            name=name, description=desc, weight=weight, max_score=max_score,
            is_active=bool(row.get("is_active")) if not pd.isna(row.get("is_active")) else False,
        ))
    return drafts


def render_criteria_tab(repo: Repository, settings: Settings) -> None:
    """Draw the Criteria tab."""
    st.subheader("Evaluation criteria")
    st.caption("Criteria are stored in SQLite and injected into the prompt at run time. Toggle "
               "**Active**, edit weights or add rows, then **Save**. Active weights must total 100. "
               "Each run snapshots the criteria when it starts, so later edits never change past runs.")

    saved = repo.list_criteria()
    frame = pd.DataFrame([c.model_dump() for c in saved], columns=COLUMNS)
    edited = st.data_editor(
        frame,
        key="criteria_editor",
        num_rows="dynamic",
        hide_index=True,
        width="stretch",
        column_order=["is_active", "name", "description", "weight", "max_score"],
        column_config={
            "criterion_id": st.column_config.NumberColumn("ID", disabled=True),
            "is_active": st.column_config.CheckboxColumn("Active", default=True,
                                                         help="Only active criteria are evaluated."),
            "name": st.column_config.TextColumn("Criterion", required=True, max_chars=80),
            "description": st.column_config.TextColumn("What the LLM should inspect", width="large",
                                                       max_chars=400),
            "weight": st.column_config.NumberColumn("Weight (%)", min_value=0.0, max_value=100.0,
                                                    step=0.5, format="%.2f", required=True),
            "max_score": st.column_config.NumberColumn("Max score", min_value=0.5, max_value=1000.0,
                                                       step=0.5, format="%.1f", required=True,
                                                       default=10.0),
        },
    )
    drafts = drafts_from_frame(edited)
    total = weight_total(drafts)

    col_total, col_save = st.columns([3, 1], vertical_alignment="center")
    with col_total:
        render_html(weight_total_html(total, settings.weight_total, settings.weight_tolerance))
    with col_save:
        save = st.button("Save criteria", type="primary", icon=":material/save:", width="stretch",
                         key="criteria_save")

    issues = validate_criteria_drafts(drafts)
    if save:
        if issues:
            st.error("Not saved. Fix these problems first:\n\n"
                     + "\n".join(f"- **{i.field}**: {i.message}" for i in issues))
        else:
            try:
                repo.save_criteria(drafts)
            except AppError as exc:
                st.error(exc.user_message)
            else:
                # Drop the editor's pending edits; otherwise they would be re-applied
                # on top of the freshly saved rows after the rerun.
                st.session_state.pop("criteria_editor", None)
                flash("success", "Criteria saved.")
                st.rerun()
    elif issues:
        st.warning("Unsaved changes have problems:\n\n"
                   + "\n".join(f"- **{i.field}**: {i.message}" for i in issues))

    db_issues = validate_active_criteria(saved, settings)
    if db_issues:
        st.error(f"Evaluation is blocked: {db_issues[0].message}", icon=":material/block:")
    else:
        st.success(f"{sum(c.is_active for c in saved)} active criteria saved; ready for evaluation.",
                   icon=":material/check_circle:")
