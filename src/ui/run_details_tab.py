"""Run Details tab: run id, status, snapshot, warnings, tie-breaks, JSON export, history."""

from __future__ import annotations

from collections import defaultdict

import pandas as pd
import streamlit as st

from src.agents.orchestrator import export_json
from src.db.repository import Repository
from src.models.schemas import RunResult, WarningRecord
from src.ui.components import badge_html, render_empty, render_html, status_badge_html
from src.ui.state import flash, set_run
from src.utils.logging_setup import get_logger

log = get_logger(__name__)


def group_warnings(warnings: list[WarningRecord]) -> dict[str, list[WarningRecord]]:
    """Warnings grouped by supplier ("Run" for run-level ones), stable order (pure)."""
    groups: dict[str, list[WarningRecord]] = defaultdict(list)
    for w in warnings:
        groups[w.supplier_name or "Run"].append(w)
    return dict(sorted(groups.items(), key=lambda kv: (kv[0] != "Run", kv[0].casefold())))


def render_run_details_tab(run: RunResult | None, repo: Repository) -> None:
    """Draw the Run Details tab (history selector is always available)."""
    st.subheader("Run details")
    _history(repo, run)
    if run is None:
        render_empty("No run selected", "Finish an evaluation or load a past run from the history above.")
        return

    st.markdown("##### RFP_RUN_ID")
    st.code(run.rfp_run_id, language=None)
    render_html(" ".join([
        status_badge_html(run.status.value),
        badge_html(f"Provider: {run.provider}", "neutral"),
        badge_html(f"Model: {run.model}", "neutral"),
        badge_html(f"Prompt {run.prompt_version}", "neutral"),
        badge_html(f"Created {run.created_at} (UTC)", "neutral"),
    ]))
    if run.provider == "demo":
        st.info("This run used the offline Demo provider (deterministic keyword scoring, not an AI model).",
                icon=":material/info:")

    st.download_button("Download complete result (JSON)", data=export_json(run).encode("utf-8"),
                       file_name=f"rfp_run_{run.rfp_run_id}.json", mime="application/json",
                       icon=":material/download:", type="primary", key=f"dl_{run.rfp_run_id}")

    st.markdown("#### Tie-break explanations")
    if run.ranking.tie_break_explanations:
        for line in run.ranking.tie_break_explanations:
            st.markdown(f"- {line}")
    else:
        st.caption("No two suppliers had the same PPI, so no tie-break rule was needed.")

    st.markdown("#### Criteria snapshot")
    st.caption("The criteria exactly as they were when this run started.")
    st.dataframe(pd.DataFrame([{"ID": c.criterion_id, "Criterion": c.name, "Description": c.description,
                                "Weight (%)": c.weight, "Max score": c.max_score}
                               for c in run.criteria_snapshot]), hide_index=True, width="stretch")

    st.markdown("#### Warnings")
    groups = group_warnings(run.all_warnings())
    if not groups:
        st.success("No warnings: every result passed validation unchanged.", icon=":material/check_circle:")
    for supplier, items in groups.items():
        with st.expander(f"{supplier} · {len(items)} warning(s)", icon=":material/warning:"):
            st.dataframe(pd.DataFrame([{"Code": w.code, "Criterion": w.criterion_id or "—",
                                        "Message": w.message} for w in items]),
                         hide_index=True, width="stretch")


def _history(repo: Repository, run: RunResult | None) -> None:
    runs = repo.list_runs()
    if not runs:
        st.caption("No saved runs yet.")
        return
    options = {f"{r.created_at} · {r.status} · {r.provider}/{r.model} · {r.supplier_count} suppliers · "
               f"{r.rfp_run_id[:8]}": r.rfp_run_id for r in runs}
    current = next((label for label, rid in options.items() if run and rid == run.rfp_run_id), None)
    labels = list(options)
    col, btn = st.columns([4, 1], vertical_alignment="bottom")
    choice = col.selectbox("Run history", labels, index=labels.index(current) if current else 0,
                           key="history_select")
    if btn.button("Load run", icon=":material/history:", width="stretch", key="history_load"):
        loaded = repo.load_run(options[choice])
        if loaded is None:
            st.error("That run has no saved results (it may have failed before completion).")
        else:
            set_run(loaded)
            flash("info", f"Loaded run {loaded.rfp_run_id[:8]}… from SQLite.")
            st.rerun()
