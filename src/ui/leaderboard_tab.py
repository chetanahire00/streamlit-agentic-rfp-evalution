"""Leaderboard tab: KPI cards, ranked table, PPI/absolute bars and a criterion radar."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.models.schemas import RunResult
from src.ui.components import (
    MAX_SERIES,
    is_dark_theme,
    kpi_html,
    medal,
    radar_chart,
    render_empty,
    render_html,
    score_bar_chart,
)


def leaderboard_frame(run: RunResult) -> pd.DataFrame:
    """Table rows in final rank order (pure, for testing)."""
    status = {c.supplier_name: c.status.value for c in run.scorecards}
    return pd.DataFrame([
        {
            "Rank": medal(r.final_rank),
            "Supplier": r.supplier_name,
            "Absolute score": r.absolute_score,
            "PPI": r.ppi,
            "Submission date": r.submission_date.isoformat(),
            "Experience rating": r.experience_rating,
            "Status": status.get(r.supplier_name, ""),
        }
        for r in run.ranking.rankings
    ])


def render_leaderboard_tab(run: RunResult | None) -> None:
    """Draw the Leaderboard tab."""
    st.subheader("Leaderboard")
    if run is None:
        render_empty("No evaluation yet",
                     "Configure a provider in the sidebar, add supplier PDFs in the Supplier Input tab "
                     "and click Evaluate. You can also reopen a past run from Run Details.")
        return

    top = run.ranking.rankings[0]
    cols = st.columns(4)
    cards = [
        ("Suppliers evaluated", str(len(run.scorecards)),
         f"{sum(c.status.value == 'OK' for c in run.scorecards)} fully evaluated"),
        ("Top supplier", top.supplier_name, f"Rank 1 · {run.model}"),
        ("Top PPI", f"{top.ppi:.2f}", f"Absolute score {top.absolute_score:.2f}"),
        ("Warnings", str(len(run.all_warnings())), "See Run Details"),
    ]
    for col, (label, value, hint) in zip(cols, cards, strict=True):
        with col:
            render_html(kpi_html(label, value, hint))

    st.markdown("#### Ranking")
    st.dataframe(
        leaderboard_frame(run),
        hide_index=True,
        width="stretch",
        column_config={
            "Rank": st.column_config.TextColumn("Rank", width="small"),
            "Absolute score": st.column_config.ProgressColumn(
                "Absolute score", min_value=0, max_value=100, format="%.2f",
                help="Σ (score ÷ max score) × weight, on a 0–100 scale"),
            "PPI": st.column_config.ProgressColumn(
                "PPI", min_value=0, max_value=100, format="%.2f",
                help="Peer Performance Index: weighted average of relative % against the best peer"),
            "Experience rating": st.column_config.NumberColumn(format="%.1f"),
        },
    )
    st.caption("Order: higher PPI → earlier submission date → higher experience rating → supplier name (A–Z). "
               "Ranks are sequential with no ties.")

    dark = is_dark_theme()
    left, right = st.columns([1, 1])
    with left:
        st.markdown("#### PPI and absolute score")
        st.plotly_chart(score_bar_chart(run, dark), width="stretch", theme="streamlit",
                        config={"displaylogo": False})
    with right:
        st.markdown("#### Criterion comparison")
        names = [r.supplier_name for r in run.ranking.rankings]
        chosen = st.multiselect(
            "Suppliers on the radar", names, default=names[: min(4, MAX_SERIES)],
            max_selections=MAX_SERIES, key=f"radar_{run.rfp_run_id}",
            help=f"Up to {MAX_SERIES} suppliers, so every line keeps a distinct colour.")
        if chosen:
            st.plotly_chart(radar_chart(run, chosen, dark), width="stretch", theme="streamlit",
                            config={"displaylogo": False})
            st.caption("Each axis shows the supplier's score as a % of that criterion's maximum.")
        else:
            render_empty("Nothing selected", "Choose at least one supplier to draw the radar chart.")
