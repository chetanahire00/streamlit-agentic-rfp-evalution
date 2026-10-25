"""Scorecards tab: per-criterion drill-down with benchmark, gap, relative % and evidence."""

from __future__ import annotations

import html

import pandas as pd
import streamlit as st

from src.models.schemas import RunResult, SupplierRanking, SupplierScorecard
from src.ui.components import badge_html, medal, render_empty, render_html, status_badge_html

GAP_NEGATIVE = "color: #B91C1C; font-weight: 600"   # danger red, 6.5:1 on white


def scorecard_frame(card: SupplierScorecard, ranking: SupplierRanking) -> pd.DataFrame:
    """One row per criterion, joining validated scores with peer metrics (pure)."""
    metrics = {m.criterion_id: m for m in ranking.metrics}
    rows = []
    for cs in card.criteria:
        m = metrics.get(cs.criterion_id)
        page = f" (p. {cs.evidence_page})" if cs.evidence_page else ""
        if cs.evidence:
            evidence_status = "⚠ Unverified" if cs.evidence_unverified else "✓ Found in PDF"
        else:
            evidence_status = "⚠ No evidence" if cs.evidence_unverified else "—"
        rows.append({
            "Criterion": cs.criterion_name,
            "Score": cs.score,
            "Max": cs.max_score,
            "Benchmark": m.benchmark if m else None,
            "Gap": m.gap if m else None,
            "Relative %": m.relative_pct if m else None,
            "Weight": cs.weight,
            "Weighted": m.weighted_score if m else None,
            "Evidence": (f'"{cs.evidence}"{page}' if cs.evidence else "—"),
            "Evidence check": evidence_status,
            "Justification": cs.justification,
        })
    return pd.DataFrame(rows)


def _style_gap(value: object) -> str:
    return GAP_NEGATIVE if isinstance(value, int | float) and value < 0 else ""


def render_scorecard_tab(run: RunResult | None) -> None:
    """Draw the Scorecards tab."""
    st.subheader("Supplier scorecards")
    if run is None:
        render_empty("No scorecards yet", "Run an evaluation (or load a past run) to see each supplier's "
                     "criterion scores, peer benchmarks and supporting evidence.")
        return

    rankings = {r.supplier_name: r for r in run.ranking.rankings}
    cards = {c.supplier_name: c for c in run.scorecards}
    ordered = [r.supplier_name for r in run.ranking.rankings]
    name = st.selectbox("Supplier", ordered, key=f"sc_{run.rfp_run_id}",
                        format_func=lambda n: f"{medal(rankings[n].final_rank)} · {n}")
    card, rank = cards[name], rankings[name]

    badges = [status_badge_html(card.status.value),
              badge_html(f"Rank {rank.final_rank} of {len(ordered)}", "info"),
              badge_html(f"PPI {rank.ppi:.2f}", "neutral"),
              badge_html(f"Absolute {rank.absolute_score:.2f}", "neutral")]
    if card.from_cache:
        badges.append(badge_html("↺ From cache", "neutral"))
    unverified = sum(c.evidence_unverified for c in card.criteria)
    if unverified:
        badges.append(badge_html(f"⚠ {unverified} unverified evidence", "warning"))
    render_html(" ".join(badges))
    st.caption(f"{card.filename} · {card.page_count} page(s) · {card.char_count:,} characters · "
               f"extracted with {card.extraction_method or 'n/a'} · submitted {card.submission_date} · "
               f"experience {card.experience_rating:g}")

    frame = scorecard_frame(card, rank)
    styled = frame.style.map(_style_gap, subset=["Gap"]).format(
        {"Score": "{:g}", "Max": "{:g}", "Benchmark": "{:g}", "Gap": "{:+g}", "Weight": "{:g}",
         "Weighted": "{:.2f}"}, na_rep="—")
    st.dataframe(
        styled, hide_index=True, width="stretch",
        column_config={
            "Relative %": st.column_config.ProgressColumn(
                "Relative %", min_value=0, max_value=100, format="%.1f%%",
                help="Score ÷ best peer score × 100 (0 when every supplier scored 0)"),
            "Gap": st.column_config.Column("Gap", help="Score − benchmark; negative values are red"),
            "Evidence": st.column_config.TextColumn("Evidence (quote, page)", width="large"),
            "Justification": st.column_config.TextColumn(width="large"),
        },
    )

    st.markdown("#### Evidence and justification")
    for cs in card.criteria:
        m = next((x for x in rank.metrics if x.criterion_id == cs.criterion_id), None)
        label = f"{cs.criterion_name}: {cs.score:g}/{cs.max_score:g}"
        if cs.evidence_unverified:
            label += " · unverified evidence"
        with st.expander(label, icon=":material/format_quote:"):
            if m:
                st.progress(min(1.0, m.relative_pct / 100), text=f"{m.relative_pct:.1f}% of the best peer "
                            f"(benchmark {m.benchmark:g}, gap {m.gap:+g}, weight {cs.weight:g}%)")
            if cs.evidence:
                cls = "rfp-quote rfp-quote--unverified" if cs.evidence_unverified else "rfp-quote"
                page = f" (page {cs.evidence_page})" if cs.evidence_page else ""
                render_html(f'<blockquote class="{cls}">“{html.escape(cs.evidence)}”{page}</blockquote>')
                if cs.evidence_unverified:
                    render_html(badge_html("⚠ Quote not found in the extracted PDF text; verify manually",
                                           "warning"))
            else:
                st.caption("No evidence quote provided.")
            st.markdown(f"**Justification:** {cs.justification}")

    left, right = st.columns(2)
    with left:
        st.markdown("#### Risks")
        if card.risks:
            st.markdown("\n".join(f"- {r}" for r in card.risks))
        else:
            st.caption("No risks reported.")
    with right:
        st.markdown("#### Overall summary")
        st.write(card.overall_summary or "—")
