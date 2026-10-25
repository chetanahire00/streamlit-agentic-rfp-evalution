"""Reusable presentation helpers: HTML snippets, formatting and Plotly charts.

Builders that return HTML strings or figures are pure (no Streamlit calls) so
they can be unit-tested; ``render_*`` helpers write to the page.
"""

from __future__ import annotations

import html
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

from src.models.schemas import RunResult, RunStatus, SupplierStatus

CSS_FILE = Path(__file__).resolve().parents[2] / "assets" / "styles.css"

# Categorical palette from the dataviz reference instance (validated: CVD-safe in
# this fixed order, light and dark steps). Colour follows the supplier, not its rank.
SERIES_LIGHT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SERIES_DARK = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"]
MAX_SERIES = len(SERIES_LIGHT)

MEDALS = {1: "🥇", 2: "🥈", 3: "🥉"}

_LOGO_SVG = (
    '<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#FFFFFF" stroke-width="2" '
    'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
    '<path d="M3 3v18h18"/><path d="M7 15l4-4 3 3 5-6"/></svg>'
)

STATUS_TONES = {
    RunStatus.COMPLETED.value: ("success", "✓ Completed"),
    RunStatus.PARTIAL.value: ("warning", "! Partial"),
    RunStatus.FAILED.value: ("danger", "✕ Failed"),
    RunStatus.RUNNING.value: ("info", "… Running"),
    RunStatus.CREATED.value: ("neutral", "○ Created"),
    SupplierStatus.OK.value: ("success", "✓ Evaluated"),
    SupplierStatus.NO_TEXT.value: ("warning", "! No text"),
    SupplierStatus.EVALUATION_FAILED.value: ("danger", "✕ Invalid output"),
    SupplierStatus.LLM_FAILED.value: ("danger", "✕ LLM failed"),
}


# ── pure HTML builders ──────────────────────────────────────────────────────
def badge_html(text: str, tone: str = "neutral") -> str:
    """A pill badge. ``tone`` is success | warning | danger | info | neutral."""
    return f'<span class="rfp-badge rfp-badge--{tone}">{html.escape(text)}</span>'


def status_badge_html(status: str) -> str:
    """Badge for a run or supplier status (icon + word, never colour alone)."""
    tone, label = STATUS_TONES.get(status, ("neutral", status))
    return badge_html(label, tone)


def kpi_html(label: str, value: str, hint: str = "") -> str:
    """A KPI card."""
    hint_html = f'<div class="rfp-kpi__hint">{html.escape(hint)}</div>' if hint else ""
    return (f'<div class="rfp-kpi" role="group" aria-label="{html.escape(label)}">'
            f'<div class="rfp-kpi__label">{html.escape(label)}</div>'
            f'<div class="rfp-kpi__value">{html.escape(value)}</div>{hint_html}</div>')


def empty_state_html(title: str, body: str) -> str:
    """Friendly placeholder shown instead of a blank screen."""
    return (f'<div class="rfp-empty" role="status"><p class="rfp-empty__title">{html.escape(title)}</p>'
            f'<p class="rfp-empty__body">{html.escape(body)}</p></div>')


def header_html(run: RunResult | None) -> str:
    """Branded page header; shows the active run id when there is one."""
    meta = ""
    if run is not None:
        meta = (f'<div class="rfp-header__meta">Active run<br><code>{html.escape(run.rfp_run_id[:8])}…</code>'
                f" · {html.escape(run.status.value)}</div>")
    return (f'<header class="rfp-header"><div class="rfp-header__logo">{_LOGO_SVG}</div>'
            '<div><div class="rfp-header__title" role="heading" aria-level="1">Agentic RFP Evaluator</div>'
            '<div class="rfp-header__subtitle">AI-assisted supplier scoring with deterministic, '
            'explainable peer ranking</div></div>' + meta + "</header>")


def footer_html(prompt_version: str) -> str:
    """Page footer."""
    return ('<footer class="rfp-footer"><span>Agentic RFP Evaluation &amp; Supplier Ranking · '
            'classroom mini project · all sample data is fictional</span>'
            f'<span>Prompt {html.escape(prompt_version)} · LLM judges content; Python does the maths</span>'
            "</footer>")


def checklist_html(items: list[tuple[bool, str]]) -> str:
    """Readiness checklist with ✓/✕ glyphs and text."""
    rows = "".join(
        f'<li><span class="{"rfp-check__ok" if ok else "rfp-check__no"}" aria-hidden="true">'
        f'{"✓" if ok else "✕"}</span><span>{html.escape(text)}'
        f'<span class="sr-only">{" (done)" if ok else " (to do)"}</span></span></li>'
        for ok, text in items
    )
    return f'<ul class="rfp-check">{rows}</ul>'


def weight_total_html(total: float, target: float, tolerance: float) -> str:
    """Green at the target total, red otherwise (with a text cue)."""
    ok = abs(total - target) <= tolerance
    tone, icon = ("ok", "✓") if ok else ("bad", "✕")
    note = "ready" if ok else f"must total {target:g}"
    return (f'<div class="rfp-total rfp-total--{tone}" role="status" aria-live="polite">'
            f'<span aria-hidden="true">{icon}</span> Active weight total: {total:g} ({note})</div>')


def medal(rank: int) -> str:
    """Medal for ranks 1–3, else ``#n``."""
    return f"{MEDALS[rank]} {rank}" if rank in MEDALS else f"#{rank}"


def friendly_supplier_name(filename: str) -> str:
    """Default supplier name from a file name: ``apex_systems.pdf`` → ``Apex Systems``."""
    stem = Path(filename).stem.replace("_", " ").replace("-", " ")
    return " ".join(w if w.isupper() else w.capitalize() for w in stem.split())


# ── charts ──────────────────────────────────────────────────────────────────
def series_colors(dark: bool = False) -> list[str]:
    """Categorical colours for the current theme."""
    return SERIES_DARK if dark else SERIES_LIGHT


def color_map(names: list[str], dark: bool = False) -> dict[str, str]:
    """Stable supplier→colour mapping (alphabetical, so colour follows the entity)."""
    colors = series_colors(dark)
    return {n: colors[i % MAX_SERIES] for i, n in enumerate(sorted(names, key=str.casefold))}


def _base_layout(fig: go.Figure, height: int) -> go.Figure:
    fig.update_layout(
        height=height, margin={"l": 8, "r": 8, "t": 36, "b": 8},
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
        hoverlabel={"font_size": 13},
    )
    return fig


def score_bar_chart(run: RunResult, dark: bool = False) -> go.Figure:
    """Horizontal grouped bars: PPI and absolute score per supplier on one 0–100 axis."""
    ranks = list(reversed(run.ranking.rankings))  # rank 1 at the top
    names = [f"{medal(r.final_rank)}  {r.supplier_name}" for r in ranks]
    colors = series_colors(dark)
    fig = go.Figure()
    fig.add_bar(name="PPI", y=names, x=[r.ppi for r in ranks], orientation="h",
                marker={"color": colors[0], "cornerradius": 4},
                hovertemplate="%{y}<br>PPI: %{x:.2f}<extra></extra>")
    fig.add_bar(name="Absolute score", y=names, x=[r.absolute_score for r in ranks], orientation="h",
                marker={"color": colors[1], "cornerradius": 4},
                hovertemplate="%{y}<br>Absolute score: %{x:.2f}<extra></extra>")
    fig.update_layout(barmode="group", bargap=0.3, bargroupgap=0.08)
    fig.update_xaxes(range=[0, 105], title_text="Score (0–100)", gridcolor="rgba(148,163,184,0.25)",
                     zeroline=False)
    fig.update_yaxes(title_text=None)
    return _base_layout(fig, max(260, 70 * len(ranks) + 80))


def radar_chart(run: RunResult, suppliers: list[str], dark: bool = False) -> go.Figure:
    """Radar of each selected supplier's score as % of max per criterion."""
    cmap = color_map([r.supplier_name for r in run.ranking.rankings], dark)
    by_name = {r.supplier_name: r for r in run.ranking.rankings}
    fig = go.Figure()
    for name in suppliers[:MAX_SERIES]:
        r = by_name[name]
        labels = [m.criterion_name for m in r.metrics]
        values = [round(m.score / m.max_score * 100, 1) if m.max_score else 0 for m in r.metrics]
        fig.add_trace(go.Scatterpolar(
            r=values + values[:1], theta=labels + labels[:1], name=name, mode="lines+markers",
            line={"color": cmap[name], "width": 2}, marker={"size": 8},
            hovertemplate=f"{html.escape(name)}<br>%{{theta}}: %{{r:.0f}}% of max<extra></extra>",
        ))
    fig.update_layout(polar={"radialaxis": {"range": [0, 100], "ticksuffix": "%", "angle": 90,
                                            "tickangle": 90, "tickfont": {"size": 10},
                                            "gridcolor": "rgba(148,163,184,0.35)"},
                             "angularaxis": {"gridcolor": "rgba(148,163,184,0.35)", "rotation": 90,
                                             "direction": "clockwise"},
                             "domain": {"x": [0.18, 0.82], "y": [0.02, 0.9]},
                             "bgcolor": "rgba(0,0,0,0)"})
    fig = _base_layout(fig, 480)
    fig.update_layout(margin={"l": 24, "r": 24, "t": 48, "b": 24})
    return fig


# ── Streamlit renderers ─────────────────────────────────────────────────────
def inject_css() -> None:
    """Inject the design-system stylesheet once per page run."""
    css = CSS_FILE.read_text(encoding="utf-8")
    st.html(f"<style>{css}</style>")


def render_html(fragment: str) -> None:
    """Write trusted, escaped HTML produced by the builders above."""
    st.markdown(fragment, unsafe_allow_html=True)


def render_empty(title: str, body: str) -> None:
    """Show an empty state."""
    render_html(empty_state_html(title, body))


def is_dark_theme() -> bool:
    """True when the viewer is using Streamlit's dark theme."""
    try:
        return st.context.theme.type == "dark"
    except Exception:  # noqa: BLE001 - older runtimes / bare execution
        return False
