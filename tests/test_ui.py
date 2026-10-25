"""Headless UI tests with Streamlit's AppTest (no browser, no network)."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from scripts import generate_sample_pdfs as gen
from src.models.schemas import RunStatus, WarningRecord
from src.ui import components as comp
from src.ui.criteria_tab import drafts_from_frame
from src.ui.leaderboard_tab import leaderboard_frame
from src.ui.run_details_tab import group_warnings
from src.ui.scorecard_tab import scorecard_frame
from src.ui.sidebar import resolve_api_key

APP = str(Path(__file__).resolve().parent.parent / "app.py")
DEMO = "Demo (offline, no key)"


@pytest.fixture(scope="module", autouse=True)
def _sample_pdfs() -> None:
    gen.build_all(Path(__file__).resolve().parent.parent / "sample_data")


@pytest.fixture()
def app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AppTest:
    monkeypatch.setenv("RFP_DB_PATH", str(tmp_path / "ui.db"))
    for k in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "GROQ_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    return at


def _demo_run(at: AppTest) -> AppTest:
    at.selectbox(key="sb_provider").select(DEMO).run()
    at.button(key="sb_test").click().run()
    at.button(key="toggle_samples").click().run()
    at.button(key="evaluate").click().run()
    return at


def test_initial_render(app: AppTest) -> None:
    assert not app.exception
    assert len(app.tabs) == 5
    assert app.button(key="evaluate").disabled  # no key, no files
    assert "Agentic RFP Evaluator" in "".join(m.value for m in app.markdown)
    assert any("No evaluation yet" in m.value for m in app.markdown)


def test_missing_key_blocks_and_explains(app: AppTest) -> None:
    app.selectbox(key="sb_provider").select("OpenAI").run()
    assert app.button(key="sb_test").disabled
    assert any("Enter an API key" in m.value for m in app.markdown)


def test_typed_key_is_masked_and_never_rendered(app: AppTest) -> None:
    secret = "sk-proj-THISISASECRETKEY-9876wxyz"
    app.selectbox(key="sb_provider").select("OpenAI").run()
    app.text_input(key="sb_key_openai").input(secret).run()
    page = " ".join(str(getattr(e, "value", "")) for e in app.main) + " ".join(
        str(getattr(e, "value", "")) for e in app.sidebar if getattr(e, "type", "") != "text_input")
    assert secret not in page
    assert any("sk-…wxyz" in c.value for c in app.caption)


def test_demo_end_to_end(app: AppTest) -> None:
    at = _demo_run(app)
    assert not at.exception
    run = at.session_state["current_run"]
    assert run.status is RunStatus.COMPLETED
    assert [r.final_rank for r in run.ranking.rankings] == [1, 2, 3, 4]
    assert any("Evaluation finished" in s.value for s in at.success)
    assert len(at.dataframe) >= 2  # leaderboard + scorecard tables
    # Reload from history.
    at.button(key="history_load").click().run()
    assert not at.exception
    assert any("Loaded run" in i.value for i in at.info)


def test_metadata_validation_inline(app: AppTest) -> None:
    app.selectbox(key="sb_provider").select(DEMO).run()
    app.button(key="toggle_samples").click().run()
    app.text_input(key="name_sample_nexaworks").input("apex systems").run()
    assert app.button(key="evaluate").disabled
    assert any("already used" in e.value for e in app.error)
    assert any("need fixing" in e.value for e in app.error)


def test_samples_toggle_and_connection_badge(app: AppTest) -> None:
    app.selectbox(key="sb_provider").select(DEMO).run()
    app.button(key="sb_test").click().run()
    assert any("Connected to" in m.value for m in app.sidebar.markdown)
    app.button(key="toggle_samples").click().run()
    assert app.button(key="toggle_samples").label == "Remove sample PDFs"
    app.button(key="toggle_samples").click().run()
    assert any("No proposals yet" in m.value for m in app.markdown)


def test_criteria_save_and_blocked_state(app: AppTest) -> None:
    app.button(key="criteria_save").click().run()
    assert not app.exception
    assert any("Criteria saved" in s.value for s in app.success)
    assert any("ready for evaluation" in s.value for s in app.success)


# ── pure helpers ────────────────────────────────────────────────────────────
def test_drafts_from_frame_handles_blank_and_new_rows() -> None:
    df = pd.DataFrame([
        {"criterion_id": 1, "is_active": True, "name": "A", "description": "d", "weight": 60, "max_score": 10},
        {"criterion_id": None, "is_active": None, "name": "New", "description": None, "weight": "40",
         "max_score": 5},
        {"criterion_id": None, "is_active": None, "name": None, "description": None, "weight": None,
         "max_score": None},
    ])
    drafts = drafts_from_frame(df)
    assert len(drafts) == 2
    assert drafts[1].criterion_id is None and drafts[1].weight == 40 and drafts[1].is_active is False


def test_component_builders() -> None:
    assert "rfp-badge--danger" in comp.status_badge_html("FAILED")
    assert "&lt;script&gt;" in comp.kpi_html("<script>", "1")
    assert comp.medal(1).startswith("🥇") and comp.medal(5) == "#5"
    assert comp.friendly_supplier_name("apex_systems-v2.pdf") == "Apex Systems V2"
    assert "rfp-total--ok" in comp.weight_total_html(100, 100, 0.01)
    assert "rfp-total--bad" in comp.weight_total_html(90, 100, 0.01)
    assert comp.color_map(["b", "A"]) == {"A": comp.SERIES_LIGHT[0], "b": comp.SERIES_LIGHT[1]}
    assert comp.series_colors(dark=True) == comp.SERIES_DARK
    assert "(to do)" in comp.checklist_html([(False, "x")])
    assert "Active run" not in comp.header_html(None)


def test_frames_and_charts(repo) -> None:  # type: ignore[no-untyped-def]
    from config.settings import Settings
    from src.agents.orchestrator import Orchestrator
    from src.llm.base import LLMConfig
    from src.models.schemas import SupplierInput
    from src.utils.hashing import sha256_bytes

    sups = []
    for i, p in enumerate(gen.MAIN_PROPOSALS):
        data = gen.render_proposal(p)
        sups.append(SupplierInput(supplier_name=p.supplier, submission_date=date(2026, 9, 1 + i),
                                  experience_rating=3, filename=p.filename, file_sha256=sha256_bytes(data),
                                  file_bytes=data))
    run = Orchestrator(repo, Settings()).run(sups, LLMConfig(provider="demo", model="heuristic-v1"))
    lb = leaderboard_frame(run)
    assert next(iter(lb["Rank"])).startswith("🥇") and len(lb) == 4
    card = run.scorecards[0]
    rank = next(r for r in run.ranking.rankings if r.supplier_name == card.supplier_name)
    sf = scorecard_frame(card, rank)
    assert {"Benchmark", "Gap", "Relative %", "Evidence check"} <= set(sf.columns)
    assert len(comp.score_bar_chart(run).data) == 2
    radar = comp.radar_chart(run, [r.supplier_name for r in run.ranking.rankings])
    assert len(radar.data) == 4
    json.dumps(radar.to_plotly_json(), default=str)
    groups = group_warnings([WarningRecord(code="X", message="m", supplier_name="Zed"),
                             WarningRecord(code="Y", message="m")])
    assert list(groups) == ["Run", "Zed"]


def test_resolve_api_key_precedence(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MY_KEY", "env-key-123456")
    assert resolve_api_key("  typed-key-123  ", "MY_KEY") == ("typed-key-123", "Sidebar input")
    key, source = resolve_api_key("", "MY_KEY")
    assert key == "env-key-123456" and "Environment" in source
    assert resolve_api_key("", None) == ("", "Not required")
    monkeypatch.delenv("MY_KEY")
    assert resolve_api_key("", "MY_KEY") == ("", "Missing")
