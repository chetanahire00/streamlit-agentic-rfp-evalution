"""Supplier Input tab: uploads, per-file metadata cards, inline validation, Evaluate."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import streamlit as st

from config.settings import PROJECT_ROOT, Settings
from src.agents.orchestrator import Orchestrator, ProgressEvent
from src.db.repository import Repository
from src.models.schemas import SupplierInput
from src.tools import document_tool
from src.tools.input_validation import (
    Issue,
    SupplierMeta,
    check_file_count,
    check_pdf_bytes,
    find_duplicate_files,
    normalise_name,
    validate_active_criteria,
    validate_metadata,
)
from src.ui.components import badge_html, checklist_html, friendly_supplier_name, render_html
from src.ui.sidebar import SidebarState
from src.ui.state import ACTIVE_TAB, SAMPLES_LOADED, flash, set_run
from src.utils.errors import AppError, DocumentError, ErrorCode
from src.utils.hashing import sha256_bytes
from src.utils.logging_setup import get_logger

log = get_logger(__name__)

SAMPLE_LOADED_KEY = SAMPLES_LOADED

SAMPLE_DIR = PROJECT_ROOT / "sample_data"
SAMPLE_META: list[tuple[str, str, date, float]] = [
    ("apex_systems.pdf", "Apex Systems", date(2026, 9, 10), 4.0),
    ("brightpath_tech.pdf", "BrightPath Tech", date(2026, 9, 8), 2.0),
    ("nexaworks.pdf", "NexaWorks", date(2026, 9, 9), 4.0),
    ("orbit_digital.pdf", "Orbit Digital", date(2026, 9, 12), 4.5),
]


@dataclass(frozen=True)
class Candidate:
    """One uploaded or sample file awaiting metadata."""

    key: str
    filename: str
    data: bytes
    sha256: str
    default_name: str
    default_date: date | None
    default_rating: float | None
    source: str


@st.cache_data(show_spinner=False, max_entries=128)
def preflight(sha256: str, _data: bytes, filename: str) -> dict[str, object]:
    """Try extraction once per file content (cached by hash) to surface problems early."""
    try:
        doc = document_tool.extract_document(_data, filename)
    except DocumentError as exc:
        return {"ok": False, "code": exc.code.value, "message": exc.user_message}
    return {"ok": True, "pages": doc.page_count, "chars": document_tool.content_length(doc.text),
            "method": doc.extraction_method}


def _candidates(uploads: list, samples: bool) -> list[Candidate]:  # type: ignore[type-arg]
    out: list[Candidate] = []
    if samples:
        for filename, name, d, rating in SAMPLE_META:
            path = SAMPLE_DIR / filename
            if path.exists():
                data = path.read_bytes()
                out.append(Candidate(f"sample_{Path(filename).stem}", filename, data, sha256_bytes(data),
                                     name, d, rating, "sample"))
    for up in uploads or []:
        data = up.getvalue()
        out.append(Candidate(f"up_{up.file_id}", up.name, data, sha256_bytes(data),
                             friendly_supplier_name(up.name), date.today(), None, "upload"))
    return out


def _metadata_inputs(c: Candidate, settings: Settings, today: date) -> SupplierMeta:
    cols = st.columns([2.2, 1.4, 1.2])
    name = cols[0].text_input("Supplier name *", value=c.default_name, key=f"name_{c.key}", max_chars=120)
    sub_date = cols[1].date_input("Submission date *", value=c.default_date, max_value=today,
                                  key=f"date_{c.key}", format="YYYY-MM-DD")
    rating = cols[2].number_input(
        f"Experience rating ({settings.experience_min:g}–{settings.experience_max:g}) *",
        min_value=settings.experience_min, max_value=settings.experience_max, value=c.default_rating,
        step=0.5, key=f"rating_{c.key}", placeholder="e.g. 3.5",
        help="Historical experience with this supplier: 0 = none/poor, 5 = excellent.")
    return SupplierMeta(c.key, name, sub_date if isinstance(sub_date, date) else None, rating)


def render_input_tab(repo: Repository, settings: Settings, sidebar: SidebarState) -> None:
    """Draw the Supplier Input tab."""
    st.subheader("Supplier proposals")
    st.caption(f"Upload at least {settings.min_suppliers} supplier RFP responses (PDF, up to "
               f"{settings.max_file_mb:g} MB each). Files are checked by content, not just extension.")

    up_col, sample_col = st.columns([3, 1], vertical_alignment="bottom")
    with up_col:
        uploads = st.file_uploader("Supplier PDFs", type=["pdf"], accept_multiple_files=True,
                                   key="uploads", help="Multiple files allowed.",
                                   max_upload_size=int(settings.max_file_mb))
    with sample_col:
        loaded = bool(st.session_state.get(SAMPLE_LOADED_KEY))
        if st.button("Remove sample PDFs" if loaded else "Load 4 sample PDFs",
                     icon=":material/folder_open:", width="stretch", key="toggle_samples"):
            st.session_state[SAMPLE_LOADED_KEY] = not loaded
            st.rerun()

    candidates = _candidates(uploads or [], bool(st.session_state.get(SAMPLE_LOADED_KEY)))
    if not candidates:
        render_html('<div class="rfp-empty" role="status"><p class="rfp-empty__title">No proposals yet</p>'
                    '<p class="rfp-empty__body">Upload supplier PDFs above, or click <b>Load 4 sample '
                    "PDFs</b> to try the fictional proposals from the brief.</p></div>")

    today = date.today()
    accepted: list[Candidate] = []
    file_issues: dict[str, list[Issue]] = {}
    for c in candidates:
        issues = check_pdf_bytes(c.filename, c.data, settings)
        if not issues:
            pf = preflight(c.sha256, c.data, c.filename)
            if not pf["ok"]:
                issues = [Issue(ErrorCode(pf["code"]), str(pf["message"]), c.filename)]
        file_issues[c.key] = issues
        if not issues:
            accepted.append(c)

    dup_issues = find_duplicate_files([(c.key, c.sha256) for c in accepted])
    dup_by_key = {i.field: i for i in dup_issues}
    first_by_sha: dict[str, str] = {}
    for c in accepted:
        first_by_sha.setdefault(c.sha256, c.filename)
    metas: list[SupplierMeta] = []

    for c in candidates:
        with st.container(border=True):
            issues = file_issues[c.key]
            head, badge_col = st.columns([4, 1.4], vertical_alignment="center")
            size_kb = len(c.data) / 1024
            info = f"{size_kb:,.0f} KB"
            if not issues:
                pf = preflight(c.sha256, c.data, c.filename)
                info += f" · {pf['pages']} page(s) · {pf['chars']:,} characters"
            head.markdown(f"**{c.filename}**  \n<span style='color:#475569'>{info}"
                          f"{' · sample' if c.source == 'sample' else ''}</span>", unsafe_allow_html=True)
            if issues:
                badge_col.markdown(badge_html("✕ Rejected", "danger"), unsafe_allow_html=True)
                for i in issues:
                    st.error(i.message, icon=":material/error:")
                continue
            pf = preflight(c.sha256, c.data, c.filename)
            if c.key in dup_by_key:
                badge_col.markdown(badge_html("✕ Duplicate", "danger"), unsafe_allow_html=True)
                st.error(f"Identical content to '{first_by_sha[c.sha256]}'. Remove one of them to continue.",
                         icon=":material/content_copy:")
                continue
            if int(pf["chars"]) < settings.min_text_chars:  # type: ignore[call-overload]
                badge_col.markdown(badge_html("! No text", "warning"), unsafe_allow_html=True)
                st.warning("Almost no text could be extracted (scanned or image-only PDF). It will be marked "
                           "NO_TEXT and scored 0 without calling the LLM.", icon=":material/warning:")
            else:
                badge_col.markdown(badge_html("✓ Readable", "success"), unsafe_allow_html=True)
            meta = _metadata_inputs(c, settings, today)
            metas.append(meta)
            # Inline: validate cumulatively so a duplicate name is flagged on the later card.
            for issue in validate_metadata(metas, settings, today):
                if issue.field == c.key:
                    st.error(issue.message, icon=":material/error:")

    meta_issues = validate_metadata(metas, settings, today)
    if meta_issues:
        st.error(f"{len(meta_issues)} supplier detail(s) need fixing; see the messages in the cards above.",
                 icon=":material/error:")

    valid_files = len(metas)
    criteria_issues = validate_active_criteria(repo.list_criteria(), settings)
    count_issues = check_file_count(valid_files, settings)
    checks = [
        (sidebar.llm_ready, "AI provider configured and connection tested (or test skipped)"
         if sidebar.llm_ready else _llm_hint(sidebar)),
        (not criteria_issues, "Criteria valid" if not criteria_issues else criteria_issues[0].message),
        (not count_issues, f"{valid_files} valid PDF(s)" if not count_issues else count_issues[0].message),
        (not dup_issues, "No duplicate files" if not dup_issues else "Remove duplicate files"),
        (not meta_issues and bool(metas), "Supplier metadata complete and valid"
         if not meta_issues else f"{len(meta_issues)} metadata problem(s) to fix"),
    ]
    st.markdown("##### Ready to evaluate?")
    render_html(checklist_html(checks))
    ready = all(ok for ok, _ in checks)
    clicked = st.button("Evaluate suppliers", type="primary", icon=":material/play_arrow:",
                        disabled=not ready, key="evaluate", width="stretch")
    if clicked and ready and sidebar.config is not None:
        meta_by_key = {m.file_key: m for m in metas}
        suppliers = [
            SupplierInput(
                supplier_name=normalise_name(meta_by_key[c.key].supplier_name),
                submission_date=meta_by_key[c.key].submission_date,  # type: ignore[arg-type]
                experience_rating=float(meta_by_key[c.key].experience_rating),  # type: ignore[arg-type]
                filename=c.filename, file_sha256=c.sha256, file_bytes=c.data,
            )
            for c in accepted if c.key in meta_by_key
        ]
        _run(repo, settings, sidebar, suppliers)


def _llm_hint(sidebar: SidebarState) -> str:
    if sidebar.problems:
        return "Sidebar: " + "; ".join(sidebar.problems)
    return "Sidebar: click Test connection (or tick Skip connection test)"


def _run(repo: Repository, settings: Settings, sidebar: SidebarState, suppliers: list[SupplierInput]) -> None:
    assert sidebar.config is not None
    progress = st.progress(0.0, text="Starting evaluation…")
    run_box = st.status("Run pipeline", expanded=False)
    boxes: dict[str, object] = {}

    def on_event(e: ProgressEvent) -> None:
        progress.progress(min(1.0, max(0.0, e.fraction)), text=f"Step {e.step}/10 · {e.step_name}: {e.message}")
        if e.supplier is None:
            run_box.write(f"**{e.step}. {e.step_name}**: {e.message}")
            return
        box = boxes.get(e.supplier)
        if box is None:
            box = st.status(e.supplier, expanded=False, state="running")
            boxes[e.supplier] = box
        box.write(f"**{e.step}. {e.step_name}**: {e.message}")  # type: ignore[attr-defined]
        state = {"error": "error", "success": "complete"}.get(e.level, "running")
        if e.level == "warning" and e.step == 4:
            state = "complete"
        box.update(label=f"{e.supplier}: {e.message}", state=state)  # type: ignore[attr-defined]

    orchestrator = Orchestrator(repo, settings, on_event=on_event)
    try:
        result = orchestrator.run(suppliers, sidebar.config, max_pdf_chars=sidebar.max_pdf_chars)
    except AppError as exc:
        run_box.update(label="Run stopped", state="error")
        st.error(exc.user_message, icon=":material/error:")
        return
    except Exception:
        log.exception("Evaluation crashed")
        run_box.update(label="Run stopped", state="error")
        st.error("Something went wrong during evaluation. The details were logged; please try again.",
                 icon=":material/error:")
        return
    run_box.update(label=f"Run {result.rfp_run_id[:8]}… finished: {result.status.value}", state="complete")
    set_run(result)
    flash("success" if result.status.value == "COMPLETED" else "warning",
          f"Evaluation finished with status {result.status.value}. Showing the leaderboard.")
    st.session_state[ACTIVE_TAB] = "leaderboard"
    st.rerun()
