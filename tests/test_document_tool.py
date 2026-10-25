"""Document tool: extraction, cleaning, fallbacks, failure modes and truncation."""

from __future__ import annotations

import pytest

from scripts import generate_sample_pdfs as gen
from src.models.schemas import ExtractedDocument
from src.tools import document_tool
from src.tools.document_tool import (
    TRUNCATION_NOTE,
    clean_page_text,
    content_length,
    extract_document,
    has_enough_text,
    join_pages,
    truncate_document,
)
from src.utils.errors import DocumentError, ErrorCode


@pytest.fixture(scope="module")
def apex_pdf() -> bytes:
    return gen.render_proposal(gen.APEX)


def test_normal_pdf(apex_pdf: bytes) -> None:
    doc = extract_document(apex_pdf, "apex.pdf")
    assert doc.extraction_method == "pymupdf"
    assert 2 <= doc.page_count <= 4
    assert doc.text.startswith("[Page 1]\n")
    assert "[Page 2]" in doc.text
    assert "ISO 27001" in doc.text
    assert has_enough_text(doc, 200)


def test_scanned_pdf_has_no_text() -> None:
    doc = extract_document(gen.render_scanned(), "scan.pdf")
    assert not has_enough_text(doc, 200)
    assert content_length(doc.text) == 0


def test_encrypted_pdf_rejected() -> None:
    with pytest.raises(DocumentError) as exc:
        extract_document(gen.render_encrypted(), "locked.pdf")
    assert exc.value.code is ErrorCode.FILE_ENCRYPTED


def test_corrupt_pdf_rejected() -> None:
    with pytest.raises(DocumentError) as exc:
        extract_document(gen.render_corrupt(), "bad.pdf")
    assert exc.value.code is ErrorCode.FILE_CORRUPT
    assert "bad.pdf" in exc.value.user_message


def test_pypdf_fallback(apex_pdf: bytes, monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*_a: object, **_k: object) -> None:
        raise RuntimeError("pymupdf unavailable")
    monkeypatch.setattr(document_tool.pymupdf, "open", boom)
    doc = extract_document(apex_pdf, "apex.pdf")
    assert doc.extraction_method == "pypdf"
    assert "ISO 27001" in doc.text


def test_pypdf_fallback_encrypted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(document_tool.pymupdf, "open", lambda *a, **k: (_ for _ in ()).throw(RuntimeError()))
    with pytest.raises(DocumentError) as exc:
        extract_document(gen.render_encrypted(), "locked.pdf")
    assert exc.value.code is ErrorCode.FILE_ENCRYPTED


def test_clean_page_text() -> None:
    raw = "Imple-\nmentation   plan\r\n\n\n\n\n  with\tspaces   here "
    assert clean_page_text(raw) == "Implementation plan\n\nwith spaces here"


def test_join_pages_markers() -> None:
    assert join_pages(["a", "b"]) == "[Page 1]\na\n\n[Page 2]\nb"


def test_truncation_adds_warning(apex_pdf: bytes) -> None:
    doc = extract_document(apex_pdf, "apex.pdf")
    short, warning = truncate_document(doc, 1000, "Apex Systems")
    assert short.truncated and warning is not None
    assert short.text.endswith(TRUNCATION_NOTE)
    assert len(short.text) <= 1000 + len(TRUNCATION_NOTE)
    assert warning.code == "DOC_TRUNCATED" and warning.supplier_name == "Apex Systems"


def test_no_truncation_when_short() -> None:
    doc = ExtractedDocument(text="short", page_count=1, char_count=5, extraction_method="x")
    same, warning = truncate_document(doc, 1000, "S")
    assert same is doc and warning is None
    same, warning = truncate_document(doc, 0, "S")
    assert warning is None


def test_generator_is_deterministic() -> None:
    assert gen.render_proposal(gen.NEXAWORKS) == gen.render_proposal(gen.NEXAWORKS)


def test_all_main_proposals_have_required_sections() -> None:
    for proposal in gen.MAIN_PROPOSALS:
        doc = extract_document(gen.render_proposal(proposal), proposal.filename)
        assert 2 <= doc.page_count <= 4, proposal.supplier
        for needle in ("Executive Summary", "Proposed Solution", "Implementation", "Timeline",
                       "Team Structure", "Price Table", "Pricing assumptions", "Security",
                       "Support", "Experience", "References"):
            assert needle in doc.text, (proposal.supplier, needle)


def test_build_all_writes_files(tmp_path) -> None:  # type: ignore[no-untyped-def]
    paths = gen.build_all(tmp_path)
    names = {p.name for p in paths}
    assert {"apex_systems.pdf", "empty_scanned.pdf", "prompt_injection.pdf", "minimal_info.pdf",
            "corrupt.pdf"} <= names
