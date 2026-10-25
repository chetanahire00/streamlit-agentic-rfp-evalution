"""Document Tool: extract clean, page-marked text from supplier PDFs.

PyMuPDF is the primary extractor; pypdf is the fallback when PyMuPDF cannot
open the file. Output text keeps ``[Page N]`` markers so evidence quotes can be
traced back to a page.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass

import pymupdf
from pypdf import PdfReader

from src.models.schemas import ExtractedDocument, WarningRecord
from src.utils.errors import DocumentError, ErrorCode
from src.utils.logging_setup import get_logger

log = get_logger(__name__)

_HYPHEN_BREAK = re.compile(r"(\w)-\n\s*(\w)")
_SPACES = re.compile(r"[ \t ]+")
_MANY_NEWLINES = re.compile(r"\n{3,}")
_PAGE_MARK = re.compile(r"\[Page \d+\]")
TRUNCATION_NOTE = "\n[… document truncated …]"


@dataclass(frozen=True)
class _RawPages:
    pages: list[str]
    method: str


def clean_page_text(text: str) -> str:
    """Normalise one page of extracted text.

    * joins words hyphenated across a line break (``imple-\\nmentation``)
    * collapses runs of spaces/tabs, trims each line
    * collapses three or more blank lines into one blank line
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _HYPHEN_BREAK.sub(r"\1\2", text)
    lines = [_SPACES.sub(" ", line).strip() for line in text.split("\n")]
    return _MANY_NEWLINES.sub("\n\n", "\n".join(lines)).strip()


def join_pages(pages: list[str]) -> str:
    """Join cleaned pages, prefixing each with a ``[Page N]`` marker."""
    return "\n\n".join(f"[Page {i}]\n{p}" for i, p in enumerate(pages, start=1))


def content_length(text: str) -> int:
    """Characters of real content (page markers and whitespace excluded)."""
    return len(re.sub(r"\s+", "", _PAGE_MARK.sub("", text)))


def _with_pymupdf(data: bytes, filename: str) -> _RawPages:
    doc = pymupdf.open(stream=data, filetype="pdf")
    try:
        if doc.needs_pass:
            raise DocumentError(ErrorCode.FILE_ENCRYPTED,
                                f"'{filename}' is password-protected. Upload an unencrypted copy.")
        if doc.page_count == 0:
            raise ValueError("no pages")
        return _RawPages([page.get_text("text") for page in doc], "pymupdf")
    finally:
        doc.close()


def _with_pypdf(data: bytes, filename: str) -> _RawPages:
    reader = PdfReader(io.BytesIO(data))
    if reader.is_encrypted:
        try:
            ok = reader.decrypt("")
        except Exception:  # noqa: BLE001 - any decrypt failure means "needs a password"
            ok = 0
        if not ok:
            raise DocumentError(ErrorCode.FILE_ENCRYPTED,
                                f"'{filename}' is password-protected. Upload an unencrypted copy.")
    pages = [page.extract_text() or "" for page in reader.pages]
    if not pages:
        raise ValueError("no pages")
    return _RawPages(pages, "pypdf")


def extract_document(data: bytes, filename: str) -> ExtractedDocument:
    """Extract and clean text from a PDF.

    Raises:
        DocumentError: ``FILE_ENCRYPTED`` for password-protected PDFs and
            ``FILE_CORRUPT`` when neither extractor can read the file.
    """
    raw: _RawPages | None = None
    try:
        raw = _with_pymupdf(data, filename)
    except DocumentError:
        raise
    except Exception as exc:  # noqa: BLE001 - PyMuPDF raises several unrelated types
        log.warning("PyMuPDF could not read %s (%s); trying pypdf", filename, type(exc).__name__)
        try:
            raw = _with_pypdf(data, filename)
        except DocumentError:
            raise
        except Exception as exc2:
            log.warning("pypdf could not read %s (%s)", filename, type(exc2).__name__)
            raise DocumentError(ErrorCode.FILE_CORRUPT,
                                f"'{filename}' appears to be damaged and could not be read. "
                                "Re-export the PDF and upload it again.") from exc2
    text = join_pages([clean_page_text(p) for p in raw.pages])
    return ExtractedDocument(text=text, page_count=len(raw.pages), char_count=len(text),
                             extraction_method=raw.method, original_char_count=len(text))


def has_enough_text(doc: ExtractedDocument, min_chars: int) -> bool:
    """False for scanned/image-only PDFs whose extractable content is below ``min_chars``."""
    return content_length(doc.text) >= min_chars


def truncate_document(
    doc: ExtractedDocument, max_chars: int, supplier_name: str
) -> tuple[ExtractedDocument, WarningRecord | None]:
    """Cut text to ``max_chars`` at a line boundary and report it.

    Returns:
        The (possibly) truncated document and a warning when truncation happened.
    """
    if max_chars <= 0 or len(doc.text) <= max_chars:
        return doc, None
    cut = doc.text[:max_chars]
    newline = cut.rfind("\n")
    if newline > max_chars * 0.8:
        cut = cut[:newline]
    text = cut + TRUNCATION_NOTE
    warning = WarningRecord(
        code="DOC_TRUNCATED",
        supplier_name=supplier_name,
        message=(f"Document truncated from {len(doc.text):,} to {len(cut):,} characters to fit the "
                 "LLM budget; content after the cut was not evaluated."),
    )
    return doc.model_copy(update={"text": text, "char_count": len(text), "truncated": True}), warning
