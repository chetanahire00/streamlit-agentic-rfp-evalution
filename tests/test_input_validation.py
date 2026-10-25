"""Input validation: metadata, files and duplicate detection."""

from __future__ import annotations

from datetime import date

import pytest

from config.settings import Settings
from src.tools.input_validation import (
    SupplierMeta,
    check_file_count,
    check_pdf_bytes,
    find_duplicate_files,
    normalise_name,
    validate_metadata,
)
from src.utils.errors import ErrorCode

S = Settings()
TODAY = date(2026, 10, 25)


def meta(key: str, name: str | None = "Apex", d: date | None = date(2026, 10, 1),
         r: float | None = 3.0) -> SupplierMeta:
    return SupplierMeta(file_key=key, supplier_name=name, submission_date=d, experience_rating=r)


def codes(issues) -> list[ErrorCode]:  # type: ignore[no-untyped-def]
    return [i.code for i in issues]


def test_valid_metadata() -> None:
    assert validate_metadata([meta("a", "Apex"), meta("b", "Orbit")], S, TODAY) == []


def test_duplicate_names_case_insensitive() -> None:
    issues = validate_metadata([meta("a", "Apex Systems"), meta("b", "  apex   systems ")], S, TODAY)
    assert codes(issues) == [ErrorCode.META_NAME_DUPLICATE]
    assert issues[0].field == "b"


@pytest.mark.parametrize("name", ["", "   ", None])
def test_name_required(name: str | None) -> None:
    assert codes(validate_metadata([meta("a", name)], S, TODAY)) == [ErrorCode.META_NAME_REQUIRED]


def test_name_trimmed() -> None:
    assert normalise_name("  Orbit   Digital ") == "Orbit Digital"


def test_future_date_rejected() -> None:
    issues = validate_metadata([meta("a", d=date(2026, 10, 26))], S, TODAY)
    assert codes(issues) == [ErrorCode.META_DATE_FUTURE]
    assert validate_metadata([meta("a", d=TODAY)], S, TODAY) == []


def test_date_required() -> None:
    assert codes(validate_metadata([meta("a", d=None)], S, TODAY)) == [ErrorCode.META_DATE_REQUIRED]


@pytest.mark.parametrize("rating", [-0.1, 5.1, None, float("nan"), float("inf")])
def test_rating_out_of_range(rating: float | None) -> None:
    assert codes(validate_metadata([meta("a", r=rating)], S, TODAY)) == [ErrorCode.META_RATING_INVALID]


@pytest.mark.parametrize("rating", [0.0, 2.5, 5.0])
def test_rating_bounds_inclusive(rating: float) -> None:
    assert validate_metadata([meta("a", r=rating)], S, TODAY) == []


def test_fewer_than_two_files() -> None:
    assert codes(check_file_count(1, S)) == [ErrorCode.FILE_TOO_FEW]
    assert codes(check_file_count(0, S)) == [ErrorCode.FILE_TOO_FEW]
    assert check_file_count(2, S) == []


def test_duplicate_file_hash() -> None:
    issues = find_duplicate_files([("a.pdf", "h1"), ("b.pdf", "h2"), ("copy of a.pdf", "h1")])
    assert codes(issues) == [ErrorCode.FILE_DUPLICATE]
    assert "a.pdf" in issues[0].message and issues[0].field == "copy of a.pdf"


def test_non_pdf_rejected_by_magic_bytes() -> None:
    fake = b"PK\x03\x04 this is a zip renamed to .pdf"
    assert codes(check_pdf_bytes("evil.pdf", fake, S)) == [ErrorCode.FILE_NOT_PDF]


def test_pdf_header_accepted() -> None:
    assert check_pdf_bytes("ok.pdf", b"%PDF-1.7\n...", S) == []
    assert check_pdf_bytes("junk-prefix.pdf", b"\x00" * 10 + b"%PDF-1.4", S) == []


def test_empty_and_oversized_files() -> None:
    assert codes(check_pdf_bytes("e.pdf", b"", S)) == [ErrorCode.FILE_EMPTY]
    big = b"%PDF-" + b"0" * (S.max_file_bytes + 1)
    assert codes(check_pdf_bytes("big.pdf", big, S)) == [ErrorCode.FILE_TOO_LARGE]
