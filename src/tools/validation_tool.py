"""Validation Tool: parse, validate and normalise LLM output into a scorecard.

Every correction emits a structured :class:`WarningRecord` so the final score
stays traceable. Rules (see docs/TECHNICAL.md, "Validation rules"):

====  =======================  ==========================================================
Code  Rule                     Behaviour
====  =======================  ==========================================================
V01   Non-JSON / malformed      Strip markdown fences and retry; caller then tries one LLM
                               repair; finally :func:`failed_scorecard` (all zeros).
V02   Missing criterion         Added with score 0, "Missing from LLM output".
V03   Unknown criterion_id      Dropped.
V04   Duplicate criterion_id    First occurrence kept.
V05   Score type / range        "8" / "8/10" coerced; non-numeric, NaN, inf → 0; clipped.
V06   max_score mismatch        Database max_score always used.
V07   Evidence grounding        Fuzzy match against the document; unverified → flag only.
V08   supplier_name mismatch    User-entered name always used.
V09   Empty / long fields       Placeholder for empty justification; long text truncated.
====  =======================  ==========================================================
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from rapidfuzz import fuzz

from config.settings import Settings
from src.models.schemas import (
    Criterion,
    CriterionScore,
    ExtractedDocument,
    SupplierInput,
    SupplierScorecard,
    SupplierStatus,
    WarningRecord,
)

PLACEHOLDER_JUSTIFICATION = "No justification provided by the model."
MISSING_JUSTIFICATION = "Missing from LLM output"
NOT_EVALUATED = "Not evaluated: {reason}"

_FENCE_RE = re.compile(r"^\s*```(?:json|JSON)?\s*\n?(.*?)\n?\s*```\s*$", re.DOTALL)
_NUMBER_RE = re.compile(r"^\s*([-+]?\d+(?:\.\d+)?)\s*(?:/\s*(\d+(?:\.\d+)?))?")
_PAGE_MARK_RE = re.compile(r"\[Page (\d+)\]")
_PAGE_REF_RE = re.compile(r"\(?\[?\s*(?:p\.|page)\s*(\d+)\s*\]?\)?", re.IGNORECASE)
_WS_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class ParseResult:
    """Outcome of :func:`parse_llm_json`."""

    data: dict[str, Any] | None
    warnings: list[WarningRecord]


def strip_fences(text: str) -> str:
    """Remove a surrounding markdown code fence, if any."""
    m = _FENCE_RE.match(text)
    return m.group(1) if m else text


def _extract_object(text: str) -> str | None:
    """Return the outermost ``{...}`` span in ``text`` (handles leading chatter)."""
    start, end = text.find("{"), text.rfind("}")
    return text[start:end + 1] if 0 <= start < end else None


def parse_llm_json(raw: str | None, supplier_name: str) -> ParseResult:
    """Parse raw LLM text into a dict (rule V01, first two attempts).

    Tries, in order: the raw text, the text with fences stripped, and the
    outermost ``{...}`` span. Returns ``data=None`` if all fail or the top level
    is not an object; the caller should then attempt one LLM repair.
    """
    warnings: list[WarningRecord] = []
    if not raw or not raw.strip():
        return ParseResult(None, [_w("V01", "The model returned an empty response.", supplier_name)])
    candidates = [("raw", raw)]
    stripped = strip_fences(raw)
    if stripped != raw:
        candidates.append(("fence", stripped))
    span = _extract_object(stripped)
    if span is not None and span != stripped:
        candidates.append(("span", span))
    for kind, text in candidates:
        try:
            value = json.loads(text)
        except (json.JSONDecodeError, ValueError):
            continue
        if not isinstance(value, dict):
            warnings.append(_w("V01", "The model returned JSON that is not an object.", supplier_name))
            return ParseResult(None, warnings)
        if kind == "fence":
            warnings.append(_w("V01", "Removed markdown code fences around the JSON.", supplier_name))
        elif kind == "span":
            warnings.append(_w("V01", "Removed extra text around the JSON object.", supplier_name))
        return ParseResult(value, warnings)
    warnings.append(_w("V01", "The model response was not valid JSON.", supplier_name))
    return ParseResult(None, warnings)


def coerce_score(value: Any, max_score: float) -> tuple[float, str | None]:
    """Convert a model-supplied score to a float in [0, max_score] (rules V05/V06).

    Returns:
        ``(score, note)`` where ``note`` describes any correction (``None`` if none).
    """
    note: str | None = None
    if isinstance(value, bool) or value is None:
        return 0.0, f"non-numeric score {value!r} set to 0"
    if isinstance(value, int | float):
        number = float(value)
    elif isinstance(value, str):
        m = _NUMBER_RE.match(value)
        if not m:
            return 0.0, f"non-numeric score {value!r} set to 0"
        number = float(m.group(1))
        if m.group(2):
            denom = float(m.group(2))
            if denom > 0 and denom != max_score:
                number = number / denom * max_score
                note = f"score {value!r} rescaled to {number:g}/{max_score:g}"
            else:
                note = f"score {value!r} coerced to {number:g}"
        else:
            note = f"score {value!r} coerced to {number:g}"
    else:
        return 0.0, f"non-numeric score of type {type(value).__name__} set to 0"
    if math.isnan(number) or math.isinf(number):
        return 0.0, "NaN/infinite score set to 0"
    if number < 0:
        return 0.0, f"score {number:g} below 0 clipped to 0"
    if number > max_score:
        return float(max_score), f"score {number:g} above maximum clipped to {max_score:g}"
    return number, note


def _normalise(text: str) -> str:
    return _WS_RE.sub(" ", _PAGE_MARK_RE.sub(" ", text)).strip().casefold()


def ground_evidence(
    evidence: str, document_text: str, threshold: float
) -> tuple[bool, int | None]:
    """Check that ``evidence`` appears in ``document_text`` (rule V07).

    Uses an exact normalised substring test, then rapidfuzz partial matching.

    Returns:
        ``(verified, page)`` where ``page`` is the ``[Page N]`` marker preceding
        the best match in the document (``None`` if not found).
    """
    quote = _normalise(_PAGE_REF_RE.sub(" ", evidence)).strip(" .\"'“”")
    if len(quote) < 8:
        return False, None
    pages = _page_spans(document_text)
    best: tuple[float, int | None] = (0.0, None)
    for page_no, page_text in pages:
        norm = _normalise(page_text)
        if quote in norm:
            return True, page_no
        if len(norm) == 0:
            continue
        score = fuzz.partial_ratio(quote, norm)
        if score > best[0]:
            best = (score, page_no)
    if best[0] >= threshold:
        return True, best[1]
    return False, None


def _page_spans(text: str) -> list[tuple[int | None, str]]:
    """Split document text on ``[Page N]`` markers into (page, text) pairs."""
    parts = _PAGE_MARK_RE.split(text)
    if len(parts) == 1:
        return [(None, text)]
    spans: list[tuple[int | None, str]] = []
    if parts[0].strip():
        spans.append((None, parts[0]))
    for i in range(1, len(parts), 2):
        spans.append((int(parts[i]), parts[i + 1] if i + 1 < len(parts) else ""))
    return spans


def _truncate_words(text: str, max_words: int) -> tuple[str, bool]:
    words = text.split()
    if len(words) <= max_words:
        return text, False
    return " ".join(words[:max_words]) + " …", True


def _truncate_chars(text: str, max_chars: int) -> tuple[str, bool]:
    if len(text) <= max_chars:
        return text, False
    return text[:max_chars].rstrip() + " …", True


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and value.strip().lstrip("+-").isdigit():
        return int(value.strip())
    return None


def validate_evaluation(
    data: dict[str, Any],
    *,
    supplier: SupplierInput,
    criteria: Sequence[Criterion],
    document: ExtractedDocument,
    settings: Settings,
    prior_warnings: Sequence[WarningRecord] = (),
) -> SupplierScorecard:
    """Validate and normalise a parsed LLM result into a :class:`SupplierScorecard`.

    Args:
        data: Parsed JSON object from the model.
        supplier: The user-entered supplier (its name always wins, rule V08).
        criteria: The run's criteria snapshot; only these ids are accepted.
        document: Extracted text used for evidence grounding.
        settings: Limits (truncation lengths, fuzzy threshold).
        prior_warnings: Warnings collected earlier (extraction, parsing).
    """
    name = supplier.supplier_name
    warnings = list(prior_warnings)
    by_id = {c.criterion_id: c for c in criteria}

    model_name = _as_text(data.get("supplier_name"))
    if model_name and model_name.casefold() != name.casefold():
        warnings.append(_w("V08", f"Model returned supplier name '{model_name[:80]}'; "
                                  f"using the entered name '{name}'.", name))

    raw_items = data.get("criteria")
    if not isinstance(raw_items, list):
        warnings.append(_w("V02", "'criteria' was missing or not a list; all criteria scored 0.", name))
        raw_items = []

    seen: dict[int, CriterionScore] = {}
    for idx, item in enumerate(raw_items):
        if not isinstance(item, dict):
            warnings.append(_w("V03", f"Dropped criteria entry #{idx + 1}: not an object.", name))
            continue
        cid = _as_int(item.get("criterion_id"))
        if cid is None or cid not in by_id:
            warnings.append(_w("V03", f"Dropped unknown criterion_id {item.get('criterion_id')!r}.", name))
            continue
        if cid in seen:
            warnings.append(_w("V04", f"Duplicate criterion_id {cid}; kept the first occurrence.", name, cid))
            continue
        seen[cid] = _normalise_item(item, by_id[cid], name, document, settings, warnings)

    results: list[CriterionScore] = []
    for crit in sorted(criteria, key=lambda c: c.criterion_id):
        if crit.criterion_id in seen:
            results.append(seen[crit.criterion_id])
            continue
        warnings.append(_w("V02", f"Criterion '{crit.name}' missing from model output; scored 0.",
                           name, crit.criterion_id))
        results.append(CriterionScore(
            criterion_id=crit.criterion_id, criterion_name=crit.name, score=0.0,
            max_score=crit.max_score, weight=crit.weight, justification=MISSING_JUSTIFICATION,
            evidence="", evidence_page=None, evidence_unverified=False,
        ))

    risks_raw = data.get("risks", [])
    if isinstance(risks_raw, str):
        risks_raw = [risks_raw]
    if not isinstance(risks_raw, list):
        warnings.append(_w("V09", "'risks' was not a list; ignored.", name))
        risks_raw = []
    risks = [_truncate_chars(_as_text(r), settings.max_risk_chars)[0] for r in risks_raw if _as_text(r)]
    if len(risks) > settings.max_risks:
        warnings.append(_w("V09", f"Kept the first {settings.max_risks} of {len(risks)} risks.", name))
        risks = risks[: settings.max_risks]

    summary, cut = _truncate_chars(_as_text(data.get("overall_summary")), settings.max_summary_chars)
    if cut:
        warnings.append(_w("V09", "Overall summary truncated.", name))
    if not summary:
        summary = "No overall summary provided by the model."
        warnings.append(_w("V09", "Overall summary was empty; placeholder used.", name))

    return SupplierScorecard(
        supplier_name=name,
        submission_date=supplier.submission_date,
        experience_rating=supplier.experience_rating,
        filename=supplier.filename,
        file_sha256=supplier.file_sha256,
        status=SupplierStatus.OK,
        criteria=results,
        risks=risks,
        overall_summary=summary,
        warnings=warnings,
        page_count=document.page_count,
        char_count=document.char_count,
        extraction_method=document.extraction_method,
    )


def _normalise_item(
    item: dict[str, Any],
    crit: Criterion,
    supplier_name: str,
    document: ExtractedDocument,
    settings: Settings,
    warnings: list[WarningRecord],
) -> CriterionScore:
    cid = crit.criterion_id
    score, note = coerce_score(item.get("score"), crit.max_score)
    if note:
        warnings.append(_w("V05", f"'{crit.name}': {note}.", supplier_name, cid))

    model_max = item.get("max_score")
    if model_max is not None and coerce_score(model_max, float("inf"))[0] != crit.max_score:
        warnings.append(_w("V06", f"'{crit.name}': model used max_score {model_max!r}; "
                                  f"database value {crit.max_score:g} applied.", supplier_name, cid))

    justification, cut = _truncate_chars(_as_text(item.get("justification")),
                                         settings.max_justification_chars)
    if cut:
        warnings.append(_w("V09", f"'{crit.name}': justification truncated.", supplier_name, cid))
    if not justification:
        justification = PLACEHOLDER_JUSTIFICATION
        warnings.append(_w("V09", f"'{crit.name}': empty justification; placeholder used.",
                           supplier_name, cid))

    evidence, cut = _truncate_words(_as_text(item.get("evidence")), settings.max_evidence_words)
    if cut:
        warnings.append(_w("V09", f"'{crit.name}': evidence truncated to "
                                  f"{settings.max_evidence_words} words.", supplier_name, cid))

    page = _as_int(item.get("page"))
    unverified = False
    if evidence:
        verified, found_page = ground_evidence(evidence.removesuffix(" …"), document.text,
                                               settings.evidence_match_threshold)
        if verified:
            page = found_page if found_page is not None else page
        else:
            unverified = True
            warnings.append(_w("V07", f"'{crit.name}': evidence quote not found in the document.",
                               supplier_name, cid))
    elif score > 0:
        unverified = True
        warnings.append(_w("V07", f"'{crit.name}': score {score:g} given without evidence.",
                           supplier_name, cid))

    return CriterionScore(
        criterion_id=cid, criterion_name=crit.name, score=score, max_score=crit.max_score,
        weight=crit.weight, justification=justification, evidence=evidence,
        evidence_page=page, evidence_unverified=unverified,
    )


def failed_scorecard(
    supplier: SupplierInput,
    criteria: Sequence[Criterion],
    status: SupplierStatus,
    reason: str,
    *,
    warnings: Sequence[WarningRecord] = (),
    document: ExtractedDocument | None = None,
) -> SupplierScorecard:
    """All-zero scorecard for a supplier that could not be evaluated."""
    return SupplierScorecard(
        supplier_name=supplier.supplier_name,
        submission_date=supplier.submission_date,
        experience_rating=supplier.experience_rating,
        filename=supplier.filename,
        file_sha256=supplier.file_sha256,
        status=status,
        criteria=[
            CriterionScore(
                criterion_id=c.criterion_id, criterion_name=c.name, score=0.0,
                max_score=c.max_score, weight=c.weight,
                justification=NOT_EVALUATED.format(reason=reason), evidence="",
            )
            for c in sorted(criteria, key=lambda c: c.criterion_id)
        ],
        risks=[],
        overall_summary=NOT_EVALUATED.format(reason=reason),
        warnings=list(warnings),
        page_count=document.page_count if document else 0,
        char_count=document.char_count if document else 0,
        extraction_method=document.extraction_method if document else "",
    )


def _w(code: str, message: str, supplier: str | None = None, criterion_id: int | None = None) -> WarningRecord:
    return WarningRecord(code=code, message=message, supplier_name=supplier, criterion_id=criterion_id)
