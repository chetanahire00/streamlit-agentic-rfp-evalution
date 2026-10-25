"""Offline "Demo" provider: a deterministic keyword-and-evidence scorer.

It lets the app run end to end without an API key (classroom demos, CI, the
sample export). It is NOT an AI model and its scores are indicative only. It
reads the same prompt the real providers receive, so the rest of the pipeline
(validation, ranking, persistence) is exercised exactly as in a real run.

Scoring per criterion:

1. Build a term list from the criterion's name and description, expanded with
   a small domain lexicon (e.g. "certifications" → "iso 27001", "soc 2").
2. Split the document into sentences, tracking the ``[Page N]`` markers.
3. Each sentence earns ``min(distinct matched terms, 3)`` points; sentences
   containing hedging phrases ("to be confirmed", "planned for", …) cost points.
4. ``score = max_score × (1 − e^(−points / 6))`` minus the hedge penalty,
   clipped to [0, max_score] and rounded to 0.5.
5. Evidence is the best-matching sentence, quoted verbatim (≤ 40 words).
"""

from __future__ import annotations

import json
import math
import re
from typing import Any

from src.llm.base import LLMProvider

_CRITERIA_RE = re.compile(r"<criteria>\s*(.*?)\s*</criteria>", re.DOTALL)
_DOC_RE = re.compile(r"<supplier_document>\s*(.*?)\s*</supplier_document>", re.DOTALL)
_NAME_RE = re.compile(r"^Supplier name:\s*(.+)$", re.MULTILINE)
_PAGE_RE = re.compile(r"\[Page (\d+)\]")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
_WORD = re.compile(r"[a-z0-9][a-z0-9\-]+")

STOPWORDS = {"and", "the", "for", "with", "plan", "value", "capability", "fit", "model"}

LEXICON: dict[str, list[str]] = {
    "architecture": ["architecture", "microservices", "kubernetes", "cloud-native", "availability",
                     "failover", "workflow engine", "data warehouse"],
    "integration": ["integration", "integrations", "api", "apis", "rest", "graphql", "connector",
                    "connectors", "single sign-on", "saml", "openid", "erp", "sap", "oracle", "dynamics",
                    "event-driven"],
    "scalab": ["scales", "scale", "scalability", "horizontally", "load-tested", "concurrent"],
    "technical": ["technical design", "p95", "sla", "recovery point"],
    "timeline": ["timeline", "week", "weeks", "month", "months", "phase", "phases", "schedule"],
    "milestone": ["milestone", "milestones", "quality gate", "exit criteria", "sign-off", "steering"],
    "staffing": ["team", "fte", "named", "specialists", "staffing", "dedicated"],
    "risk": ["risk register", "risk plan", "mitigation", "rollback", "rehearsed", "rehearsals", "hypercare"],
    "pricing": ["price", "prices", "pricing", "fixed", "subscription", "fee", "vat"],
    "cost": ["total cost", "total three-year cost", "no additional charges", "includes", "covers"],
    "assumption": ["assumption", "assumptions", "exclude", "excludes", "include"],
    "controls": ["controls", "role-based", "multi-factor", "mfa", "access control", "encrypted",
                 "encryption", "aes-256", "tls"],
    "certification": ["iso 27001", "soc 2", "cyber essentials", "certified", "crest"],
    "privacy": ["gdpr", "data protection", "data processing agreement", "data residency", "uk data centres"],
    "auditab": ["audit logs", "audit", "auditability", "penetration testing", "security operations"],
    "support": ["support", "24x7", "service desk", "response", "resolution", "service credits",
                "customer success", "knowledge base", "service level"],
    "similar": ["similar", "deployments", "delivered", "councils", "public sector", "awards"],
    "references": ["references", "reference", "clients", "contacts", "customers"],
}

HEDGES = ["to be confirmed", "tbc", "to be agreed", "planned for", "working towards", "on request",
          "will be scoped", "as needed", "after contract award", "available after", "will be described",
          "not included", "best practice", "will be agreed", "subject to further", "later"]

INJECTION_RE = re.compile(r"ignore (all|previous|prior) instructions|score 10|system note", re.IGNORECASE)


def _terms_for(criterion: dict[str, Any]) -> list[str]:
    text = f"{criterion.get('name', '')} {criterion.get('description', '')}".lower()
    words = {w for w in _WORD.findall(text) if len(w) >= 4 and w not in STOPWORDS}
    terms = set(words)
    for stem, extra in LEXICON.items():
        if any(w.startswith(stem[:6]) for w in words):
            terms.update(extra)
    return sorted(terms)


_SHORT_LINE = 70
_TERMINAL = (".", "!", "?", ":", ";")


def _blocks(lines: list[str]) -> list[str]:
    """Re-join wrapped lines into sentences; keep headings/table cells separate.

    A line that is short and lacks closing punctuation (a heading, table cell or
    page footer) stands alone when no sentence is in progress. Other lines are
    appended until a line ends with sentence punctuation.
    """
    blocks: list[str] = []
    buffer: list[str] = []
    for raw in lines:
        line = raw.strip().lstrip("•·▪*- ").strip()
        if not line:
            continue
        if not buffer and len(line) < _SHORT_LINE and not line.endswith(_TERMINAL):
            blocks.append(line)
            continue
        buffer.append(line)
        if line.endswith(_TERMINAL):
            blocks.append(" ".join(buffer))
            buffer = []
    if buffer:
        blocks.append(" ".join(buffer))
    return blocks


def _sentences(document: str) -> list[tuple[int | None, str]]:
    """Split into (page, sentence) so evidence quotes never start mid-sentence."""
    out: list[tuple[int | None, str]] = []
    page: int | None = None
    for chunk in _PAGE_RE.split(document):
        if chunk.isdigit():
            page = int(chunk)
            continue
        for block in _blocks(chunk.splitlines()):
            for s in _SENTENCE_SPLIT.split(block):
                s = s.strip()
                if len(s) >= 12:
                    out.append((page, s))
    return out


def _is_statement(sentence: str) -> bool:
    """A real sentence (ends with punctuation, ≥ 6 words) rather than a heading or table cell."""
    return sentence.endswith((".", "!", "?")) and len(sentence.split()) >= 6


def _matches(sentence: str, terms: list[str]) -> int:
    low = sentence.lower()
    return sum(1 for t in terms if re.search(rf"(?<![a-z0-9]){re.escape(t)}(?![a-z0-9])", low))


def _hedged(sentence: str) -> bool:
    low = sentence.lower()
    return any(h in low for h in HEDGES)


def score_document(supplier_name: str, criteria: list[dict[str, Any]], document: str) -> dict[str, Any]:
    """Deterministically score ``document`` against ``criteria`` (brief §8 output shape)."""
    sentences = [(p, s) for p, s in _sentences(document) if not INJECTION_RE.search(s)]
    results = []
    strengths: list[tuple[float, str]] = []
    for crit in criteria:
        terms = _terms_for(crit)
        max_score = float(crit.get("max_score", 10))
        scored = [(min(_matches(s, terms), 3), p, s) for p, s in sentences]
        relevant = [x for x in scored if x[0] > 0]
        points = sum(x[0] for x in relevant if not _hedged(x[2]))
        hedges = sum(1 for x in relevant if _hedged(x[2]))
        raw = max_score * (1 - math.exp(-points / 6)) - hedges * 0.12 * max_score
        score = round(max(0.0, min(max_score, raw)) * 2) / 2
        best = max((x for x in relevant if not _hedged(x[2])),
                   key=lambda x: (_is_statement(x[2]), x[0], -abs(len(x[2].split()) - 25)),
                   default=None)
        if best is None or score == 0:
            justification = "Not addressed in proposal: no specific statements found for this criterion."
            evidence, page = "", None
        else:
            words = best[2].split()
            evidence = " ".join(words[:40])
            page = best[1]
            justification = (f"{len(relevant)} relevant statement(s) found"
                             + (f", {hedges} of them vague or deferred" if hedges else "")
                             + ". Score reflects the depth of specific, evidenced detail.")
        strengths.append((score / max_score if max_score else 0, str(crit.get("name", ""))))
        results.append({"criterion_id": crit["criterion_id"], "score": score, "max_score": max_score,
                        "justification": justification, "evidence": evidence, "page": page})

    risks = [s for _, s in sentences if _hedged(s)][:4]
    if INJECTION_RE.search(document):
        risks.insert(0, "Document contains text that tries to instruct the evaluator "
                        "(possible prompt injection); it was ignored.")
    ordered = sorted(strengths, key=lambda t: (-t[0], t[1]))
    summary = ("Deterministic demo evaluation (keyword and evidence heuristics, not an AI model). "
               f"Strongest area: {ordered[0][1]}; weakest area: {ordered[-1][1]}." if ordered
               else "Deterministic demo evaluation.")
    return {"supplier_name": supplier_name, "criteria": results, "risks": risks, "overall_summary": summary}


class DemoProvider(LLMProvider):
    """Offline provider that implements the same ``complete_json`` contract."""

    provider_id = "demo"
    requires_key = False

    def _build_client(self) -> Any:
        return None

    def _complete(self, system: str, user: str, schema: dict[str, Any]) -> str:
        if "criteria" not in schema.get("properties", {}):
            return json.dumps({"ok": True})
        crit_match, doc_match = _CRITERIA_RE.search(user), _DOC_RE.search(user)
        if not crit_match or not doc_match:
            return json.dumps({"supplier_name": "", "criteria": [], "risks": [], "overall_summary": ""})
        name_match = _NAME_RE.search(user)
        return json.dumps(
            score_document(name_match.group(1).strip() if name_match else "",
                           json.loads(crit_match.group(1)), doc_match.group(1)),
            sort_keys=True,
        )
