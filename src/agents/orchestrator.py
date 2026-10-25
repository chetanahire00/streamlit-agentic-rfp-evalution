"""Orchestrator Agent: the brief's 10-step pipeline as a LangGraph ``StateGraph``.

Each step is a graph node; the per-supplier loop and the failure paths are
conditional edges. Nodes only move data between tools: they never score,
rank or call the LLM directly (the Evaluation Agent and tools do that).

==  =================  =========================================================
#   Node               What happens (tool used)
==  =================  =========================================================
1   setup              Load active criteria from SQLite (Repository)
2   check_inputs       ≥2 suppliers, unique names and files (input_validation)
3   create_batch       RFP_RUN_ID, criteria snapshot, run row, LLM provider
4   extract_document   Extract + clean text; NO_TEXT / corrupt → skip LLM (Document Tool)
4   evaluate_supplier  One independent LLM call per supplier (Evaluation Agent)
5   validate_scorecard Normalise and ground the result (Validation Tool)
·   record_supplier    Append the scorecard; loop to the next supplier or continue
6   score              Absolute weighted scores (Ranking Tool)
7   benchmark          Best score per criterion (Ranking Tool)
8   rank               Gap, relative %, PPI, tie-breaks, ranks (Ranking Tool)
9   persist            All results in one transaction (Repository)
10  present            Build the RunResult for the UI and JSON export
==  =================  =========================================================

::

    START → setup → check_inputs → create_batch → extract_document
    extract_document ─ok──────→ evaluate_supplier ─ok──→ validate_scorecard → record_supplier
                     └skipped─→ record_supplier   └failed─→ record_supplier
    record_supplier ─more─→ extract_document
                    └done─→ score → benchmark → rank → persist → present → END
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from config.settings import PROMPT_VERSION, Settings
from src.agents.evaluation_agent import AgentOutcome, EvaluationAgent
from src.db.connection import utc_now_iso
from src.db.repository import Repository
from src.llm.base import LLMConfig
from src.llm.factory import ProviderFactory, create_provider
from src.models.schemas import (
    Criterion,
    ExtractedDocument,
    RankingResult,
    RunResult,
    RunStatus,
    SupplierInput,
    SupplierScorecard,
    SupplierStatus,
    WarningRecord,
)
from src.tools import document_tool, ranking_tool, validation_tool
from src.tools.input_validation import check_file_count, normalise_name, validate_active_criteria
from src.utils.errors import (
    DocumentError,
    ErrorCode,
    LLMError,
    PersistenceError,
    RunAbortedError,
    ValidationError,
)
from src.utils.logging_setup import get_logger

log = get_logger(__name__)

STEP_NAMES = {
    1: "Setup", 2: "Input", 3: "Batch", 4: "Evaluate", 5: "Validate",
    6: "Score", 7: "Benchmark", 8: "Rank", 9: "Persist", 10: "Present",
}


@dataclass(frozen=True)
class ProgressEvent:
    """A progress update for the UI.

    Attributes:
        step: Pipeline step number (1–10).
        message: Human-readable description.
        supplier: Supplier the event refers to, if any.
        level: ``info``, ``success``, ``warning`` or ``error``.
        fraction: Overall progress in [0, 1].
    """

    step: int
    message: str
    supplier: str | None = None
    level: str = "info"
    fraction: float = 0.0

    @property
    def step_name(self) -> str:
        return STEP_NAMES[self.step]


EventHandler = Callable[[ProgressEvent], None]


class RunState(TypedDict, total=False):
    """Data flowing through the graph (no live objects such as clients or connections)."""

    suppliers: list[SupplierInput]
    llm_config: LLMConfig
    budget: int
    snapshot: list[Criterion]
    run_id: str
    created_at: str
    index: int
    document: ExtractedDocument | None
    pending_warnings: list[WarningRecord]
    outcome: AgentOutcome | None
    card: SupplierScorecard | None
    cards: list[SupplierScorecard]
    absolute_scores: dict[str, float]
    benchmarks: dict[int, float]
    ranking: RankingResult
    status: RunStatus
    result: RunResult


class Orchestrator:
    """Builds and runs the LangGraph workflow; owns no business logic itself.

    Args:
        repo: Persistence (criteria, runs, LLM cache).
        settings: Limits.
        provider_factory: Builds the LLM adapter (tests inject fakes).
        on_event: Progress callback (the UI renders these in ``st.status``).
        id_factory: RFP_RUN_ID generator (tests inject fixed ids).
        clock: UTC ISO timestamp source (tests inject a fixed time).
    """

    def __init__(
        self,
        repo: Repository,
        settings: Settings,
        provider_factory: ProviderFactory = create_provider,
        on_event: EventHandler | None = None,
        id_factory: Callable[[], str] = lambda: str(uuid.uuid4()),
        clock: Callable[[], str] = utc_now_iso,
    ) -> None:
        self.repo = repo
        self.settings = settings
        self.provider_factory = provider_factory
        self.on_event = on_event or (lambda _event: None)
        self.id_factory = id_factory
        self.clock = clock
        self._agent: EvaluationAgent | None = None
        self.graph = self._build_graph()

    # ── public API ────────────────────────────────────────────────────────
    def run(self, suppliers: Sequence[SupplierInput], llm_config: LLMConfig,
            *, max_pdf_chars: int | None = None) -> RunResult:
        """Execute the graph (steps 1–10) and return the completed run.

        Raises:
            ValidationError: criteria or supplier inputs are invalid (nothing is created).
            RunAbortedError: the batch stopped (e.g. invalid API key); run marked FAILED.
            PersistenceError: results could not be saved; run marked FAILED.
        """
        budget = max_pdf_chars if max_pdf_chars is not None else self.settings.max_pdf_chars
        initial: RunState = {"suppliers": list(suppliers), "llm_config": llm_config, "budget": budget}
        # Each supplier visits at most four nodes; leave headroom for the fixed steps.
        limit = 25 + 5 * len(suppliers)
        try:
            final = self.graph.invoke(initial, {"recursion_limit": limit})
        finally:
            self._agent = None
        return final["result"]

    def mermaid(self) -> str:
        """Mermaid source for the compiled graph (used in docs/ARCHITECTURE.md)."""
        return self.graph.get_graph().draw_mermaid()

    # ── graph definition ─────────────────────────────────────────────────
    def _build_graph(self) -> Any:
        g = StateGraph(RunState)
        g.add_node("setup", self._setup)
        g.add_node("check_inputs", self._check_inputs)
        g.add_node("create_batch", self._create_batch)
        g.add_node("extract_document", self._extract_document)
        g.add_node("evaluate_supplier", self._evaluate_supplier)
        g.add_node("validate_scorecard", self._validate_scorecard)
        g.add_node("record_supplier", self._record_supplier)
        g.add_node("score", self._score)
        g.add_node("benchmark", self._benchmark)
        g.add_node("rank", self._rank)
        g.add_node("persist", self._persist)
        g.add_node("present", self._present)

        g.add_edge(START, "setup")
        g.add_edge("setup", "check_inputs")
        g.add_edge("check_inputs", "create_batch")
        g.add_edge("create_batch", "extract_document")
        g.add_conditional_edges("extract_document", _after_extract,
                                {"evaluate": "evaluate_supplier", "skip": "record_supplier"})
        g.add_conditional_edges("evaluate_supplier", _after_evaluate,
                                {"validate": "validate_scorecard", "skip": "record_supplier"})
        g.add_edge("validate_scorecard", "record_supplier")
        g.add_conditional_edges("record_supplier", _after_record,
                                {"next": "extract_document", "done": "score"})
        g.add_edge("score", "benchmark")
        g.add_edge("benchmark", "rank")
        g.add_edge("rank", "persist")
        g.add_edge("persist", "present")
        g.add_edge("present", END)
        return g.compile(name="rfp_evaluation")

    # ── helpers ───────────────────────────────────────────────────────────
    def _emit(self, step: int, message: str, *, supplier: str | None = None,
              level: str = "info", fraction: float = 0.0) -> None:
        self.on_event(ProgressEvent(step, message, supplier, level, fraction))

    @staticmethod
    def _progress(state: RunState, part: float) -> float:
        """Overall progress for the current supplier: 8% → 88% across all suppliers."""
        n = max(1, len(state["suppliers"]))
        return 0.08 + 0.8 * (state["index"] + part) / n

    # ── nodes ─────────────────────────────────────────────────────────────
    def _setup(self, state: RunState) -> RunState:
        """1. Load the latest active criteria and check they are usable."""
        criteria = self.repo.list_criteria(active_only=True)
        issues = validate_active_criteria(criteria, self.settings)
        if issues:
            raise ValidationError(issues[0].code, issues[0].message)
        self._emit(1, f"Loaded {len(criteria)} active criteria", fraction=0.02)
        return {"snapshot": sorted(criteria, key=lambda c: c.criterion_id)}

    def _check_inputs(self, state: RunState) -> RunState:
        """2. Defensive re-check of supplier inputs (the UI validates first)."""
        suppliers = state["suppliers"]
        issues = check_file_count(len(suppliers), self.settings)
        if issues:
            raise ValidationError(issues[0].code, issues[0].message)
        names = [normalise_name(s.supplier_name).casefold() for s in suppliers]
        if len(set(names)) != len(names) or not all(names):
            raise ValidationError(ErrorCode.META_NAME_DUPLICATE,
                                  "Supplier names must be present and unique within the batch.")
        hashes = [s.file_sha256 for s in suppliers]
        if len(set(hashes)) != len(hashes):
            raise ValidationError(ErrorCode.FILE_DUPLICATE, "The same file was uploaded more than once.")
        self._emit(2, f"{len(suppliers)} suppliers ready for evaluation", fraction=0.04)
        return {}

    def _create_batch(self, state: RunState) -> RunState:
        """3. Create the run with an immutable criteria snapshot and build the LLM agent."""
        cfg = state["llm_config"]
        run_id, created_at = self.id_factory(), self.clock()
        self.repo.create_run(run_id, created_at, cfg.provider, cfg.model, PROMPT_VERSION, state["snapshot"])
        self._emit(3, f"Created run {run_id}", fraction=0.06)
        try:
            provider = self.provider_factory(cfg)
        except LLMError as exc:
            self.repo.update_run_status(run_id, RunStatus.FAILED)
            raise RunAbortedError(ErrorCode.RUN_ABORTED, exc.user_message) from exc
        self._agent = EvaluationAgent(provider, cache=self.repo)
        self.repo.update_run_status(run_id, RunStatus.RUNNING)
        return {"run_id": run_id, "created_at": created_at, "index": 0, "cards": []}

    def _extract_document(self, state: RunState) -> RunState:
        """4a. Extract text; short-circuit corrupt and image-only PDFs."""
        supplier = state["suppliers"][state["index"]]
        name, snapshot = supplier.supplier_name, state["snapshot"]
        self._emit(4, "Extracting text", supplier=name, fraction=self._progress(state, 0))
        try:
            doc = document_tool.extract_document(supplier.file_bytes, supplier.filename)
        except DocumentError as exc:
            self._emit(4, exc.user_message, supplier=name, level="error", fraction=self._progress(state, 0))
            card = validation_tool.failed_scorecard(
                supplier, snapshot, SupplierStatus.EVALUATION_FAILED, exc.user_message,
                warnings=[WarningRecord(code=exc.code.value, supplier_name=name, message=exc.user_message)])
            return {"card": card, "document": None}

        if not document_tool.has_enough_text(doc, self.settings.min_text_chars):
            msg = (f"Only {document_tool.content_length(doc.text)} characters of text could be extracted "
                   f"(minimum {self.settings.min_text_chars}); the PDF looks scanned or image-only. "
                   "LLM call skipped and all scores set to 0.")
            self._emit(4, msg, supplier=name, level="warning", fraction=self._progress(state, 1))
            card = validation_tool.failed_scorecard(
                supplier, snapshot, SupplierStatus.NO_TEXT, "no extractable text",
                warnings=[WarningRecord(code="NO_TEXT", supplier_name=name, message=msg)], document=doc)
            return {"card": card, "document": doc}

        doc, trunc_warning = document_tool.truncate_document(doc, state["budget"], name)
        return {"card": None, "document": doc, "pending_warnings": [trunc_warning] if trunc_warning else []}

    def _evaluate_supplier(self, state: RunState) -> RunState:
        """4b. One independent LLM evaluation (cache-aware) via the Evaluation Agent."""
        supplier = state["suppliers"][state["index"]]
        doc = state["document"]
        assert doc is not None and self._agent is not None
        name = supplier.supplier_name
        self._emit(4, f"Extracted {doc.page_count} pages via {doc.extraction_method}; calling the LLM",
                   supplier=name, fraction=self._progress(state, 0.3))
        try:
            outcome = self._agent.evaluate(name, state["snapshot"], doc)
        except LLMError as exc:
            if exc.code is ErrorCode.LLM_AUTH:
                self.repo.update_run_status(state["run_id"], RunStatus.FAILED)
                self._emit(4, exc.user_message, supplier=name, level="error")
                raise RunAbortedError(ErrorCode.RUN_ABORTED, exc.user_message) from exc
            self._emit(4, f"LLM failed after retries: {exc.user_message}", supplier=name, level="error")
            card = validation_tool.failed_scorecard(
                supplier, state["snapshot"], SupplierStatus.LLM_FAILED, exc.user_message,
                warnings=[WarningRecord(code=exc.code.value, supplier_name=name, message=exc.user_message)])
            return {"card": card, "outcome": None}
        return {"outcome": outcome}

    def _validate_scorecard(self, state: RunState) -> RunState:
        """5. Normalise the model output into a scorecard (or a failed one)."""
        supplier = state["suppliers"][state["index"]]
        outcome, doc = state["outcome"], state["document"]
        assert outcome is not None and doc is not None
        name = supplier.supplier_name
        warnings = [*state.get("pending_warnings", []), *outcome.warnings]
        self._emit(5, "Validating and normalising the scorecard", supplier=name,
                   fraction=self._progress(state, 0.8))
        if outcome.data is None:
            card = validation_tool.failed_scorecard(
                supplier, state["snapshot"], SupplierStatus.EVALUATION_FAILED,
                "the model did not return valid JSON", warnings=warnings, document=doc)
            return {"card": card}
        card = validation_tool.validate_evaluation(
            outcome.data, supplier=supplier, criteria=state["snapshot"], document=doc,
            settings=self.settings, prior_warnings=warnings)
        card = card.model_copy(update={"from_cache": outcome.from_cache})
        note = " (cached result)" if outcome.from_cache else ""
        self._emit(5, f"Scorecard ready with {len(card.warnings)} warning(s){note}",
                   supplier=name, level="success", fraction=self._progress(state, 1))
        return {"card": card}

    def _record_supplier(self, state: RunState) -> RunState:
        """Append this supplier's scorecard and move to the next one."""
        card = state["card"]
        assert card is not None
        return {"cards": [*state["cards"], card], "index": state["index"] + 1, "card": None,
                "document": None, "outcome": None, "pending_warnings": []}

    def _score(self, state: RunState) -> RunState:
        """6. Absolute weighted score for every supplier."""
        scores = {c.supplier_name: ranking_tool.absolute_score(c, state["snapshot"], self.settings.decimals)
                  for c in state["cards"]}
        self._emit(6, "Calculated absolute weighted scores", fraction=0.9)
        return {"absolute_scores": scores}

    def _benchmark(self, state: RunState) -> RunState:
        """7. Best validated score per criterion across all suppliers."""
        benchmarks = ranking_tool.compute_benchmarks(state["cards"], state["snapshot"], self.settings.decimals)
        self._emit(7, "Benchmarked suppliers on every criterion", fraction=0.92)
        return {"benchmarks": benchmarks}

    def _rank(self, state: RunState) -> RunState:
        """8. Gap, relative %, PPI, tie-breaks and sequential ranks (pure Ranking Tool)."""
        ranking = ranking_tool.rank_suppliers(state["cards"], state["snapshot"], self.settings.decimals)
        self._emit(8, f"Ranked {len(ranking.rankings)} suppliers by PPI with tie-breaks", fraction=0.94)
        return {"ranking": ranking, "status": _overall_status(state["cards"])}

    def _persist(self, state: RunState) -> RunState:
        """9. Write the complete run in one transaction."""
        cfg = state["llm_config"]
        result = RunResult(
            rfp_run_id=state["run_id"], created_at=state["created_at"], status=state["status"],
            provider=cfg.provider, model=cfg.model, prompt_version=PROMPT_VERSION,
            criteria_snapshot=state["snapshot"],
            scorecards=sorted(state["cards"], key=lambda c: c.supplier_name.casefold()),
            ranking=state["ranking"],
        )
        try:
            self.repo.save_run_results(result)
        except PersistenceError:
            self.repo.update_run_status(state["run_id"], RunStatus.FAILED)
            raise
        self._emit(9, "Saved results to SQLite", fraction=0.98)
        return {"result": result}

    def _present(self, state: RunState) -> RunState:
        """10. Hand the run to the UI."""
        status = state["result"].status
        level = "success" if status is RunStatus.COMPLETED else "warning"
        self._emit(10, f"Run finished with status {status.value}", level=level, fraction=1.0)
        return {}


# ── routing functions (conditional edges) ───────────────────────────────────
def _after_extract(state: RunState) -> str:
    """Skip the LLM when extraction already produced a (failed / NO_TEXT) scorecard."""
    return "skip" if state.get("card") is not None else "evaluate"


def _after_evaluate(state: RunState) -> str:
    """Skip validation when the LLM call failed and produced a failed scorecard."""
    return "skip" if state.get("card") is not None else "validate"


def _after_record(state: RunState) -> str:
    """Loop until every supplier has a scorecard."""
    return "next" if state["index"] < len(state["suppliers"]) else "done"


def _overall_status(cards: Sequence[SupplierScorecard]) -> RunStatus:
    ok = sum(1 for c in cards if c.status is SupplierStatus.OK)
    if ok == len(cards):
        return RunStatus.COMPLETED
    if ok == 0:
        return RunStatus.FAILED
    return RunStatus.PARTIAL


def export_json(result: RunResult) -> str:
    """Canonical, pretty JSON export of a run (contains no secrets).

    Suppliers appear in final rank order so the file reads like the leaderboard.
    """
    payload = result.model_dump(mode="json")
    rank_order = {r.supplier_name: r.final_rank for r in result.ranking.rankings}
    payload["scorecards"].sort(key=lambda c: rank_order.get(c["supplier_name"], 10**6))
    payload["export_format"] = "rfp-evaluator/v1"
    return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False)
