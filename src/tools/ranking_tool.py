"""Ranking Tool: deterministic scoring, peer benchmarks, PPI, tie-breaks and ranks.

Pure functions only: no I/O, no LLM, no clock, no randomness. Given the same
validated scorecards and criteria snapshot the output is always identical,
whatever order the scorecards arrive in.

Formulas (all values rounded to ``decimals`` places before comparison):

* weighted_score_c = score_c / max_score_c × weight_c
* absolute_score   = Σ_c weighted_score_c                 (0–100 when weights total 100)
* benchmark_c      = max over suppliers of score_c
* gap_c            = score_c − benchmark_c                 (0 for the leader, else negative)
* relative_pct_c   = score_c / benchmark_c × 100           (0 when benchmark_c = 0, with a warning)
* PPI              = Σ_c (relative_pct_c × weight_c) / Σ_c weight_c

Tie-break order (stable sort): PPI desc → submission date asc → experience
rating desc → supplier name asc (case-folded). Ranks are then 1..N with no
shared ranks.
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import pairwise

from src.models.schemas import (
    Criterion,
    CriterionMetric,
    RankingResult,
    SupplierRanking,
    SupplierScorecard,
    WarningRecord,
)

DEFAULT_DECIMALS = 4

TIE_RULES = (
    "earlier submission date",
    "higher historical experience rating",
    "supplier name in ascending order",
)


def _r(value: float, decimals: int) -> float:
    """Round and normalise ``-0.0`` to ``0.0`` so output is byte-stable."""
    out = round(value, decimals)
    return 0.0 if out == 0 else out


def compute_benchmarks(
    scorecards: Sequence[SupplierScorecard], criteria: Sequence[Criterion], decimals: int = DEFAULT_DECIMALS
) -> dict[int, float]:
    """Highest validated score per criterion across all suppliers."""
    bench: dict[int, float] = {}
    for crit in criteria:
        values = [_score_of(card, crit.criterion_id) for card in scorecards]
        bench[crit.criterion_id] = _r(max(values, default=0.0), decimals)
    return bench


def absolute_score(card: SupplierScorecard, criteria: Sequence[Criterion],
                   decimals: int = DEFAULT_DECIMALS) -> float:
    """Σ (score / max_score) × weight, using the criteria snapshot's max and weight."""
    total = sum(_score_of(card, c.criterion_id) / c.max_score * c.weight for c in criteria)
    return _r(total, decimals)


def relative_pct(score: float, benchmark: float, decimals: int = DEFAULT_DECIMALS) -> float:
    """score / benchmark × 100, defined as 0 when the benchmark is 0."""
    if benchmark == 0:
        return 0.0
    return _r(score / benchmark * 100.0, decimals)


def peer_performance_index(metrics: Sequence[CriterionMetric], decimals: int = DEFAULT_DECIMALS) -> float:
    """Weight-averaged relative performance: Σ(rel% × w) / Σw (0 if Σw = 0)."""
    total_weight = sum(m.weight for m in metrics)
    if total_weight == 0:
        return 0.0
    return _r(sum(m.relative_pct * m.weight for m in metrics) / total_weight, decimals)


def sort_key(r: SupplierRanking) -> tuple[float, str, float, str]:
    """Composite key for the mandatory tie-break order (ascending sort)."""
    return (-r.ppi, r.submission_date.isoformat(), -r.experience_rating, r.supplier_name.casefold())


def explain_tie(upper: SupplierRanking, lower: SupplierRanking) -> str:
    """Sentence naming the rule that put ``upper`` above ``lower`` (equal PPI assumed)."""
    if upper.submission_date != lower.submission_date:
        rule = TIE_RULES[0]
        detail = f"{upper.submission_date.isoformat()} vs {lower.submission_date.isoformat()}"
    elif upper.experience_rating != lower.experience_rating:
        rule = TIE_RULES[1]
        detail = f"{upper.experience_rating:g} vs {lower.experience_rating:g}"
    else:
        rule = TIE_RULES[2]
        detail = f"'{upper.supplier_name}' before '{lower.supplier_name}'"
    return (f"{upper.supplier_name} ranked above {lower.supplier_name}: equal PPI {upper.ppi:g}; "
            f"{rule} ({detail}).")


def rank_suppliers(
    scorecards: Sequence[SupplierScorecard],
    criteria: Sequence[Criterion],
    decimals: int = DEFAULT_DECIMALS,
) -> RankingResult:
    """Score, benchmark and rank every supplier.

    Args:
        scorecards: Validated scorecards (any order). Names must be unique.
        criteria: The run's criteria snapshot (active criteria only).
        decimals: Rounding applied to every computed value.

    Returns:
        Rankings in final order, per-criterion benchmarks, tie-break
        explanations and ranking warnings.
    """
    ordered_criteria = sorted(criteria, key=lambda c: c.criterion_id)
    # Sort inputs first so that nothing downstream depends on arrival order.
    cards = sorted(scorecards, key=lambda c: c.supplier_name.casefold())
    benchmarks = compute_benchmarks(cards, ordered_criteria, decimals)

    warnings: list[WarningRecord] = [
        WarningRecord(
            code="ZERO_BENCHMARK",
            criterion_id=c.criterion_id,
            message=(f"Every supplier scored 0 on '{c.name}'; relative % for this criterion is "
                     "defined as 0 for all suppliers."),
        )
        for c in ordered_criteria
        if benchmarks[c.criterion_id] == 0
    ]

    rankings: list[SupplierRanking] = []
    for card in cards:
        metrics = []
        for c in ordered_criteria:
            score = _r(_score_of(card, c.criterion_id), decimals)
            bench = benchmarks[c.criterion_id]
            metrics.append(CriterionMetric(
                criterion_id=c.criterion_id,
                criterion_name=c.name,
                score=score,
                max_score=c.max_score,
                weight=c.weight,
                weighted_score=_r(score / c.max_score * c.weight, decimals),
                benchmark=bench,
                gap=_r(score - bench, decimals),
                relative_pct=relative_pct(score, bench, decimals),
            ))
        rankings.append(SupplierRanking(
            supplier_name=card.supplier_name,
            submission_date=card.submission_date,
            experience_rating=card.experience_rating,
            absolute_score=absolute_score(card, ordered_criteria, decimals),
            ppi=peer_performance_index(metrics, decimals),
            final_rank=0,
            metrics=metrics,
        ))

    rankings.sort(key=sort_key)
    final = [r.model_copy(update={"final_rank": i}) for i, r in enumerate(rankings, start=1)]
    explanations = [
        explain_tie(upper, lower)
        for upper, lower in pairwise(final)
        if upper.ppi == lower.ppi
    ]
    return RankingResult(
        rankings=final, benchmarks=benchmarks, tie_break_explanations=explanations, warnings=warnings
    )


def _score_of(card: SupplierScorecard, criterion_id: int) -> float:
    for cs in card.criteria:
        if cs.criterion_id == criterion_id:
            return cs.score
    return 0.0
