"""Ranking tool: formulas, benchmarks, PPI, every tie-break level and determinism."""

from __future__ import annotations

import random
from datetime import date

import pytest

from src.models.schemas import Criterion
from src.tools.ranking_tool import (
    absolute_score,
    peer_performance_index,
    rank_suppliers,
    relative_pct,
)
from tests.conftest import make_criteria, make_scorecard


def test_absolute_score_weighted_math(criteria: list[Criterion]) -> None:
    card = make_scorecard("A", {1: 8, 2: 6, 3: 10, 4: 5, 5: 0}, criteria)
    # 0.8*30 + 0.6*20 + 1.0*20 + 0.5*20 + 0*10 = 24 + 12 + 20 + 10 + 0
    assert absolute_score(card, criteria) == 66.0


def test_absolute_score_respects_max_score() -> None:
    crits = [Criterion(criterion_id=1, name="A", weight=60, max_score=5),
             Criterion(criterion_id=2, name="B", weight=40, max_score=20)]
    card = make_scorecard("A", {1: 4, 2: 10}, crits)
    assert absolute_score(card, crits) == pytest.approx(48 + 20)


def test_benchmark_gap_relative(criteria: list[Criterion]) -> None:
    a = make_scorecard("A", {1: 8, 2: 4}, criteria)
    b = make_scorecard("B", {1: 6, 2: 5}, criteria)
    res = rank_suppliers([a, b], criteria)
    assert res.benchmarks[1] == 8 and res.benchmarks[2] == 5
    b_rank = next(r for r in res.rankings if r.supplier_name == "B")
    m1 = b_rank.metrics[0]
    assert m1.gap == -2.0
    assert m1.relative_pct == 75.0
    a_rank = next(r for r in res.rankings if r.supplier_name == "A")
    assert a_rank.metrics[0].gap == 0.0
    assert a_rank.metrics[0].relative_pct == 100.0


def test_relative_pct_zero_benchmark() -> None:
    assert relative_pct(0, 0) == 0.0
    assert relative_pct(3, 4) == 75.0


def test_zero_benchmark_warning(criteria: list[Criterion]) -> None:
    a = make_scorecard("A", {1: 8}, criteria)
    b = make_scorecard("B", {1: 6}, criteria)
    res = rank_suppliers([a, b], criteria)
    zero = [w for w in res.warnings if w.code == "ZERO_BENCHMARK"]
    assert [w.criterion_id for w in zero] == [2, 3, 4, 5]
    assert all(m.relative_pct == 0 for r in res.rankings for m in r.metrics if m.criterion_id != 1)


def test_ppi_formula(criteria: list[Criterion]) -> None:
    a = make_scorecard("A", {1: 10, 2: 5, 3: 8, 4: 4, 5: 2}, criteria)
    b = make_scorecard("B", {1: 5, 2: 10, 3: 8, 4: 8, 5: 4}, criteria)
    res = rank_suppliers([a, b], criteria)
    ra = next(r for r in res.rankings if r.supplier_name == "A")
    # rel%: 100, 50, 100, 50, 50 -> (100*30 + 50*20 + 100*20 + 50*20 + 50*10) / 100
    assert ra.ppi == 75.0
    assert peer_performance_index(ra.metrics) == 75.0
    rb = next(r for r in res.rankings if r.supplier_name == "B")
    # rel%: 50, 100, 100, 100, 100 -> (1500 + 2000 + 2000 + 2000 + 1000) / 100
    assert rb.ppi == 85.0
    assert [r.supplier_name for r in res.rankings] == ["B", "A"]


def test_ppi_zero_total_weight() -> None:
    assert peer_performance_index([]) == 0.0


def _equal(names_dates_ratings):  # type: ignore[no-untyped-def]
    crits = make_criteria()
    return [make_scorecard(n, {1: 7}, crits, submitted=d, rating=r) for n, d, r in names_dates_ratings], crits


def test_tiebreak_submission_date() -> None:
    cards, crits = _equal([("Late", date(2026, 3, 2), 5), ("Early", date(2026, 3, 1), 1)])
    res = rank_suppliers(cards, crits)
    assert [r.supplier_name for r in res.rankings] == ["Early", "Late"]
    assert "earlier submission date" in res.tie_break_explanations[0]
    # Only criterion 1 (weight 30) has a non-zero benchmark: PPI = 100 * 30 / 100.
    assert res.tie_break_explanations[0].startswith("Early ranked above Late: equal PPI 30;")


def test_tiebreak_experience_rating() -> None:
    d = date(2026, 3, 1)
    cards, crits = _equal([("Low", d, 2.5), ("High", d, 4.5)])
    res = rank_suppliers(cards, crits)
    assert [r.supplier_name for r in res.rankings] == ["High", "Low"]
    assert "higher historical experience rating" in res.tie_break_explanations[0]


def test_tiebreak_name_casefolded() -> None:
    d = date(2026, 3, 1)
    cards, crits = _equal([("beta", d, 3), ("Alpha", d, 3)])
    res = rank_suppliers(cards, crits)
    assert [r.supplier_name for r in res.rankings] == ["Alpha", "beta"]
    assert "supplier name in ascending order" in res.tie_break_explanations[0]


def test_four_way_tie_uses_each_rule() -> None:
    cards, crits = _equal([
        ("Delta", date(2026, 3, 1), 3.0),   # same date+rating as Charlie -> name
        ("Charlie", date(2026, 3, 1), 3.0),
        ("Bravo", date(2026, 3, 1), 4.0),   # higher rating -> before Charlie
        ("Alpha", date(2026, 2, 1), 1.0),   # earliest date -> first
    ])
    res = rank_suppliers(cards, crits)
    assert [r.supplier_name for r in res.rankings] == ["Alpha", "Bravo", "Charlie", "Delta"]
    assert [r.final_rank for r in res.rankings] == [1, 2, 3, 4]
    rules = [e.split("; ")[1] for e in res.tie_break_explanations]
    assert rules[0].startswith("earlier submission date")
    assert rules[1].startswith("higher historical experience rating")
    assert rules[2].startswith("supplier name in ascending order")


def test_no_explanations_when_ppi_differs(criteria: list[Criterion]) -> None:
    res = rank_suppliers([make_scorecard("A", {1: 9}, criteria), make_scorecard("B", {1: 3}, criteria)],
                         criteria)
    assert res.tie_break_explanations == []


def test_sequential_unique_ranks(criteria: list[Criterion]) -> None:
    cards = [make_scorecard(f"S{i}", {1: i % 3}, criteria, rating=i % 2) for i in range(7)]
    res = rank_suppliers(cards, criteria)
    assert [r.final_rank for r in res.rankings] == list(range(1, 8))


def test_determinism_shuffled_input(criteria: list[Criterion]) -> None:
    cards = [
        make_scorecard(f"Supplier {i}", {1: (i * 7) % 11, 2: (i * 3) % 10, 3: i % 4, 4: 5, 5: (i * 5) % 9},
                       criteria, submitted=date(2026, 1, 1 + i % 3), rating=(i % 5) / 1.0)
        for i in range(10)
    ]
    baseline = rank_suppliers(cards, criteria).model_dump_json()
    rng = random.Random(1234)
    for _ in range(25):
        shuffled = cards[:]
        rng.shuffle(shuffled)
        assert rank_suppliers(shuffled, criteria).model_dump_json() == baseline


def test_missing_criterion_counts_as_zero(criteria: list[Criterion]) -> None:
    card = make_scorecard("A", {1: 10}, criteria)
    card = card.model_copy(update={"criteria": card.criteria[:1]})
    assert absolute_score(card, criteria) == 30.0


def test_rounding_to_four_decimals() -> None:
    crits = [Criterion(criterion_id=1, name="A", weight=100, max_score=3)]
    card = make_scorecard("A", {1: 1}, crits)
    assert absolute_score(card, crits) == 33.3333
