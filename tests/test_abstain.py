"""Unit tests for the retrieval-confidence gate in abstain.py."""

from __future__ import annotations

import pytest

from corpcheck.models import ChunkResult
from corpcheck.retrieval.abstain import (
    AbstainDecision,
    evaluate_confidence,
    should_abstain,
)

# Values from evaluation/calibrate_abstain.py: in-domain queries bottom out
# around 0.55, out-of-domain queries top out around 0.34.
STRONG = 0.70
WEAK = 0.25
TOP1_MIN = 0.42
MEAN3_MIN = 0.40


def _chunk(
    cos_sim: float | None,
    chunk_id: str = "c1",
    score: float = 1.0,
) -> ChunkResult:
    """A candidate row. ``score`` defaults high to prove it is *not* consulted."""
    return ChunkResult(
        chunk_id=chunk_id,
        text="Sample text",
        score=score,
        cos_sim=cos_sim,
        company="AAPL",
    )


class TestEvaluateConfidence:
    def test_empty_results_abstain(self) -> None:
        decision = evaluate_confidence([])
        assert decision.abstain is True
        assert "No matching passages" in decision.reason

    def test_strong_dense_similarity_answers(self) -> None:
        results = [_chunk(0.72, "c1"), _chunk(0.68, "c2"), _chunk(0.66, "c3")]
        decision = evaluate_confidence(results, TOP1_MIN, MEAN3_MIN)
        assert decision.abstain is False
        assert decision.reason == ""
        assert decision.top1 == 0.72

    def test_uniformly_weak_similarity_abstains(self) -> None:
        # Shape of a real out-of-domain query: nothing clears the floor.
        results = [_chunk(0.34, "c1"), _chunk(0.31, "c2"), _chunk(0.29, "c3")]
        decision = evaluate_confidence(results, TOP1_MIN, MEAN3_MIN)
        assert decision.abstain is True
        assert "not contain passages relevant enough" in decision.reason

    def test_single_strong_hit_among_noise_abstains(self) -> None:
        # top1 clears its floor, but mean of top 3 does not: one lucky match is
        # not enough supporting evidence.
        results = [_chunk(0.60, "c1"), _chunk(0.20, "c2"), _chunk(0.18, "c3")]
        decision = evaluate_confidence(results, TOP1_MIN, MEAN3_MIN)
        assert decision.abstain is True
        assert "not enough supporting evidence" in decision.reason

    def test_gate_ignores_the_fused_score(self) -> None:
        """The regression this module exists to prevent.

        RRF min-max normalisation pins the best candidate's ``score`` to 1.0
        regardless of relevance, so a gate reading ``score`` can never fire.
        Every chunk here has the maximal fused score and junk similarity.
        """
        results = [
            _chunk(0.22, "c1", score=1.0),
            _chunk(0.21, "c2", score=0.98),
            _chunk(0.20, "c3", score=0.97),
        ]
        decision = evaluate_confidence(results, TOP1_MIN, MEAN3_MIN)
        assert decision.abstain is True

    def test_boost_reordering_does_not_hide_a_strong_match(self) -> None:
        """Similarities are read from the pool, not from the fused ordering.

        Metadata boosts can push a mediocre chunk to rank 1, so the gate sorts
        by cos_sim itself rather than trusting position 0.
        """
        results = [
            _chunk(0.44, "boosted", score=1.0),  # rank 1 by boosted score
            _chunk(0.71, "best", score=0.6),  # actually the closest match
            _chunk(0.69, "next", score=0.5),
        ]
        decision = evaluate_confidence(results, TOP1_MIN, MEAN3_MIN)
        assert decision.abstain is False
        assert decision.top1 == 0.71

    def test_sparse_only_candidates_are_allowed_through(self) -> None:
        # No cos_sim anywhere: refusing on a missing measurement would be wrong.
        results = [_chunk(None, "c1"), _chunk(None, "c2")]
        decision = evaluate_confidence(results, TOP1_MIN, MEAN3_MIN)
        assert decision.abstain is False
        assert decision.top1 is None

    def test_missing_similarities_are_skipped_not_zeroed(self) -> None:
        # Counting the sparse-only chunks as 0.0 would drag mean_top3 to 0.24
        # and abstain incorrectly.
        results = [_chunk(0.72, "c1"), _chunk(None, "c2"), _chunk(None, "c3")]
        decision = evaluate_confidence(results, TOP1_MIN, MEAN3_MIN)
        assert decision.abstain is False
        assert decision.mean_top3 == 0.72

    def test_fewer_than_three_results_averages_what_exists(self) -> None:
        results = [_chunk(0.70, "c1"), _chunk(0.60, "c2")]
        decision = evaluate_confidence(results, TOP1_MIN, MEAN3_MIN)
        assert decision.abstain is False
        assert decision.mean_top3 == pytest.approx(0.65)

    def test_threshold_boundary_is_inclusive(self) -> None:
        # Exactly at the floor should pass; the check is a strict `<`.
        results = [_chunk(TOP1_MIN, "c1"), _chunk(TOP1_MIN, "c2"), _chunk(TOP1_MIN, "c3")]
        decision = evaluate_confidence(results, TOP1_MIN, mean_top3_min=TOP1_MIN)
        assert decision.abstain is False


class TestAbstainDecision:
    def test_is_truthy_when_abstaining(self) -> None:
        assert bool(AbstainDecision(True, "nope"))
        assert not bool(AbstainDecision(False))

    def test_detail_formats_measured_values(self) -> None:
        decision = AbstainDecision(True, "r", top1=0.1234, mean_top3=0.5678)
        assert decision.detail == "top1=0.1234 mean_top3=0.5678"

    def test_detail_handles_absent_measurements(self) -> None:
        assert AbstainDecision(False).detail == "top1=n/a mean_top3=n/a"


class TestShouldAbstainWrapper:
    def test_returns_tuple(self) -> None:
        abstain, reason = should_abstain(
            [_chunk(0.20, "c1")], TOP1_MIN, MEAN3_MIN
        )
        assert abstain is True
        assert isinstance(reason, str) and reason

    def test_passing_case_returns_empty_reason(self) -> None:
        abstain, reason = should_abstain(
            [_chunk(STRONG, "c1"), _chunk(STRONG, "c2")], TOP1_MIN, MEAN3_MIN
        )
        assert abstain is False
        assert reason == ""


class TestCalibratedDefaults:
    """Guards the thresholds against the measured populations."""

    def test_in_domain_floor_passes_with_defaults(self) -> None:
        # Weakest in-domain query measured: top1=0.566, mean3=0.554.
        results = [_chunk(0.566, "c1"), _chunk(0.556, "c2"), _chunk(0.540, "c3")]
        assert evaluate_confidence(results).abstain is False

    def test_out_of_domain_ceiling_abstains_with_defaults(self) -> None:
        # Strongest out-of-domain query measured: top1=0.342, mean3=0.335.
        results = [_chunk(0.342, "c1"), _chunk(0.335, "c2"), _chunk(0.328, "c3")]
        assert evaluate_confidence(results).abstain is True
