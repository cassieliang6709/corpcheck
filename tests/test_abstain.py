"""Unit tests for double-threshold abstain gating in abstain.py."""

from __future__ import annotations

import pytest

from corpcheck.models import ChunkResult
from corpcheck.retrieval.abstain import should_abstain


def _chunk(score: float, chunk_id: str = "c1") -> ChunkResult:
    return ChunkResult(
        chunk_id=chunk_id,
        text="Sample text",
        score=score,
        company="AAPL",
    )


class TestShouldAbstain:
    def test_empty_results_abstains(self) -> None:
        abstain, reason = should_abstain([])
        assert abstain is True
        assert "No candidate" in reason

    def test_high_confidence_does_not_abstain(self) -> None:
        results = [_chunk(0.85, "c1"), _chunk(0.75, "c2"), _chunk(0.65, "c3")]
        abstain, reason = should_abstain(results, top1_min=0.35, mean_top3_min=0.25)
        assert abstain is False
        assert reason == ""

    def test_low_top1_score_abstains(self) -> None:
        results = [_chunk(0.30, "c1"), _chunk(0.28, "c2"), _chunk(0.26, "c3")]
        abstain, reason = should_abstain(results, top1_min=0.35, mean_top3_min=0.25)
        assert abstain is True
        assert "Top-1 score 0.3000 is below minimum" in reason

    def test_low_mean_top3_score_abstains(self) -> None:
        # Top-1 is 0.40 (above 0.35 threshold), but top 3 are [0.40, 0.10, 0.10] -> mean = 0.20 < 0.25
        results = [_chunk(0.40, "c1"), _chunk(0.10, "c2"), _chunk(0.10, "c3")]
        abstain, reason = should_abstain(results, top1_min=0.35, mean_top3_min=0.25)
        assert abstain is True
        assert "Mean top-3 score 0.2000 is below minimum" in reason

    def test_fewer_than_3_results_calculates_mean_over_available(self) -> None:
        # 2 results: [0.40, 0.20] -> mean = 0.30 >= 0.25
        results = [_chunk(0.40, "c1"), _chunk(0.20, "c2")]
        abstain, reason = should_abstain(results, top1_min=0.35, mean_top3_min=0.25)
        assert abstain is False
