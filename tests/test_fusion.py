"""Unit tests for retrieval score fusion algorithms (Min-Max and RRF)."""

from __future__ import annotations

import pytest

from corpcheck.retrieval.fusion import (
    compute_rrf_scores,
    fuse_candidates,
    fuse_scores,
    minmax_normalize,
)


def _row(chunk_id: str, score_v: float = 0.0, score_b: float = 0.0) -> dict[str, float | str]:
    return {"chunk_id": chunk_id, "score_v": score_v, "score_b": score_b}


# ── minmax_normalize ────────────────────────────────────────────────────────

class TestMinMaxNormalize:
    def test_basic_scaling(self) -> None:
        scores = [10.0, 20.0, 30.0]
        assert minmax_normalize(scores) == [0.0, 0.5, 1.0]

    def test_degenerate_all_equal(self) -> None:
        assert minmax_normalize([5.0, 5.0, 5.0]) == [1.0, 1.0, 1.0]

    def test_empty(self) -> None:
        assert minmax_normalize([]) == []

    def test_single_element(self) -> None:
        assert minmax_normalize([42.0]) == [1.0]


# ── fuse_scores ─────────────────────────────────────────────────────────────

class TestFuseScores:
    def test_linear_blend(self) -> None:
        # alpha * score_v + (1 - alpha) * score_b
        assert fuse_scores(1.0, 0.0, alpha=0.7) == pytest.approx(0.7)
        assert fuse_scores(0.5, 0.5, alpha=0.5) == pytest.approx(0.5)
        assert fuse_scores(0.0, 1.0, alpha=0.7) == pytest.approx(0.3)


# ── compute_rrf_scores ──────────────────────────────────────────────────────

class TestComputeRRF:
    def test_basic_rrf_ranking(self) -> None:
        vec_rows = [_row("c1"), _row("c2"), _row("c3")]
        bm25_rows = [_row("c2"), _row("c1"), _row("c4")]

        # c2 is rank 2 in vec, rank 1 in bm25 -> high combined rank
        # c1 is rank 1 in vec, rank 2 in bm25 -> high combined rank
        # c3 is rank 3 in vec, absent in bm25
        # c4 is absent in vec, rank 3 in bm25
        scores = compute_rrf_scores(vec_rows, bm25_rows, alpha=0.5, k_rrf=60, normalize=False)

        assert "c1" in scores and "c2" in scores and "c3" in scores and "c4" in scores
        # c1 raw score = 0.5 * (1/61) + 0.5 * (1/62)
        # c2 raw score = 0.5 * (1/62) + 0.5 * (1/61) -> equal with alpha=0.5
        assert scores["c1"] == pytest.approx(scores["c2"])
        # c1 and c2 should score higher than c3 and c4
        assert scores["c1"] > scores["c3"]
        assert scores["c1"] > scores["c4"]

    def test_alpha_weighting(self) -> None:
        vec_rows = [_row("c1"), _row("c2")]
        bm25_rows = [_row("c2"), _row("c1")]

        # alpha=1.0 -> Vector search only
        v_only = compute_rrf_scores(vec_rows, bm25_rows, alpha=1.0, normalize=False)
        assert v_only["c1"] > v_only["c2"]

        # alpha=0.0 -> BM25 search only
        b_only = compute_rrf_scores(vec_rows, bm25_rows, alpha=0.0, normalize=False)
        assert b_only["c2"] > b_only["c1"]

    def test_normalization_scales_to_unit_interval(self) -> None:
        vec_rows = [_row("c1"), _row("c2")]
        bm25_rows = [_row("c2"), _row("c3")]
        scores = compute_rrf_scores(vec_rows, bm25_rows, alpha=0.7, normalize=True)

        assert min(scores.values()) == pytest.approx(0.0)
        assert max(scores.values()) == pytest.approx(1.0)

    def test_empty_candidate_lists(self) -> None:
        assert compute_rrf_scores([], [], alpha=0.7) == {}

    def test_custom_k_rrf(self) -> None:
        vec_rows = [_row("c1")]
        bm25_rows = [_row("c1")]
        # Raw RRF score for rank 1 in both: alpha/(k+1) + (1-alpha)/(k+1) = 1/(k+1)
        raw = compute_rrf_scores(vec_rows, bm25_rows, k_rrf=10, normalize=False)
        assert raw["c1"] == pytest.approx(1.0 / 11.0)


# ── fuse_candidates ─────────────────────────────────────────────────────────

class TestFuseCandidates:
    def test_rrf_strategy(self) -> None:
        vec_rows = [_row("c1", score_v=0.9), _row("c2", score_v=0.8)]
        bm25_rows = [_row("c2", score_b=15.0), _row("c1", score_b=10.0)]

        res = fuse_candidates(vec_rows, bm25_rows, strategy="rrf")
        assert "c1" in res and "c2" in res

    def test_minmax_strategy(self) -> None:
        vec_rows = [_row("c1", score_v=0.9), _row("c2", score_v=0.1)]
        bm25_rows = [_row("c1", score_b=10.0), _row("c2", score_b=5.0)]

        res = fuse_candidates(vec_rows, bm25_rows, alpha=0.5, strategy="minmax")
        # c1 has top scores in both streams -> score 1.0
        # c2 has lowest scores in both streams -> score 0.0
        assert res["c1"] == pytest.approx(1.0)
        assert res["c2"] == pytest.approx(0.0)

    def test_invalid_strategy_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown fusion strategy"):
            fuse_candidates([], [], strategy="invalid_strategy")
