"""Unit tests for the retrieval-confidence gate in abstain.py."""

from __future__ import annotations

import pytest

from corpcheck.models import ChunkResult
from corpcheck.retrieval.abstain import (
    AbstainDecision,
    evaluate_answerability,
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
    company: str = "AAPL",
    fiscal_year: int | None = None,
    text: str = "Sample text",
) -> ChunkResult:
    """A candidate row. ``score`` defaults high to prove it is *not* consulted."""
    return ChunkResult(
        chunk_id=chunk_id,
        text=text,
        score=score,
        cos_sim=cos_sim,
        company=company,
        fiscal_year=fiscal_year,
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

    def test_status_defaults_preserve_manual_construction(self) -> None:
        assert AbstainDecision(True, "nope").status == "abstain"
        assert AbstainDecision(False).status == "pass"


class TestEvaluateAnswerability:
    @pytest.fixture(autouse=True)
    def company_caches(self, monkeypatch) -> None:
        from corpcheck.retrieval import query_parse

        names = {
            "AAPL": "Apple Inc",
            "AMZN": "Amazon.com, Inc.",
            "COST": "Costco Wholesale Corporation",
        }
        aliases: dict[str, str] = {}
        for ticker, name in names.items():
            for alias in query_parse._generate_company_aliases(name):
                aliases.setdefault(alias, ticker)
        monkeypatch.setattr(query_parse, "_known_tickers", set(names))
        monkeypatch.setattr(query_parse, "_company_alias_to_ticker", aliases)

    def test_future_year_abstains_despite_strong_costco_chunks(self) -> None:
        results = [
            _chunk(STRONG, f"c{year}", company="COST", fiscal_year=year)
            for year in range(2020, 2027)
        ]
        decision = evaluate_answerability(
            "What were Costco's total assets at the end of FY2099?", results
        )
        assert decision.abstain is True
        assert decision.status == "year_mismatch"

    def test_matching_company_and_year_reach_cosine_gate(self) -> None:
        results = [_chunk(STRONG, company="COST", fiscal_year=2021)]
        decision = evaluate_answerability("Costco FY2021 total assets", results)
        assert decision.abstain is False
        assert decision.status == "pass"

    def test_known_company_mismatch_abstains(self) -> None:
        results = [_chunk(STRONG, company="AAPL", fiscal_year=2021)]
        decision = evaluate_answerability("Costco FY2021 total assets", results)
        assert decision.abstain is True
        assert decision.status == "company_mismatch"

    def test_unknown_sec_issuer_abstains(self) -> None:
        results = [_chunk(STRONG, company="AAPL", fiscal_year=2023)]
        decision = evaluate_answerability(
            "According to its SEC annual filing, what was OpenAI's net income in FY2023?",
            results,
        )
        assert decision.abstain is True
        assert decision.status == "unknown_company"

    def test_explicit_company_filter_ignores_other_possessive_names(self) -> None:
        decision = evaluate_answerability(
            "According to the SEC filing, what was CEO's compensation in FY2023?",
            [_chunk(STRONG, company="AAPL", fiscal_year=2023)],
            expected_company="AAPL",
        )
        assert decision.abstain is False
        assert decision.status == "pass"

    @pytest.mark.parametrize(
        "query",
        [
            "What was the company's FY2023 net income in its SEC filing?",
            "What was management's FY2023 outlook in the SEC filing?",
        ],
    )
    def test_generic_possessives_do_not_trigger_unknown_company(self, query) -> None:
        decision = evaluate_answerability(
            query, [_chunk(STRONG, company="AAPL", fiscal_year=2023)]
        )
        assert decision.abstain is False

    @pytest.mark.parametrize(
        ("query", "company"),
        [
            ("What did Costco's SEC annual filing report for FY2023?", "COST"),
            ("What did Amazon's SEC annual filing report for FY2023?", "AMZN"),
            ("What did Apple Inc's SEC annual filing report for FY2023?", "AAPL"),
        ],
    )
    def test_known_possessive_forms_resolve_normally(self, query, company) -> None:
        decision = evaluate_answerability(
            query, [_chunk(STRONG, company=company, fiscal_year=2023)]
        )
        assert decision.abstain is False

    def test_comparative_query_needs_year_intersection_not_every_year(self) -> None:
        result = [_chunk(STRONG, company="AAPL", fiscal_year=2023)]
        assert not evaluate_answerability(
            "Compare AAPL FY2023 versus FY2022", result
        ).abstain
        decision = evaluate_answerability(
            "Compare AAPL FY2023 versus FY2022",
            [_chunk(STRONG, company="AAPL", fiscal_year=2021)],
        )
        assert decision.status == "year_mismatch"

    def test_later_filing_text_can_cover_an_earlier_requested_year(self) -> None:
        result = _chunk(
            STRONG,
            company="AAPL",
            fiscal_year=2023,
            text="Comparative net income for 2022 was $99 million.",
        )
        decision = evaluate_answerability("AAPL net income in FY2022", [result])
        assert decision.abstain is False
        assert decision.status == "pass"

    def test_year_must_be_covered_by_the_requested_company(self) -> None:
        results = [
            _chunk(STRONG, "cost", company="COST", fiscal_year=2022),
            _chunk(
                STRONG,
                "apple",
                company="AAPL",
                fiscal_year=2023,
                text="Results for 2023.",
            ),
        ]
        decision = evaluate_answerability("Costco total assets in FY2023", results)
        assert decision.abstain is True
        assert decision.status == "year_mismatch"

    def test_no_company_or_year_preserves_confidence_behavior(self) -> None:
        results = [_chunk(WEAK)]
        expected = evaluate_confidence(results)
        actual = evaluate_answerability("What are the main business risks?", results)
        assert (actual.abstain, actual.status) == (expected.abstain, expected.status)

    def test_empty_results_preserve_existing_reason(self) -> None:
        decision = evaluate_answerability("OpenAI's SEC annual filing", [])
        assert decision.status == "no_results"
        assert decision.reason == "No matching passages were retrieved."


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
