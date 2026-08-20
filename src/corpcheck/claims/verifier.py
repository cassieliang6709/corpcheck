"""Deterministic claim verifier (first pass, no LLM).

The strategy is intentionally conservative:
- no dynamic scoring models,
- no retrieval re-ranking beyond existing `retrieve()`,
- no fuzzy entailment, only lexical + numeric heuristics.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Iterable, Optional

from corpcheck.models import ChunkResult

from .normalizer import _extract_metric as _infer_metric
from .normalizer import extract_numeric_bindings
from .schema import ClaimVerdict, EvidenceItem, FinancialClaim

_STOP_WORDS = {
    "has",
    "have",
    "the",
    "is",
    "it",
    "on",
    "or",
    "that",
    "this",
    "to",
    "was",
    "were",
    "a",
    "an",
    "and",
    "of",
    "in",
    "for",
    "as",
    "by",
    "with",
    "from",
    "at",
    "will",
}

# Keep alias families explicit so comparison can match "total revenue" and
# "net sales" as the same high-level intent.
_METRIC_ALIASES = {
    "revenue": {
        "revenue",
        "total revenue",
        "net sales",
        "sales",
        "营业收入",
        "营收",
    },
    "profit": {
        "profit",
        "net income",
        "income",
        "净利润",
        "盈利",
        "利润",
    },
    "eps": {"eps", "each share", "每股收益", "每股盈余"},
    "margin": {"margin", "毛利", "净利率", "margin rate"},
    "cash flow": {"cash flow", "现金流"},
    "share": {"share", "shares", "股份", "股票"},
}


def _short_text(text: str, limit: int = 260) -> str:
    if len(text) <= limit:
        return text
    half = max(1, (limit - 3) // 2)
    return f"{text[:half]}…{text[-half:]}"


def _metric_aliases(metric: Optional[str]) -> set[str]:
    if not metric:
        return set()
    lowered = metric.lower()
    return _METRIC_ALIASES.get(lowered, {lowered})


def _metric_match(metric: Optional[str], chunk_text: str) -> bool:
    if not metric:
        return True
    lowered = chunk_text.lower()
    return any(alias in lowered for alias in _metric_aliases(metric))


def _to_floatish(value: Decimal) -> float:
    return float(value)


def _numeric_equal(a: Decimal, b: Decimal) -> bool:
    delta = abs(a - b)
    scale = max(Decimal("1"), abs(a), abs(b))
    return _to_floatish(delta) <= 1e-6 * _to_floatish(scale)


def _numeric_compare(
    actual: Decimal, expected: Decimal, comparator: Optional[str]
) -> bool:
    if comparator in {None, "eq"}:
        return _numeric_equal(actual, expected)
    if comparator == "gt":
        return actual > expected
    if comparator == "lt":
        return actual < expected
    if comparator == "gte":
        return actual >= expected
    if comparator == "lte":
        return actual <= expected
    return _numeric_equal(actual, expected)


def _as_chunk_list(chunks: Iterable[Any]) -> list[ChunkResult]:
    return [
        chunk if isinstance(chunk, ChunkResult) else ChunkResult(**chunk)
        for chunk in chunks
    ]


def _evidence_from_chunk(
    claim_id: str,
    chunk: ChunkResult,
    value: Optional[Decimal] = None,
    unit: Optional[str] = None,
) -> EvidenceItem:
    return EvidenceItem(
        claim_id=claim_id,
        source=(chunk.source_url or chunk.chunk_id),
        excerpt=_short_text(chunk.text),
        filing_id=(chunk.chunk_id if "-" in chunk.chunk_id else None),
        score=(chunk.cos_sim),
        value=value,
        unit=unit,
    )


def _keyword_overlap(a: str, b: str) -> float:
    a_tokens = {t for t in a.lower().split() if t and t not in _STOP_WORDS}
    b_tokens = {t for t in b.lower().split() if t and t not in _STOP_WORDS}
    if not a_tokens:
        return 0.0
    if not b_tokens:
        return 0.0
    return len(a_tokens & b_tokens) / len(a_tokens)


def _verify_with_chunk_set(
    claim: FinancialClaim,
    chunks: list[ChunkResult],
) -> ClaimVerdict:
    if claim.checkability == "non_verifiable":
        return ClaimVerdict(
            claim_id=claim.claim_id,
            verdict="non_verifiable",
            reason_code="unsupported_claim_type",
        )
    if claim.checkability == "watch_later":
        return ClaimVerdict(
            claim_id=claim.claim_id,
            verdict="not_yet_decidable",
            reason_code="prediction_requires_future_evidence",
        )

    if not chunks:
        return ClaimVerdict(
            claim_id=claim.claim_id,
            verdict="insufficient_evidence",
            reason_code="no_evidence_retrieved",
        )

    support: list[EvidenceItem] = []
    oppose: list[EvidenceItem] = []

    expected = claim.value
    metric = claim.metric or _infer_metric(claim.normalized_text)
    as_of_year = claim.as_of.year if claim.as_of is not None else None

    for chunk in chunks:
        if as_of_year is not None and chunk.fiscal_year and chunk.fiscal_year > as_of_year:
            continue

        lowered_chunk = chunk.text.lower()
        if (
            claim.checkability == "check_now"
            and metric
            and not _metric_match(metric, lowered_chunk)
        ):
            # Keep deterministic behavior: don't infer metric from the paragraph.
            continue

        if expected is not None and claim.value is not None:
            for value, unit, _raw in extract_numeric_bindings(chunk.text):
                if not _metric_match(metric, lowered_chunk):
                    continue
                if _numeric_compare(value, expected, claim.comparator):
                    support.append(
                        _evidence_from_chunk(claim.claim_id, chunk, value, unit)
                    )
                elif _metric_match(metric, lowered_chunk):
                    oppose.append(
                        _evidence_from_chunk(claim.claim_id, chunk, value, unit)
                    )
            continue

        # Non-numeric claim (or missing explicit value): deterministic keyword overlap.
        overlap = _keyword_overlap(claim.normalized_text, lowered_chunk)
        if overlap >= 0.55:
            support.append(_evidence_from_chunk(claim.claim_id, chunk))
        else:
            oppose.append(_evidence_from_chunk(claim.claim_id, chunk))

    if support:
        if expected is not None:
            return ClaimVerdict(
                claim_id=claim.claim_id,
                verdict="verified",
                reason_code="evidence_binds_value_and_metric",
                evidence_for=support,
                evidence_against=[],
            )
        return ClaimVerdict(
            claim_id=claim.claim_id,
            verdict="verified",
            reason_code="narrative_support_in_retrieved_chunks",
            evidence_for=support,
            evidence_against=[],
        )

    if oppose and expected is not None:
        return ClaimVerdict(
            claim_id=claim.claim_id,
            verdict="conflicting",
            reason_code="retrieved_chunks_disagree_with_numeric_assertion",
            evidence_for=[],
            evidence_against=oppose,
        )

    return ClaimVerdict(
        claim_id=claim.claim_id,
        verdict="insufficient_evidence",
        reason_code="retrieved_chunks_do_not_bind_expected_numeric_claim",
        evidence_for=[],
        evidence_against=oppose,
    )


def verify_claims(
    claims: list[FinancialClaim],
    retrieve_chunks: Iterable[list[Any]],
) -> tuple[list[ClaimVerdict], list[str]]:
    """Run deterministic policy for each claim.

    ``retrieve_chunks`` is intentionally a sequence of per-claim chunk lists in the
    same order as claims, making this function easy to test with monkeypatches.
    """
    verdicts: list[ClaimVerdict] = []
    obligations: list[str] = []

    for claim, chunk_payload in zip(claims, retrieve_chunks, strict=False):
        chunks = _as_chunk_list(chunk_payload)
        verdict = _verify_with_chunk_set(claim, chunks)
        verdicts.append(verdict)

    return verdicts, obligations
