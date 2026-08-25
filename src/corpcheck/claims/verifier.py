"""Deterministic claim verifier (first pass, no LLM).

The strategy is intentionally conservative:
- no dynamic scoring models,
- no retrieval re-ranking beyond existing `retrieve()`,
- no fuzzy entailment, only lexical + numeric heuristics.

中文：核验器不用模糊语义猜测。数值必须绑定指标和可比单位；期间检查目前只在
双方都有财年时排除年份冲突，不比较季度，缺失期间元数据会记录为待补义务。
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Iterable, Optional

from corpcheck.models import ChunkResult

from .normalizer import (
    _PERCENT_UNITS,
    _UNIT_SCALE,
    extract_numeric_bindings_with_spans,
    find_metric_mentions,
    parse_fiscal_period,
)
from .normalizer import _extract_metric as _infer_metric
from .schema import ClaimVerdict, EvidenceItem, FinancialClaim

# A number is only treated as evidence for a metric when a mention of that
# metric sits within this many characters of it. Beyond that the number is in a
# different sentence or a different table row and binding it would be a guess.
_METRIC_BINDING_WINDOW = 120

# How far back to look for wording that marks a number as a movement.
_CHANGE_WINDOW = 48
_CHANGE_KEYWORDS = (
    "increased", "decreased", "increase", "decrease", "grew", "growth",
    "declined", "decline", "rose", "fell", "change", "changed", "higher",
    "lower", "up by", "down by", "增长", "下降", "减少", "增加",
)

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

def _short_text(text: str, limit: int = 260) -> str:
    """Trim a long excerpt symmetrically while preserving its two ends.

    中文：无精确匹配位置时仍给人工审核保留上下文两端，而不是只截取开头。
    """
    if len(text) <= limit:
        return text
    half = max(1, (limit - 3) // 2)
    return f"{text[:half]}…{text[-half:]}"


def _excerpt_around(text: str, start: int, end: int, width: int = 220) -> str:
    """Excerpt centred on the matched number, so the receipt shows the evidence.

    A receipt whose excerpt does not contain the number it claims to bind is
    useless to a human reviewer, so the window follows the match rather than
    always quoting the head of the chunk.

    中文：证据摘录必须包含被比较的数字，因此窗口围绕匹配位置而非固定从文本开头截取。
    """
    if not text:
        return ""
    half = max(1, width // 2)
    left = max(0, start - half)
    right = min(len(text), end + half)
    excerpt = text[left:right].strip()
    if left > 0:
        excerpt = f"…{excerpt}"
    if right < len(text):
        excerpt = f"{excerpt}…"
    return excerpt


def _bound_metric(
    mentions: list[tuple[int, int, str]], start: int, end: int
) -> Optional[str]:
    """Canonical metric that a number at ``[start, end)`` reports.

    Financial prose names the metric *before* its value — "net sales were
    $383.3 billion", "repurchased $19.1 billion". So the closest *preceding*
    mention wins, and a following mention is only used when nothing precedes.
    Pure nearest-mention binding is a coin flip on the very common
    "net sales were $383.3 billion and net income was $97.0 billion", where the
    next metric's name sits fewer characters away than the owning one.

    中文：财务文本通常先给指标再给数值，故优先选择最近的前置指标；这比纯距离匹配更不易错绑。
    """
    before: Optional[tuple[int, str]] = None
    after: Optional[tuple[int, str]] = None

    for m_start, m_end, metric in mentions:
        if m_start < end and m_end > start:
            return metric
        if m_end <= start:
            distance = start - m_end
            if distance <= _METRIC_BINDING_WINDOW and (
                before is None or distance < before[0]
            ):
                before = (distance, metric)
        else:
            distance = m_start - end
            if distance <= _METRIC_BINDING_WINDOW and (
                after is None or distance < after[0]
            ):
                after = (distance, metric)

    if before is not None:
        return before[1]
    return after[1] if after is not None else None


def _is_change_amount(text: str, start: int) -> bool:
    """True when a number reports a movement rather than a level.

    "total net sales decreased 3% or $11.0 billion" states a delta, not a
    balance. Scoring it against a claimed level produces a false refutation.
    "increased to $383.3 billion" is excluded, since there the number *is* the
    resulting level.

    中文：变化额不能反驳余额类声明；但“增至”后的结果数值仍可作为余额证据。
    """
    window = text[max(0, start - _CHANGE_WINDOW):start].lower()
    match = None
    for keyword in _CHANGE_KEYWORDS:
        position = window.rfind(keyword)
        if position != -1 and (match is None or position > match):
            match = position
    if match is None:
        return False
    return " to " not in window[match:]


def _to_floatish(value: Decimal) -> float:
    """Convert only for the relative-tolerance comparison below.

    中文：主要比较仍保留 Decimal 精度；此转换仅服务于微小的相对误差阈值。
    """
    return float(value)


_CURRENCY_MARKS = ("$", "¥", "€", "£", "dollar", "usd", "美元")


def _is_monetary(text: str) -> bool:
    """Detect whether a raw numeric expression carries a currency marker.

    中文：金额和股数即使单位倍率相同也不可互证，因此单独保留货币维度。
    """
    lowered = text.lower()
    return any(mark in lowered for mark in _CURRENCY_MARKS)


def _comparable_units(claim_unit: Optional[str], candidate_unit: Optional[str]) -> bool:
    """True when two quantities are the same kind of thing.

    Financial prose is full of bare integers — week counts, note numbers,
    segment counts. Without this check "fiscal year 2023 spanned 53 weeks"
    becomes counter-evidence against a revenue figure.

    中文：先保证量纲相同，再比较大小；防止周数、股数等裸整数与金额互相误判。
    """
    claim_scaled = claim_unit in _UNIT_SCALE
    candidate_scaled = candidate_unit in _UNIT_SCALE
    if claim_scaled or candidate_scaled:
        return claim_scaled and candidate_scaled
    if claim_unit in _PERCENT_UNITS or candidate_unit in _PERCENT_UNITS:
        return claim_unit in _PERCENT_UNITS and candidate_unit in _PERCENT_UNITS
    return claim_unit == candidate_unit


def _rounding_tolerance(expected: Decimal, unit: Optional[str]) -> Decimal:
    """Half a unit of the precision the claim was actually written at.

    A press release saying "$383.3 billion" is not contradicted by a 10-K
    saying "$383,285 million"; it is the same number quoted to fewer digits.
    Comparing at the claim's own stated precision keeps that from being scored
    as a refutation, without loosening into a blanket percentage tolerance.

    中文：容差取自声明的书写精度，而不是统一按百分比放宽，兼顾四舍五入与严格性。
    """
    scale = _UNIT_SCALE.get(unit or "", Decimal("1"))
    stated = expected / scale if scale != 0 else expected
    exponent = stated.normalize().as_tuple().exponent
    if not isinstance(exponent, int):
        return Decimal("0")
    # Clamp at 10^0: normalize() strips trailing zeros, so "900" reports an
    # exponent of +2 and would otherwise buy a ±50 billion tolerance.
    quantum = Decimal(1).scaleb(min(0, exponent))
    return (quantum / 2) * scale


def _numeric_equal(a: Decimal, b: Decimal, tolerance: Decimal = Decimal("0")) -> bool:
    """Compare exact Decimal values with optional stated-precision tolerance.

    中文：先使用显式容差，再允许极小的浮点式相对误差以处理计算路径差异。
    """
    delta = abs(a - b)
    if tolerance > 0 and delta <= tolerance:
        return True
    scale = max(Decimal("1"), abs(a), abs(b))
    return _to_floatish(delta) <= 1e-6 * _to_floatish(scale)


def _numeric_compare(
    actual: Decimal,
    expected: Decimal,
    comparator: Optional[str],
    tolerance: Decimal = Decimal("0"),
) -> bool:
    """Evaluate the claim's equality or inequality operator against evidence.

    中文：无比较符默认相等；未知操作符安全地回退为相等性检查。
    """
    if comparator in {None, "eq"}:
        return _numeric_equal(actual, expected, tolerance)
    if comparator == "gt":
        return actual > expected
    if comparator == "lt":
        return actual < expected
    if comparator == "gte":
        return actual >= expected
    if comparator == "lte":
        return actual <= expected
    return _numeric_equal(actual, expected, tolerance)


def _as_chunk_list(chunks: Iterable[Any]) -> list[ChunkResult]:
    """Normalize dict-like test or retrieval payloads into ``ChunkResult`` values.

    中文：边界兼容字典和模型实例，后续核验逻辑只面对一种类型。
    """
    return [
        chunk if isinstance(chunk, ChunkResult) else ChunkResult(**chunk)
        for chunk in chunks
    ]


def _evidence_from_chunk(
    claim_id: str,
    chunk: ChunkResult,
    value: Optional[Decimal] = None,
    unit: Optional[str] = None,
    span: Optional[tuple[int, int]] = None,
) -> EvidenceItem:
    """Create an auditable evidence receipt from one retrieved chunk.

    中文：若已知数字位置则提取邻近片段，否则保留缩短后的整段摘要。
    """
    excerpt = (
        _excerpt_around(chunk.text, span[0], span[1])
        if span is not None
        else _short_text(chunk.text)
    )
    return EvidenceItem(
        claim_id=claim_id,
        source=(chunk.source_url or chunk.chunk_id),
        excerpt=excerpt,
        filing_id=(chunk.chunk_id if "-" in chunk.chunk_id else None),
        score=(chunk.cos_sim),
        value=value,
        unit=unit,
    )


def _is_after_cutoff(chunk: ChunkResult, claim: FinancialClaim) -> bool:
    """True when the chunk was disclosed after the claim's ``as_of`` instant.

    Filing date is checked at day granularity when available; fiscal year is
    only a fallback, because a same-fiscal-year filing published months after
    ``as_of`` is still future information the claim's author could not have had.

    中文：优先用提交日期阻止“事后知识”；没有日期时才以财年作较粗的回退。
    """
    if claim.as_of is None:
        return False
    if chunk.filed_date is not None:
        return chunk.filed_date > claim.as_of.date()
    if chunk.fiscal_year is not None:
        return chunk.fiscal_year > claim.as_of.year
    return False


def _period_mismatch(chunk: ChunkResult, period: Optional[tuple[int, Optional[int]]]) -> bool:
    """Return true only when metadata proves the chunk covers another period.

    中文：未知期间不是不匹配的证据；调用方会记录缺失元数据而不是静默丢弃。
    """
    if period is None:
        return False
    year, _quarter = period
    if chunk.fiscal_year is None:
        # Unknown provenance is not proof of mismatch; the caller records an
        # obligation instead of silently trusting or silently dropping it.
        return False
    return chunk.fiscal_year != year


def _keyword_overlap(a: str, b: str) -> float:
    """Return asymmetric meaningful-token coverage from claim text to evidence text.

    中文：非数值声明只能获得词面支持，不能因词面不同被判为反驳。
    """
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
    """Verify one normalized claim against its already-retrieved candidate chunks.

    中文：该函数不负责检索；它按可验证性、截止时间、可用的财年信息、指标绑定
    和数值比较生成裁决。缺失期间信息不会单独阻止 ``verified``。
    """
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
    obligations: list[str] = []

    expected = claim.value
    metric = claim.metric or _infer_metric(claim.normalized_text)
    period = parse_fiscal_period(claim.normalized_text)
    tolerance = (
        _rounding_tolerance(expected, claim.unit)
        if expected is not None
        else Decimal("0")
    )
    claim_is_monetary = _is_monetary(claim.normalized_text)

    if period is None:
        obligations.append("claim_fiscal_period_not_identified")

    # Fail closed: an unidentified metric used to mean "match any number", which
    # let a claim about repurchases be "verified" by a dividend figure.
    if expected is not None and metric is None:
        return ClaimVerdict(
            claim_id=claim.claim_id,
            verdict="insufficient_evidence",
            reason_code="claim_metric_not_identified",
            missing_obligations=["claim_metric_not_identified"],
        )

    in_scope = 0
    for chunk in chunks:
        if _is_after_cutoff(chunk, claim):
            continue
        if _period_mismatch(chunk, period):
            continue
        if period is not None and chunk.fiscal_year is None:
            obligations.append("chunk_period_metadata_missing")
        in_scope += 1

        if expected is not None:
            mentions = find_metric_mentions(chunk.text)
            for value, unit, raw, start, end in extract_numeric_bindings_with_spans(
                chunk.text
            ):
                if not _comparable_units(claim.unit, unit):
                    continue
                if claim_is_monetary != _is_monetary(raw):
                    # "repurchased 133 million shares" is a share count, not the
                    # dollar amount a "$3.7 billion" claim is talking about.
                    continue
                if _bound_metric(mentions, start, end) != metric:
                    continue
                if _is_change_amount(chunk.text, start):
                    continue
                item = _evidence_from_chunk(
                    claim.claim_id, chunk, value, unit, span=(start, end)
                )
                if _numeric_compare(value, expected, claim.comparator, tolerance):
                    support.append(item)
                else:
                    oppose.append(item)
            continue

        # Non-numeric claim: deterministic keyword overlap can support a claim
        # but can never refute one, so no opposing evidence is produced here.
        if _keyword_overlap(claim.normalized_text, chunk.text.lower()) >= 0.55:
            support.append(_evidence_from_chunk(claim.claim_id, chunk))

    deduped_obligations = list(dict.fromkeys(obligations))

    if not in_scope:
        return ClaimVerdict(
            claim_id=claim.claim_id,
            verdict="insufficient_evidence",
            reason_code="no_evidence_within_period_and_cutoff_scope",
            missing_obligations=deduped_obligations,
        )

    if support and oppose:
        return ClaimVerdict(
            claim_id=claim.claim_id,
            verdict="conflicting",
            reason_code="in_scope_evidence_disagrees_with_itself",
            evidence_for=support,
            evidence_against=oppose,
            missing_obligations=deduped_obligations,
        )

    if support:
        return ClaimVerdict(
            claim_id=claim.claim_id,
            verdict="verified",
            reason_code=(
                "evidence_binds_metric_period_and_value"
                if expected is not None
                else "narrative_support_in_retrieved_chunks"
            ),
            evidence_for=support,
            missing_obligations=deduped_obligations,
        )

    if oppose:
        return ClaimVerdict(
            claim_id=claim.claim_id,
            verdict="refuted",
            reason_code="evidence_binds_metric_and_period_but_value_differs",
            evidence_against=oppose,
            missing_obligations=deduped_obligations,
        )

    return ClaimVerdict(
        claim_id=claim.claim_id,
        verdict="insufficient_evidence",
        reason_code="retrieved_chunks_do_not_bind_claim_metric",
        missing_obligations=deduped_obligations,
    )


def verify_claims(
    claims: list[FinancialClaim],
    retrieve_chunks: Iterable[list[Any]],
) -> tuple[list[ClaimVerdict], list[str]]:
    """Run deterministic policy for each claim.

    ``retrieve_chunks`` is intentionally a sequence of per-claim chunk lists in the
    same order as claims, making this function easy to test with monkeypatches.

    中文：每条声明对应一组已检索证据，保持顺序配对让批量调用和测试都可复现。
    """
    verdicts: list[ClaimVerdict] = []
    obligations: list[str] = []

    for claim, chunk_payload in zip(claims, retrieve_chunks, strict=False):
        chunks = _as_chunk_list(chunk_payload)
        verdict = _verify_with_chunk_set(claim, chunks)
        verdicts.append(verdict)

    return verdicts, obligations
