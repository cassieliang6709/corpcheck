"""Deterministic claim normalizer for financial statements and comments."""

from __future__ import annotations

import hashlib
import re
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from typing import Optional

from .schema import ClaimEvaluationRequest, FinancialClaim

_SENTENCE_SPLIT = re.compile(r"[\.;!?]\s+|\n+")

_NUMERIC_WITH_UNIT_RE = re.compile(
    r"(?<![A-Za-z0-9])"
    r"(?P<paren>\()?"
    r"\s*(?P<sign>[+-]|−)?\s*"
    r"(?P<currency>[$¥€£])?\s*"
    r"(?P<number>\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?)"
    r"\s*(?P<unit>%|％|percent|pct|basis points|bps|trillion|tr|" +
    "billion|bn|b|million|m|thousand|k|万|千|亿)?"
    r"(?!-)(?![A-Za-z0-9])",
    re.IGNORECASE,
)

_NUMERIC_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")

_UNIT_SCALE = {
    "thousand": Decimal("1000"),
    "k": Decimal("1000"),
    "million": Decimal("1000000"),
    "m": Decimal("1000000"),
    "billion": Decimal("1000000000"),
    "bn": Decimal("1000000000"),
    "b": Decimal("1000000000"),
    "trillion": Decimal("1000000000000"),
    "tr": Decimal("1000000000000"),
    "万": Decimal("10000"),
    "千": Decimal("1000"),
    "亿": Decimal("100000000"),
}

_PERCENT_UNITS = {"%", "％", "percent", "pct"}

_PREDICTION_KEYWORDS = (
    "will", "预计", "expect", "预计", "forecast", "projection", "预计将",
    "下一", "未来", "下个", "下季度", "next quarter", "next year",
    "next fiscal",
)
_CLAIM_KEYWORDS = {
    "reported_numeric": [
        "revenue", "营收", "净利润", "eps", "%", "亿", "million", "billion", "k"
    ],
    "causal": [
        "导致", "because", "因为", "due to", "attributed", "attribution", "驱动"
    ],
    "comparative": [
        "同比", "环比", "比", "vs", "versus", "increase", "decrease", "下降",
        "上升", "greater", "less",
    ],
    "opinion": [
        "likely", "likely", "unlikely", "believe", "应该", "可能", "或许",
        "might", "could", "估计",
    ],
}


def _canonical_id(claim_text: str, company_name: Optional[str] = None) -> str:
    raw = f"{(company_name or '').strip()}::{claim_text.strip().lower()}"
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
    return f"claim_{digest}"


def normalize_text(text: str) -> str:
    cleaned = text.replace("\u201c", '"').replace("\u201d", '"')
    cleaned = cleaned.replace("\u2018", "'").replace("\u2019", "'")
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" \t")
    return cleaned


def split_atomic_claims(text: str) -> list[str]:
    normalized = normalize_text(text)
    pieces = [segment.strip() for segment in _SENTENCE_SPLIT.split(normalized)]
    pieces = [p for p in pieces if p]
    atomic: list[str] = []

    for piece in pieces:
        # also split long coordination phrases that often mix multiple claims
        clauses = [c.strip() for c in re.split(r"\s*,\s*(and|and/or|or)\s+", piece)]
        for clause in clauses:
            if clause.lower() in {"and", "or", "and/or"}:
                continue
            if clause:
                atomic.append(clause)

    deduped: list[str] = []
    seen = set()
    for candidate in atomic:
        lowered = candidate.lower()
        if lowered in seen:
            continue
        deduped.append(candidate)
        seen.add(lowered)

    return deduped


def _is_prediction(claim: str) -> bool:
    lower = claim.lower()
    return any(k in lower for k in _PREDICTION_KEYWORDS)


def _guess_claim_type(claim: str) -> str:
    lower = claim.lower()
    if _is_prediction(claim):
        return "prediction"

    if _NUMERIC_RE.search(claim) and any(
        k in lower for k in [
            "revenue", "profit", "loss", "净利润", "增长", "margin", "eps",
        ]
    ):
        return "reported_numeric"

    for claim_type, keywords in _CLAIM_KEYWORDS.items():
        if any(k in lower for k in keywords):
            return claim_type

    # conservative default; narrative facts are most common and easiest to audit.
    return "narrative_disclosure"


def _guess_checkability(claim: str, claim_type: str, company: Optional[str]) -> str:
    if claim_type == "prediction":
        return "watch_later"
    if claim_type == "opinion":
        return "non_verifiable"
    if (
        claim_type in {
            "reported_numeric", "comparative", "causal", "narrative_disclosure",
            "derived_numeric"
        } and company
    ):
        return "check_now"
    return "non_verifiable"


def _normalize_unit(unit: Optional[str]) -> Optional[str]:
    if not unit:
        return None
    lowered = unit.lower().strip().strip(". ")

    if lowered in _UNIT_SCALE:
        return lowered
    if lowered == "亿":
        return "亿"
    if lowered in {"basis points", "bps"}:
        return "bps"
    if lowered in _PERCENT_UNITS:
        return "%"
    return lowered or None


def _looks_like_year(value: str, claim: str, start: int, end: int) -> bool:
    if len(value) != 4:
        return False
    try:
        numeric_year = int(value)
    except ValueError:
        return False
    if not 1900 <= numeric_year <= 2100:
        return False

    left = claim[max(0, start - 18):start].lower()
    right = claim[end : min(len(claim), end + 18)].lower()

    # Year-like numerics near explicit period markers are usually metadata labels.
    if re.search(
        r"\b(fiscal|fiscal year|year|fy|q[1-4]|\bq\d\b|period|quarter)\b",
        left,
    ):
        return True
    if re.search(
        r"\b(fiscal|fiscal year|year|fy|q[1-4]|\bq\d\b|period|quarter|" +
        "for|in|as of|on)\b",
        right,
    ):
        return True
    # bare "as of 2026", "in 2026" etc.
    if re.search(r"\b(as of|in|for|by|till|through|until)\s+$", left):
        return True
    return False


def _coerce_decimal(value: str) -> Optional[Decimal]:
    try:
        return Decimal(value.replace(",", ""))
    except (InvalidOperation, ValueError):
        return None


def extract_numeric_bindings(text: str) -> list[tuple[Decimal, Optional[str], str]]:
    """Return candidate numeric bindings as (value, unit, raw) in sentence order."""
    if not text:
        return []

    bindings: list[tuple[Decimal, Optional[str], str]] = []
    for match in _NUMERIC_WITH_UNIT_RE.finditer(text):
        number_text = match.group("number")
        unit = _normalize_unit(match.group("unit"))
        if _looks_like_year(number_text, text, match.start(), match.end()):
            continue

        value = _coerce_decimal(number_text)
        if value is None:
            continue

        if match.group("paren") and match.group("sign") in {None, ""}:
            value = -value
        elif (match.group("sign") in {"-", "−"}):
            value = -value

        if value == 0:
            raw = match.group(0).strip()
            bindings.append((value, unit, raw))
            continue

        if unit and unit in _UNIT_SCALE:
            # Keep raw numeric value and scale to help deterministic matching across
            # mixed expressions such as "$11.6 billion" and "$11,600 million".
            scaled = value * _UNIT_SCALE[unit]
            if scaled == value:
                # Defensive for non-scale or malformed unit.
                scaled = value
            value = scaled
            # Store normalized unit to support downstream explainability.
        bindings.append((value, unit, match.group(0).strip()))

    return bindings


def _extract_primary_numeric_binding(
    claim: str,
) -> tuple[Optional[Decimal], Optional[str], Optional[str]]:
    candidates = extract_numeric_bindings(claim)
    if not candidates:
        return None, None, None

    # Prefer explicit units (e.g., $10 million) over bare years/numbers.
    unit_candidates = [c for c in candidates if c[1]]
    chosen = unit_candidates[0] if unit_candidates else candidates[0]
    return chosen[0], chosen[1], chosen[2]


def normalize_claims(req: ClaimEvaluationRequest) -> list[FinancialClaim]:
    as_of = req.as_of or datetime.now(tz=UTC)
    claims = []

    for idx, raw in enumerate(split_atomic_claims(req.source_text), 1):
        normalized = normalize_text(raw)
        claim_type = _guess_claim_type(normalized)
        checkability = _guess_checkability(normalized, claim_type, req.company_name)

        extracted_value, extracted_unit, _ = _extract_primary_numeric_binding(normalized)
        claim_id = _canonical_id(normalized, req.company_name)
        claim = FinancialClaim(
            claim_id=claim_id,
            text=raw,
            normalized_text=normalized,
            claim_type=claim_type,
            company_name=req.company_name,
            as_of=as_of,
            checkability=checkability,
            fiscal_period=None,
            metric=_extract_metric(normalized),
            comparator=_extract_comparator(normalized),
            value=extracted_value,
            unit=extracted_unit,
            qualifiers=[],
        )
        claim.claim_id = f"{claim.claim_id}_{idx:02d}"
        claims.append(claim)

    return claims


def _extract_metric(claim: str) -> Optional[str]:
    lowered = claim.lower()
    if "revenue" in lowered or "营收" in claim:
        return "revenue"
    if "profit" in lowered or "利润" in claim or "net income" in lowered:
        return "profit"
    if "margin" in lowered or "毛利" in claim:
        return "margin"
    if "eps" in lowered or "每股" in claim:
        return "eps"
    if "share" in lowered and "repurchase" in lowered:
        return "share"
    if "cash" in lowered and "flow" in lowered:
        return "cash flow"
    return None


def _extract_comparator(claim: str) -> Optional[str]:
    lowered = claim.lower()
    if " above " in lowered or " > " in lowered or ">" in lowered:
        return "gt"
    if " below " in lowered or " < " in lowered or "<" in lowered:
        return "lt"
    if " no less " in lowered or "at least" in lowered or "至少" in claim:
        return "gte"
    if "no more" in lowered or "at most" in lowered or "最多" in claim:
        return "lte"
    if " to " in lowered or " at " in lowered:
        # keep deterministic signal for absolute statements like "was 100"/"rose to 100"
        return "eq"
    return None
