"""Typed data structures for the claim verification overlay.

中文：这些 Pydantic 模型同时是 API 契约，字段描述和类文档会进入 JSON Schema，修改前需评估兼容性。
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from typing import Literal, Optional

from pydantic import BaseModel, Field

ClaimType = Literal[
    "reported_numeric",
    "derived_numeric",
    "narrative_disclosure",
    "comparative",
    "causal",
    "prediction",
    "opinion",
]

Checkability = Literal["check_now", "watch_later", "non_verifiable"]
Verdict = Literal[
    "verified",
    "refuted",
    "conflicting",
    "insufficient_evidence",
    "not_yet_decidable",
    "non_verifiable",
]


class FinancialClaim(BaseModel):
    """Normalized financial claim emitted by the extraction layer."""

    claim_id: str = Field(..., description="Stable deterministic claim identifier")
    text: str = Field(..., min_length=1, description="Original claim text")
    normalized_text: str = Field(..., min_length=1)
    claim_type: ClaimType
    company_name: Optional[str] = None
    cik: Optional[str] = None
    fiscal_period: Optional[str] = None
    as_of: datetime = Field(default_factory=lambda: datetime.now(tz=UTC))
    metric: Optional[str] = None
    comparator: Optional[str] = None
    value: Optional[Decimal] = None
    unit: Optional[str] = None
    qualifiers: list[str] = Field(default_factory=list)
    checkability: Checkability
    observation_window_start: Optional[datetime] = None
    observation_window_end: Optional[datetime] = None


class ClaimEvaluationRequest(BaseModel):
    """API input for one or more claim checks."""

    source_text: str = Field(..., min_length=1)
    company_name: Optional[str] = None
    as_of: Optional[datetime] = None


class EvidenceItem(BaseModel):
    """Evidence that can support/refute one claim."""

    claim_id: str
    source: str
    excerpt: str
    filing_id: Optional[str] = None
    score: Optional[float] = None
    value: Optional[Decimal] = None
    unit: Optional[str] = None


class ClaimVerdict(BaseModel):
    """Decision for one claim."""

    claim_id: str
    verdict: Verdict
    reason_code: str
    evidence_for: list[EvidenceItem] = Field(default_factory=list)
    evidence_against: list[EvidenceItem] = Field(default_factory=list)
    missing_obligations: list[str] = Field(default_factory=list)
    as_of: datetime = Field(default_factory=lambda: datetime.now(tz=UTC))


class ClaimEvaluationResponse(BaseModel):
    """API response for batch claim verification."""

    source_text: str
    total_claims: int
    claims: list[FinancialClaim]
    verdicts: list[ClaimVerdict] = Field(default_factory=list)
