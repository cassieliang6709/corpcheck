"""Claim verification overlay types and deterministic helpers.

中文：声明核验是检索之上的确定性规则层，负责把原文拆成可检查声明并产出可审计裁决。
"""

from corpcheck.claims.normalizer import (
    extract_numeric_bindings,
    normalize_claims,
    normalize_text,
    split_atomic_claims,
)
from corpcheck.claims.schema import (
    ClaimEvaluationRequest,
    ClaimEvaluationResponse,
    ClaimVerdict,
    EvidenceItem,
    FinancialClaim,
)

__all__ = [
    "normalize_claims",
    "normalize_text",
    "split_atomic_claims",
    "extract_numeric_bindings",
    "ClaimEvaluationRequest",
    "ClaimEvaluationResponse",
    "ClaimVerdict",
    "EvidenceItem",
    "FinancialClaim",
]
