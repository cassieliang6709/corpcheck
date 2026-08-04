"""Double-threshold abstain gating to short-circuit low-confidence retrieval.

In audit-grade financial QA, refusing to answer ("abstaining") is strictly
preferred over serving a confident hallucination derived from low-relevance context.

The gating rule evaluates two score bounds on candidate results:
1. Top-1 score threshold (``ABSTAIN_TOP1_MIN``): ensures at least one highly relevant
   anchor document exists.
2. Mean top-3 score threshold (``ABSTAIN_MEAN_TOP3_MIN``): ensures consistent support
   across multiple retrieved passages.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from corpcheck.settings import ABSTAIN_MEAN_TOP3_MIN, ABSTAIN_TOP1_MIN

if TYPE_CHECKING:
    from corpcheck.retrieval.pipeline import ChunkResult


def should_abstain(
    results: list[ChunkResult],
    top1_min: float = ABSTAIN_TOP1_MIN,
    mean_top3_min: float = ABSTAIN_MEAN_TOP3_MIN,
) -> tuple[bool, str]:
    """Evaluate whether candidate retrieval results pass confidence thresholds.

    Returns:
        tuple[bool, str]: (should_abstain, reason_message)
    """
    if not results:
        return True, "No candidate chunks were retrieved."

    top1_score = results[0].score
    if top1_score < top1_min:
        return (
            True,
            f"Top-1 score {top1_score:.4f} is below minimum threshold {top1_min:.4f}.",
        )

    top3 = [r.score for r in results[:3]]
    mean_top3 = sum(top3) / len(top3)
    if mean_top3 < mean_top3_min:
        return (
            True,
            f"Mean top-3 score {mean_top3:.4f} is below minimum threshold {mean_top3_min:.4f}.",
        )

    return False, ""
