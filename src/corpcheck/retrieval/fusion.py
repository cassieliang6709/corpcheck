"""Score fusion for hybrid dense + sparse retrieval.

Currently min-max normalisation followed by a linear ``alpha`` blend. Reciprocal
Rank Fusion lands here in a later sprint, which is why the strategy is isolated
in its own module.
"""

from __future__ import annotations


def minmax_normalize(scores: list[float]) -> list[float]:
    """Scale scores into [0, 1]. A degenerate (all-equal) list maps to all 1.0."""
    if not scores:
        return []
    mn = min(scores)
    mx = max(scores)
    if mx == mn:
        return [1.0] * len(scores)
    return [(s - mn) / (mx - mn) for s in scores]


def fuse_scores(score_v: float, score_b: float, alpha: float) -> float:
    """Linear blend of a dense and a sparse score, both assumed pre-normalised."""
    return alpha * score_v + (1.0 - alpha) * score_b
