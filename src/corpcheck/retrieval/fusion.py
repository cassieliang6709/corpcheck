"""Score fusion for hybrid dense + sparse retrieval.

Supports both Min-Max linear score blending and Reciprocal Rank Fusion (RRF).
Reciprocal Rank Fusion evaluates document position rather than raw relevance
magnitude, making it robust against disparate score distributions between vector
embedding cosine similarities and BM25 BM25 text scores.
"""

from __future__ import annotations

from typing import Any


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


def compute_rrf_scores(
    vec_rows: list[dict[str, Any]],
    bm25_rows: list[dict[str, Any]],
    alpha: float = 0.7,
    k_rrf: int = 60,
    normalize: bool = True,
) -> dict[str, float]:
    """Compute Reciprocal Rank Fusion (RRF) scores for candidate chunks.

    RRF formula for document d:
        RRF(d) = alpha * (1 / (k_rrf + rank_v(d))) + (1 - alpha) * (1 / (k_rrf + rank_b(d)))

    Ranks are 1-based indices in the pre-sorted ``vec_rows`` and ``bm25_rows``.
    If normalized is True, min-max scales output scores into [0, 1] so downstream
    additive rerank bonuses retain their relative magnitude.
    """
    # 1-based ranks
    vec_ranks = {row["chunk_id"]: idx + 1 for idx, row in enumerate(vec_rows)}
    bm25_ranks = {row["chunk_id"]: idx + 1 for idx, row in enumerate(bm25_rows)}

    all_cids = list(dict.fromkeys(list(vec_ranks.keys()) + list(bm25_ranks.keys())))
    if not all_cids:
        return {}

    raw_scores: list[float] = []
    for cid in all_cids:
        score = 0.0
        if cid in vec_ranks:
            score += alpha * (1.0 / (k_rrf + vec_ranks[cid]))
        if cid in bm25_ranks:
            score += (1.0 - alpha) * (1.0 / (k_rrf + bm25_ranks[cid]))
        raw_scores.append(score)

    if normalize:
        scaled = minmax_normalize(raw_scores)
        return dict(zip(all_cids, scaled, strict=True))
    return dict(zip(all_cids, raw_scores, strict=True))


def fuse_candidates(
    vec_rows: list[dict[str, Any]],
    bm25_rows: list[dict[str, Any]],
    alpha: float = 0.7,
    strategy: str = "rrf",
    k_rrf: int = 60,
) -> dict[str, float]:
    """Unified entry point for score fusion across vector and BM25 candidate sets.

    Supported strategies:
    - ``"rrf"``: Reciprocal Rank Fusion
    - ``"minmax"``: Min-Max normalisation followed by linear alpha blend
    """
    strat = strategy.lower().strip()
    if strat == "rrf":
        return compute_rrf_scores(vec_rows, bm25_rows, alpha=alpha, k_rrf=k_rrf, normalize=True)
    elif strat == "minmax":
        vec_map = {r["chunk_id"]: r for r in vec_rows}
        bm25_map = {r["chunk_id"]: r for r in bm25_rows}
        all_ids = list({**vec_map, **bm25_map}.keys())

        raw_v = [vec_map[cid]["score_v"] if cid in vec_map else 0.0 for cid in all_ids]
        raw_b = [bm25_map[cid]["score_b"] if cid in bm25_map else 0.0 for cid in all_ids]

        norm_v = minmax_normalize(raw_v) if vec_rows else [0.0] * len(all_ids)
        norm_b = minmax_normalize(raw_b) if bm25_rows else [0.0] * len(all_ids)

        return {
            cid: fuse_scores(norm_v[i], norm_b[i], alpha)
            for i, cid in enumerate(all_ids)
        }
    else:
        raise ValueError(f"Unknown fusion strategy: {strategy!r}. Expected 'rrf' or 'minmax'.")
