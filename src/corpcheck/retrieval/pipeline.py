"""The retrieval orchestrator: query understanding → hybrid search → fusion → top-k.

This is the single entry point used by both the HTTP API and the offline IR
evaluation harness, so both measure exactly the same code path.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

import asyncpg

from corpcheck.models import ChunkResult
from corpcheck.retrieval.fusion import fuse_scores, minmax_normalize
from corpcheck.retrieval.query_parse import (
    detect_company_in_query,
    detect_filing_type_hint_in_query,
    detect_filing_type_in_query,
    detect_years_in_query,
    query_prefers_explanatory_chunks,
    query_prefers_quantitative_chunks,
    resolve_company_filter,
    sanitize_bm25_query,
)
from corpcheck.retrieval.rerank import compose_article_title, compute_rerank_bonus
from corpcheck.retrieval.search import (
    bm25_search,
    build_filter_clause,
    embed_query,
    vector_search,
)

logger = logging.getLogger(__name__)


async def retrieve(
    pool: asyncpg.Pool,
    query: str,
    k: int,
    alpha: float,
    sector: Optional[str],
    company: Optional[str],
    filing_type: Optional[str],
    fiscal_year: Optional[int] = None,
) -> list[ChunkResult]:
    """Return the top-``k`` chunks for ``query`` under the given metadata filters."""
    resolved_company = resolve_company_filter(company)
    detected_company = None if resolved_company else detect_company_in_query(query)
    query_vec = embed_query(query)
    bm25_query = sanitize_bm25_query(query, resolved_company or detected_company)
    needs_quant = query_prefers_quantitative_chunks(query)
    needs_explanatory = query_prefers_explanatory_chunks(query)
    overfetch = max(k * 3, 30) if needs_quant else k * 3
    explicit_filing_type = None if filing_type else detect_filing_type_in_query(query)
    hinted_filing_type = None if filing_type else detect_filing_type_hint_in_query(query)
    effective_filing_type = filing_type or explicit_filing_type
    filter_where, filter_params = build_filter_clause(
        sector, resolved_company, effective_filing_type, fiscal_year
    )

    # Detect signals before retrieval so boosts are embedded in SQL ORDER BY,
    # affecting which chunks the DB returns rather than just reordering them after.
    boost_ticker = detected_company
    boost_filing_type = None if effective_filing_type else hinted_filing_type
    boost_years = detect_years_in_query(query)

    vec_rows, bm25_rows = await asyncio.gather(
        vector_search(
            pool,
            query_vec,
            overfetch,
            filter_where,
            filter_params,
            boost_ticker,
            boost_filing_type,
            boost_years,
        ),
        bm25_search(
            pool,
            bm25_query,
            overfetch,
            filter_where,
            filter_params,
            boost_ticker,
            boost_filing_type,
            boost_years,
        ),
    )

    vec_map: dict[str, dict] = {r["chunk_id"]: r for r in vec_rows}
    bm25_map: dict[str, dict] = {r["chunk_id"]: r for r in bm25_rows}
    all_ids = list({**vec_map, **bm25_map}.keys())

    raw_v = [vec_map[cid]["score_v"] if cid in vec_map else 0.0 for cid in all_ids]
    raw_b = [bm25_map[cid]["score_b"] if cid in bm25_map else 0.0 for cid in all_ids]
    raw_d = [
        float((vec_map.get(cid) or bm25_map.get(cid) or {}).get("data_signal_score") or 0.0)
        for cid in all_ids
    ]

    norm_v = minmax_normalize(raw_v) if vec_rows else [0.0] * len(all_ids)
    norm_b = minmax_normalize(raw_b) if bm25_rows else [0.0] * len(all_ids)
    norm_d = minmax_normalize(raw_d) if needs_quant and raw_d else [0.0] * len(all_ids)

    scored = []
    for i, cid in enumerate(all_ids):
        row = vec_map.get(cid) or bm25_map.get(cid)
        source_type = row.get("source_type") or "sec"
        content_kind = row.get("content_kind")
        final_score = (
            fuse_scores(norm_v[i], norm_b[i], alpha)
            + (0.15 * norm_d[i] if needs_quant else 0.0)
            + compute_rerank_bonus(
                needs_quant=needs_quant,
                needs_explanatory=needs_explanatory,
                content_kind=content_kind,
                source_type=source_type,
            )
        )
        scored.append((final_score, cid))
    scored.sort(reverse=True)

    results: list[ChunkResult] = []
    for score, cid in scored[:k]:
        row = vec_map.get(cid) or bm25_map[cid]
        content_kind = row.get("content_kind")
        source_type = row.get("source_type") or "sec"

        ft = row.get("filing_type") or ""
        results.append(
            ChunkResult(
                chunk_id=cid,
                text=row["text"],
                score=round(score, 6),
                company=row["company"],
                sector=row["sector"],
                filing_type=ft,
                filed_date=row.get("filed_date"),
                source_url=row.get("source_url"),
                article_title=compose_article_title(row) or None,
                page_num=None,
                source_type=source_type,
                content_kind=content_kind,
                chunk_strategy=row.get("chunk_strategy"),
                display_title=row.get("display_title"),
            )
        )
    results.sort(key=lambda chunk: chunk.score, reverse=True)
    return results
