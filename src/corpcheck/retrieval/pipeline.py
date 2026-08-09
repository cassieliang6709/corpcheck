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
from corpcheck.retrieval.fusion import fuse_candidates, minmax_normalize
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
from corpcheck.retrieval.revision import (
    filter_superseded_rows,
    load_superseded_filings,
    log_dropped,
)
from corpcheck.retrieval.search import (
    bm25_search,
    build_filter_clause,
    embed_query,
    table_child_vector_search,
    vector_search,
)
from corpcheck.settings import (
    COMPANY_SCOPE_ENABLED,
    FUSION_STRATEGY,
    REVISION_FILTER_ENABLED,
    RRF_K,
    TABLE_CHILD_RETRIEVAL_ENABLED,
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
    fusion_strategy: Optional[str] = None,
) -> list[ChunkResult]:
    """Return the top-``k`` chunks for ``query`` under the given metadata filters."""
    strat = fusion_strategy or FUSION_STRATEGY
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
    # Detect signals before retrieval so boosts are embedded in SQL ORDER BY,
    # affecting which chunks the DB returns rather than just reordering them after.
    boost_ticker = detected_company
    boost_filing_type = None if effective_filing_type else hinted_filing_type
    boost_years = detect_years_in_query(query)

    # Scoping to the detected issuer is a filter, not a preference; see
    # COMPANY_SCOPE_ENABLED. The unscoped pool is still reachable as a fallback.
    scope_company = detected_company if COMPANY_SCOPE_ENABLED else None

    async def _candidates(company_filter: Optional[str]):
        filter_where, filter_params = build_filter_clause(
            sector, resolved_company or company_filter, effective_filing_type, fiscal_year
        )
        searches = [
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
        ]
        if TABLE_CHILD_RETRIEVAL_ENABLED and needs_quant:
            searches.append(
                table_child_vector_search(
                    pool,
                    query_vec,
                    filter_where,
                    filter_params,
                    boost_ticker,
                    boost_filing_type,
                    boost_years,
                )
            )
        return await asyncio.gather(*searches)

    candidate_arms = await _candidates(scope_company)
    vec_rows, bm25_rows = candidate_arms[:2]
    child_rows = candidate_arms[2] if len(candidate_arms) == 3 else []
    if scope_company and not vec_rows and not bm25_rows and not child_rows:
        # The detection was wrong, or the issuer has no chunks. Degrade to the
        # whole corpus rather than returning nothing.
        logger.info("Company scope %s yielded no candidates; retrying unscoped", scope_company)
        candidate_arms = await _candidates(None)
        vec_rows, bm25_rows = candidate_arms[:2]
        child_rows = candidate_arms[2] if len(candidate_arms) == 3 else []

    if child_rows:
        dense_by_id = {row["chunk_id"]: row for row in vec_rows}
        for row in child_rows:
            current = dense_by_id.get(row["chunk_id"])
            if current is None or float(row["score_v"]) > float(current["score_v"]):
                dense_by_id[row["chunk_id"]] = row
        vec_rows = sorted(
            dense_by_id.values(), key=lambda row: float(row["score_v"]), reverse=True
        )[:overfetch]

    # Supersession is resolved here — after candidate generation, before fusion.
    # Filtering the candidate pool (rather than the final top-k) lets surviving
    # chunks move up and fill the vacated slots, so the caller still gets k results.
    if REVISION_FILTER_ENABLED:
        superseded = await load_superseded_filings(pool, vec_rows + bm25_rows)
        vec_rows, dropped_vec = filter_superseded_rows(vec_rows, superseded)
        bm25_rows, dropped_bm25 = filter_superseded_rows(bm25_rows, superseded)
        log_dropped(dropped_vec + dropped_bm25)

    vec_map: dict[str, dict] = {r["chunk_id"]: r for r in vec_rows}
    bm25_map: dict[str, dict] = {r["chunk_id"]: r for r in bm25_rows}
    all_ids = list({**vec_map, **bm25_map}.keys())

    # Fused base scores using specified fusion strategy ("rrf" or "minmax")
    fused_scores = fuse_candidates(
        vec_rows, bm25_rows, alpha=alpha, strategy=strat, k_rrf=RRF_K
    )

    raw_d = [
        float((vec_map.get(cid) or bm25_map.get(cid) or {}).get("data_signal_score") or 0.0)
        for cid in all_ids
    ]
    norm_d = minmax_normalize(raw_d) if needs_quant and raw_d else [0.0] * len(all_ids)

    scored = []
    for i, cid in enumerate(all_ids):
        row = vec_map.get(cid) or bm25_map.get(cid)
        source_type = row.get("source_type") or "sec"
        content_kind = row.get("content_kind")
        fused_score = fused_scores.get(cid, 0.0)
        final_score = (
            fused_score
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
        # Read from vec_map specifically: cos_sim only exists on the dense arm's
        # rows, and `row` may have come from bm25_map.
        dense_row = vec_map.get(cid)
        cos_sim = dense_row.get("cos_sim") if dense_row else None

        results.append(
            ChunkResult(
                chunk_id=cid,
                text=row["text"],
                score=round(score, 6),
                cos_sim=round(float(cos_sim), 6) if cos_sim is not None else None,
                company=row["company"],
                sector=row["sector"],
                filing_type=ft,
                fiscal_year=row.get("fiscal_year"),
                period_label=row.get("period_label"),
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
