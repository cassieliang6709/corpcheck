"""SQL-level candidate generation: dense (pgvector) and sparse (ts_rank) search.

Both retrievers read from the ``v_retrieval_chunks`` view and share the same
metadata filter and boost machinery, so their candidate sets are directly
comparable before fusion.
"""

from __future__ import annotations

import logging
from typing import Optional, Union

import asyncpg
from sentence_transformers import SentenceTransformer

from corpcheck.settings import (
    COMPANY_BOOST,
    EMBEDDING_MODEL,
    FILING_TYPE_BOOST,
    FISCAL_YEAR_BOOST,
)

logger = logging.getLogger(__name__)

_model: Optional[SentenceTransformer] = None

# Columns every candidate row must expose so downstream fusion, revision
# filtering, and result assembly can all work off a uniform shape.
_SELECT_COLUMNS = """
            chunk_id,
            ticker                                                AS company,
            sector,
            filing_type,
            event_date                                            AS filed_date,
            source_url,
            content                                               AS text,
            company_name,
            fiscal_year,
            period_label,
            section_name,
            source_type,
            content_kind,
            chunk_strategy,
            display_title,
            data_signal_score,
            is_quantitative"""


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        logger.info("Loading embedding model %s", EMBEDDING_MODEL)
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def embed_query(query: str) -> list[float]:
    return get_model().encode(query, normalize_embeddings=True).tolist()


def build_boost_expression(
    boost_ticker: Optional[str],
    boost_filing_type: Optional[str],
    boost_years: list[str],
    start_idx: int,
) -> tuple[str, list, int]:
    """Return a SQL multiplicative boost expression, its positional param values, and next index.

    Boost values are inlined as float literals (server-controlled config); only the
    comparison values (ticker, filing_type, years) are parameterised for safety.
    """
    parts: list[str] = []
    values: list = []
    idx = start_idx
    if boost_ticker:
        parts.append(f"CASE WHEN ticker = ${idx} THEN {COMPANY_BOOST}::float ELSE 1.0 END")
        values.append(boost_ticker)
        idx += 1
    if boost_filing_type:
        parts.append(f"CASE WHEN filing_type = ${idx} THEN {FILING_TYPE_BOOST}::float ELSE 1.0 END")
        values.append(boost_filing_type)
        idx += 1
    if boost_years:
        parts.append(
            f"CASE WHEN fiscal_year::text = ANY(${idx}::text[]) THEN {FISCAL_YEAR_BOOST}::float ELSE 1.0 END"
        )
        values.append(boost_years)
        idx += 1
    expr = " * ".join(parts) if parts else "1.0"
    return expr, values, idx


def build_filter_clause(
    sector: Optional[str],
    company: Optional[str],
    filing_type: Optional[str],
    fiscal_year: Optional[int],
) -> tuple[str, dict]:
    """Return a SQL WHERE fragment using :name placeholders, and a params dict."""
    conditions: list[str] = []
    params: dict[str, Union[str, int]] = {}
    if sector is not None:
        conditions.append("sector = :sector")
        params["sector"] = sector
    if company is not None:
        conditions.append("(ticker = :company OR lower(company_name) = lower(:company))")
        params["company"] = company
    if filing_type is not None:
        conditions.append("filing_type = :filing_type")
        params["filing_type"] = filing_type
    if fiscal_year is not None:
        conditions.append("fiscal_year = :fiscal_year")
        params["fiscal_year"] = fiscal_year
    return " AND ".join(conditions), params


def apply_filter(
    base_sql: str,
    where: str,
    params: dict,
    next_idx: int,
) -> tuple[str, list, int]:
    """Replace :name placeholders with $N positional params for asyncpg."""
    positional_where = where
    values: list = []
    idx = next_idx
    for key, val in params.items():
        positional_where = positional_where.replace(f":{key}", f"${idx}")
        values.append(val)
        idx += 1
    replacement = f"AND {positional_where}" if positional_where else ""
    return base_sql.replace("__FILTER__", replacement), values, idx


async def vector_search(
    pool: asyncpg.Pool,
    query_vec: list[float],
    limit: int,
    filter_where: str,
    filter_params: dict,
    boost_ticker: Optional[str] = None,
    boost_filing_type: Optional[str] = None,
    boost_years: Optional[list[str]] = None,
) -> list[dict]:
    """Dense retrieval: cosine similarity against the chunk embeddings."""
    boost_expr, boost_vals, _ = build_boost_expression(
        boost_ticker, boost_filing_type, boost_years or [], start_idx=3 + len(filter_params)
    )
    # cos_sim is the same similarity without the metadata boost applied. The
    # boosted score_v decides *which* rows come back and in what order; cos_sim
    # is the only column on the row that still means something on an absolute
    # scale, so it is what the abstain gate reads. Boosting can multiply a score
    # by up to COMPANY_BOOST * FILING_TYPE_BOOST * FISCAL_YEAR_BOOST, which
    # would make any absolute threshold on score_v meaningless.
    sql = f"""
        SELECT
{_SELECT_COLUMNS},
            (1 - (embedding <=> $1::vector))                      AS cos_sim,
            (1 - (embedding <=> $1::vector)) * {boost_expr}       AS score_v
        FROM v_retrieval_chunks
        WHERE embedding IS NOT NULL
        __FILTER__
        ORDER BY score_v DESC
        LIMIT $2
    """
    sql, filter_vals, _ = apply_filter(sql, filter_where, filter_params, next_idx=3)
    async with pool.acquire() as conn:
        rows = await conn.fetch(sql, query_vec, limit, *filter_vals, *boost_vals)
    return [dict(r) for r in rows]


async def bm25_search(
    pool: asyncpg.Pool,
    query: str,
    limit: int,
    filter_where: str,
    filter_params: dict,
    boost_ticker: Optional[str] = None,
    boost_filing_type: Optional[str] = None,
    boost_years: Optional[list[str]] = None,
) -> list[dict]:
    """Sparse retrieval: Postgres full-text ``ts_rank`` over the chunk tsvector."""
    boost_expr, boost_vals, _ = build_boost_expression(
        boost_ticker, boost_filing_type, boost_years or [], start_idx=3 + len(filter_params)
    )
    # bm25_raw mirrors cos_sim on the dense side: the unboosted score, kept for
    # observability and eval. It is deliberately *not* used as an absolute
    # abstain threshold — ts_rank varies with query term count and document
    # length, so a one-word query scores low whether or not it matched well.
    sql = f"""
        SELECT
{_SELECT_COLUMNS},
            ts_rank(content_tsv, plainto_tsquery('english', $1)) AS bm25_raw,
            ts_rank(content_tsv, plainto_tsquery('english', $1)) * {boost_expr} AS score_b
        FROM v_retrieval_chunks
        WHERE content_tsv @@ plainto_tsquery('english', $1)
        __FILTER__
        ORDER BY score_b DESC
        LIMIT $2
    """
    sql, filter_vals, _ = apply_filter(sql, filter_where, filter_params, next_idx=3)
    async with pool.acquire() as conn:
        rows = await conn.fetch(sql, query, limit, *filter_vals, *boost_vals)
    return [dict(r) for r in rows]
