"""SQL-level candidate generation: dense (pgvector) and sparse (ts_rank) search.

Both retrievers read from the ``v_retrieval_chunks`` view and share the same
metadata filter and boost machinery, so their candidate sets are directly
comparable before fusion.

中文：本模块只生成数据库候选及其原始/加权分数；融合、修订过滤和响应模型转换由上层管线负责。
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
    """Return the process-cached embedding model, loading it on first use.

    中文：模型对象昂贵且可复用，因此进程内只创建一次；启动路径可主动调用以尽早暴露配置错误。
    """
    global _model
    if _model is None:
        logger.info("Loading embedding model %s", EMBEDDING_MODEL)
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def embed_query(query: str) -> list[float]:
    """Embed one query with L2 normalization for cosine-distance search.

    中文：归一化使 pgvector 的余弦距离与跨查询的相似度阈值具有一致含义。
    """
    return get_model().encode(query, normalize_embeddings=True).tolist()


def build_boost_expression(
    boost_ticker: Optional[str],
    boost_filing_type: Optional[str],
    boost_years: list[str],
    start_idx: int,
) -> tuple[str, list, int]:
    """Build a SQL boost expression, positional parameter values, and next index.

    Boost values are inlined as float literals (server-controlled config); only the
    comparison values (ticker, filing_type, years) are parameterised for safety.

    中文：配置中的 boost 数值可信并内联；用户相关的公司、文件和年份始终作为 SQL 参数传入。
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
            "CASE WHEN fiscal_year::text = "
            f"ANY(${idx}::text[]) THEN {FISCAL_YEAR_BOOST}::float ELSE 1.0 END"
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
    """Build an optional SQL WHERE fragment with named intermediate placeholders.

    中文：先保留可读的 ``:name`` 占位符，再由 ``apply_filter`` 转换为 asyncpg 的位置参数。
    """
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
    """Replace named placeholders with asyncpg ``$N`` parameters.

    中文：返回替换后的 SQL、按出现顺序的值和下一个参数索引；不拼接用户值到 SQL 文本。
    """
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
    """Run dense cosine retrieval and retain both raw and metadata-boosted scores.

    中文：``score_v`` 决定候选排序，``cos_sim`` 保持未加权，供拒答门控进行跨查询比较。
    """
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


async def table_child_vector_search(
    pool: asyncpg.Pool,
    query_vec: list[float],
    filter_where: str,
    filter_params: dict,
    boost_ticker: Optional[str] = None,
    boost_filing_type: Optional[str] = None,
    boost_years: Optional[list[str]] = None,
) -> list[dict]:
    """Retrieve SEC parent chunks through their best matching table-row child.

    The experimental table is intentionally optional. A production database
    without it keeps serving the standard dense and sparse arms.

    中文：表格子行用于提高数值问题的召回；实验表缺失时安全返回空候选，不中断主检索。
    """
    limit = 30
    boost_expr, boost_vals, _ = build_boost_expression(
        boost_ticker, boost_filing_type, boost_years or [], start_idx=3 + len(filter_params)
    )
    sql = f"""
        WITH child_scores AS (
            SELECT
                parent.chunk_id,
                parent.ticker                                      AS company,
                parent.sector,
                parent.filing_type,
                parent.event_date                                  AS filed_date,
                parent.source_url,
                parent.content                                     AS text,
                parent.company_name,
                parent.fiscal_year,
                parent.period_label,
                parent.section_name,
                parent.source_type,
                parent.content_kind,
                parent.chunk_strategy,
                parent.display_title,
                parent.data_signal_score,
                parent.is_quantitative,
                child.child_index,
                (1 - (child.embedding <=> $1::vector))             AS cos_sim,
                (1 - (child.embedding <=> $1::vector)) * {boost_expr} AS score_v
            FROM eval_table_child_chunks AS child
            JOIN v_retrieval_chunks AS parent
              ON parent.chunk_id = child.parent_chunk_id::text
             AND parent.source_type = 'sec'
            WHERE child.embedding IS NOT NULL
            __FILTER__
            ORDER BY score_v DESC
            LIMIT $2
        ), ranked AS (
            SELECT
                *,
                ROW_NUMBER() OVER (
                    PARTITION BY chunk_id
                    ORDER BY score_v DESC, child_index
                ) AS parent_rank
            FROM child_scores
        )
        SELECT *
        FROM ranked
        WHERE parent_rank = 1
        ORDER BY score_v DESC
        LIMIT $2
    """
    sql, filter_vals, _ = apply_filter(sql, filter_where, filter_params, next_idx=3)
    try:
        async with pool.acquire() as conn:
            rows = await conn.fetch(sql, query_vec, limit, *filter_vals, *boost_vals)
    except asyncpg.UndefinedTableError:
        logger.warning(
            "Table-child retrieval is enabled but eval_table_child_chunks is missing; "
            "continuing without the experimental arm"
        )
        return []
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
    """Run sparse Postgres full-text retrieval with the same metadata boosts.

    中文：``score_b`` 用于候选融合；原始 ``bm25_raw`` 仅用于可观察性，不能用作绝对置信度。
    """
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
