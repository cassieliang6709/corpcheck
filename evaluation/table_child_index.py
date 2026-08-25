#!/usr/bin/env python3
"""Build a resumable, evaluation-only PostgreSQL index of table-row children.

中文：建立可续跑的评测专用表格子行索引，并将进度与生产检索隔离；数据库操作
只应针对显式的评测表，连接或嵌入条件不满足时应停止。
"""

from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence

import numpy as np

from corpcheck.ingestion.config import DATABASE_URL, EMBEDDING_BATCH_SIZE, EMBEDDING_DIM
from corpcheck.ingestion.processors.embedder import Embedder
from evaluation.table_child_experiment import parse_table_children

logger = logging.getLogger(__name__)

TABLE_NAME = "eval_table_child_chunks"
TRACKER_TABLE_NAME = "eval_table_child_indexed_parents"

SCHEMA_STATEMENTS = (
    f"""
    CREATE TABLE IF NOT EXISTS {TRACKER_TABLE_NAME} (
        parent_chunk_id BIGINT PRIMARY KEY REFERENCES chunks(id) ON DELETE CASCADE,
        child_count INTEGER NOT NULL,
        indexed_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
    """,
    f"""
    CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
        parent_chunk_id BIGINT NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
        child_index INTEGER NOT NULL,
        content TEXT NOT NULL,
        embedding vector({EMBEDDING_DIM}),
        content_tsv TSVECTOR,
        UNIQUE (parent_chunk_id, child_index)
    )
    """,
    f"""
    CREATE INDEX IF NOT EXISTS idx_eval_table_child_embedding_hnsw
    ON {TABLE_NAME} USING hnsw (embedding vector_cosine_ops)
    """,
    f"""
    CREATE INDEX IF NOT EXISTS idx_eval_table_child_content_tsv
    ON {TABLE_NAME} USING GIN (content_tsv)
    """,
    f"""
    CREATE INDEX IF NOT EXISTS idx_eval_table_child_parent
    ON {TABLE_NAME} (parent_chunk_id)
    """,
)

UPSERT_SQL = f"""
    INSERT INTO {TABLE_NAME} (
        parent_chunk_id, child_index, content, embedding, content_tsv
    ) VALUES (
        %(parent_chunk_id)s,
        %(child_index)s,
        %(content)s,
        %(embedding)s,
        to_tsvector('english', %(content)s)
    )
    ON CONFLICT (parent_chunk_id, child_index) DO UPDATE SET
        content = EXCLUDED.content,
        embedding = EXCLUDED.embedding,
        content_tsv = EXCLUDED.content_tsv
"""

TRACKER_UPSERT_SQL = f"""
    INSERT INTO {TRACKER_TABLE_NAME} (parent_chunk_id, child_count, indexed_at)
    VALUES (%(parent_chunk_id)s, %(child_count)s, NOW())
    ON CONFLICT (parent_chunk_id) DO UPDATE SET
        child_count = EXCLUDED.child_count,
        indexed_at = EXCLUDED.indexed_at
"""


@dataclass(frozen=True)
class IndexStats:
    """Progress counters for one index run."""

    parents_scanned: int = 0
    parents_with_children: int = 0
    parents_without_children: int = 0
    children_upserted: int = 0


BatchWriter = Callable[..., Any]


def ensure_schema(conn: Any) -> None:
    """Create the evaluation table and its search indexes."""
    with conn.cursor() as cursor:
        for statement in SCHEMA_STATEMENTS:
            cursor.execute(statement)
    conn.commit()


def _scope_clause(
    tickers: Optional[Sequence[str]],
    years: Optional[Sequence[int]],
    *,
    alias: str,
) -> tuple[str, list[Any]]:
    clauses: list[str] = []
    params: list[Any] = []
    if tickers:
        clauses.append(f"{alias}.ticker = ANY(%s)")
        params.append(list(tickers))
    if years:
        clauses.append(f"{alias}.fiscal_year = ANY(%s)")
        params.append(list(years))
    return (" AND ".join(clauses), params)


def build_parent_query(
    tickers: Optional[Sequence[str]],
    years: Optional[Sequence[int]],
    *,
    after_id: int,
    fetch_size: int,
) -> tuple[str, tuple[Any, ...]]:
    """Return a keyset query for table parents not yet represented in the child index."""
    scope_sql, params = _scope_clause(tickers, years, alias="c")
    scope_filter = f"\n          AND {scope_sql}" if scope_sql else ""
    sql = f"""
        SELECT c.id, c.content
        FROM chunks c
        WHERE c.content_kind = 'table'
          AND c.id > %s
          AND NOT EXISTS (
              SELECT 1
              FROM {TRACKER_TABLE_NAME} indexed
              WHERE indexed.parent_chunk_id = c.id
          ){scope_filter}
        ORDER BY c.id
        LIMIT %s
    """
    return sql, (after_id, *params, fetch_size)


def rebuild_scope(
    conn: Any,
    tickers: Optional[Sequence[str]],
    years: Optional[Sequence[int]],
    *,
    limit: Optional[int] = None,
) -> int:
    """Delete child and completion rows only for the requested parent scope."""
    scope_sql, params = _scope_clause(tickers, years, alias="c")
    if limit is not None:
        scope_filter = f"AND {scope_sql}" if scope_sql else ""
        selected_sql = f"""
            SELECT c.id
            FROM chunks c
            WHERE c.content_kind = 'table'
              {scope_filter}
            ORDER BY c.id
            LIMIT %s
        """
        child_sql = f"""
            DELETE FROM {TABLE_NAME} child
            USING (
                {selected_sql}
            ) selected
            WHERE child.parent_chunk_id = selected.id
        """
        params.append(limit)
        tracker_sql = f"""
            DELETE FROM {TRACKER_TABLE_NAME} indexed
            USING (
                {selected_sql}
            ) selected
            WHERE indexed.parent_chunk_id = selected.id
        """
    elif scope_sql:
        child_sql = f"""
            DELETE FROM {TABLE_NAME} child
            USING chunks c
            WHERE child.parent_chunk_id = c.id
              AND c.content_kind = 'table'
              AND {scope_sql}
        """
        tracker_sql = f"""
            DELETE FROM {TRACKER_TABLE_NAME} indexed
            USING chunks c
            WHERE indexed.parent_chunk_id = c.id
              AND c.content_kind = 'table'
              AND {scope_sql}
        """
    else:
        child_sql = f"DELETE FROM {TABLE_NAME}"
        tracker_sql = f"DELETE FROM {TRACKER_TABLE_NAME}"
    with conn.cursor() as cursor:
        cursor.execute(child_sql, tuple(params))
        deleted = max(int(cursor.rowcount), 0)
        cursor.execute(tracker_sql, tuple(params))
        deleted += max(int(cursor.rowcount), 0)
    conn.commit()
    return deleted


def prepare_child_rows(
    parents: Sequence[tuple[int, str]],
    embedder: Any,
) -> tuple[list[dict[str, Any]], int, int]:
    """Parse and embed one parent batch, returning rows and parent-level counts."""
    parsed: list[tuple[int, int, str]] = []
    parents_with_children = 0
    for parent_id, content in parents:
        children = parse_table_children(content, str(parent_id))
        if children:
            parents_with_children += 1
        parsed.extend((parent_id, child.child_index, child.text) for child in children)

    if not parsed:
        return [], 0, len(parents)

    embeddings = embedder.encode([content for _, _, content in parsed])
    if len(embeddings) != len(parsed):
        raise ValueError("Embedder returned a different number of vectors than child texts")

    rows = [
        {
            "parent_chunk_id": parent_id,
            "child_index": child_index,
            "content": content,
            "embedding": np.asarray(embeddings[index], dtype=np.float32),
        }
        for index, (parent_id, child_index, content) in enumerate(parsed)
    ]
    return rows, parents_with_children, len(parents) - parents_with_children


def prepare_tracker_rows(
    parents: Sequence[tuple[int, str]],
    child_rows: Sequence[dict[str, Any]],
) -> list[dict[str, int]]:
    """Record every scanned parent, including parents that produced zero children."""
    child_counts: dict[int, int] = {}
    for row in child_rows:
        parent_id = int(row["parent_chunk_id"])
        child_counts[parent_id] = child_counts.get(parent_id, 0) + 1
    return [
        {"parent_chunk_id": parent_id, "child_count": child_counts.get(parent_id, 0)}
        for parent_id, _ in parents
    ]


def build_index(
    conn: Any,
    embedder: Any,
    *,
    tickers: Optional[Sequence[str]] = None,
    years: Optional[Sequence[int]] = None,
    batch_size: int = EMBEDDING_BATCH_SIZE,
    limit: Optional[int] = None,
    rebuild: bool = False,
    all_scope: bool = False,
    batch_writer: Optional[BatchWriter] = None,
) -> IndexStats:
    """Build or resume the child index for an explicitly selected scope."""
    if not all_scope and not tickers and not years:
        raise ValueError("A ticker/year scope is required; use the CLI --all flag intentionally")
    if all_scope and (tickers or years):
        raise ValueError("all_scope cannot be combined with ticker/year filters")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")

    if batch_writer is None:
        from psycopg2.extras import execute_batch

        batch_writer = execute_batch

    ensure_schema(conn)
    if rebuild:
        deleted = rebuild_scope(conn, tickers, years, limit=limit)
        logger.info("Removed %d existing child rows in the selected scope", deleted)

    scanned = with_children = without_children = upserted = 0
    after_id = 0
    while limit is None or scanned < limit:
        fetch_size = batch_size if limit is None else min(batch_size, limit - scanned)
        sql, params = build_parent_query(
            tickers,
            years,
            after_id=after_id,
            fetch_size=fetch_size,
        )
        with conn.cursor() as cursor:
            cursor.execute(sql, params)
            parents = list(cursor.fetchall())
        if not parents:
            break

        rows, batch_with, batch_without = prepare_child_rows(parents, embedder)
        tracker_rows = prepare_tracker_rows(parents, rows)
        if rows:
            with conn.cursor() as cursor:
                batch_writer(cursor, UPSERT_SQL, rows, page_size=batch_size)
        with conn.cursor() as cursor:
            batch_writer(cursor, TRACKER_UPSERT_SQL, tracker_rows, page_size=batch_size)
        conn.commit()

        scanned += len(parents)
        with_children += batch_with
        without_children += batch_without
        upserted += len(rows)
        after_id = int(parents[-1][0])
        logger.info(
            "Progress: parents=%d, with_children=%d, without_children=%d, children=%d",
            scanned,
            with_children,
            without_children,
            upserted,
        )

    return IndexStats(scanned, with_children, without_children, upserted)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse evaluation-index options before connecting to PostgreSQL.

    中文：参数不执行索引写入；连接、嵌入和评测表的失败边界由运行路径保留。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dsn", default=DATABASE_URL, help="PostgreSQL DSN")
    parser.add_argument("--tickers", nargs="+", help="Only index these tickers")
    parser.add_argument("--years", nargs="+", type=int, help="Only index these fiscal years")
    parser.add_argument("--all", action="store_true", help="Intentionally index all table chunks")
    parser.add_argument("--batch-size", type=int, default=EMBEDDING_BATCH_SIZE)
    parser.add_argument("--limit", type=int, help="Maximum parent chunks to inspect")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild the selected scope")
    args = parser.parse_args(argv)
    if args.all and (args.tickers or args.years):
        parser.error("--all cannot be combined with --tickers or --years")
    if not args.all and not args.tickers and not args.years:
        parser.error("provide --tickers and/or --years, or explicitly pass --all")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.tickers:
        args.tickers = [ticker.upper() for ticker in args.tickers]
    return args


def main(argv: Optional[list[str]] = None) -> int:
    """Build or resume the evaluation-only child index from the command line.

    中文：入口只操作专用评测表；数据库错误不会被转换为看似完成的进度。
    """
    args = parse_args(argv)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    import psycopg2
    from pgvector.psycopg2 import register_vector

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    conn = psycopg2.connect(args.dsn)
    conn.autocommit = False
    register_vector(conn)
    try:
        embedder = Embedder(batch_size=args.batch_size)
        stats = build_index(
            conn,
            embedder,
            tickers=args.tickers,
            years=args.years,
            batch_size=args.batch_size,
            limit=args.limit,
            rebuild=args.rebuild,
            all_scope=args.all,
        )
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    logger.info("Index complete: %s", stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
