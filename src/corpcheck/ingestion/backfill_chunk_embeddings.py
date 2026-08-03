"""
Backfill missing SEC chunk embeddings in PostgreSQL.

This script reads existing rows from ``chunks`` where ``embedding`` is NULL,
generates embeddings using the project's configured sentence-transformers
model, and updates the rows in-place.

It defaults to offline Hugging Face mode so it can reuse a locally cached
model in restricted environments.
"""

from __future__ import annotations

import argparse
import logging
import os

import numpy as np
import psycopg2
import psycopg2.extras
from pgvector.psycopg2 import register_vector

from corpcheck.ingestion.config import DATABASE_URL, EMBEDDING_BATCH_SIZE
from corpcheck.ingestion.processors.embedder import Embedder


logger = logging.getLogger(__name__)


def _count_missing_chunks(
    conn: psycopg2.extensions.connection,
    limit: int | None = None,
) -> int:
    with conn.cursor() as cur:
        if limit is None:
            cur.execute("SELECT COUNT(*) FROM chunks WHERE embedding IS NULL")
        else:
            cur.execute(
                """
                SELECT COUNT(*)
                FROM (
                    SELECT 1
                    FROM chunks
                    WHERE embedding IS NULL
                    ORDER BY id
                    LIMIT %s
                ) limited
                """,
                (limit,),
            )
        return int(cur.fetchone()[0])


def _fetch_missing_chunk_batch(
    conn: psycopg2.extensions.connection,
    batch_size: int,
    *,
    after_id: int = 0,
    remaining_limit: int | None = None,
) -> list[tuple[int, str]]:
    sql = """
        SELECT id, content
        FROM chunks
        WHERE embedding IS NULL
          AND id > %s
        ORDER BY id
        LIMIT %s
    """
    fetch_size = batch_size if remaining_limit is None else min(batch_size, remaining_limit)
    if fetch_size <= 0:
        return []

    with conn.cursor() as cur:
        cur.execute(sql, (after_id, fetch_size))
        return list(cur.fetchall())


def backfill_embeddings(
    dsn: str,
    batch_size: int,
    limit: int | None = None,
) -> int:
    # Prefer local cache instead of attempting a network fetch.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    conn = psycopg2.connect(dsn)
    conn.autocommit = False
    register_vector(conn)

    try:
        total_missing = _count_missing_chunks(conn, limit=limit)
        if total_missing == 0:
            logger.info("No missing chunk embeddings found")
            return 0

        logger.info("Found %d chunks with NULL embeddings", total_missing)
        embedder = Embedder(batch_size=batch_size)
        updated = 0
        last_id = 0

        update_sql = """
            UPDATE chunks
            SET embedding = %(embedding)s
            WHERE id = %(id)s
        """

        with conn.cursor() as cur:
            while True:
                remaining_limit = None if limit is None else max(limit - updated, 0)
                batch = _fetch_missing_chunk_batch(
                    conn,
                    batch_size,
                    after_id=last_id,
                    remaining_limit=remaining_limit,
                )
                if not batch:
                    break

                ids = [row_id for row_id, _ in batch]
                texts = [content for _, content in batch]
                embeddings = embedder.encode(texts)

                payload = [
                    {
                        "id": row_id,
                        "embedding": np.asarray(embeddings[i], dtype=np.float32),
                    }
                    for i, row_id in enumerate(ids)
                ]

                psycopg2.extras.execute_batch(cur, update_sql, payload, page_size=batch_size)
                conn.commit()
                updated += len(payload)
                last_id = ids[-1]
                logger.info("Updated %d / %d chunk embeddings", updated, total_missing)

        return updated
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill missing chunk embeddings")
    parser.add_argument("--dsn", default=DATABASE_URL, help="PostgreSQL DSN")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=EMBEDDING_BATCH_SIZE,
        help="Embedding / update batch size",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit for testing",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

    updated = backfill_embeddings(
        dsn=args.dsn,
        batch_size=args.batch_size,
        limit=args.limit,
    )
    logger.info("Backfill complete: %d chunks updated", updated)


if __name__ == "__main__":
    main()
