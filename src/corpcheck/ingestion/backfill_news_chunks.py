"""
Backfill news article chunks into ``news_chunks``.

Reads existing rows from ``news_articles``, chunks each article body, and
stores the result in ``news_chunks`` so news can participate in text-level
retrieval alongside SEC filing chunks.
"""

from __future__ import annotations

import argparse
import logging
from typing import Any

from corpcheck.ingestion.config import DATABASE_URL, LOG_FILE, LOG_LEVEL
from corpcheck.ingestion.chunk_features import compute_chunk_features
from corpcheck.ingestion.loaders.db_loader import DBLoader
from corpcheck.ingestion.pipeline import _setup_logging
from corpcheck.ingestion.processors.chunker import Chunker
from corpcheck.ingestion.processors.embedder import Embedder
from corpcheck.ingestion.processors.segment_builders import build_news_segments

logger = logging.getLogger(__name__)


def _fetch_news_articles(
    loader: DBLoader,
    ticker: str | None = None,
) -> list[dict[str, Any]]:
    """Return existing news article rows from the database."""
    sql = """
        SELECT id, ticker, title, content, summary, published_date, source, source_url
        FROM news_articles
        WHERE COALESCE(content, summary, '') <> ''
    """
    params: tuple[Any, ...] = ()
    if ticker:
        sql += " AND ticker = %s"
        params = (ticker,)
    sql += " ORDER BY published_date NULLS LAST, id"

    with loader.conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()

    return [
        {
            "id": row[0],
            "ticker": row[1],
            "title": row[2],
            "content": row[3],
            "summary": row[4],
            "published_date": row[5],
            "source": row[6],
            "source_url": row[7],
        }
        for row in rows
    ]


def _article_text(article: dict[str, Any]) -> str:
    """Assemble a chunkable text body for one article."""
    title = (article.get("title") or "").strip()
    summary = (article.get("summary") or "").strip()
    content = (article.get("content") or "").strip()
    parts = [part for part in (title, summary, content) if part]
    return "\n\n".join(parts)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Backfill news_chunks from news_articles")
    parser.add_argument("--dsn", default=DATABASE_URL, help="PostgreSQL DSN")
    parser.add_argument("--ticker", default=None, help="Optional ticker filter")
    parser.add_argument("--skip-embed", action="store_true", help="Skip embedding generation")
    parser.add_argument("--log-level", default=LOG_LEVEL, choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--log-file", default=LOG_FILE, help="Path to log file")
    args = parser.parse_args(argv)

    _setup_logging(args.log_level, args.log_file)

    chunker = Chunker()
    embedder = None if args.skip_embed else Embedder()

    with DBLoader(dsn=args.dsn) as loader:
        loader.init_schema()
        articles = _fetch_news_articles(loader, ticker=args.ticker)
        logger.info("Loaded %d news articles to backfill", len(articles))

        total_chunks = 0
        for article in articles:
            segments = build_news_segments(article)
            if not segments:
                continue

            chunk_payloads = chunker.chunk_segments(segments)
            if not chunk_payloads:
                continue

            chunk_rows: list[dict[str, Any]] = []
            for idx, payload in enumerate(chunk_payloads):
                chunk_features = compute_chunk_features(payload.section_name, payload.text)
                chunk_rows.append(
                    {
                        "news_article_id": article["id"],
                        "ticker": article["ticker"],
                        "published_date": article["published_date"],
                        "source": article["source"],
                        "chunk_index": idx,
                        "content": payload.text,
                        "token_count": payload.token_count,
                        "numeric_token_count": chunk_features["numeric_token_count"],
                        "number_density": chunk_features["number_density"],
                        "data_signal_score": chunk_features["data_signal_score"],
                        "is_quantitative": chunk_features["is_quantitative"],
                        "content_kind": payload.content_kind,
                        "chunk_strategy": payload.chunk_strategy,
                        "display_title": payload.display_title,
                        "chunk_group_key": payload.chunk_group_key,
                        "structure_meta": payload.structure_meta,
                        "source_url": article["source_url"],
                        "embedding": None,
                    }
                )

            if embedder is not None:
                embeddings = embedder.encode([row["content"] for row in chunk_rows])
                for i, row in enumerate(chunk_rows):
                    row["embedding"] = embeddings[i]

            loader.prune_news_chunks_for_article(article["id"])
            loader.load_news_chunks(chunk_rows)
            total_chunks += len(chunk_rows)

        logger.info("Backfilled %d news_chunks", total_chunks)


if __name__ == "__main__":
    main()
