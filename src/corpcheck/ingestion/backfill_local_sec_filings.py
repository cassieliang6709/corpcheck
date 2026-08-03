"""
Backfill local SEC filings that already exist on disk.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import psycopg2

from corpcheck.ingestion.config import DATABASE_URL, EMBEDDING_BATCH_SIZE, SEC_DOWNLOAD_DIR
from corpcheck.ingestion.downloaders.sec_downloader import SECDownloader
from corpcheck.ingestion.loaders.db_loader import DBLoader
from corpcheck.ingestion.pipeline import _process_filings
from corpcheck.ingestion.processors.embedder import Embedder


logger = logging.getLogger(__name__)

DEFAULT_LOCAL_SEC_YEARS = list(range(2018, 2026))
DEFAULT_LOCAL_SEC_FILING_TYPES = ["10-K", "10-Q"]


def _discover_local_tickers(download_dir: str) -> list[str]:
    root = Path(download_dir) / "sec-edgar-filings"
    if not root.exists():
        return []
    return sorted(path.name for path in root.iterdir() if path.is_dir())


def _existing_filing_keys(dsn: str, tickers: list[str]) -> set[tuple[str, str, int, str]]:
    with psycopg2.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT ticker, filing_type, fiscal_year, period
                FROM filings
                WHERE ticker = ANY(%s)
                """,
                (tickers,),
            )
            return set(cur.fetchall())


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill local SEC filings already present on disk")
    parser.add_argument("--tickers", nargs="+", default=None, help="Tickers to process; defaults to all local SEC tickers")
    parser.add_argument("--years", nargs="+", type=int, default=None, help="Years to process; defaults to 2018-2025")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit after filtering and sorting missing filings")
    parser.add_argument(
        "--include-existing",
        action="store_true",
        help="Reprocess all discovered local filings, not just DB-missing ones",
    )
    parser.add_argument("--db-url", default=DATABASE_URL, help="PostgreSQL DSN")
    parser.add_argument("--batch-size", type=int, default=EMBEDDING_BATCH_SIZE, help="Embedding batch size")
    parser.add_argument(
        "--skip-embed",
        action="store_true",
        help="Skip embedding generation and write NULL embeddings for regenerated chunks",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    tickers = args.tickers or _discover_local_tickers(SEC_DOWNLOAD_DIR)
    years = args.years or DEFAULT_LOCAL_SEC_YEARS
    if not tickers:
        raise SystemExit("No local SEC tickers found under SEC_DOWNLOAD_DIR")

    downloader = SECDownloader()
    discovered = downloader._collect_metadata(tickers, DEFAULT_LOCAL_SEC_FILING_TYPES, years)
    existing = _existing_filing_keys(args.db_url, tickers)
    missing = [
        filing_meta
        for filing_meta in discovered
        if (filing_meta[0], filing_meta[1], filing_meta[2], filing_meta[3]) not in existing
    ]
    missing.sort(key=lambda meta: (-meta[2], meta[0], meta[1], meta[3]))
    remaining = discovered if args.include_existing else missing
    if args.limit is not None:
        remaining = remaining[: args.limit]

    logger.info("Local SEC tickers: %s", ", ".join(tickers))
    logger.info("Years: %s", years)
    logger.info("Discovered filing keys: %d", len(discovered))
    logger.info("Existing filing keys: %d", len(existing))
    logger.info("Remaining filing keys: %d", len(missing))
    logger.info("Selected filing keys for this run: %d", len(remaining))

    if not remaining:
        logger.info("No missing local SEC filings remain")
        return

    with DBLoader(dsn=args.db_url) as loader:
        embedder = None if args.skip_embed else Embedder(batch_size=args.batch_size)
        loaded_chunks = _process_filings(
            remaining,
            DEFAULT_LOCAL_SEC_FILING_TYPES,
            embedder,
            loader,
            skip_embed=args.skip_embed,
            skip_load=False,
        )

    logger.info("Loaded chunk rows: %d", loaded_chunks)


if __name__ == "__main__":
    main()
