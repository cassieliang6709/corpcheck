"""
Main pipeline orchestration script
=====================================
Runs the full financial research RAG data pipeline end-to-end:

  1. Download SEC filings (10-K / 10-Q / 8-K)
  2. Download market data (yfinance)
  3. Download macro data (FRED)
  4. Download news articles (RSS)
  5. Download earnings call transcripts (Motley Fool)
  6. Clean & chunk all text documents
  7. Embed chunks and articles
  8. Load everything into PostgreSQL

Each stage can be skipped via CLI flags for incremental runs.

Usage
-----
    python -m corpcheck.ingestion.pipeline [options]

    --tickers AAPL MSFT ...     (default: all 50)
    --years   2019 2020 ...     (default: 2019-2023)
    --filing-types 10-K 10-Q   (default: 10-K 10-Q)
    --skip-download             skip all download stages
    --skip-sec                  skip SEC filing download
    --skip-market               skip yfinance download
    --skip-macro                skip FRED download
    --skip-news                 skip news download
    --skip-transcripts          skip transcript download
    --skip-embed                skip embedding stage (reuse existing vectors)
    --skip-load                 skip DB load stage (dry run)
    --schema-only               only run schema.sql, then exit
    --schema-path PATH          path to schema.sql (default: ./corpcheck.ingestion/schema.sql)
    --db-url URL                override DATABASE_URL env var
    --log-level LEVEL           DEBUG / INFO / WARNING (default: INFO)
    --log-file PATH             path to log file
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

from tqdm import tqdm

# ---------------------------------------------------------------------------
# Bootstrap logging before importing heavy modules
# ---------------------------------------------------------------------------

def _setup_logging(level: str, log_file: str) -> None:
    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file:
        os.makedirs(os.path.dirname(log_file) if os.path.dirname(log_file) else ".", exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO), format=fmt, handlers=handlers)


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pipeline imports (after logging is configured)
# ---------------------------------------------------------------------------

from corpcheck.ingestion.config import (
    ALL_TICKERS,
    DATABASE_URL,
    DEFAULT_FILING_TYPES,
    EMBEDDING_BATCH_SIZE,
    LOG_FILE,
    LOG_LEVEL,
    YEARS,
)
from corpcheck.ingestion.chunk_features import compute_chunk_features
from corpcheck.ingestion.downloaders.market_downloader import MarketDownloader
from corpcheck.ingestion.downloaders.macro_downloader import MacroDownloader
from corpcheck.ingestion.downloaders.news_downloader import NewsDownloader
from corpcheck.ingestion.downloaders.sec_downloader import SECDownloader
from corpcheck.ingestion.downloaders.transcript_downloader import TranscriptDownloader
from corpcheck.ingestion.loaders.db_loader import DBLoader
from corpcheck.ingestion.metadata import resolve_company_metadata
from corpcheck.ingestion.processors.chunker import Chunker
from corpcheck.ingestion.processors.embedder import Embedder
from corpcheck.ingestion.processors.html_cleaner import HTMLCleaner
from corpcheck.ingestion.processors.segment_builders import build_transcript_segments


# ---------------------------------------------------------------------------
# Stage helpers
# ---------------------------------------------------------------------------

def _run_sec_stage(
    tickers: list[str],
    filing_types: list[str],
    years: list[int],
) -> list[tuple]:
    """Download SEC filings and return filing metadata tuples."""
    logger.info("=== Stage: SEC filing download ===")
    dl = SECDownloader()
    metas = dl.download(tickers, filing_types, years)
    logger.info("SEC stage complete: %d filing documents", len(metas))
    return metas


def _run_market_stage(
    tickers: list[str],
) -> tuple[list[dict], list[dict], list[dict]]:
    """Download market data and return (company_infos, price_rows, fin_rows)."""
    logger.info("=== Stage: Market data download ===")
    dl = MarketDownloader()
    result = dl.download_all(tickers)
    logger.info(
        "Market stage complete: %d companies, %d price rows, %d financial rows",
        len(result[0]),
        len(result[1]),
        len(result[2]),
    )
    return result


def _run_macro_stage() -> list[dict]:
    """Download FRED macro data."""
    logger.info("=== Stage: Macro data download ===")
    dl = MacroDownloader()
    rows = dl.download_all()
    logger.info("Macro stage complete: %d rows", len(rows))
    return rows


def _run_news_stage(
    tickers: list[str],
    years: list[int],
    official_only: bool = False,
) -> list[dict]:
    """Download news articles."""
    logger.info("=== Stage: News download ===")
    dl = NewsDownloader(tickers=tickers, years=years, official_only=official_only)
    rows = dl.download_all()
    logger.info("News stage complete: %d articles", len(rows))
    return rows


def _run_transcript_stage(
    tickers: list[str],
    years: list[int],
) -> list[dict]:
    """Download earnings call transcripts."""
    logger.info("=== Stage: Transcript download ===")
    dl = TranscriptDownloader(tickers=tickers, years=years)
    rows = dl.download_all()
    logger.info("Transcript stage complete: %d transcripts", len(rows))
    return rows


def _process_filings(
    filing_metas: list[tuple],
    filing_types: list[str],
    embedder: Embedder,
    loader: DBLoader,
    skip_embed: bool,
    skip_load: bool,
) -> int:
    """
    Clean, chunk, embed, and load SEC filings.
    Returns the total number of chunks loaded.
    """
    logger.info("=== Stage: Clean / chunk / embed SEC filings ===")
    cleaner_cache: dict[str, HTMLCleaner] = {}
    chunker = Chunker()
    total_chunks = 0

    # Group filings by filing type to reuse cleaner
    for filing_meta in tqdm(filing_metas, desc="Processing filings", unit="filing"):
        try:
            period_of_report = None
            accession_number = ""
            cik = ""
            if len(filing_meta) >= 10:
                (
                    ticker,
                    filing_type,
                    fiscal_year,
                    period,
                    local_path,
                    source_url,
                    filed_date,
                    period_of_report,
                    accession_number,
                    cik,
                    *_,
                ) = filing_meta
            else:
                ticker, filing_type, fiscal_year, period, local_path, source_url, filed_date = filing_meta

            if filing_type not in cleaner_cache:
                cleaner_cache[filing_type] = HTMLCleaner(filing_type=filing_type)
            cleaner = cleaner_cache[filing_type]

            segments = cleaner.clean_segments(local_path)
            if not segments:
                logger.warning("No sections extracted from %s", local_path)
                continue

            chunk_payloads = chunker.chunk_segments(segments)
            if not chunk_payloads:
                continue

            # Ensure filing row exists in DB
            company_metadata = resolve_company_metadata(ticker)
            filing_row = {
                "ticker": ticker,
                "filing_type": filing_type,
                "fiscal_year": fiscal_year,
                "period": period,
                "filed_date": filed_date,
                "period_of_report": period_of_report,
                "accession_number": accession_number,
                "cik": cik,
                "source_url": source_url,
                "local_path": str(local_path),
            }

            filing_id_map: dict[tuple, int] = {}
            if not skip_load:
                loader.load_company(
                    [
                        {
                            "ticker": ticker,
                            "name": company_metadata["name"],
                            "sector": company_metadata["sector"],
                            "industry": company_metadata["industry"],
                            "market_cap": company_metadata["market_cap"],
                            "description": company_metadata["description"],
                        }
                    ]
                )
                filing_id_map = loader.load_filing([filing_row])

            filing_key = (ticker, filing_type, fiscal_year, period)
            filing_id = filing_id_map.get(filing_key)

            # Embed
            texts = [payload.text for payload in chunk_payloads]
            if skip_embed:
                embeddings = [None] * len(texts)
            else:
                emb_array = embedder.encode(texts)
                embeddings = [emb_array[i] for i in range(len(texts))]

            # Build chunk rows
            sector = company_metadata["sector"]
            chunk_rows = []
            for i, payload in enumerate(chunk_payloads):
                chunk_features = compute_chunk_features(payload.section_name, payload.text)
                chunk_rows.append(
                    {
                        "filing_id": filing_id,
                        "ticker": ticker,
                        "sector": sector,
                        "filing_type": filing_type,
                        "fiscal_year": fiscal_year,
                        "period": period,
                        "filed_date": filed_date,
                        "section_name": payload.section_name,
                        "chunk_index": i,
                        "content": payload.text,
                        "char_count": len(payload.text),
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
                        "embedding": embeddings[i],
                        "source_url": source_url,
                    }
                )

            if not skip_load and chunk_rows:
                loader.load_chunks(chunk_rows)
                if filing_id is not None:
                    loader.prune_chunks_for_filing(
                        filing_id,
                        [row["chunk_index"] for row in chunk_rows],
                    )

            total_chunks += len(chunk_rows)

        except Exception as exc:
            logger.error(
                "Failed processing %s %s %s: %s",
                ticker,
                filing_type,
                fiscal_year,
                exc,
                exc_info=True,
            )

    logger.info("Filing processing complete: %d total chunks", total_chunks)
    return total_chunks


def _process_news(
    news_rows: list[dict],
    embedder: Embedder,
    loader: DBLoader,
    skip_embed: bool,
    skip_load: bool,
) -> int:
    """Embed and load news articles."""
    logger.info("=== Stage: Embed / load news ===")
    if not news_rows:
        return 0

    if not skip_embed:
        texts = [
            (r.get("title") or "") + " " + (r.get("content") or "")
            for r in news_rows
        ]
        logger.info("Embedding %d news articles", len(texts))
        emb = embedder.encode_large(texts, show_progress_bar=True)
        for i, row in enumerate(news_rows):
            row["embedding"] = emb[i]

    if not skip_load:
        loader.load_news(news_rows)

    logger.info("News stage complete: %d articles", len(news_rows))
    return len(news_rows)


def _process_transcripts(
    transcript_rows: list[dict],
    embedder: Embedder,
    loader: DBLoader,
    chunker: Chunker,
    skip_embed: bool,
    skip_load: bool,
) -> int:
    """Chunk, embed, and load transcripts and their chunks."""
    logger.info("=== Stage: Process transcripts ===")
    if not transcript_rows:
        return 0

    total_chunks = 0

    # First, load the transcript parent rows
    parent_rows = [
        {
            "ticker": r["ticker"],
            "fiscal_year": r["fiscal_year"],
            "quarter": r["quarter"],
            "content": r["content"],
            "published_date": r.get("published_date"),
            "source_url": r.get("source_url"),
        }
        for r in transcript_rows
        if r.get("fiscal_year") and r.get("quarter")
    ]

    transcript_id_map: dict[tuple, int] = {}
    if not skip_load and parent_rows:
        transcript_id_map = loader.load_transcript(parent_rows)

    # Now chunk each transcript section
    for t_row in tqdm(transcript_rows, desc="Chunking transcripts", unit="transcript"):
        try:
            if not t_row.get("fiscal_year") or not t_row.get("quarter"):
                continue

            ticker = t_row["ticker"]
            fiscal_year = t_row["fiscal_year"]
            quarter = t_row["quarter"]
            sections = t_row.get("sections", {})

            t_key = (ticker, fiscal_year, quarter)
            transcript_id = transcript_id_map.get(t_key)

            all_chunk_rows: list[dict] = []
            chunk_index = 0

            transcript_segments = build_transcript_segments(
                ticker=ticker,
                fiscal_year=fiscal_year,
                quarter=quarter,
                sections=sections,
            )
            chunk_payloads = chunker.chunk_segments(transcript_segments)
            for payload in chunk_payloads:
                chunk_features = compute_chunk_features(payload.section_name, payload.text)
                all_chunk_rows.append(
                    {
                        "transcript_id": transcript_id,
                        "ticker": ticker,
                        "fiscal_year": fiscal_year,
                        "quarter": quarter,
                        "section": payload.section_name,
                        "chunk_index": chunk_index,
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
                        "embedding": None,
                    }
                )
                chunk_index += 1

            if all_chunk_rows and not skip_embed:
                texts = [r["content"] for r in all_chunk_rows]
                emb = embedder.encode(texts)
                for i, r in enumerate(all_chunk_rows):
                    r["embedding"] = emb[i]

            if not skip_load and all_chunk_rows:
                if transcript_id is not None:
                    loader.prune_transcript_chunks(transcript_id)
                loader.load_transcript_chunks(all_chunk_rows)

            total_chunks += len(all_chunk_rows)

        except Exception as exc:
            logger.error(
                "Transcript processing failed for %s: %s", t_row.get("ticker"), exc, exc_info=True
            )

    logger.info("Transcript processing complete: %d chunks", total_chunks)
    return total_chunks


def _print_stats(loader: DBLoader, elapsed: float) -> None:
    """Print final pipeline statistics."""
    try:
        stats = loader.get_stats()
        db_size = loader.get_db_size()
    except Exception as exc:
        logger.warning("Could not retrieve stats: %s", exc)
        return

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print(f"Elapsed: {elapsed:.1f}s")
    print("-" * 60)
    print(f"{'Table':<30} {'Rows':>10}")
    print("-" * 60)
    for table, count in stats.items():
        print(f"{table:<30} {count:>10,}")
    print("-" * 60)
    print(f"{'Database size':<30} {db_size:>10}")
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Financial Research RAG Data Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--tickers",
        nargs="+",
        default=None,
        metavar="TICKER",
        help="Tickers to process (default: all 50)",
    )
    p.add_argument(
        "--years",
        nargs="+",
        type=int,
        default=None,
        metavar="YEAR",
        help="Fiscal years to process (default: 2019-2023)",
    )
    p.add_argument(
        "--filing-types",
        nargs="+",
        default=None,
        metavar="TYPE",
        dest="filing_types",
        help="Filing types (default: 10-K 10-Q)",
    )
    p.add_argument("--skip-download",    action="store_true", help="Skip all download stages")
    p.add_argument("--skip-sec",         action="store_true", help="Skip SEC download")
    p.add_argument("--skip-market",      action="store_true", help="Skip market data download")
    p.add_argument("--skip-macro",       action="store_true", help="Skip macro data download")
    p.add_argument("--skip-news",        action="store_true", help="Skip news download")
    p.add_argument("--skip-transcripts", action="store_true", help="Skip transcript download")
    p.add_argument("--skip-embed",       action="store_true", help="Skip embedding stage")
    p.add_argument("--skip-load",        action="store_true", help="Skip DB load (dry run)")
    p.add_argument(
        "--official-news-only",
        action="store_true",
        help="Use official company newsroom sources for news when supported",
    )
    p.add_argument("--schema-only",      action="store_true", help="Run schema.sql and exit")
    p.add_argument(
        "--schema-path",
        default="./corpcheck.ingestion/schema.sql",
        help="Path to schema.sql",
    )
    p.add_argument("--db-url",   default=None, help="Override DATABASE_URL")
    p.add_argument("--log-level", default=LOG_LEVEL, choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    p.add_argument("--log-file",  default=LOG_FILE, help="Path to log file")
    return p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    _setup_logging(args.log_level, args.log_file)

    tickers: list[str] = args.tickers or ALL_TICKERS
    years: list[int] = args.years or YEARS
    filing_types: list[str] = args.filing_types or DEFAULT_FILING_TYPES
    db_url: str = args.db_url or DATABASE_URL

    skip_sec         = args.skip_download or args.skip_sec
    skip_market      = args.skip_download or args.skip_market
    skip_macro       = args.skip_download or args.skip_macro
    skip_news        = args.skip_download or args.skip_news
    skip_transcripts = args.skip_download or args.skip_transcripts
    skip_embed       = args.skip_embed
    skip_load        = args.skip_load
    official_news_only = args.official_news_only

    logger.info("Starting Financial RAG Pipeline")
    logger.info("  Tickers      : %d", len(tickers))
    logger.info("  Years        : %s", years)
    logger.info("  Filing types : %s", filing_types)
    logger.info("  Skip download: %s", args.skip_download)
    logger.info("  Skip embed   : %s", skip_embed)
    logger.info("  Skip load    : %s", skip_load)

    t_start = time.time()

    with DBLoader(dsn=db_url) as loader:
        # ----------------------------------------------------------------
        # Schema
        # ----------------------------------------------------------------
        if not skip_load:
            loader.init_schema(args.schema_path)

        if args.schema_only:
            logger.info("--schema-only flag set; exiting after schema init")
            return

        # ----------------------------------------------------------------
        # Embedder (initialise once and reuse)
        # ----------------------------------------------------------------
        embedder = Embedder() if not skip_embed else None  # type: ignore[assignment]
        chunker = Chunker()

        # ----------------------------------------------------------------
        # Market data
        # ----------------------------------------------------------------
        if not skip_market:
            company_infos, price_rows, fin_rows = _run_market_stage(tickers)
            if not skip_load:
                loader.load_company(company_infos)
                loader.load_market_data(price_rows)
                loader.load_financials(fin_rows)
        else:
            # Ensure companies table has at minimum a stub row per ticker
            # so FK constraints on other tables are satisfied.
            if not skip_load:
                stub_companies = [
                    resolve_company_metadata(t, {})
                    for t in tickers
                ]
                loader.load_company(stub_companies)

        # ----------------------------------------------------------------
        # Macro data
        # ----------------------------------------------------------------
        if not skip_macro:
            macro_rows = _run_macro_stage()
            if not skip_load:
                loader.load_macro(macro_rows)

        # ----------------------------------------------------------------
        # News
        # ----------------------------------------------------------------
        if not skip_news:
            news_rows = _run_news_stage(tickers, years, official_only=official_news_only)
            if news_rows:
                _process_news(news_rows, embedder, loader, skip_embed, skip_load)

        # ----------------------------------------------------------------
        # Transcripts
        # ----------------------------------------------------------------
        if not skip_transcripts:
            transcript_rows = _run_transcript_stage(tickers, years)
            if transcript_rows:
                _process_transcripts(
                    transcript_rows, embedder, loader, chunker, skip_embed, skip_load
                )

        # ----------------------------------------------------------------
        # SEC filings
        # ----------------------------------------------------------------
        if not skip_sec:
            filing_metas = _run_sec_stage(tickers, filing_types, years)
            if filing_metas:
                _process_filings(
                    filing_metas,
                    filing_types,
                    embedder,
                    loader,
                    skip_embed,
                    skip_load,
                )

        # ----------------------------------------------------------------
        # Final stats
        # ----------------------------------------------------------------
        elapsed = time.time() - t_start
        if not skip_load:
            _print_stats(loader, elapsed)
        else:
            logger.info("Dry-run complete in %.1fs (no DB writes)", elapsed)


if __name__ == "__main__":
    main()
