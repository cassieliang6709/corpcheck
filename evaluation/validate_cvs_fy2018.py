#!/usr/bin/env python3
"""Repair and validate the fixture-locked CVS FY2018 10-K in isolation.

This runner is intentionally not a general ingestion command. It downloads one
allowlisted SEC accession into a pristine, explicit directory, imports it into
an existing pristine database, and proves that CorpCheck can retrieve both
FinanceBench evidence spans without changing the public benchmark corpus.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from datetime import date
from pathlib import Path
from typing import Any, Optional, Sequence

import asyncpg
import psycopg2
from pgvector.asyncpg import register_vector

from corpcheck.ingestion.config import (
    DATABASE_URL,
    EMBEDDING_BATCH_SIZE,
    SEC_USER_AGENT,
)
from corpcheck.ingestion.downloaders.sec_downloader import SECDownloader
from corpcheck.ingestion.loaders.db_loader import DBLoader
from corpcheck.ingestion.pipeline import _process_filings
from corpcheck.ingestion.processors.embedder import Embedder
from corpcheck.models import ChunkResult
from corpcheck.retrieval import load_known_tickers, retrieve
from evaluation.financebench import chunk_matches_gold_doc, gold_spans, parse_gold_doc
from evaluation.metrics import token_overlap
from evaluation.validate_amendment_pair import (
    BenchmarkSnapshot,
    ValidationError,
    assert_clean_benchmark,
    assert_empty_target,
    assert_pristine_target,
    benchmark_snapshot,
    validate_isolation,
)

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = REPO_ROOT / "src" / "corpcheck" / "db" / "schema.sql"
FINANCEBENCH_PATH = REPO_ROOT / "evaluation" / "datasets" / "financebench_filtered.json"

TICKER = "CVS"
FILING_TYPE = "10-K"
FISCAL_YEAR = 2018
PERIOD = "annual"
ACCESSION = "0000064803-19-000013"
CIK = "0000064803"
FILED_DATE = date(2019, 2, 28)
PERIOD_OF_REPORT = date(2018, 12, 31)
FINANCEBENCH_ID = "financebench_id_05915"
EVIDENCE_VALUES = ("194,579", "11,349", "10,292")
STRICT_OVERLAP_THRESHOLD = 0.5


def assert_pristine_download_dir(download_dir: Path) -> None:
    """Reject a download directory containing any pre-existing entry."""
    if download_dir.exists() and any(download_dir.iterdir()):
        raise ValidationError(
            "isolated download directory must be absent or empty before download"
        )


def exact_filing_metadata(discovered: Sequence[tuple[Any, ...]]) -> tuple[Any, ...]:
    """Select the exact CVS accession and reject any metadata disagreement."""
    matches = [meta for meta in discovered if str(meta[8]) == ACCESSION]
    if len(matches) != 1:
        raise ValidationError(
            f"download must contain exactly one {ACCESSION} filing, found {len(matches)}"
        )

    meta = matches[0]
    ticker, filing_type, fiscal_year, period = meta[:4]
    filed_date, period_of_report, cik = meta[6], meta[7], str(meta[9])
    if (
        ticker != TICKER
        or filing_type != FILING_TYPE
        or fiscal_year != FISCAL_YEAR
        or period != PERIOD
        or filed_date != FILED_DATE
        or period_of_report != PERIOD_OF_REPORT
        or cik.lstrip("0") != CIK.lstrip("0")
    ):
        raise ValidationError(
            "downloaded accession metadata does not match the locked CVS FY2018 10-K"
        )
    return meta


def prepare_filing(
    download_dir: Path,
    *,
    downloader: Optional[SECDownloader] = None,
) -> tuple[Any, ...]:
    """Download a narrow CVS 10-K filing window and select the exact accession."""
    active_downloader = downloader or SECDownloader(download_dir=str(download_dir))
    active_downloader._rate_limit()
    active_downloader._downloader.get(
        FILING_TYPE,
        TICKER,
        after="2019-02-01",
        before="2019-03-31",
        limit=1,
        include_amends=False,
        download_details=True,
    )
    discovered = active_downloader._collect_metadata(
        [TICKER], [FILING_TYPE], [FISCAL_YEAR]
    )
    return exact_filing_metadata(discovered)


def load_benchmark_case(path: Path = FINANCEBENCH_PATH) -> dict[str, Any]:
    """Load only the post-ingestion evaluation case for this locked repair."""
    rows = json.loads(path.read_text(encoding="utf-8"))
    matches = [row for row in rows if row.get("financebench_id") == FINANCEBENCH_ID]
    if len(matches) != 1:
        raise ValidationError(
            f"expected exactly one {FINANCEBENCH_ID} benchmark row, found {len(matches)}"
        )
    return matches[0]


def assert_imported_filing(database_url: str) -> None:
    """Assert exact filing metadata, embeddings, section, and required values."""
    with psycopg2.connect(database_url) as conn:
        conn.set_session(readonly=True)
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT ticker, filing_type, fiscal_year, period, filed_date,
                       period_of_report, accession_number, cik
                FROM filings
                """
            )
            filings = list(cursor.fetchall())
            cursor.execute(
                """
                SELECT c.content, c.embedding IS NOT NULL
                FROM chunks c
                JOIN filings f ON f.id = c.filing_id
                WHERE f.accession_number = %s
                  AND c.section_name = 'Financial Statements'
                ORDER BY c.chunk_index
                """,
                (ACCESSION,),
            )
            financial_chunks = list(cursor.fetchall())

    expected_without_cik = (
        TICKER,
        FILING_TYPE,
        FISCAL_YEAR,
        PERIOD,
        FILED_DATE,
        PERIOD_OF_REPORT,
        ACCESSION,
    )
    if (
        len(filings) != 1
        or tuple(filings[0][:-1]) != expected_without_cik
        or str(filings[0][-1]).lstrip("0") != CIK.lstrip("0")
    ):
        raise ValidationError(f"isolated database has wrong CVS filing metadata: {filings}")
    if not financial_chunks or any(not embedded for _, embedded in financial_chunks):
        raise ValidationError("Financial Statements chunks are missing or not fully embedded")

    content = "\n".join(str(text) for text, _ in financial_chunks)
    missing = [value for value in EVIDENCE_VALUES if value not in content]
    if missing:
        raise ValidationError(
            f"Financial Statements chunks are missing required values: {missing}"
        )


def assert_strict_overlap(
    chunks: Sequence[ChunkResult],
    benchmark_case: dict[str, Any],
    *,
    threshold: float = STRICT_OVERLAP_THRESHOLD,
) -> list[int]:
    """Require every gold evidence span to have a provenance-gated top-10 hit."""
    gold_doc = parse_gold_doc(benchmark_case)
    spans = gold_spans(benchmark_case)
    ranks: list[int] = []
    best_overlaps: list[float] = []
    for span in spans:
        overlaps = [
            token_overlap(chunk.text, span, remove_boilerplate=True)
            if chunk_matches_gold_doc(chunk, gold_doc, match_quarter=False)
            else 0.0
            for chunk in chunks[:10]
        ]
        best_overlaps.append(max(overlaps, default=0.0))
        rank = next(
            (index for index, overlap in enumerate(overlaps, start=1) if overlap >= threshold),
            None,
        )
        if rank is None:
            raise ValidationError(
                "CVS FY2018 strict Recall@10 failed; "
                f"best evidence overlaps={best_overlaps}"
            )
        ranks.append(rank)
    if not ranks:
        raise ValidationError("CVS benchmark case has no gold evidence spans")
    return ranks


async def _retrieval_smoke(database_url: str, benchmark_case: dict[str, Any]) -> list[int]:
    async def init_connection(connection: asyncpg.Connection) -> None:
        await register_vector(connection)
        await connection.execute("SET ivfflat.probes = 10")

    pool = await asyncpg.create_pool(
        dsn=database_url,
        min_size=1,
        max_size=1,
        init=init_connection,
    )
    try:
        await load_known_tickers(pool)
        chunks = await retrieve(
            pool=pool,
            query=str(benchmark_case["question"]),
            k=10,
            alpha=0.7,
            sector=None,
            company=None,
            filing_type=None,
            fiscal_year=None,
        )
    finally:
        await pool.close()
    return assert_strict_overlap(chunks, benchmark_case)


def _assert_benchmark_unchanged(
    before: BenchmarkSnapshot,
    benchmark_database_url: str,
) -> None:
    after = benchmark_snapshot(benchmark_database_url)
    if after != before:
        raise ValidationError(
            f"benchmark database changed during isolated validation: {before} -> {after}"
        )


def import_and_validate(
    database_url: str,
    benchmark_database_url: str,
    filing: tuple[Any, ...],
    *,
    batch_size: int,
) -> tuple[int, list[int]]:
    """Import the filing and prove evidence plus question-only strict retrieval."""
    before = benchmark_snapshot(benchmark_database_url)
    assert_clean_benchmark(before)
    assert_pristine_target(database_url)
    try:
        with DBLoader(dsn=database_url) as loader:
            loader.init_schema(str(SCHEMA_PATH))
            assert_empty_target(database_url)
            embedder = Embedder(batch_size=batch_size)
            loaded = _process_filings(
                [filing],
                [FILING_TYPE],
                embedder,
                loader,
                skip_embed=False,
                skip_load=False,
            )
        assert_imported_filing(database_url)
        benchmark_case = load_benchmark_case()
        ranks = asyncio.run(_retrieval_smoke(database_url, benchmark_case))
    finally:
        _assert_benchmark_unchanged(before, benchmark_database_url)
    return loaded, ranks


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database-url", required=True, help="Existing pristine target DB URL")
    parser.add_argument("--download-dir", required=True, type=Path)
    parser.add_argument(
        "--benchmark-database-url",
        default=DATABASE_URL,
        help="Read-only public benchmark DB used for before/after invariants",
    )
    parser.add_argument("--batch-size", type=int, default=EMBEDDING_BATCH_SIZE)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args(argv)
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    return args


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    sec_user_agent = os.getenv("SEC_USER_AGENT", SEC_USER_AGENT)
    validate_isolation(
        args.database_url,
        args.download_dir,
        sec_user_agent,
        benchmark_database_url=args.benchmark_database_url,
    )
    assert_pristine_download_dir(args.download_dir)
    args.download_dir.mkdir(parents=True, exist_ok=True)
    filing = prepare_filing(args.download_dir)
    logger.info("Prepared exact accession: %s", filing[8])
    if args.prepare_only:
        return 0

    loaded, ranks = import_and_validate(
        args.database_url,
        args.benchmark_database_url,
        filing,
        batch_size=args.batch_size,
    )
    logger.info(
        "CVS FY2018 isolated validation complete: %d chunks loaded, evidence ranks=%s",
        loaded,
        ranks,
    )
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    raise SystemExit(main())
