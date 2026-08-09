#!/usr/bin/env python3
"""Validate the locked AMZN 2019 and NKE 2018 table filings in isolation."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Optional, Sequence

import asyncpg
import psycopg2
from pgvector.asyncpg import register_vector

from corpcheck.ingestion.config import DATABASE_URL, EMBEDDING_BATCH_SIZE, SEC_USER_AGENT
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
from evaluation.validate_cvs_fy2018 import assert_pristine_download_dir

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = REPO_ROOT / "src" / "corpcheck" / "db" / "schema.sql"
FINANCEBENCH_PATH = REPO_ROOT / "evaluation" / "datasets" / "financebench_filtered.json"
FILING_TYPE = "10-K"
PERIOD = "annual"
STRICT_OVERLAP_THRESHOLD = 0.5


@dataclass(frozen=True)
class FilingSpec:
    ticker: str
    fiscal_year: int
    accession: str
    cik: str
    filed_date: date
    period_of_report: date
    after: str
    before: str
    financebench_id: str
    evidence_values: tuple[str, ...]
    title_phrase: str
    header_tokens: tuple[str, ...]


FILING_SPECS = (
    FilingSpec(
        ticker="AMZN",
        fiscal_year=2019,
        accession="0001018724-20-000004",
        cik="0001018724",
        filed_date=date(2020, 1, 31),
        period_of_report=date(2019, 12, 31),
        after="2020-01-01",
        before="2020-02-29",
        financebench_id="financebench_id_08286",
        evidence_values=("11,588",),
        title_phrase="consolidated statements of operations",
        header_tokens=("year ended december 31", "2019"),
    ),
    FilingSpec(
        ticker="NKE",
        fiscal_year=2018,
        accession="0000320187-18-000142",
        cik="0000320187",
        filed_date=date(2018, 7, 25),
        period_of_report=date(2018, 5, 31),
        after="2018-07-01",
        before="2018-08-31",
        financebench_id="financebench_id_04302",
        evidence_values=("36,397", "20,441"),
        title_phrase="consolidated statements of income",
        header_tokens=("year ended may 31", "2018"),
    ),
)


def exact_pair_metadata(discovered: Sequence[tuple[Any, ...]]) -> list[tuple[Any, ...]]:
    """Require exactly the two locked accessions with exact SEC metadata."""
    expected = {spec.accession: spec for spec in FILING_SPECS}
    accessions = [str(meta[8]) for meta in discovered]
    if len(accessions) != len(expected) or set(accessions) != set(expected):
        raise ValidationError(
            "downloaded metadata must match the locked AMZN/NKE pair exactly; "
            f"accessions={accessions}"
        )

    pair: list[tuple[Any, ...]] = []
    for meta in discovered:
        spec = expected[str(meta[8])]
        ticker, filing_type, fiscal_year, period = meta[:4]
        filed_date, period_of_report, cik = meta[6], meta[7], str(meta[9])
        if (
            ticker != spec.ticker
            or filing_type != FILING_TYPE
            or fiscal_year != spec.fiscal_year
            or period != PERIOD
            or filed_date != spec.filed_date
            or period_of_report != spec.period_of_report
            or cik.lstrip("0") != spec.cik.lstrip("0")
        ):
            raise ValidationError(
                f"downloaded metadata does not match locked filing {spec.accession}"
            )
        pair.append(meta)
    return sorted(pair, key=lambda meta: str(meta[0]))


def prepare_pair(
    download_dir: Path,
    *,
    downloader: Optional[SECDownloader] = None,
) -> list[tuple[Any, ...]]:
    """Download only narrow filing windows and select the exact locked pair."""
    active_downloader = downloader or SECDownloader(download_dir=str(download_dir))
    discovered: list[tuple[Any, ...]] = []
    for spec in FILING_SPECS:
        active_downloader._rate_limit()
        active_downloader._downloader.get(
            FILING_TYPE,
            spec.ticker,
            after=spec.after,
            before=spec.before,
            limit=1,
            include_amends=False,
            download_details=True,
        )
        discovered.extend(
            active_downloader._collect_metadata(
                [spec.ticker], [FILING_TYPE], [spec.fiscal_year]
            )
        )
    return exact_pair_metadata(discovered)


def load_benchmark_cases(path: Path = FINANCEBENCH_PATH) -> list[dict[str, Any]]:
    """Load exactly the two post-ingestion evaluation questions."""
    rows = json.loads(path.read_text(encoding="utf-8"))
    expected_ids = {spec.financebench_id for spec in FILING_SPECS}
    matches = [row for row in rows if row.get("financebench_id") in expected_ids]
    if len(matches) != len(expected_ids) or {
        str(row.get("financebench_id")) for row in matches
    } != expected_ids:
        raise ValidationError("expected exactly the two locked FinanceBench questions")
    return sorted(matches, key=lambda row: str(row["financebench_id"]))


def _filing_row_matches(row: Sequence[Any], spec: FilingSpec) -> bool:
    return (
        tuple(row[:-1])
        == (
            spec.ticker,
            FILING_TYPE,
            spec.fiscal_year,
            PERIOD,
            spec.filed_date,
            spec.period_of_report,
            spec.accession,
        )
        and str(row[-1]).lstrip("0") == spec.cik.lstrip("0")
    )


def _assert_semantic_table(
    spec: FilingSpec,
    rows: Sequence[Sequence[Any]],
) -> None:
    candidates = [
        (str(content), bool(embedded), str(title or ""), str(header or ""))
        for content, embedded, title, header in rows
        if all(value in str(content) for value in spec.evidence_values)
    ]
    if not candidates:
        raise ValidationError(
            f"{spec.ticker} Financial Statements tables are missing {spec.evidence_values}"
        )
    for _, embedded, title, header in candidates:
        normalized_title = " ".join(title.lower().split())
        normalized_header = " ".join(header.lower().split())
        if not embedded:
            continue
        if spec.title_phrase not in normalized_title:
            continue
        if not all(token in normalized_header for token in spec.header_tokens):
            continue
        return
    raise ValidationError(
        f"{spec.ticker} evidence table lacks embedding, semantic title, or period header"
    )


def assert_imported_pair(database_url: str) -> None:
    """Assert exact filings and semantic table metadata around the target values."""
    with psycopg2.connect(database_url) as conn:
        conn.set_session(readonly=True)
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT ticker, filing_type, fiscal_year, period, filed_date,
                       period_of_report, accession_number, cik
                FROM filings
                ORDER BY ticker
                """
            )
            filings = list(cursor.fetchall())
            tables: dict[str, list[Sequence[Any]]] = {}
            for spec in FILING_SPECS:
                cursor.execute(
                    """
                    SELECT c.content, c.embedding IS NOT NULL, c.display_title,
                           c.structure_meta->>'header_text'
                    FROM chunks c
                    JOIN filings f ON f.id = c.filing_id
                    WHERE f.accession_number = %s
                      AND c.section_name = 'Financial Statements'
                      AND c.content_kind = 'table'
                    ORDER BY c.chunk_index
                    """,
                    (spec.accession,),
                )
                tables[spec.accession] = list(cursor.fetchall())

    if len(filings) != len(FILING_SPECS) or not all(
        any(_filing_row_matches(row, spec) for row in filings) for spec in FILING_SPECS
    ):
        raise ValidationError(f"isolated database does not contain the exact pair: {filings}")
    for spec in FILING_SPECS:
        _assert_semantic_table(spec, tables[spec.accession])


def assert_strict_overlap(
    chunks: Sequence[ChunkResult],
    benchmark_case: dict[str, Any],
    *,
    threshold: float = STRICT_OVERLAP_THRESHOLD,
) -> list[int]:
    """Require every evidence span to have a provenance-gated strict top-10 hit."""
    gold_doc = parse_gold_doc(benchmark_case)
    ranks: list[int] = []
    best_overlaps: list[float] = []
    for span in gold_spans(benchmark_case):
        overlaps = [
            token_overlap(chunk.text, span, remove_boilerplate=True)
            if chunk_matches_gold_doc(chunk, gold_doc, match_quarter=False)
            else 0.0
            for chunk in chunks[:10]
        ]
        best_overlaps.append(max(overlaps, default=0.0))
        rank = next(
            (index for index, overlap in enumerate(overlaps, 1) if overlap >= threshold),
            None,
        )
        if rank is None:
            raise ValidationError(
                f"{benchmark_case.get('financebench_id')} strict Recall@10 failed; "
                f"best evidence overlaps={best_overlaps}"
            )
        ranks.append(rank)
    if not ranks:
        raise ValidationError("locked FinanceBench question has no evidence spans")
    return ranks


async def _retrieval_smoke(
    database_url: str,
    benchmark_cases: Sequence[dict[str, Any]],
) -> dict[str, list[int]]:
    async def init_connection(connection: asyncpg.Connection) -> None:
        await register_vector(connection)
        await connection.execute("SET ivfflat.probes = 10")

    pool = await asyncpg.create_pool(
        dsn=database_url, min_size=1, max_size=1, init=init_connection
    )
    results: dict[str, list[int]] = {}
    try:
        await load_known_tickers(pool)
        for case in benchmark_cases:
            chunks = await retrieve(
                pool=pool,
                query=str(case["question"]),
                k=10,
                alpha=0.7,
                sector=None,
                company=None,
                filing_type=None,
                fiscal_year=None,
            )
            case_id = str(case["financebench_id"])
            results[case_id] = assert_strict_overlap(chunks, case)
    finally:
        await pool.close()
    return results


def _assert_benchmark_unchanged(
    before: BenchmarkSnapshot,
    benchmark_database_url: str,
) -> None:
    after = benchmark_snapshot(benchmark_database_url)
    assert_clean_benchmark(after)
    if after != before:
        raise ValidationError(
            f"benchmark database changed during isolated validation: {before} -> {after}"
        )


def import_and_validate(
    database_url: str,
    benchmark_database_url: str,
    pair: list[tuple[Any, ...]],
    *,
    batch_size: int,
) -> tuple[int, dict[str, list[int]]]:
    """Import only the pair and prove table structure plus question-only retrieval."""
    pair = exact_pair_metadata(pair)
    before = benchmark_snapshot(benchmark_database_url)
    assert_clean_benchmark(before)
    assert_pristine_target(database_url)
    try:
        with DBLoader(dsn=database_url) as loader:
            loader.init_schema(str(SCHEMA_PATH))
            assert_empty_target(database_url)
            loaded = _process_filings(
                pair,
                [FILING_TYPE],
                Embedder(batch_size=batch_size),
                loader,
                skip_embed=False,
                skip_load=False,
            )
        assert_imported_pair(database_url)
        ranks = asyncio.run(_retrieval_smoke(database_url, load_benchmark_cases()))
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
    validate_isolation(
        args.database_url,
        args.download_dir,
        os.getenv("SEC_USER_AGENT", SEC_USER_AGENT),
        benchmark_database_url=args.benchmark_database_url,
    )
    assert_pristine_download_dir(args.download_dir)
    args.download_dir.mkdir(parents=True, exist_ok=True)
    pair = prepare_pair(args.download_dir)
    logger.info("Prepared exact accessions: %s", ", ".join(str(meta[8]) for meta in pair))
    if args.prepare_only:
        return 0

    loaded, ranks = import_and_validate(
        args.database_url,
        args.benchmark_database_url,
        pair,
        batch_size=args.batch_size,
    )
    logger.info("Table pair validation complete: %d chunks, ranks=%s", loaded, ranks)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    raise SystemExit(main())
