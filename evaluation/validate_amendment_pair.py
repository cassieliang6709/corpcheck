#!/usr/bin/env python3
"""Validate one real SEC original/amendment pair in an isolated database.

中文：使用允许列表中的原件/修订件在隔离数据库验证修订处理路径；保护检查会拒绝
基准库、默认下载目录和非预期输入，以防评测脚本影响真实语料。

The runner is deliberately fixture-locked and refuses CorpCheck's benchmark
database and default SEC download directory. It is not a general ingestion CLI.
"""

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
from urllib.parse import unquote, urlparse

import asyncpg
import psycopg2

from corpcheck.ingestion.config import (
    DATABASE_URL,
    EMBEDDING_BATCH_SIZE,
    SEC_DOWNLOAD_DIR,
    SEC_USER_AGENT,
)
from corpcheck.ingestion.downloaders.sec_downloader import (
    SECDownloader,
    parse_sec_user_agent,
)
from corpcheck.ingestion.loaders.db_loader import DBLoader
from corpcheck.ingestion.pipeline import _process_filings
from corpcheck.ingestion.processors.embedder import Embedder
from corpcheck.retrieval.revision import filter_superseded_rows, load_superseded_filings

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PATH = REPO_ROOT / "tests" / "fixtures" / "sec_real_amendment_gme_2024.json"
SCHEMA_PATH = REPO_ROOT / "src" / "corpcheck" / "db" / "schema.sql"
BENCHMARK_DATABASE_NAME = "financial_rag"
PLACEHOLDER_USER_AGENT_MARKERS = (
    "example.com",
    "your-email",
    "user@example",
    "corpcheck-pipeline your-email",
)


class ValidationError(RuntimeError):
    """A fail-closed validation or isolation error."""


@dataclass(frozen=True)
class BenchmarkSnapshot:
    """Baseline corpus counts used to prove the benchmark remained untouched.

    中文：隔离验证前后的公开基准快照；计数发生变化即表示该验证不再可信。
    """

    companies: int
    filings: int
    chunks: int
    gme_filings: int
    amendments: int


def database_name(database_url: str) -> str:
    """Extract the decoded database name from a PostgreSQL URL."""
    parsed = urlparse(database_url)
    if parsed.scheme not in {"postgres", "postgresql"} or not parsed.path.strip("/"):
        raise ValidationError("--database-url must be a PostgreSQL URL with a database name")
    return unquote(parsed.path.strip("/"))


def validate_real_sec_user_agent(sec_user_agent: str) -> None:
    """Require the SEC's two-token identity format and a non-placeholder email."""
    try:
        _, contact_email = parse_sec_user_agent(sec_user_agent)
    except ValueError as exc:
        raise ValidationError(str(exc)) from exc
    normalized_agent = " ".join(sec_user_agent.split()).lower()
    if (
        "@" not in contact_email
        or any(marker in normalized_agent for marker in PLACEHOLDER_USER_AGENT_MARKERS)
    ):
        raise ValidationError("SEC_USER_AGENT must contain a real, non-placeholder contact email")


def validate_isolation(
    database_url: str,
    download_dir: Path,
    sec_user_agent: str,
    *,
    benchmark_database_url: str,
) -> None:
    """Reject unsafe target, download, and SEC identity configuration."""
    target_name = database_name(database_url)
    benchmark_name = database_name(benchmark_database_url)
    if target_name.lower() == BENCHMARK_DATABASE_NAME:
        raise ValidationError(f"refusing benchmark database {BENCHMARK_DATABASE_NAME!r}")
    if target_name.lower() == benchmark_name.lower():
        raise ValidationError("target and benchmark database names must differ")
    if benchmark_name.lower() != BENCHMARK_DATABASE_NAME:
        raise ValidationError(
            f"benchmark URL must point to {BENCHMARK_DATABASE_NAME!r}, got {benchmark_name!r}"
        )

    resolved_download = download_dir.expanduser().resolve()
    default_download = Path(SEC_DOWNLOAD_DIR).expanduser().resolve()
    if resolved_download == default_download or tuple(resolved_download.parts[-2:]) == (
        "data",
        "sec_filings",
    ):
        raise ValidationError("refusing the default data/sec_filings download directory")

    validate_real_sec_user_agent(sec_user_agent)


def load_fixture(path: Path = FIXTURE_PATH) -> dict[str, Any]:
    """Load and validate the fixture's exact two-document contract."""
    fixture = json.loads(path.read_text(encoding="utf-8"))
    if fixture.get("ticker") != "GME":
        raise ValidationError("fixture ticker must be GME")
    accessions = [
        fixture.get(name, {}).get("accession_number") for name in ("original", "amendment")
    ]
    if len(set(accessions)) != 2 or any(not value for value in accessions):
        raise ValidationError("fixture must define exactly two distinct accession numbers")
    if fixture["original"].get("filing_type") != "10-K":
        raise ValidationError("fixture original must be a 10-K")
    if fixture["amendment"].get("filing_type") != "10-K/A":
        raise ValidationError("fixture amendment must be a 10-K/A")
    return fixture


def _fixture_meta(
    discovered: tuple[Any, ...],
    fixture: dict[str, Any],
) -> tuple[Any, ...]:
    """Apply fixture-authoritative version metadata to one discovered local file."""
    ticker, _, _, _, local_path, _, _, _, accession, _ = discovered[:10]
    versions = {
        fixture[name]["accession_number"]: fixture[name] for name in ("original", "amendment")
    }
    version = versions[accession]
    grouping = fixture["corpcheck_grouping"]
    return (
        ticker,
        version["filing_type"],
        int(grouping["fiscal_year"]),
        grouping["period"],
        local_path,
        version["source_url"],
        date.fromisoformat(version["filed_date"]),
        date.fromisoformat(version["period_of_report"]),
        accession,
        fixture["cik"],
    )


def exact_pair_metadata(
    discovered: Sequence[tuple[Any, ...]],
    fixture: dict[str, Any],
) -> list[tuple[Any, ...]]:
    """Return the exact allowlisted pair, rejecting missing, duplicate, or extra accessions."""
    expected = {
        fixture["original"]["accession_number"],
        fixture["amendment"]["accession_number"],
    }
    discovered_accessions = [str(meta[8]) for meta in discovered]
    if len(discovered_accessions) != 2 or set(discovered_accessions) != expected:
        missing = sorted(expected - set(discovered_accessions))
        unexpected = sorted(set(discovered_accessions) - expected)
        raise ValidationError(
            "downloaded metadata must match the fixture exactly; "
            f"missing={missing}, unexpected={unexpected}, rows={len(discovered_accessions)}"
        )
    pair = [_fixture_meta(meta, fixture) for meta in discovered]
    return sorted(pair, key=lambda meta: meta[6])


def prepare_pair(
    download_dir: Path,
    fixture: dict[str, Any],
    *,
    downloader: Optional[SECDownloader] = None,
) -> list[tuple[Any, ...]]:
    """Download GME's base 10-K with amendments and validate the exact pair."""
    active_downloader = downloader or SECDownloader(download_dir=str(download_dir))
    active_downloader._rate_limit()
    active_downloader._downloader.get(
        "10-K",
        fixture["ticker"],
        after="2024-01-01",
        before="2024-12-31",
        limit=2,
        include_amends=True,
        download_details=True,
    )
    discovered = active_downloader._collect_metadata(
        [fixture["ticker"]],
        ["10-K"],
        [int(fixture["corpcheck_grouping"]["fiscal_year"])],
    )
    return exact_pair_metadata(discovered, fixture)


def benchmark_snapshot(database_url: str) -> BenchmarkSnapshot:
    """Read benchmark invariants through a read-only transaction."""
    with psycopg2.connect(database_url) as conn:
        conn.set_session(readonly=True)
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    (SELECT COUNT(*) FROM companies),
                    (SELECT COUNT(*) FROM filings),
                    (SELECT COUNT(*) FROM chunks),
                    (SELECT COUNT(*) FROM filings WHERE ticker = 'GME'),
                    (SELECT COUNT(*) FROM filings WHERE filing_type LIKE '%/A')
                """
            )
            values = cursor.fetchone()
    if values is None:
        raise ValidationError("benchmark snapshot query returned no row")
    return BenchmarkSnapshot(*(int(value) for value in values))


def assert_clean_benchmark(snapshot: BenchmarkSnapshot) -> None:
    """Require the public corpus to remain free of the validation issuer/pair."""
    if snapshot.gme_filings or snapshot.amendments:
        raise ValidationError(
            "public benchmark is already contaminated with GME or amendment filings: "
            f"{snapshot}"
        )


def assert_pristine_target(database_url: str) -> None:
    """Reject a target containing any user-created relation before schema init."""
    with psycopg2.connect(database_url) as conn:
        conn.set_session(readonly=True)
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT n.nspname, c.relname, c.relkind
                FROM pg_catalog.pg_class AS c
                JOIN pg_catalog.pg_namespace AS n ON n.oid = c.relnamespace
                WHERE n.nspname NOT IN ('pg_catalog', 'information_schema')
                  AND n.nspname NOT LIKE 'pg_toast%'
                  AND c.relkind IN ('r', 'p', 'v', 'm', 'S', 'f')
                ORDER BY n.nspname, c.relname
                LIMIT 10
                """
            )
            relations = list(cursor.fetchall())
    if relations:
        names = [f"{schema}.{name}" for schema, name, _ in relations]
        raise ValidationError(
            "isolated target must be a pristine database before schema init; "
            f"found relations: {names}"
        )


def assert_empty_target(database_url: str) -> None:
    """Refuse to mix the fixture pair with data already present in the target DB."""
    with psycopg2.connect(database_url) as conn:
        conn.set_session(readonly=True)
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT (SELECT COUNT(*) FROM filings), (SELECT COUNT(*) FROM chunks)"
            )
            values = cursor.fetchone()
    if values is None or any(int(value) for value in values):
        raise ValidationError(f"isolated target must be empty after schema init, got {values}")


def assert_imported_pair(database_url: str, fixture: dict[str, Any]) -> None:
    """Assert exact filing metadata, embedded chunks, and safe amendment sections."""
    expected = {
        fixture["original"]["accession_number"]: fixture["original"]["filing_type"],
        fixture["amendment"]["accession_number"]: fixture["amendment"]["filing_type"],
    }
    grouping = fixture["corpcheck_grouping"]
    with psycopg2.connect(database_url) as conn:
        conn.set_session(readonly=True)
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT accession_number, filing_type, fiscal_year, period,
                       period_of_report, filed_date
                FROM filings
                ORDER BY accession_number
                """
            )
            filings = list(cursor.fetchall())
            if len(filings) != 2 or {row[0]: row[1] for row in filings} != expected:
                raise ValidationError(
                    f"isolated database does not contain the exact pair: {filings}"
                )
            for row in filings:
                version_name = (
                    "original"
                    if row[0] == fixture["original"]["accession_number"]
                    else "amendment"
                )
                version = fixture[version_name]
                if (
                    row[2] != grouping["fiscal_year"]
                    or row[3] != grouping["period"]
                    or row[4] != date.fromisoformat(version["period_of_report"])
                    or row[5] != date.fromisoformat(version["filed_date"])
                ):
                    raise ValidationError(
                        "pair fiscal-year, period, report-date, or filed-date metadata is wrong"
                    )

            cursor.execute(
                """
                SELECT f.accession_number, COUNT(c.id), COUNT(c.embedding),
                       ARRAY_AGG(DISTINCT c.section_name ORDER BY c.section_name)
                FROM filings f
                LEFT JOIN chunks c ON c.filing_id = f.id
                GROUP BY f.accession_number
                ORDER BY f.accession_number
                """
            )
            chunk_rows = list(cursor.fetchall())
    if len(chunk_rows) != 2 or any(
        total == 0 or total != embedded for _, total, embedded, _ in chunk_rows
    ):
        raise ValidationError(f"pair chunks are missing or not fully embedded: {chunk_rows}")
    amendment_accession = fixture["amendment"]["accession_number"]
    amendment_sections = next(row[3] or [] for row in chunk_rows if row[0] == amendment_accession)
    unsafe = {str(section).strip().upper() for section in amendment_sections} & {
        "FULL DOCUMENT",
        "UNKNOWN",
    }
    if unsafe:
        raise ValidationError(f"amendment produced unsafe wildcard sections: {sorted(unsafe)}")


async def _revision_composition_smoke(database_url: str) -> None:
    pool = await asyncpg.create_pool(dsn=database_url, min_size=1, max_size=1)
    try:
        records = await pool.fetch(
            """
            SELECT c.id::text AS chunk_id, 'sec'::text AS source_type,
                   c.ticker AS company, c.filing_type, c.fiscal_year,
                   c.period AS period_label, c.section_name
            FROM chunks c
            WHERE c.ticker = 'GME'
            ORDER BY c.filing_type, c.chunk_index
            """
        )
        candidates = [dict(record) for record in records]
        superseded = await load_superseded_filings(pool, candidates)
        kept, dropped = filter_superseded_rows(candidates, superseded)
    finally:
        await pool.close()

    def _ids(rows: Sequence[dict[str, Any]], filing_type: str, section: str) -> set[str]:
        return {
            row["chunk_id"]
            for row in rows
            if row["filing_type"] == filing_type
            and str(row["section_name"]).strip().upper() == section
        }

    original_item5 = _ids(candidates, "10-K", "MARKET FOR COMMON EQUITY") | _ids(
        candidates, "10-K", "ITEM 5"
    )
    original_item8 = _ids(candidates, "10-K", "FINANCIAL STATEMENTS") | _ids(
        candidates, "10-K", "ITEM 8"
    )
    amendment_item5 = _ids(candidates, "10-K/A", "MARKET FOR COMMON EQUITY") | _ids(
        candidates, "10-K/A", "ITEM 5"
    )
    kept_ids = {row["chunk_id"] for row in kept}
    dropped_ids = {row["chunk_id"] for row in dropped}
    if not original_item5 or not original_item8 or not amendment_item5:
        raise ValidationError("revision smoke could not find original Item 5/8 and amended Item 5")
    if not original_item5 <= dropped_ids:
        raise ValidationError("original Item 5 was not fully suppressed")
    if not (original_item8 | amendment_item5) <= kept_ids:
        raise ValidationError("unchanged Item 8 or amended Item 5 was incorrectly suppressed")


def import_and_validate(
    database_url: str,
    benchmark_database_url: str,
    pair: list[tuple[Any, ...]],
    fixture: dict[str, Any],
    *,
    batch_size: int,
) -> int:
    """Import into the isolated DB, assert composition, and prove benchmark stability."""
    before = benchmark_snapshot(benchmark_database_url)
    assert_clean_benchmark(before)
    assert_pristine_target(database_url)
    with DBLoader(dsn=database_url) as loader:
        loader.init_schema(str(SCHEMA_PATH))
        assert_empty_target(database_url)
        embedder = Embedder(batch_size=batch_size)
        loaded = _process_filings(
            pair,
            ["10-K", "10-K/A"],
            embedder,
            loader,
            skip_embed=False,
            skip_load=False,
        )
    assert_imported_pair(database_url, fixture)
    asyncio.run(_revision_composition_smoke(database_url))
    after = benchmark_snapshot(benchmark_database_url)
    assert_clean_benchmark(after)
    if after != before:
        raise ValidationError(
            f"benchmark database changed during isolated validation: {before} -> {after}"
        )
    return loaded


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse isolated amendment-fixture validation settings without side effects.

    中文：仅收集明确的隔离路径与数据库参数；安全边界和 fixture 身份在运行阶段 fail-closed 检查。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database-url", required=True, help="Existing isolated target DB URL")
    parser.add_argument("--download-dir", required=True, type=Path)
    parser.add_argument(
        "--benchmark-database-url",
        default=DATABASE_URL,
        help="Read-only public benchmark DB used for before/after invariants",
    )
    parser.add_argument("--fixture", type=Path, default=FIXTURE_PATH)
    parser.add_argument("--batch-size", type=int, default=EMBEDDING_BATCH_SIZE)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args(argv)
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    return args


def main(argv: Optional[list[str]] = None) -> int:
    """Validate the allowlisted amendment pair and return its CLI status.

    中文：入口拒绝运行在公开基准库或默认下载目录上，避免把评测修复误用于真实语料。
    """
    args = parse_args(argv)
    sec_user_agent = os.getenv("SEC_USER_AGENT", SEC_USER_AGENT)
    validate_isolation(
        args.database_url,
        args.download_dir,
        sec_user_agent,
        benchmark_database_url=args.benchmark_database_url,
    )
    fixture = load_fixture(args.fixture)
    args.download_dir.mkdir(parents=True, exist_ok=True)
    pair = prepare_pair(args.download_dir, fixture)
    logger.info("Prepared exact accessions: %s", ", ".join(meta[8] for meta in pair))
    if args.prepare_only:
        return 0

    loaded = import_and_validate(
        args.database_url,
        args.benchmark_database_url,
        pair,
        fixture,
        batch_size=args.batch_size,
    )
    logger.info("Isolated amendment validation complete: %d chunks loaded", loaded)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    raise SystemExit(main())
