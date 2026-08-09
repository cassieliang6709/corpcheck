#!/usr/bin/env python3
"""Snapshot exact SEC filing identities from a PostgreSQL corpus.

The utility is intentionally read-only and does not download filings.  Its
write-once output can later drive an exact raw-submission recovery step without
depending on mutable ticker/year discovery.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Optional, Sequence
from urllib.parse import urlsplit

import asyncpg

ACCESSION_RE = re.compile(r"^(\d{10})-(\d{2})-(\d{6})$")
CIK_RE = re.compile(r"^\d{1,10}$")
SCHEMA_VERSION = 1


class ManifestError(RuntimeError):
    """The corpus or requested output violates the snapshot contract."""


def validate_database_url(database_url: str) -> None:
    parsed = urlsplit(database_url)
    if parsed.scheme not in {"postgres", "postgresql"} or not parsed.hostname:
        raise ManifestError("--database-url must be an explicit PostgreSQL URL")
    if not parsed.path.lstrip("/"):
        raise ManifestError("--database-url must name a database")


def full_submission_url(cik: str, accession: str) -> str:
    """Return the deterministic SEC Archives URL for the raw submission."""
    return (
        "https://www.sec.gov/Archives/edgar/data/"
        f"{int(cik)}/{accession.replace('-', '')}/{accession}.txt"
    )


def _date_value(value: Any, field: str) -> str:
    if isinstance(value, dt.datetime) or not isinstance(value, dt.date):
        raise ManifestError(f"filing has invalid {field}")
    return value.isoformat()


def _required_text(row: Any, field: str) -> str:
    value = row[field]
    if not isinstance(value, str) or not value or value != value.strip():
        raise ManifestError(f"filing has missing or invalid {field}")
    return value


def normalize_filing(row: Any) -> dict[str, Any]:
    ticker = _required_text(row, "ticker")
    form = _required_text(row, "form")
    accession = _required_text(row, "accession")
    cik = _required_text(row, "cik")

    if ACCESSION_RE.fullmatch(accession) is None:
        raise ManifestError(f"filing has invalid accession: {accession!r}")
    if CIK_RE.fullmatch(cik) is None or int(cik) <= 0:
        raise ManifestError(f"filing {accession} has invalid CIK")

    source_url = _required_text(row, "source_url")
    parsed_source = urlsplit(source_url)
    if parsed_source.username is not None or parsed_source.password is not None:
        raise ManifestError(f"filing {accession} source_url contains credentials")
    if parsed_source.scheme != "https" or parsed_source.hostname not in {
        "sec.gov",
        "www.sec.gov",
    }:
        raise ManifestError(f"filing {accession} has non-SEC source_url")

    fiscal_year = row["fiscal_year"]
    if isinstance(fiscal_year, bool) or not isinstance(fiscal_year, int):
        raise ManifestError(f"filing {accession} has invalid fiscal_year")
    period = _required_text(row, "period")

    return {
        "ticker": ticker,
        "form": form,
        "fiscal_year": fiscal_year,
        "period": period,
        "filed_date": _date_value(row["filed_date"], "filed_date"),
        "period_of_report": _date_value(row["period_of_report"], "period_of_report"),
        "accession": accession,
        "cik": cik,
        "source_url": source_url,
        "full_submission_url": full_submission_url(cik, accession),
    }


def build_envelope(
    filing_rows: Sequence[Any],
    *,
    chunk_count: int,
    embedded_chunk_count: int,
) -> dict[str, Any]:
    if chunk_count < 0 or not 0 <= embedded_chunk_count <= chunk_count:
        raise ManifestError("corpus returned invalid chunk counts")

    filings = sorted(
        (normalize_filing(row) for row in filing_rows),
        key=lambda row: row["accession"],
    )
    if not filings:
        raise ManifestError("corpus contains no filings")
    accessions = [row["accession"] for row in filings]
    if len(accessions) != len(set(accessions)):
        raise ManifestError("corpus contains duplicate accession numbers")

    payload = {
        "schema_version": SCHEMA_VERSION,
        "corpus_counts": {
            "ticker_count": len({row["ticker"] for row in filings}),
            "filing_count": len(filings),
            "chunk_count": chunk_count,
            "embedded_chunk_count": embedded_chunk_count,
        },
        "filings": filings,
    }
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        **payload,
        "digest": {
            "algorithm": "sha256",
            "payload_sha256": hashlib.sha256(canonical).hexdigest(),
        },
    }


async def snapshot_corpus(pool: Any) -> dict[str, Any]:
    """Read one transactionally consistent corpus snapshot."""
    async with pool.acquire() as connection:
        async with connection.transaction(isolation="repeatable_read", readonly=True):
            filing_rows = await connection.fetch(
                """
                SELECT ticker, filing_type AS form, fiscal_year, period,
                       filed_date, period_of_report,
                       accession_number AS accession, cik, source_url
                FROM filings
                ORDER BY accession_number
                """
            )
            counts = await connection.fetchrow(
                """
                SELECT COUNT(*) AS chunk_count,
                       COUNT(*) FILTER (WHERE embedding IS NOT NULL)
                           AS embedded_chunk_count
                FROM chunks
                """
            )
    return build_envelope(
        filing_rows,
        chunk_count=int(counts["chunk_count"]),
        embedded_chunk_count=int(counts["embedded_chunk_count"]),
    )


def render_manifest(envelope: dict[str, Any]) -> str:
    return json.dumps(envelope, indent=2, ensure_ascii=False, sort_keys=True) + "\n"


def write_once(path: Path, content: str) -> None:
    encoded = content.encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(encoded)
    except FileExistsError as exc:
        if path.is_file() and path.read_bytes() == encoded:
            return
        raise ManifestError(f"refusing to overwrite different existing output: {path}") from exc


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database-url", required=True)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


async def run(database_url: str) -> dict[str, Any]:
    validate_database_url(database_url)
    pool = await asyncpg.create_pool(
        dsn=database_url,
        min_size=1,
        max_size=1,
        server_settings={"default_transaction_read_only": "on"},
    )
    try:
        return await snapshot_corpus(pool)
    finally:
        await pool.close()


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    envelope = asyncio.run(run(args.database_url))
    write_once(args.output, render_manifest(envelope))
    print(
        f"Snapshotted {envelope['corpus_counts']['filing_count']} filings: {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
