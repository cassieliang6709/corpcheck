"""
Backfill SEC filing metadata and chunk-level data signals for an existing DB.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from psycopg2.extras import execute_batch

from corpcheck.ingestion.chunk_features import compute_chunk_features
from corpcheck.ingestion.config import DATABASE_URL
from corpcheck.ingestion.downloaders.sec_downloader import (
    _build_source_url,
    _infer_fiscal_year,
    _infer_period,
    _parse_submission_metadata,
    _parse_yyyymmdd,
)
from corpcheck.ingestion.loaders.db_loader import _connect


def _load_filing_rows(conn) -> list[dict]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, ticker, filing_type, fiscal_year, period, filed_date, local_path, source_url
            FROM filings
            WHERE local_path IS NOT NULL
              AND local_path <> ''
              AND filing_type IN ('10-K', '10-Q')
            ORDER BY id
            """
        )
        cols = [desc[0] for desc in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]


def _prepare_filing_updates(rows: list[dict]) -> list[dict]:
    updates: list[dict] = []
    for row in rows:
        local_path = Path(row["local_path"])
        accession_dir = local_path.parent
        accession = accession_dir.name
        metadata = _parse_submission_metadata(accession_dir)
        report_date = _parse_yyyymmdd(metadata.get("period_of_report"))
        filed_date = _parse_yyyymmdd(metadata.get("filed_date")) or row["filed_date"]
        cik = metadata.get("cik") or ""
        if report_date is None:
            continue

        new_fiscal_year = _infer_fiscal_year(
            row["filing_type"],
            report_date,
            metadata.get("fiscal_year_end"),
        )
        new_period = _infer_period(
            row["filing_type"],
            report_date,
            metadata.get("fiscal_year_end"),
        )
        new_source_url = _build_source_url(cik, accession) if cik else (row["source_url"] or "")
        updates.append(
            {
                "id": row["id"],
                "ticker": row["ticker"],
                "filing_type": row["filing_type"],
                "old_fiscal_year": row["fiscal_year"],
                "old_period": row["period"],
                "new_fiscal_year": new_fiscal_year,
                "new_period": new_period,
                "filed_date": filed_date,
                "period_of_report": report_date,
                "accession_number": accession,
                "cik": cik,
                "source_url": new_source_url,
            }
        )

    target_keys = [
        (row["ticker"], row["filing_type"], row["new_fiscal_year"], row["new_period"])
        for row in updates
    ]
    duplicates = [key for key, count in Counter(target_keys).items() if count > 1]
    if duplicates:
        raise RuntimeError(f"Duplicate corrected filing keys detected: {duplicates[:5]}")
    return updates


def _apply_filing_updates(conn, updates: list[dict]) -> int:
    changed_keys = [
        row
        for row in updates
        if row["old_fiscal_year"] != row["new_fiscal_year"] or row["old_period"] != row["new_period"]
    ]

    with conn.cursor() as cur:
        if changed_keys:
            execute_batch(
                cur,
                """
                UPDATE filings
                SET fiscal_year = fiscal_year + 100,
                    period = period || '__tmp__'
                WHERE id = %(id)s
                """,
                changed_keys,
                page_size=500,
            )

        execute_batch(
            cur,
            """
            UPDATE filings
            SET fiscal_year = %(new_fiscal_year)s,
                period = %(new_period)s,
                filed_date = %(filed_date)s,
                period_of_report = %(period_of_report)s,
                accession_number = %(accession_number)s,
                cik = %(cik)s,
                source_url = %(source_url)s
            WHERE id = %(id)s
            """,
            updates,
            page_size=500,
        )

        execute_batch(
            cur,
            """
            UPDATE chunks
            SET fiscal_year = %(new_fiscal_year)s,
                period = %(new_period)s,
                filed_date = %(filed_date)s,
                source_url = %(source_url)s
            WHERE filing_id = %(id)s
            """,
            updates,
            page_size=500,
        )

    return len(updates)


def _backfill_chunk_features(conn) -> int:
    total = 0
    last_id = 0
    while True:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, section_name, content
                FROM chunks
                WHERE id > %s
                ORDER BY id
                LIMIT 2000
                """,
                (last_id,),
            )
            rows = cur.fetchall()
        if not rows:
            break

        payload = []
        for chunk_id, section_name, content in rows:
            features = compute_chunk_features(section_name, content)
            payload.append({"id": chunk_id, **features})

        with conn.cursor() as cur:
            execute_batch(
                cur,
                """
                UPDATE chunks
                SET numeric_token_count = %(numeric_token_count)s,
                    number_density = %(number_density)s,
                    data_signal_score = %(data_signal_score)s,
                    is_quantitative = %(is_quantitative)s
                WHERE id = %(id)s
                """,
                payload,
                page_size=2000,
            )
        total += len(payload)
        last_id = rows[-1][0]
        conn.commit()
    return total


def main() -> None:
    conn = _connect(DATABASE_URL)
    try:
        filing_rows = _load_filing_rows(conn)
        updates = _prepare_filing_updates(filing_rows)
        _apply_filing_updates(conn, updates)
        conn.commit()
        chunk_count = _backfill_chunk_features(conn)
        conn.commit()
    finally:
        conn.close()

    print(f"Updated {len(updates)} filings and recomputed features for {chunk_count} chunks.")


if __name__ == "__main__":
    main()
