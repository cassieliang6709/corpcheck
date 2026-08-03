"""
Backfill company metadata into the database for arbitrary dumps.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from corpcheck.ingestion.config import DATABASE_URL
from corpcheck.ingestion.loaders.db_loader import _connect
from corpcheck.ingestion.metadata import (
    is_unresolved_company_name,
    is_unresolved_sector,
    resolve_company_metadata,
)


def _fetch_all_tickers(conn) -> list[str]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT DISTINCT ticker
            FROM (
                SELECT ticker FROM companies
                UNION
                SELECT ticker FROM chunks
                UNION
                SELECT ticker FROM filings
                UNION
                SELECT ticker FROM market_data
                UNION
                SELECT ticker FROM financials
            ) tickers
            WHERE ticker IS NOT NULL AND ticker <> ''
            ORDER BY ticker
            """
        )
        return [row[0] for row in cur.fetchall()]


def main() -> None:
    conn = _connect(DATABASE_URL)
    inserted_companies = 0
    updated_companies = 0
    updated_chunks = 0
    try:
        tickers = _fetch_all_tickers(conn)
        with conn.cursor() as cur:
            for ticker in tickers:
                metadata = resolve_company_metadata(ticker)
                cur.execute(
                    """
                    SELECT name, sector, industry, market_cap, description
                    FROM companies
                    WHERE ticker = %s
                    """,
                    (ticker,),
                )
                row = cur.fetchone()

                if row is None:
                    cur.execute(
                        """
                        INSERT INTO companies (
                            ticker, name, sector, industry, market_cap, description, updated_at
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, NOW())
                        """,
                        (
                            ticker,
                            metadata["name"],
                            metadata["sector"],
                            metadata["industry"],
                            metadata["market_cap"],
                            metadata["description"],
                        ),
                    )
                    inserted_companies += cur.rowcount
                else:
                    current_name, current_sector, current_industry, current_market_cap, current_description = row
                    next_name = current_name
                    next_sector = current_sector
                    next_industry = current_industry
                    next_market_cap = current_market_cap
                    next_description = current_description

                    if is_unresolved_company_name(current_name, ticker) and metadata["name"] != ticker:
                        next_name = metadata["name"]
                    if is_unresolved_sector(current_sector) and not is_unresolved_sector(metadata["sector"]):
                        next_sector = metadata["sector"]
                    if current_industry in (None, "") and metadata["industry"]:
                        next_industry = metadata["industry"]
                    if current_market_cap is None and metadata["market_cap"] is not None:
                        next_market_cap = metadata["market_cap"]
                    if current_description in (None, "") and metadata["description"]:
                        next_description = metadata["description"]

                    cur.execute(
                        """
                        UPDATE companies
                        SET name = %s,
                            sector = %s,
                            industry = %s,
                            market_cap = %s,
                            description = %s,
                            updated_at = NOW()
                        WHERE ticker = %s
                          AND (
                              name IS DISTINCT FROM %s OR
                              sector IS DISTINCT FROM %s OR
                              industry IS DISTINCT FROM %s OR
                              market_cap IS DISTINCT FROM %s OR
                              description IS DISTINCT FROM %s
                          )
                        """,
                        (
                            next_name,
                            next_sector,
                            next_industry,
                            next_market_cap,
                            next_description,
                            ticker,
                            next_name,
                            next_sector,
                            next_industry,
                            next_market_cap,
                            next_description,
                        ),
                    )
                    updated_companies += cur.rowcount

                cur.execute(
                    """
                    UPDATE chunks
                    SET sector = %s
                    WHERE ticker = %s
                      AND sector IS DISTINCT FROM %s
                    """,
                    (metadata["sector"], ticker, metadata["sector"]),
                )
                updated_chunks += cur.rowcount
        conn.commit()
    finally:
        conn.close()

    print(
        "Inserted {inserted} companies, updated {updated_companies} companies, "
        "updated {updated_chunks} chunk rows.".format(
            inserted=inserted_companies,
            updated_companies=updated_companies,
            updated_chunks=updated_chunks,
        )
    )


if __name__ == "__main__":
    main()
