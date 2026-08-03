import os
import unittest
from pathlib import Path

import psycopg2

from corpcheck.ingestion.config import SEC_DOWNLOAD_DIR
from corpcheck.ingestion.downloaders.sec_downloader import SECDownloader


TEST_DATABASE_URL = os.getenv(
    "TEST_DATABASE_URL",
    os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/financial_rag"),
)

SEC_TICKER_ALIASES = {
    "0001038357": "PXD",
}


class DatabaseAcceptanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.conn = psycopg2.connect(TEST_DATABASE_URL)
        cls.conn.autocommit = True

    @classmethod
    def tearDownClass(cls) -> None:
        cls.conn.close()

    def fetchone(self, sql: str, params: tuple = ()) -> tuple:
        with self.conn.cursor() as cur:
            cur.execute(sql, params)
            row = cur.fetchone()
        assert row is not None
        return row

    def fetchall(self, sql: str, params: tuple = ()) -> list[tuple]:
        with self.conn.cursor() as cur:
            cur.execute(sql, params)
            return list(cur.fetchall())

    def test_database_meets_10k_volume_requirements(self) -> None:
        total_rows = self.fetchone(
            """
            SELECT
                (SELECT COUNT(*) FROM companies) +
                (SELECT COUNT(*) FROM filings) +
                (SELECT COUNT(*) FROM chunks) +
                (SELECT COUNT(*) FROM market_data) +
                (SELECT COUNT(*) FROM financials) +
                (SELECT COUNT(*) FROM macro_indicators) +
                (SELECT COUNT(*) FROM news_articles) +
                (SELECT COUNT(*) FROM earnings_transcripts) +
                (SELECT COUNT(*) FROM transcript_chunks)
            """
        )[0]
        chunk_rows = self.fetchone("SELECT COUNT(*) FROM chunks")[0]

        self.assertGreaterEqual(total_rows, 10_000)
        self.assertGreaterEqual(chunk_rows, 10_000)

    def test_all_chunk_embeddings_are_present(self) -> None:
        total_chunks, embedded_chunks = self.fetchone(
            "SELECT COUNT(*), COUNT(embedding) FROM chunks"
        )
        self.assertEqual(embedded_chunks, total_chunks)

    def test_no_duplicate_chunk_groups_exist(self) -> None:
        duplicate_groups = self.fetchone(
            """
            WITH duplicate_groups AS (
                SELECT filing_id, chunk_index
                FROM chunks
                GROUP BY filing_id, chunk_index
                HAVING COUNT(*) > 1
            )
            SELECT COUNT(*) FROM duplicate_groups
            """
        )[0]
        self.assertEqual(duplicate_groups, 0)

    def test_company_metadata_is_backfilled(self) -> None:
        non_numeric_ticker_like_names = self.fetchone(
            """
            SELECT COUNT(*)
            FROM v_chunk_search
            WHERE company_name = ticker
              AND ticker !~ '^[0-9]+$'
            """
        )[0]
        aliased_numeric_ticker_placeholders = self.fetchone(
            """
            SELECT COUNT(DISTINCT ticker)
            FROM v_chunk_search
            WHERE company_name = ticker
              AND ticker = ANY(%s)
            """,
            (list(SEC_TICKER_ALIASES.keys()),),
        )[0]
        unresolved_company_sectors = self.fetchone(
            "SELECT COUNT(*) FROM companies WHERE lower(coalesce(sector, '')) IN ('', 'unknown')"
        )[0]
        unresolved_chunk_sectors = self.fetchone(
            "SELECT COUNT(*) FROM chunks WHERE lower(coalesce(sector, '')) IN ('', 'unknown')"
        )[0]
        missing_company_links = self.fetchone(
            """
            SELECT COUNT(DISTINCT c.ticker)
            FROM chunks c
            LEFT JOIN companies co ON co.ticker = c.ticker
            WHERE co.ticker IS NULL
            """
        )[0]

        self.assertEqual(non_numeric_ticker_like_names, 0)
        self.assertLessEqual(aliased_numeric_ticker_placeholders, len(SEC_TICKER_ALIASES))
        self.assertEqual(unresolved_company_sectors, 0)
        self.assertEqual(unresolved_chunk_sectors, 0)
        self.assertEqual(missing_company_links, 0)

    def test_validated_sec_subset_is_intact(self) -> None:
        validated_tickers = ("AAPL", "JPM", "UNH", "XOM")

        filing_count, filing_dates, filing_urls = self.fetchone(
            """
            SELECT COUNT(*), COUNT(filed_date), COUNT(source_url)
            FROM filings
            WHERE ticker = ANY(%s)
              AND fiscal_year = 2023
              AND filing_type IN ('10-K', '10-Q')
            """,
            (list(validated_tickers),),
        )
        period_rows = self.fetchone(
            """
            SELECT COUNT(*)
            FROM filings
            WHERE ticker = ANY(%s)
              AND fiscal_year = 2023
              AND (
                    (filing_type = '10-K' AND period = 'annual') OR
                    (filing_type = '10-Q' AND period IN ('Q1', 'Q2', 'Q3'))
                  )
            """,
            (list(validated_tickers),),
        )[0]
        sec_chunk_count, sec_embedded_count = self.fetchone(
            """
            SELECT COUNT(*), COUNT(embedding)
            FROM chunks
            WHERE ticker = ANY(%s)
              AND fiscal_year = 2023
              AND filing_type IN ('10-K', '10-Q')
            """,
            (list(validated_tickers),),
        )
        min_chunks_per_filing, filings_with_tables, filings_with_narrative = self.fetchone(
            """
            WITH per_filing AS (
                SELECT
                    filing_id,
                    COUNT(*) AS chunk_count,
                    COUNT(*) FILTER (WHERE content_kind = 'table') AS table_chunks,
                    COUNT(*) FILTER (WHERE content_kind = 'narrative') AS narrative_chunks
                FROM chunks
                WHERE ticker = ANY(%s)
                  AND fiscal_year = 2023
                  AND filing_type IN ('10-K', '10-Q')
                GROUP BY filing_id
            )
            SELECT
                MIN(chunk_count),
                COUNT(*) FILTER (WHERE table_chunks > 0),
                COUNT(*) FILTER (WHERE narrative_chunks > 0)
            FROM per_filing
            """,
            (list(validated_tickers),),
        )

        self.assertEqual(filing_count, 16)
        self.assertEqual(filing_dates, 16)
        self.assertEqual(filing_urls, 16)
        self.assertEqual(period_rows, 16)
        self.assertEqual(sec_embedded_count, sec_chunk_count)
        self.assertGreaterEqual(sec_chunk_count, 1_500)
        self.assertGreaterEqual(min_chunks_per_filing, 3)
        self.assertEqual(filings_with_tables, filing_count)
        self.assertEqual(filings_with_narrative, filing_count)

    def test_local_sec_disk_coverage_is_complete(self) -> None:
        local_root = Path(SEC_DOWNLOAD_DIR) / "sec-edgar-filings"
        local_dirs = sorted(path.name for path in local_root.iterdir() if path.is_dir())
        years = list(range(2018, 2026))
        local_metas = SECDownloader()._collect_metadata(local_dirs, ["10-K", "10-Q"], years)
        local_keys = {
            (SEC_TICKER_ALIASES.get(ticker, ticker), filing_type, fiscal_year, period)
            for ticker, filing_type, fiscal_year, period, *_ in local_metas
        }
        local_tickers = sorted({SEC_TICKER_ALIASES.get(ticker, ticker) for ticker in local_dirs})

        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT ticker, filing_type, fiscal_year, period
                FROM filings
                WHERE ticker = ANY(%s)
                """,
                (local_tickers,),
            )
            db_keys = set(cur.fetchall())

        self.assertGreater(len(local_keys), 0)
        self.assertEqual(local_keys - db_keys, set())
        self.assertEqual(db_keys - local_keys, set())

    def test_sec_filings_have_complete_metadata(self) -> None:
        missing_meta_count = self.fetchone(
            """
            SELECT COUNT(*)
            FROM filings
            WHERE filing_type IN ('10-K', '10-Q')
              AND (
                    filed_date IS NULL OR
                    period_of_report IS NULL OR
                    nullif(accession_number, '') IS NULL OR
                    nullif(cik, '') IS NULL OR
                    nullif(source_url, '') IS NULL
                  )
            """
        )[0]
        self.assertEqual(missing_meta_count, 0)

    def test_sec_fiscal_year_and_period_are_consistent(self) -> None:
        bad_10k_fiscal_years = self.fetchone(
            """
            SELECT COUNT(*)
            FROM filings
            WHERE filing_type = '10-K'
              AND period_of_report IS NOT NULL
              AND EXTRACT(YEAR FROM period_of_report) <> fiscal_year
            """
        )[0]
        bad_10q_periods = self.fetchone(
            """
            SELECT COUNT(*)
            FROM filings
            WHERE filing_type = '10-Q'
              AND period NOT IN ('Q1', 'Q2', 'Q3')
            """
        )[0]

        self.assertEqual(bad_10k_fiscal_years, 0)
        self.assertEqual(bad_10q_periods, 0)

    def test_chunk_metadata_matches_parent_filing(self) -> None:
        inconsistent_chunk_count = self.fetchone(
            """
            SELECT COUNT(*)
            FROM chunks c
            JOIN filings f ON f.id = c.filing_id
            WHERE c.ticker IS DISTINCT FROM f.ticker
               OR c.filing_type IS DISTINCT FROM f.filing_type
               OR c.fiscal_year IS DISTINCT FROM f.fiscal_year
               OR c.period IS DISTINCT FROM f.period
               OR c.filed_date IS DISTINCT FROM f.filed_date
               OR c.source_url IS DISTINCT FROM f.source_url
            """
        )[0]
        self.assertEqual(inconsistent_chunk_count, 0)

    def test_search_support_columns_are_populated(self) -> None:
        missing_search_columns = self.fetchone(
            """
            SELECT COUNT(*)
            FROM chunks
            WHERE content_tsv IS NULL
               OR embedding IS NULL
               OR numeric_token_count IS NULL
               OR number_density IS NULL
               OR data_signal_score IS NULL
               OR is_quantitative IS NULL
            """
        )[0]
        self.assertEqual(missing_search_columns, 0)

    def test_quantitative_sections_score_above_narrative_sections(self) -> None:
        rows = self.fetchall(
            """
            SELECT section_name, AVG(data_signal_score) AS avg_score
            FROM chunks
            WHERE section_name IN (
                'Financial Statements',
                'Selected Financial Data',
                'Business Description',
                'Risk Factors'
            )
            GROUP BY section_name
            """
        )
        score_map = {section_name: float(avg_score) for section_name, avg_score in rows}
        self.assertGreater(score_map["Financial Statements"], score_map["Business Description"])
        self.assertGreater(score_map["Selected Financial Data"], score_map["Risk Factors"])

    def test_exact_company_name_is_filterable_in_search_view(self) -> None:
        amd_company_name, amd_chunk_count = self.fetchone(
            """
            SELECT company_name, COUNT(*)
            FROM v_chunk_search
            WHERE lower(company_name) = lower('Advanced Micro Devices, Inc.')
              AND ticker = 'AMD'
              AND filing_type = '10-K'
              AND fiscal_year = 2023
            GROUP BY company_name
            """
        )
        self.assertEqual(amd_company_name, "Advanced Micro Devices, Inc.")
        self.assertGreater(amd_chunk_count, 0)

    def test_retrieval_view_covers_all_available_chunk_sources(self) -> None:
        retrieval_source_types = {
            source_type
            for (source_type,) in self.fetchall(
                "SELECT DISTINCT source_type FROM v_retrieval_chunks"
            )
        }
        transcript_chunk_count = self.fetchone(
            "SELECT COUNT(*) FROM transcript_chunks"
        )[0]

        self.assertIn("sec", retrieval_source_types)
        self.assertIn("news", retrieval_source_types)
        if transcript_chunk_count > 0:
            self.assertIn("transcript", retrieval_source_types)

    def test_structured_chunk_metadata_columns_are_populated(self) -> None:
        missing_structured_metadata = self.fetchone(
            """
            SELECT COUNT(*)
            FROM v_retrieval_chunks
            WHERE content_kind IS NULL
               OR nullif(chunk_strategy, '') IS NULL
               OR structure_meta IS NULL
            """
        )[0]
        self.assertEqual(missing_structured_metadata, 0)

    def test_financial_statement_table_chunks_exist(self) -> None:
        total_table_chunks = self.fetchone(
            "SELECT COUNT(*) FROM chunks WHERE content_kind = 'table'"
        )[0]
        financial_statement_tables = self.fetchone(
            """
            SELECT COUNT(*)
            FROM chunks
            WHERE content_kind = 'table'
              AND section_name = 'Financial Statements'
            """
        )[0]

        self.assertGreater(total_table_chunks, 0)
        self.assertGreater(financial_statement_tables, 0)


if __name__ == "__main__":
    unittest.main()
