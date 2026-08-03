import datetime as dt
import unittest

from corpcheck.ingestion.downloaders.sec_downloader import (
    _download_limit_for_range,
    _infer_fiscal_year,
    _infer_period,
)


class SECDownloaderLimitTests(unittest.TestCase):
    def test_10q_limit_expands_for_long_backfills(self) -> None:
        years = list(range(2018, 2026))
        self.assertGreaterEqual(_download_limit_for_range("10-Q", years), 36)

    def test_10k_limit_never_drops_below_default_floor(self) -> None:
        years = [2024]
        self.assertEqual(_download_limit_for_range("10-K", years), 20)

    def test_10k_fiscal_year_uses_report_year_for_week_based_year_end(self) -> None:
        # AMD's FY2023 10-K reports period-of-report 2023-12-30 with fiscal-year-end 12/28.
        # The filing was previously mislabeled as FY2024 because the old logic treated any
        # report date after MM/DD year-end as the next fiscal year.
        self.assertEqual(
            _infer_fiscal_year("10-K", dt.date(2023, 12, 30), "1228"),
            2023,
        )

    def test_10q_fiscal_year_still_rolls_against_fiscal_year_end(self) -> None:
        self.assertEqual(
            _infer_fiscal_year("10-Q", dt.date(2024, 9, 28), "1228"),
            2024,
        )

    def test_10q_period_inference_respects_custom_fiscal_year_end(self) -> None:
        self.assertEqual(
            _infer_period("10-Q", dt.date(2024, 6, 29), "1228"),
            "Q2",
        )


if __name__ == "__main__":
    unittest.main()
