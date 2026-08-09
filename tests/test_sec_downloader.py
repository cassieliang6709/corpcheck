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

    def test_10k_year_that_closes_just_after_new_year_keeps_the_prior_year(self) -> None:
        # JNJ runs a 52/53-week calendar tracking 31 December and reports
        # FISCAL YEAR END 0101; FY2022 closed on 2023-01-01. Labelling it 2023
        # collides with the real FY2023 10-K and the upsert drops one of them.
        for report_date, expected in (
            (dt.date(2021, 1, 3), 2020),
            (dt.date(2022, 1, 2), 2021),
            (dt.date(2023, 1, 1), 2022),
        ):
            with self.subTest(report_date=report_date):
                self.assertEqual(_infer_fiscal_year("10-K", report_date, "0101"), expected)

    def test_10k_late_january_year_end_is_named_after_its_closing_year(self) -> None:
        # Retail calendars (WMT 0131, CRM 0131) name the fiscal year after the
        # calendar year it closes in, so they must NOT be shifted back.
        self.assertEqual(_infer_fiscal_year("10-K", dt.date(2024, 1, 31), "0131"), 2024)
        # TGT/TJX can close in early February for the same reason.
        self.assertEqual(_infer_fiscal_year("10-K", dt.date(2024, 2, 3), "0201"), 2024)

    def test_10q_of_a_year_turning_over_new_year_keeps_the_calendar_year(self) -> None:
        # JNJ's Q3 FY2022 ended 2022-10-02: still fiscal 2022, not 2023.
        self.assertEqual(_infer_fiscal_year("10-Q", dt.date(2022, 10, 2), "0101"), 2022)
        self.assertEqual(_infer_period("10-Q", dt.date(2022, 10, 2), "0101"), "Q3")
        self.assertEqual(_infer_period("10-Q", dt.date(2022, 4, 3), "0101"), "Q1")

    def test_10q_of_a_genuinely_offset_year_still_rolls_forward(self) -> None:
        # WMT's fiscal year ends 31 January, so 2024-04-30 falls in FY2025 Q1.
        self.assertEqual(_infer_fiscal_year("10-Q", dt.date(2024, 4, 30), "0131"), 2025)
        self.assertEqual(_infer_period("10-Q", dt.date(2024, 4, 30), "0131"), "Q1")
        self.assertEqual(_infer_period("10-Q", dt.date(2024, 10, 31), "0131"), "Q3")

    def test_malformed_fiscal_year_end_falls_back_to_calendar_quarters(self) -> None:
        self.assertEqual(_infer_period("10-Q", dt.date(2024, 6, 30), "9999"), "Q2")
        self.assertEqual(_infer_period("10-Q", dt.date(2024, 6, 30), None), "Q2")


if __name__ == "__main__":
    unittest.main()
