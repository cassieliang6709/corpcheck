import datetime as dt
import unittest
from pathlib import Path

from corpcheck.ingestion.downloaders import sec_downloader
from corpcheck.ingestion.downloaders.sec_downloader import (
    SECDownloader,
    _download_limit_for_range,
    _infer_fiscal_year,
    _infer_period,
    _parse_submission_metadata,
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


class _FakeDownloader:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict]] = []

    def get(self, filing_type: str, ticker: str, **kwargs) -> int:
        self.calls.append((filing_type, ticker, kwargs))
        return 0


def _write_submission(
    download_dir: Path,
    accession: str,
    filing_type: str,
) -> Path:
    accession_dir = (
        download_dir / "sec-edgar-filings" / "TEST" / "10-K" / accession
    )
    accession_dir.mkdir(parents=True)
    submission = accession_dir / "full-submission.txt"
    submission.write_text(
        "\n".join(
            (
                f"CONFORMED SUBMISSION TYPE: {filing_type}",
                "CENTRAL INDEX KEY: 0000123456",
                "FILED AS OF DATE: 20240315",
                "CONFORMED PERIOD OF REPORT: 20230101",
                "FISCAL YEAR END: 0101",
            )
        ),
        encoding="utf-8",
    )
    return submission


def test_download_can_explicitly_include_amendments(monkeypatch, tmp_path) -> None:
    fake = _FakeDownloader()
    monkeypatch.setattr(sec_downloader, "_make_downloader", lambda _: fake)
    downloader = SECDownloader(download_dir=str(tmp_path))

    downloader.download(["TEST"], ["10-K"], [2022], include_amends=True)

    assert len(fake.calls) == 1
    filing_type, ticker, kwargs = fake.calls[0]
    assert (filing_type, ticker) == ("10-K", "TEST")
    assert kwargs["include_amends"] is True


def test_default_download_still_excludes_amendments(monkeypatch, tmp_path) -> None:
    fake = _FakeDownloader()
    monkeypatch.setattr(sec_downloader, "_make_downloader", lambda _: fake)
    downloader = SECDownloader(download_dir=str(tmp_path))

    downloader.download(["TEST"], ["10-K"], [2022])

    assert fake.calls[0][2]["include_amends"] is False


def test_submission_header_preserves_original_and_amended_filing_types(tmp_path) -> None:
    original = _write_submission(tmp_path, "0000123456-24-000001", "10-K")
    amended = _write_submission(tmp_path, "0000123456-24-000002", "10-K/A")
    downloader = object.__new__(SECDownloader)
    downloader.download_dir = str(tmp_path)

    assert _parse_submission_metadata(original.parent)["filing_type"] == "10-K"
    assert _parse_submission_metadata(amended.parent)["filing_type"] == "10-K/A"

    metadata = downloader._collect_metadata(["TEST"], ["10-K"], [2022])
    by_accession = {row[8]: row for row in metadata}

    assert by_accession["0000123456-24-000001"][1:4] == ("10-K", 2022, "annual")
    assert by_accession["0000123456-24-000002"][1:4] == ("10-K/A", 2022, "annual")


if __name__ == "__main__":
    unittest.main()
