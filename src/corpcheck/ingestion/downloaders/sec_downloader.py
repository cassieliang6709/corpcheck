"""
SEC EDGAR downloader
=====================
Downloads 10-K, 10-Q, and 8-K filings for a list of tickers using the
sec-edgar-downloader library, then returns metadata tuples for downstream
processing.

Rate limiting is enforced via an asyncio Semaphore (max 10 req/s as required
by EDGAR's fair-use policy).
"""

from __future__ import annotations

import asyncio
import datetime as dt
import logging
import os
import re
import time
from pathlib import Path
from typing import Generator, Optional

from sec_edgar_downloader import Downloader

from corpcheck.ingestion.config import (
    SEC_DOWNLOAD_DIR,
    SEC_MAX_REQUESTS_PER_SECOND,
    SEC_USER_AGENT,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type alias for filing metadata tuples returned to the pipeline
# ---------------------------------------------------------------------------
FilingMeta = tuple[
    str,
    str,
    int,
    str,
    Path,
    str,
    Optional[dt.date],
    Optional[dt.date],
    str,
    str,
]
#              ticker  type  fiscal_year  period  local_path  source_url  filed_date
#              period_of_report  accession  cik


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _make_downloader(download_dir: str) -> Downloader:
    """Create an EDGAR Downloader pointed at *download_dir*."""
    # sec-edgar-downloader >= 0.5 accepts a company name + email for user-agent
    parts = SEC_USER_AGENT.split()
    company = parts[0] if parts else "financial-rag"
    email = parts[1] if len(parts) > 1 else "user@example.com"
    return Downloader(company, email, download_dir)


def _filing_root(download_dir: str, ticker: str, filing_type: str) -> Path:
    """Return the directory where sec-edgar-downloader stores filings."""
    return Path(download_dir) / "sec-edgar-filings" / ticker / filing_type


def _iter_filing_paths(
    root: Path,
) -> Generator[tuple[Path, str], None, None]:
    """
    Walk *root* and yield (document_path, accession_number) for every
    primary document (htm/html/txt) inside each accession sub-directory.
    """
    if not root.exists():
        return
    for accession_dir in sorted(root.iterdir()):
        if not accession_dir.is_dir():
            continue
        accession = accession_dir.name
        # sec-edgar-downloader places the primary doc as 'full-submission.txt'
        # or individual htm files.  We prefer the largest .htm/.html file.
        candidates = sorted(
            list(accession_dir.glob("*.htm")) + list(accession_dir.glob("*.html")),
            key=lambda p: p.stat().st_size,
            reverse=True,
        )
        if not candidates:
            # Fall back to the full-submission text file
            full = accession_dir / "full-submission.txt"
            if full.exists():
                candidates = [full]
        if candidates:
            yield candidates[0], accession


def _build_source_url(cik: str, accession: str) -> str:
    """Construct the EDGAR viewer URL for a given CIK + accession number."""
    accession_clean = accession.replace("-", "")
    cik_clean = str(int(cik))
    return (
        f"https://www.sec.gov/Archives/edgar/data/"
        f"{cik_clean}/{accession_clean}/{accession}-index.htm"
    )


def _download_limit_for_range(filing_type: str, years: list[int]) -> int:
    """
    Estimate a safe SEC download limit for the requested filing window.

    ``sec-edgar-downloader`` truncates results at ``limit`` per
    ticker/filing-type request. For multi-year 10-Q backfills, a hard-coded
    limit of 20 silently misses filings once the window exceeds roughly
    6 years. We intentionally over-allocate here so 2018-2025 local backfills
    collect the full on-disk set in one pass.
    """
    year_count = max(1, len(set(years)))
    if filing_type == "10-Q":
        return max(20, year_count * 4 + 4)
    if filing_type == "10-K":
        return max(20, year_count * 2 + 2)
    return max(20, year_count * 8)


def _parse_submission_metadata(accession_dir: Path) -> dict[str, str]:
    """
    Parse SEC header metadata from ``full-submission.txt``.
    """
    submission_path = accession_dir / "full-submission.txt"
    if not submission_path.exists():
        return {}

    try:
        text = submission_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return {}

    header = text[:8000]
    patterns = {
        "cik": r"CENTRAL INDEX KEY:\s+([0-9]+)",
        "filed_date": r"FILED AS OF DATE:\s+(\d{8})",
        "period_of_report": r"CONFORMED PERIOD OF REPORT:\s+(\d{8})",
        "fiscal_year_end": r"FISCAL YEAR END:\s+(\d{4})",
    }

    metadata: dict[str, str] = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, header)
        if match:
            metadata[key] = match.group(1)
    return metadata


def _parse_yyyymmdd(raw: str | None) -> dt.date | None:
    """Convert ``YYYYMMDD`` strings to ``date`` objects."""
    if not raw or len(raw) != 8 or not raw.isdigit():
        return None
    return dt.datetime.strptime(raw, "%Y%m%d").date()


def _infer_fiscal_year(
    filing_type: str,
    report_date: dt.date | None,
    fiscal_year_end_mmdd: str | None,
) -> int:
    """
    Infer the filing's fiscal year from period-of-report and fiscal year end.
    """
    if report_date is None:
        return 0

    # Annual filings should align to the reported fiscal period, not the
    # subsequent filing year. This avoids mislabeling 52/53-week calendars
    # like AMD's FY2023 10-K (period-of-report 2023-12-30, filed 2024-01-31)
    # as FY2024.
    if filing_type == "10-K":
        return report_date.year

    if not fiscal_year_end_mmdd or len(fiscal_year_end_mmdd) != 4 or not fiscal_year_end_mmdd.isdigit():
        return report_date.year

    fy_end_month = int(fiscal_year_end_mmdd[:2])
    fy_end_day = int(fiscal_year_end_mmdd[2:])

    if (report_date.month, report_date.day) > (fy_end_month, fy_end_day):
        return report_date.year + 1
    return report_date.year


def _infer_period(
    filing_type: str,
    report_date: dt.date | None,
    fiscal_year_end_mmdd: str | None,
) -> str:
    """
    Infer the period label for the filing.
    """
    if filing_type == "10-K":
        return "annual"
    if filing_type == "8-K":
        return "event"
    if filing_type != "10-Q" or report_date is None:
        return "unknown"

    if not fiscal_year_end_mmdd or len(fiscal_year_end_mmdd) != 4 or not fiscal_year_end_mmdd.isdigit():
        return f"Q{((report_date.month - 1) // 3) + 1}"

    fy_end_month = int(fiscal_year_end_mmdd[:2])
    fy_end_day = int(fiscal_year_end_mmdd[2:])
    fiscal_year = _infer_fiscal_year(filing_type, report_date, fiscal_year_end_mmdd)

    try:
        prior_fy_end = dt.date(fiscal_year - 1, fy_end_month, fy_end_day)
    except ValueError:
        # Fallback for unusual fiscal-year-end dates like Feb 29 in non-leap years.
        prior_fy_end = dt.date(fiscal_year - 1, fy_end_month, min(fy_end_day, 28))

    delta_days = (report_date - prior_fy_end).days
    quarter_num = max(1, min(4, int(round(delta_days / 91.0))))
    return f"Q{quarter_num}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class SECDownloader:
    """
    Download SEC filings for the specified tickers / filing types / years.

    Parameters
    ----------
    download_dir:
        Root directory for downloaded filings (defaults to config value).
    max_rps:
        Maximum EDGAR requests per second.
    """

    def __init__(
        self,
        download_dir: str = SEC_DOWNLOAD_DIR,
        max_rps: int = SEC_MAX_REQUESTS_PER_SECOND,
    ) -> None:
        self.download_dir = download_dir
        self.max_rps = max_rps
        os.makedirs(download_dir, exist_ok=True)
        self._downloader = _make_downloader(download_dir)
        # Semaphore used in async context; also track wall-clock time for sync
        self._last_request_times: list[float] = []

    # ------------------------------------------------------------------
    # Rate limiting (synchronous token-bucket style)
    # ------------------------------------------------------------------

    def _rate_limit(self) -> None:
        """Block until we are within the allowed requests-per-second budget."""
        now = time.monotonic()
        # Keep only requests within the last 1 second
        self._last_request_times = [
            t for t in self._last_request_times if now - t < 1.0
        ]
        if len(self._last_request_times) >= self.max_rps:
            sleep_for = 1.0 - (now - self._last_request_times[0])
            if sleep_for > 0:
                time.sleep(sleep_for)
        self._last_request_times.append(time.monotonic())

    # ------------------------------------------------------------------
    # Core download logic
    # ------------------------------------------------------------------

    def _download_one(
        self,
        ticker: str,
        filing_type: str,
        after_date: str,
        before_date: str,
        limit: int = 20,
    ) -> None:
        """
        Download filings for a single ticker/type combination.
        Already-downloaded filings are skipped automatically by the library.
        """
        self._rate_limit()
        try:
            self._downloader.get(
                filing_type,
                ticker,
                limit=limit,
                after=after_date,
                before=before_date,
            )
            logger.debug("Downloaded %s %s (%s – %s)", ticker, filing_type, after_date, before_date)
        except Exception as exc:
            logger.warning(
                "Failed to download %s %s: %s", ticker, filing_type, exc
            )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def download(
        self,
        tickers: list[str],
        filing_types: list[str],
        years: list[int],
    ) -> list[FilingMeta]:
        """
        Download filings for *tickers* / *filing_types* / *years* and return
        a list of ``FilingMeta`` tuples for all documents on disk (including
        previously downloaded ones).

        Returns
        -------
        list[FilingMeta]
            Each element: (ticker, filing_type, year, local_path, source_url)
        """
        after_date = f"{min(years) - 1}-07-01"
        before_date = f"{max(years) + 1}-06-30"

        total = len(tickers) * len(filing_types)
        processed = 0

        for ticker in tickers:
            for filing_type in filing_types:
                processed += 1
                logger.info(
                    "[%d/%d] Downloading %s %s", processed, total, ticker, filing_type
                )
                self._download_one(
                    ticker,
                    filing_type,
                    after_date,
                    before_date,
                    limit=_download_limit_for_range(filing_type, years),
                )

        # Collect metadata for everything on disk (current run + prior runs)
        return self._collect_metadata(tickers, filing_types, years)

    def _collect_metadata(
        self,
        tickers: list[str],
        filing_types: list[str],
        years: list[int],
    ) -> list[FilingMeta]:
        """
        Walk the download directory and build FilingMeta tuples.
        """
        results: list[FilingMeta] = []
        year_set = set(years)

        for ticker in tickers:
            for filing_type in filing_types:
                root = _filing_root(self.download_dir, ticker, filing_type)
                for doc_path, accession in _iter_filing_paths(root):
                    metadata = _parse_submission_metadata(doc_path.parent)
                    report_date = _parse_yyyymmdd(metadata.get("period_of_report"))
                    filed_date = _parse_yyyymmdd(metadata.get("filed_date"))
                    fiscal_year = _infer_fiscal_year(
                        filing_type,
                        report_date,
                        metadata.get("fiscal_year_end"),
                    )
                    if fiscal_year not in year_set:
                        continue
                    period = _infer_period(
                        filing_type,
                        report_date,
                        metadata.get("fiscal_year_end"),
                    )
                    cik = metadata.get("cik")
                    source_url = _build_source_url(cik, accession) if cik else ""
                    results.append(
                        (
                            ticker,
                            filing_type,
                            fiscal_year,
                            period,
                            doc_path,
                            source_url,
                            filed_date,
                            report_date,
                            accession,
                            cik or "",
                        )
                    )

        logger.info("Collected %d filing documents from disk", len(results))
        return results


# ---------------------------------------------------------------------------
# Async wrapper (for use in async orchestration contexts)
# ---------------------------------------------------------------------------

async def download_async(
    tickers: list[str],
    filing_types: list[str],
    years: list[int],
    download_dir: str = SEC_DOWNLOAD_DIR,
    max_rps: int = SEC_MAX_REQUESTS_PER_SECOND,
) -> list[FilingMeta]:
    """
    Async wrapper around :class:`SECDownloader`.  Downloads each
    ticker/type pair concurrently while honouring the rate limit via a
    semaphore.

    Because sec-edgar-downloader's ``get()`` is synchronous, we run it in
    a thread-pool executor.
    """
    semaphore = asyncio.Semaphore(max_rps)
    dl = SECDownloader(download_dir=download_dir, max_rps=max_rps)
    loop = asyncio.get_event_loop()

    after_date = f"{min(years) - 1}-07-01"
    before_date = f"{max(years) + 1}-06-30"

    async def _task(ticker: str, filing_type: str) -> None:
        async with semaphore:
            await loop.run_in_executor(
                None,
                dl._download_one,
                ticker,
                filing_type,
                after_date,
                before_date,
            )

    tasks = [
        _task(ticker, filing_type)
        for ticker in tickers
        for filing_type in filing_types
    ]
    await asyncio.gather(*tasks, return_exceptions=True)
    return dl._collect_metadata(tickers, filing_types, years)


# ---------------------------------------------------------------------------
# CLI entry point for standalone testing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    tickers = sys.argv[1:] or ["AAPL", "MSFT"]
    downloader = SECDownloader()
    metas = downloader.download(tickers, ["10-K"], [2022, 2023])
    for m in metas:
        print(m[0], m[1], m[2], m[3])
