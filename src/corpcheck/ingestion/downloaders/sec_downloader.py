"""
SEC EDGAR downloader
=====================
Downloads 10-K, 10-Q, and 8-K filings for a list of tickers using the
sec-edgar-downloader library, then returns metadata tuples for downstream
processing.

Rate limiting is enforced via an asyncio Semaphore (max 10 req/s as required
by EDGAR's fair-use policy).

中文：该模块负责按 SEC 规则下载并从本地文件推导 filing 元数据；下载与元数据收集分离，
因此重跑时能复用已经落盘的文档。
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
    SEC_TICKER_ALIASES,
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

def parse_sec_user_agent(user_agent: str) -> tuple[str, str]:
    """Parse the required ``ProjectName email@example.com`` SEC identity.

    中文：SEC 要求可联系的身份标识；严格校验可在发请求前暴露错误配置。
    """
    parts = user_agent.split()
    if len(parts) != 2:
        raise ValueError("SEC_USER_AGENT must be exactly 'ProjectName email@example.com'")
    project_name, email = parts
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", project_name):
        raise ValueError("SEC_USER_AGENT project name must be one token")
    if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", email):
        raise ValueError("SEC_USER_AGENT must contain a valid contact email")
    return project_name, email


def _make_downloader(download_dir: str) -> Downloader:
    """Create an EDGAR Downloader pointed at *download_dir*."""
    # Read the environment at construction time so callers validate and use
    # the same identity even when the module was imported earlier.
    company, email = parse_sec_user_agent(os.getenv("SEC_USER_AGENT", SEC_USER_AGENT))
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

    中文：优先选择最大的 HTML 文件，缺失时回退到完整 submission 文本；不解析文件内容。
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

    中文：这是请求上限的安全估计，而不是公司实际披露数量的断言；宁可多取也不能静默漏档。
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

    中文：只读取文件开头的 EDGAR header；缺文件或不可读时返回空字典，供调用方决定兜底。
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
        "filing_type": r"CONFORMED SUBMISSION TYPE:\s+([^\r\n]+)",
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


def _parse_mmdd(raw: str | None) -> tuple[int, int] | None:
    """Convert the SEC header's ``FISCAL YEAR END`` (``MMDD``) to (month, day)."""
    if not raw or len(raw) != 4 or not raw.isdigit():
        return None
    month, day = int(raw[:2]), int(raw[2:])
    if not 1 <= month <= 12 or not 1 <= day <= 31:
        return None
    return month, day


def _base_filing_type(filing_type: str) -> str:
    """Return the base SEC form used for period and fiscal-year inference."""
    normalized = filing_type.strip().upper()
    return normalized[:-2] if normalized.endswith("/A") else normalized


def _turns_over_new_year(report_date: dt.date, fy_end: tuple[int, int] | None) -> bool:
    """
    Whether the fiscal year closes at the turn of the calendar year.

    A 52/53-week calendar tracking 31 December drifts either side of New Year:
    JNJ reports ``FISCAL YEAR END 0101`` and closed FY2022 on 2023-01-01. Such a
    year is named after the calendar year it mostly covers, and its quarters need
    no shift. Retail calendars closing at the end of January report ``0131``
    (WMT, CRM) or even ``0201`` (TGT, TJX, whose 52/53-week year can end in early
    February) and are named after the calendar year they close in -- treating
    those the same way would move every label back a year.

    中文：这里区分跨年周历与真正偏移的财政年度，避免把 1 月初结账的 10-K 错标到下一年。
    """
    if fy_end is not None:
        return fy_end[0] == 12 or (fy_end[0] == 1 and fy_end[1] <= 14)
    return report_date.day <= 14


def _infer_fiscal_year(
    filing_type: str,
    report_date: dt.date | None,
    fiscal_year_end_mmdd: str | None,
) -> int:
    """
    Infer the filing's fiscal year from period-of-report and fiscal year end.

    中文：以报告期而非提交日标记财年；没有报告期时返回 ``0``，明确表示不能可靠推断。
    """
    if report_date is None:
        return 0

    fy_end = _parse_mmdd(fiscal_year_end_mmdd)

    # Annual filings should align to the reported fiscal period, not the
    # subsequent filing year. This avoids mislabeling 52/53-week calendars
    # like AMD's FY2023 10-K (period-of-report 2023-12-30, filed 2024-01-31)
    # as FY2024.
    if filing_type == "10-K":
        # Labelling JNJ's FY2022 (closed 2023-01-01) as 2023 collides with the
        # real FY2023 10-K, and the upsert then drops one of them.
        if report_date.month == 1 and _turns_over_new_year(report_date, fy_end):
            return report_date.year - 1
        return report_date.year

    if fy_end is None:
        return report_date.year

    # A year that turns over New Year is named for the calendar year it spans,
    # so its quarters keep that year. Only a genuinely offset year -- WMT closing
    # 31 January -- rolls its later quarters into the next fiscal year.
    if _turns_over_new_year(report_date, fy_end):
        return report_date.year

    fy_end_month, fy_end_day = fy_end

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

    中文：10-K 和 8-K 有固定标签；10-Q 则相对实际财政年末计算，兼容非自然年公司。
    """
    if filing_type == "10-K":
        return "annual"
    if filing_type == "8-K":
        return "event"
    if filing_type != "10-Q" or report_date is None:
        return "unknown"

    fy_end = _parse_mmdd(fiscal_year_end_mmdd)
    if fy_end is None:
        return f"Q{((report_date.month - 1) // 3) + 1}"

    fy_end_month, fy_end_day = fy_end

    # Anchor on the fiscal year end that actually precedes this report rather than
    # deriving one from the fiscal-year label: for a company whose year turns over
    # New Year the year ends in January *after* the year it is named for, and
    # `fiscal_year - 1` lands a full year early.
    def _fy_end_in(year: int) -> dt.date:
        try:
            return dt.date(year, fy_end_month, fy_end_day)
        except ValueError:
            # Fallback for unusual fiscal-year-end dates like Feb 29 in non-leap years.
            return dt.date(year, fy_end_month, min(fy_end_day, 28))

    prior_fy_end = _fy_end_in(report_date.year)
    if prior_fy_end >= report_date:
        prior_fy_end = _fy_end_in(report_date.year - 1)

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

    中文：封装 SEC 的礼貌限速和落盘目录约定。它返回本地文档元数据，不会在这里清洗或写库。
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
        """Block until we are within the allowed requests-per-second budget.

        中文：使用滑动的一秒窗口记录本进程请求，保证单线程下载不超过 ``max_rps``。
        """
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
        include_amends: bool = False,
    ) -> None:
        """
        Download filings for a single ticker/type combination.
        Already-downloaded filings are skipped automatically by the library.

        中文：底层库负责跳过已有文件；本方法只负责限速和将单个请求失败降级为日志告警。
        """
        self._rate_limit()
        try:
            self._downloader.get(
                filing_type,
                ticker,
                limit=limit,
                after=after_date,
                before=before_date,
                include_amends=include_amends,
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
        *,
        include_amends: bool = False,
    ) -> list[FilingMeta]:
        """
        Download filings for *tickers* / *filing_types* / *years* and return
        a list of ``FilingMeta`` tuples for all documents on disk (including
        previously downloaded ones).

        Returns
        -------
        list[FilingMeta]
            Each element: (ticker, filing_type, year, local_path, source_url)

        中文：返回范围内磁盘上的所有文档，包括此前下载的文件，便于下游做幂等入库。
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
                    include_amends=include_amends,
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

        中文：从 SEC header 而非目录名恢复报告期和 CIK，并过滤到调用方指定的财年。
        """
        results: list[FilingMeta] = []
        year_set = set(years)

        for ticker in tickers:
            # Directories named after a CIK belong to an issuer whose ticker the
            # downloader could not resolve; store them under the real ticker.
            resolved_ticker = SEC_TICKER_ALIASES.get(ticker, ticker)
            for filing_type in filing_types:
                root = _filing_root(self.download_dir, ticker, filing_type)
                for doc_path, accession in _iter_filing_paths(root):
                    metadata = _parse_submission_metadata(doc_path.parent)
                    actual_filing_type = metadata.get("filing_type", filing_type).strip().upper()
                    base_filing_type = _base_filing_type(actual_filing_type)
                    report_date = _parse_yyyymmdd(metadata.get("period_of_report"))
                    filed_date = _parse_yyyymmdd(metadata.get("filed_date"))
                    fiscal_year = _infer_fiscal_year(
                        base_filing_type,
                        report_date,
                        metadata.get("fiscal_year_end"),
                    )
                    if fiscal_year not in year_set:
                        continue
                    period = _infer_period(
                        base_filing_type,
                        report_date,
                        metadata.get("fiscal_year_end"),
                    )
                    cik = metadata.get("cik")
                    source_url = _build_source_url(cik, accession) if cik else ""
                    results.append(
                        (
                            resolved_ticker,
                            actual_filing_type,
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

    中文：异步层只协调并发；实际下载仍调用同步库，完成后与同步接口一样重新扫描本地目录。
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
