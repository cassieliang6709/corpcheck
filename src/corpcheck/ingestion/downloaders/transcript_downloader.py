"""
Earnings call transcript downloader (Motley Fool)
====================================================
Scrapes earnings call transcripts from Motley Fool for the configured
ticker universe and year range.  Transcripts are split into three sections:
  - ``prepared_remarks`` – management's opening comments
  - ``qa``               – analyst Q&A
  - ``closing``          – operator closing / disclosures

Robots.txt is respected; a polite delay is added between requests.

中文：这是公开网页抓取适配器，遵守 robots.txt 并保留来源 URL；解析失败的网页不会成为空记录。
"""

from __future__ import annotations

import logging
import re
import time
import urllib.robotparser
from datetime import date
from typing import Any
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from corpcheck.ingestion.config import (
    ALL_TICKERS,
    END_YEAR,
    REQUEST_DELAY_SECONDS,
    REQUEST_TIMEOUT,
    START_YEAR,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MOTLEY_FOOL_BASE = "https://www.fool.com"
SEARCH_URL = "https://www.fool.com/earnings-call-transcripts/"
TRANSCRIPT_SEARCH_URL = (
    "https://www.fool.com/search/solr.aspx"
    "?q={query}&page={page}&filter=articletype%3ATranscript"
)
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; FinancialRAGBot/1.0; "
        "+https://github.com/example/financial-rag)"
    ),
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "en-US,en;q=0.9",
}

# Regex patterns for section detection
PREPARED_REMARKS_PATTERNS = [
    re.compile(r"prepared\s+remarks", re.I),
    re.compile(r"opening\s+remarks", re.I),
    re.compile(r"management\s+(?:discussion|remarks)", re.I),
]
QA_PATTERNS = [
    re.compile(r"question[s]?\s+and\s+answer", re.I),
    re.compile(r"q\s*&\s*a\s+session", re.I),
    re.compile(r"question-and-answer", re.I),
]
CLOSING_PATTERNS = [
    re.compile(r"closing\s+remarks", re.I),
    re.compile(r"end\s+of\s+(?:the\s+)?(?:call|conference)", re.I),
    re.compile(r"conclud(?:e|ing)\s+remarks", re.I),
]

TranscriptRow = dict[str, Any]

# ---------------------------------------------------------------------------
# Robots.txt gate
# ---------------------------------------------------------------------------

class _RobotsGate:
    """Cache robots.txt rules per domain.

    中文：同一域名只读取一次 robots.txt；读取失败时沿用项目既有的允许策略。
    """

    def __init__(self) -> None:
        self._parsers: dict[str, urllib.robotparser.RobotFileParser] = {}

    def can_fetch(self, url: str, user_agent: str = "*") -> bool:
        """Return whether cached or freshly read robots.txt permits a request.

        中文：该方法只做访问许可判断，不执行网络内容下载。
        """
        parsed = urlparse(url)
        domain = f"{parsed.scheme}://{parsed.netloc}"
        if domain not in self._parsers:
            rp = urllib.robotparser.RobotFileParser()
            rp.set_url(f"{domain}/robots.txt")
            try:
                rp.read()
            except Exception:
                # If we can't read robots.txt, assume allowed
                self._parsers[domain] = rp
                return True
            self._parsers[domain] = rp
        return self._parsers[domain].can_fetch(user_agent, url)


_robots = _RobotsGate()


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _quarter_from_title(title: str) -> tuple[int | None, int | None]:
    """
    Extract (fiscal_year, quarter) from a transcript title like
    'Apple Q3 2021 Earnings Call Transcript'.
    Returns (None, None) if not found.

    中文：标题格式不可靠时保留 ``None``，避免从正文或发布日期猜测财政季度。
    """
    year_match = re.search(r"\b(20\d{2})\b", title)
    quarter_match = re.search(r"\bQ([1-4])\b", title, re.I)

    year = int(year_match.group(1)) if year_match else None
    quarter = int(quarter_match.group(1)) if quarter_match else None
    return year, quarter


def _split_into_sections(text: str) -> dict[str, str]:
    """
    Split a full transcript text into prepared_remarks, qa, and closing
    sections based on header patterns.

    Returns a dict with keys matching the section names.  If a section
    boundary isn't found the text is placed under 'prepared_remarks'.

    中文：分段依赖启发式标题；未识别的内容有意保留在 prepared remarks，而不是丢弃。
    """
    sections: dict[str, str] = {
        "prepared_remarks": "",
        "qa": "",
        "closing": "",
    }

    # Find QA boundary
    qa_start = len(text)
    for pattern in QA_PATTERNS:
        m = pattern.search(text)
        if m and m.start() < qa_start:
            qa_start = m.start()

    # Find closing boundary (must be after QA)
    closing_start = len(text)
    for pattern in CLOSING_PATTERNS:
        m = pattern.search(text, qa_start)
        if m and m.start() < closing_start:
            closing_start = m.start()

    if qa_start < len(text):
        sections["prepared_remarks"] = text[:qa_start].strip()
        if closing_start < len(text) and closing_start > qa_start:
            sections["qa"] = text[qa_start:closing_start].strip()
            sections["closing"] = text[closing_start:].strip()
        else:
            sections["qa"] = text[qa_start:].strip()
    else:
        sections["prepared_remarks"] = text.strip()

    return sections


def _extract_transcript_text(soup: BeautifulSoup) -> str:
    """
    Extract the main transcript body from a Motley Fool article page.
    Returns plain text with paragraph breaks.

    中文：优先站点正文容器，找不到时才逐级回退，避免不同页面模板导致整个下载失败。
    """
    # Motley Fool transcript article body is in div.article-body
    body = soup.find("div", class_="article-body")
    if body is None:
        # Fallback: largest <article> or <main> tag
        body = soup.find("article") or soup.find("main") or soup

    # Remove navigation, ads, related links, etc.
    for tag in body.find_all(  # type: ignore[union-attr]
        ["nav", "aside", "footer", "script", "style", "figure"]
    ):
        tag.decompose()

    paragraphs = []
    for p in body.find_all(["p", "h2", "h3"]):  # type: ignore[union-attr]
        text = p.get_text(separator=" ", strip=True)
        if text:
            paragraphs.append(text)

    return "\n\n".join(paragraphs)


def _extract_published_date(soup: BeautifulSoup) -> date | None:
    """Extract publication date from Motley Fool article metadata."""
    # Try <time> element with datetime attribute
    time_tag = soup.find("time", attrs={"datetime": True})
    if time_tag:
        try:
            dt_str = time_tag["datetime"]  # type: ignore[index]
            # Handles 2021-08-01 and 2021-08-01T10:00:00Z
            return date.fromisoformat(dt_str[:10])
        except (ValueError, KeyError):
            pass

    # Try meta tag
    meta = soup.find("meta", attrs={"property": "article:published_time"})
    if meta:
        try:
            return date.fromisoformat(str(meta.get("content", ""))[:10])  # type: ignore[arg-type]
        except ValueError:
            pass
    return None


# ---------------------------------------------------------------------------
# Downloader class
# ---------------------------------------------------------------------------

class TranscriptDownloader:
    """
    Download earnings call transcripts from Motley Fool.

    Parameters
    ----------
    tickers:
        List of tickers to search for.
    years:
        List of fiscal years to include.
    request_delay:
        Seconds between HTTP requests.
    timeout:
        HTTP timeout in seconds.
    max_pages_per_ticker:
        Max search result pages to walk per ticker.

    中文：调用 Motley Fool 搜索与文章页的下载器。URL 在一次运行内去重，且所有 HTTP 请求先经过
    robots.txt 检查。
    """

    def __init__(
        self,
        tickers: list[str] | None = None,
        years: list[int] | None = None,
        request_delay: float = REQUEST_DELAY_SECONDS * 2,  # be extra polite
        timeout: int = REQUEST_TIMEOUT,
        max_pages_per_ticker: int = 5,
    ) -> None:
        self.tickers = tickers or ALL_TICKERS
        self.years = set(years) if years else set(range(START_YEAR, END_YEAR + 1))
        self.request_delay = request_delay
        self.timeout = timeout
        self.max_pages_per_ticker = max_pages_per_ticker
        self._session = requests.Session()
        self._session.headers.update(HEADERS)
        self._seen_urls: set[str] = set()

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------

    def _get(self, url: str) -> requests.Response | None:
        """
        Fetch *url* if robots.txt permits.  Returns None on any error.

        中文：网络、状态码或 robots 拒绝都统一返回 ``None``，让批处理继续处理其他页面。
        """
        if not _robots.can_fetch(url):
            logger.info("robots.txt disallows %s, skipping", url)
            return None
        try:
            resp = self._session.get(url, timeout=self.timeout)
            resp.raise_for_status()
            return resp
        except requests.RequestException as exc:
            logger.warning("Request failed for %s: %s", url, exc)
            return None

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def _search_ticker(self, ticker: str) -> list[str]:
        """
        Return a list of transcript article URLs for *ticker*.
        Walks multiple search-result pages.

        中文：仅收集看起来像 transcript 的链接，是否可用还由后续文章解析验证。
        """
        urls: list[str] = []
        # Use the company name pattern commonly used in Motley Fool search
        query = f"{ticker} earnings call transcript"

        for page in range(1, self.max_pages_per_ticker + 1):
            search_url = TRANSCRIPT_SEARCH_URL.format(
                query=query.replace(" ", "+"), page=page
            )
            resp = self._get(search_url)
            if resp is None:
                break

            soup = BeautifulSoup(resp.text, "lxml")
            # Result links are <a> tags within search result items
            found_any = False
            for a in soup.select("a[href]"):
                href: str = a["href"]  # type: ignore[assignment]
                if "earnings-call-transcript" in href or "transcript" in href.lower():
                    full_url = href if href.startswith("http") else urljoin(MOTLEY_FOOL_BASE, href)
                    if full_url not in self._seen_urls:
                        urls.append(full_url)
                        found_any = True

            if not found_any:
                break

            time.sleep(self.request_delay)

        return urls

    # ------------------------------------------------------------------
    # Parse one transcript page
    # ------------------------------------------------------------------

    def _parse_transcript(
        self,
        url: str,
        ticker: str,
    ) -> TranscriptRow | None:
        """
        Fetch and parse a single Motley Fool transcript page.
        Returns a ``TranscriptRow`` dict or None if parsing fails.

        中文：过短正文、重复 URL 和超出年份范围的文章均返回 ``None``，不会写入半成品记录。
        """
        if url in self._seen_urls:
            return None
        self._seen_urls.add(url)

        resp = self._get(url)
        if resp is None:
            return None

        soup = BeautifulSoup(resp.text, "lxml")

        # Title
        title_tag = soup.find("h1")
        title = title_tag.get_text(strip=True) if title_tag else ""

        fiscal_year, quarter = _quarter_from_title(title)

        # Reject if outside our year range
        if fiscal_year is not None and fiscal_year not in self.years:
            return None

        content = _extract_transcript_text(soup)
        if len(content) < 200:
            logger.debug("Skipping short transcript at %s (%d chars)", url, len(content))
            return None

        published_date = _extract_published_date(soup)
        sections = _split_into_sections(content)

        return {
            "ticker": ticker,
            "fiscal_year": fiscal_year,
            "quarter": quarter,
            "content": content,
            "published_date": published_date,
            "source_url": url,
            "sections": sections,  # dict: prepared_remarks / qa / closing
        }

    # ------------------------------------------------------------------
    # Full download
    # ------------------------------------------------------------------

    def download_all(self) -> list[TranscriptRow]:
        """
        Search for and download earnings call transcripts for all tickers.

        Returns
        -------
        list[TranscriptRow]
            Each row has: ticker, fiscal_year, quarter, content,
            published_date, source_url, sections (dict).

        中文：逐 ticker 顺序执行以保持对来源网站的礼貌节流；返回值不触及数据库。
        """
        all_rows: list[TranscriptRow] = []

        for i, ticker in enumerate(self.tickers, 1):
            logger.info(
                "[%d/%d] Searching transcripts for %s",
                i,
                len(self.tickers),
                ticker,
            )
            candidate_urls = self._search_ticker(ticker)
            logger.debug("Found %d candidate URLs for %s", len(candidate_urls), ticker)

            for url in candidate_urls:
                row = self._parse_transcript(url, ticker)
                if row is not None:
                    all_rows.append(row)
                    logger.info(
                        "Parsed transcript: %s Q%s %s",
                        ticker,
                        row.get("quarter"),
                        row.get("fiscal_year"),
                    )
                time.sleep(self.request_delay)

        logger.info(
            "Transcript download complete: %d transcripts across %d tickers",
            len(all_rows),
            len(self.tickers),
        )
        return all_rows


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------

def download_transcripts(
    tickers: list[str] | None = None,
    years: list[int] | None = None,
) -> list[TranscriptRow]:
    """Shorthand for ``TranscriptDownloader().download_all()``.

    中文：为一次性脚本提供入口；需要修改请求间隔或搜索页数时请使用类接口。
    """
    dl = TranscriptDownloader(tickers=tickers, years=years)
    return dl.download_all()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    rows = download_transcripts(tickers=["AAPL"], years=[2022])
    print(f"Downloaded {len(rows)} transcripts")
    for r in rows:
        print(f"  {r['ticker']} Q{r['quarter']} {r['fiscal_year']} — {r['source_url']}")
