"""
News downloader
================
Fetches financial news from a mix of sources:
  - Official company newsroom sources when available
  - Yahoo Finance RSS
  - Reuters RSS

Articles are deduplicated by URL. Content and title are combined for embedding.

中文：新闻来源质量与格式各异，本模块只提取可溯源的文章文本并按 URL 去重；它不判断内容真假。
"""

from __future__ import annotations

import logging
import re
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any
from urllib.parse import urljoin

import feedparser
import requests
from bs4 import BeautifulSoup

from corpcheck.ingestion.config import ALL_TICKERS, REQUEST_DELAY_SECONDS, REQUEST_TIMEOUT

logger = logging.getLogger(__name__)

NewsRow = dict[str, Any]

# ---------------------------------------------------------------------------
# RSS feed templates
# ---------------------------------------------------------------------------

YAHOO_RSS_URL = "https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"

REUTERS_RSS_FEEDS = [
    "https://feeds.reuters.com/reuters/businessNews",
    "https://feeds.reuters.com/reuters/companyNews",
]

OFFICIAL_NEWSROOM_SOURCES: dict[str, dict[str, Any]] = {
    "AAPL": {
        "kind": "rss",
        "url": "https://www.apple.com/newsroom/rss-feed.rss",
        "source": "apple_newsroom",
    },
    "UNH": {
        "kind": "listing",
        "url": "https://www.unitedhealthgroup.com/newsroom/news.html?contenttype=press-release",
        "source": "unitedhealthgroup_newsroom",
        "article_path_pattern": re.compile(r"^/newsroom/(?:posts/)?(20\d{2})/"),
    },
    "XOM": {
        "kind": "listing",
        "url": "https://corporate.exxonmobil.com/news/news-releases",
        "source": "exxonmobil_newsroom",
        "article_path_pattern": re.compile(r"^/news/news-releases/(20\d{2})/"),
    },
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_published(entry: Any) -> datetime | None:
    """Parse the published date from a feedparser entry.

    中文：按多个常见字段依次尝试；无法识别日期时返回 ``None``，由筛选逻辑决定是否保留。
    """
    for field in ("published", "updated", "created"):
        raw = getattr(entry, field, None)
        if raw:
            try:
                return parsedate_to_datetime(raw)
            except Exception:
                pass
    if hasattr(entry, "published_parsed") and entry.published_parsed:
        try:
            return datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
        except Exception:
            pass
    return None


def _clean_html(html_text: str) -> str:
    """Strip HTML tags from a string.

    中文：解析失败时返回原文本，避免一个格式异常的摘要中断整批下载。
    """
    try:
        soup = BeautifulSoup(html_text, "lxml")
        return soup.get_text(separator=" ", strip=True)
    except Exception:
        return html_text


def _normalise_datetime(value: datetime | None) -> datetime | None:
    """Return a timezone-aware UTC datetime when possible.

    中文：无时区的来源时间按 UTC 解释；该约定保证年筛选在不同 RSS 来源间一致。
    """
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _extract_year_from_url(url: str) -> int | None:
    """Extract a 4-digit year from a newsroom-style URL."""
    match = re.search(r"/(20\d{2})/", url)
    if match:
        return int(match.group(1))
    return None


def _extract_published_from_soup(soup: BeautifulSoup) -> datetime | None:
    """Extract article published datetime from metadata or <time> tags."""
    meta_candidates = [
        ("meta", {"property": "article:published_time"}, "content"),
        ("meta", {"name": "article:published_time"}, "content"),
        ("meta", {"property": "og:published_time"}, "content"),
        ("meta", {"name": "publish-date"}, "content"),
        ("meta", {"name": "date"}, "content"),
    ]
    for tag_name, attrs, attr_name in meta_candidates:
        tag = soup.find(tag_name, attrs=attrs)
        if tag and tag.get(attr_name):
            raw = str(tag.get(attr_name)).strip()
            try:
                return _normalise_datetime(datetime.fromisoformat(raw.replace("Z", "+00:00")))
            except ValueError:
                try:
                    return _normalise_datetime(parsedate_to_datetime(raw))
                except Exception:
                    pass

    time_tag = soup.find("time", attrs={"datetime": True})
    if time_tag and time_tag.get("datetime"):
        raw = str(time_tag.get("datetime")).strip()
        try:
            return _normalise_datetime(datetime.fromisoformat(raw.replace("Z", "+00:00")))
        except ValueError:
            try:
                return _normalise_datetime(parsedate_to_datetime(raw))
            except Exception:
                return None
    return None


def _extract_article_text(soup: BeautifulSoup) -> str:
    """Extract the article body from a newsroom page.

    中文：优先正文容器，移除导航和脚本，并去掉响应式页面中常见的重复段落。
    """
    body = soup.find("article") or soup.find("main") or soup.body or soup
    for tag in body.find_all(["nav", "aside", "footer", "script", "style", "noscript", "form"]):
        tag.decompose()

    paragraphs: list[str] = []
    for tag in body.find_all(["p", "li"]):
        text = " ".join(tag.get_text(" ", strip=True).split())
        if text and len(text) >= 40:
            paragraphs.append(text)

    # Keep order but drop exact duplicates that often appear in sticky/mobile UI.
    deduped: list[str] = []
    seen: set[str] = set()
    for text in paragraphs:
        if text in seen:
            continue
        seen.add(text)
        deduped.append(text)
    return "\n\n".join(deduped)


def _entry_to_row(entry: Any, ticker: str | None, source: str) -> NewsRow | None:
    """
    Convert a feedparser entry dict into a ``NewsRow`` dict.
    Returns None if the entry has no URL (cannot deduplicate).

    中文：URL 是跨来源去重和溯源的最小条件；缺少 URL 的条目有意丢弃。
    """
    url: str | None = getattr(entry, "link", None)
    if not url:
        return None

    title: str = getattr(entry, "title", "") or ""
    summary: str = getattr(entry, "summary", "") or ""
    content_raw: str = ""

    # feedparser sometimes puts full content in entry.content list
    if hasattr(entry, "content") and entry.content:
        content_raw = entry.content[0].get("value", "") or ""

    # Prefer content over summary; fall back to summary
    content = _clean_html(content_raw) if content_raw else _clean_html(summary)

    published = _parse_published(entry)
    author = getattr(entry, "author", None)

    return {
        "ticker": ticker,
        "title": title.strip(),
        "content": content.strip(),
        "summary": _clean_html(summary).strip(),
        "author": author,
        "published_date": published,
        "source_url": url.strip(),
        "source": source,
        # embedding filled in later by embedder
        "embedding": None,
    }


# ---------------------------------------------------------------------------
# Downloader class
# ---------------------------------------------------------------------------

class NewsDownloader:
    """
    Download financial news articles from Yahoo Finance and Reuters RSS feeds.

    Parameters
    ----------
    tickers:
        List of tickers to fetch Yahoo Finance news for.
    request_delay:
        Seconds to pause between requests.
    timeout:
        HTTP request timeout in seconds.
    include_reuters:
        If True, also fetch Reuters RSS feeds (not ticker-specific).

    中文：新闻采集器维护一次运行范围内的 URL 集合。官方 newsroom 可用时优先使用，以提高
    可追溯性；否则才回退到 Yahoo RSS。
    """

    def __init__(
        self,
        tickers: list[str] | None = None,
        years: list[int] | None = None,
        request_delay: float = REQUEST_DELAY_SECONDS,
        timeout: int = REQUEST_TIMEOUT,
        include_reuters: bool = True,
        official_only: bool = False,
    ) -> None:
        self.tickers = tickers or ALL_TICKERS
        self.years = set(years or [])
        self.request_delay = request_delay
        self.timeout = timeout
        self.include_reuters = include_reuters
        self.official_only = official_only
        self._seen_urls: set[str] = set()

    # ------------------------------------------------------------------
    # Internal fetch
    # ------------------------------------------------------------------

    def _fetch_url(self, url: str) -> requests.Response | None:
        """Fetch a URL and return the response on success."""
        try:
            resp = requests.get(url, timeout=self.timeout)
            resp.raise_for_status()
            return resp
        except requests.RequestException as exc:
            logger.warning("Failed to fetch %s: %s", url, exc)
            return None

    def _fetch_feed(self, url: str) -> feedparser.FeedParserDict:
        """Fetch and parse an RSS feed URL, returning a feedparser dict."""
        resp = self._fetch_url(url)
        if resp is None:
            return feedparser.FeedParserDict()
        return feedparser.parse(resp.content)

    def _year_allowed(self, year: int | None) -> bool:
        """Return True when a row should be kept for the configured years."""
        if year is None or not self.years:
            return True
        return year in self.years

    def _process_feed(
        self,
        feed: feedparser.FeedParserDict,
        ticker: str | None,
        source: str,
    ) -> list[NewsRow]:
        """Convert feedparser entries to NewsRow dicts, deduplicating by URL.

        中文：年份筛选优先使用发布日期，缺失时才从 URL 中提取年份。
        """
        rows: list[NewsRow] = []
        for entry in getattr(feed, "entries", []):
            row = _entry_to_row(entry, ticker, source)
            if row is None:
                continue
            published = _normalise_datetime(row.get("published_date"))
            if not self._year_allowed(published.year if published else _extract_year_from_url(row["source_url"])):
                continue
            row["published_date"] = published
            url = row["source_url"]
            if url in self._seen_urls:
                continue
            self._seen_urls.add(url)
            rows.append(row)
        return rows

    def _extract_official_article_row(
        self,
        ticker: str,
        article_url: str,
        source: str,
        fallback_title: str = "",
    ) -> NewsRow | None:
        """Fetch one official newsroom article and convert it to a NewsRow.

        中文：只有同时得到标题和正文的页面才可入库，避免导航页或错误页成为检索材料。
        """
        resp = self._fetch_url(article_url)
        if resp is None:
            return None

        soup = BeautifulSoup(resp.text, "lxml")
        title = ""
        for attrs in (
            {"property": "og:title"},
            {"name": "twitter:title"},
        ):
            meta = soup.find("meta", attrs=attrs)
            if meta and meta.get("content"):
                title = str(meta.get("content")).strip()
                break
        if not title:
            title_tag = soup.find("title")
            if title_tag:
                title = " ".join(title_tag.get_text(" ", strip=True).split())
        if not title:
            title = fallback_title

        summary = ""
        for attrs in (
            {"property": "og:description"},
            {"name": "description"},
            {"name": "twitter:description"},
        ):
            meta = soup.find("meta", attrs=attrs)
            if meta and meta.get("content"):
                summary = _clean_html(str(meta.get("content")).strip())
                break

        published = _extract_published_from_soup(soup)
        if not self._year_allowed(published.year if published else _extract_year_from_url(article_url)):
            return None

        content = _extract_article_text(soup)
        if not title or not content:
            return None

        return {
            "ticker": ticker,
            "title": title,
            "content": content,
            "summary": summary,
            "author": None,
            "published_date": published,
            "source_url": article_url,
            "source": source,
            "embedding": None,
        }

    def _extract_links_from_listing(
        self,
        listing_url: str,
        article_path_pattern: re.Pattern[str],
    ) -> list[tuple[str, str]]:
        """Extract matching article links from an official newsroom listing page.

        中文：调用方提供路径模式，避免把同站点的导航、招聘等无关链接当成新闻。
        """
        resp = self._fetch_url(listing_url)
        if resp is None:
            return []

        soup = BeautifulSoup(resp.text, "lxml")
        links: list[tuple[str, str]] = []
        seen_urls: set[str] = set()
        for anchor in soup.find_all("a", href=True):
            href = str(anchor["href"]).strip()
            text = " ".join(anchor.get_text(" ", strip=True).split())
            if not href or not text:
                continue
            if not article_path_pattern.match(href):
                continue
            full_url = urljoin(listing_url, href)
            year = _extract_year_from_url(full_url)
            if not self._year_allowed(year):
                continue
            if full_url in seen_urls:
                continue
            seen_urls.add(full_url)
            links.append((full_url, text))
        return links

    # ------------------------------------------------------------------
    # Official company newsroom sources
    # ------------------------------------------------------------------

    def fetch_official_news(self, ticker: str) -> list[NewsRow]:
        """Fetch official newsroom items for one ticker when supported.

        中文：没有配置官方来源的 ticker 返回空列表，供上层决定是否回退到 RSS。
        """
        config = OFFICIAL_NEWSROOM_SOURCES.get(ticker)
        if config is None:
            return []

        source = str(config["source"])
        rows: list[NewsRow] = []

        if config["kind"] == "rss":
            feed = self._fetch_feed(str(config["url"]))
            rows = self._process_feed(feed, ticker, source)
            logger.debug("Official newsroom RSS for %s: %d new articles", ticker, len(rows))
            return rows

        listing_links = self._extract_links_from_listing(
            str(config["url"]),
            config["article_path_pattern"],
        )
        for article_url, title in listing_links:
            if article_url in self._seen_urls:
                continue
            row = self._extract_official_article_row(ticker, article_url, source, fallback_title=title)
            if row is None:
                continue
            self._seen_urls.add(article_url)
            rows.append(row)
            time.sleep(self.request_delay)

        logger.debug("Official newsroom listing for %s: %d new articles", ticker, len(rows))
        return rows

    # ------------------------------------------------------------------
    # Per-ticker Yahoo Finance
    # ------------------------------------------------------------------

    def fetch_yahoo_news(self, ticker: str) -> list[NewsRow]:
        """Fetch Yahoo Finance RSS for a single ticker.

        中文：结果仍经过运行级 URL 去重和年份筛选。
        """
        url = YAHOO_RSS_URL.format(ticker=ticker)
        feed = self._fetch_feed(url)
        rows = self._process_feed(feed, ticker, "yahoo_rss")
        logger.debug("Yahoo RSS for %s: %d new articles", ticker, len(rows))
        return rows

    # ------------------------------------------------------------------
    # Reuters feeds (general market news)
    # ------------------------------------------------------------------

    def fetch_reuters_news(self) -> list[NewsRow]:
        """Fetch Reuters business/company RSS feeds (not ticker-specific).

        中文：Reuters 行没有强行绑定 ticker，避免将宏观新闻错误归属给公司。
        """
        rows: list[NewsRow] = []
        for feed_url in REUTERS_RSS_FEEDS:
            feed = self._fetch_feed(feed_url)
            batch = self._process_feed(feed, None, "reuters_rss")
            rows.extend(batch)
            logger.debug("Reuters feed %s: %d new articles", feed_url, len(batch))
            time.sleep(self.request_delay)
        return rows

    # ------------------------------------------------------------------
    # Full download
    # ------------------------------------------------------------------

    def download_all(self) -> list[NewsRow]:
        """
        Download all news articles for every configured ticker (Yahoo) and
        all Reuters feeds.

        Returns
        -------
        list[NewsRow]
            Deduplicated list of news article dicts.

        中文：每次完整下载都会清空本次运行的去重状态，不会跨进程隐式保留旧 URL。
        """
        all_rows: list[NewsRow] = []
        self._seen_urls.clear()

        for i, ticker in enumerate(self.tickers, 1):
            official_rows = self.fetch_official_news(ticker)
            if official_rows:
                logger.info("[%d/%d] Fetched %d official news items for %s", i, len(self.tickers), len(official_rows), ticker)
                all_rows.extend(official_rows)
            elif self.official_only:
                logger.info("[%d/%d] No supported official newsroom source for %s", i, len(self.tickers), ticker)
            else:
                logger.info("[%d/%d] Fetching Yahoo news for %s", i, len(self.tickers), ticker)
                rows = self.fetch_yahoo_news(ticker)
                all_rows.extend(rows)
            time.sleep(self.request_delay)

        # Reuters
        if self.include_reuters and not self.official_only:
            logger.info("Fetching Reuters RSS feeds")
            rows = self.fetch_reuters_news()
            all_rows.extend(rows)

        logger.info(
            "News download complete: %d articles (%d unique URLs)",
            len(all_rows),
            len(self._seen_urls),
        )
        return all_rows


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------

def download_news(
    tickers: list[str] | None = None,
    years: list[int] | None = None,
    include_reuters: bool = True,
    official_only: bool = False,
) -> list[NewsRow]:
    """Shorthand for ``NewsDownloader().download_all()``.

    中文：为一次性任务提供便捷入口；复杂抓取配置请直接构造下载器。
    """
    dl = NewsDownloader(
        tickers=tickers,
        years=years,
        include_reuters=include_reuters,
        official_only=official_only,
    )
    return dl.download_all()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    articles = download_news(tickers=["AAPL", "UNH", "XOM"], years=[2024, 2025, 2026], official_only=True)
    print(f"Downloaded {len(articles)} articles")
    for a in articles[:3]:
        print(f"  [{a['ticker']}] {a['title']} ({a['published_date']})")
