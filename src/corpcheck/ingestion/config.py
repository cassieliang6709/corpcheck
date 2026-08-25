"""
Pipeline configuration: tickers, years, FRED series, DB settings,
and model hyperparameters.

中文：集中保存离线管道的默认配置。环境变量只覆盖指定项；不要在业务代码中重复这些
默认值，以免下载、处理和加载阶段使用不一致的配置。
"""

import os

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Company universe
# ---------------------------------------------------------------------------

SECTORS: dict[str, list[str]] = {
    "banking": ["JPM", "BAC", "WFC", "GS", "MS", "C", "USB", "PNC", "TFC", "SCHW"],
    "tech": [
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
        "META", "TSLA", "AMD", "INTC", "CRM",
    ],
    "healthcare": [
        "JNJ", "UNH", "PFE", "ABBV", "MRK",
        "LLY", "BMY", "AMGN", "GILD", "CVS",
    ],
    "energy": [
        "XOM", "CVX", "COP", "SLB", "EOG",
        "PXD", "VLO", "MPC", "PSX", "OXY",
    ],
    "consumer": [
        "WMT", "HD", "MCD", "NKE", "SBUX",
        "TGT", "COST", "LOW", "TJX", "DG",
    ],
}

# Canonical company display names for the project universe.
# 中文：当上游市场元数据缺失时，用这些稳定的人工维护名称兜底。
TICKER_TO_COMPANY_NAME: dict[str, str] = {
    "JPM": "JPMorgan Chase & Co.",
    "BAC": "Bank of America Corporation",
    "WFC": "Wells Fargo & Company",
    "GS": "The Goldman Sachs Group, Inc.",
    "MS": "Morgan Stanley",
    "C": "Citigroup Inc.",
    "USB": "U.S. Bancorp",
    "PNC": "The PNC Financial Services Group, Inc.",
    "TFC": "Truist Financial Corporation",
    "SCHW": "The Charles Schwab Corporation",
    "AAPL": "Apple Inc.",
    "MSFT": "Microsoft Corporation",
    "GOOGL": "Alphabet Inc.",
    "AMZN": "Amazon.com, Inc.",
    "NVDA": "NVIDIA Corporation",
    "META": "Meta Platforms, Inc.",
    "TSLA": "Tesla, Inc.",
    "AMD": "Advanced Micro Devices, Inc.",
    "INTC": "Intel Corporation",
    "CRM": "Salesforce, Inc.",
    "JNJ": "Johnson & Johnson",
    "UNH": "UnitedHealth Group Incorporated",
    "PFE": "Pfizer Inc.",
    "ABBV": "AbbVie Inc.",
    "MRK": "Merck & Co., Inc.",
    "LLY": "Eli Lilly and Company",
    "BMY": "Bristol-Myers Squibb Company",
    "AMGN": "Amgen Inc.",
    "GILD": "Gilead Sciences, Inc.",
    "CVS": "CVS Health Corporation",
    "XOM": "Exxon Mobil Corporation",
    "CVX": "Chevron Corporation",
    "COP": "ConocoPhillips",
    "SLB": "SLB N.V.",
    "EOG": "EOG Resources, Inc.",
    "PXD": "Pioneer Natural Resources Company",
    "VLO": "Valero Energy Corporation",
    "MPC": "Marathon Petroleum Corporation",
    "PSX": "Phillips 66",
    "OXY": "Occidental Petroleum Corporation",
    "WMT": "Walmart Inc.",
    "HD": "The Home Depot, Inc.",
    "MCD": "McDonald's Corporation",
    "NKE": "NIKE, Inc.",
    "SBUX": "Starbucks Corporation",
    "TGT": "Target Corporation",
    "COST": "Costco Wholesale Corporation",
    "LOW": "Lowe's Companies, Inc.",
    "TJX": "The TJX Companies, Inc.",
    "DG": "Dollar General Corporation",
    # Company-disjoint FinanceBench held-out issuers. These are intentionally
    # absent from SECTORS, so they do not expand the default ingestion universe.
    "MMM": "3M Company",
    "AES": "The AES Corporation",
    "ATVI": "Activision Blizzard, Inc.",
    "ADBE": "Adobe Inc.",
    "AMCR": "Amcor plc",
    "AXP": "American Express Company",
    "AWK": "American Water Works Company, Inc.",
    "BBY": "Best Buy Co., Inc.",
    "XYZ": "Block, Inc.",
    "BA": "The Boeing Company",
    "KO": "The Coca-Cola Company",
    "GLW": "Corning Incorporated",
    "GIS": "General Mills, Inc.",
    "KHC": "The Kraft Heinz Company",
    "LMT": "Lockheed Martin Corporation",
    "MGM": "MGM Resorts International",
    "NFLX": "Netflix, Inc.",
    "PYPL": "PayPal Holdings, Inc.",
    "PEP": "PepsiCo, Inc.",
    "ULTA": "Ulta Beauty, Inc.",
    "VZ": "Verizon Communications Inc.",
}

# Flat list of all project tickers.
# 中文：默认抓取范围由行业映射派生，避免维护第二份容易漂移的列表。
ALL_TICKERS: list[str] = [t for tickers in SECTORS.values() for t in tickers]

# Reverse lookup: ticker -> sector.
# 中文：元数据解析优先使用此映射，避免依赖外部供应商的非标准行业标签。
TICKER_TO_SECTOR: dict[str, str] = {
    ticker: sector
    for sector, tickers in SECTORS.items()
    for ticker in tickers
}

# ---------------------------------------------------------------------------
# Time range
# ---------------------------------------------------------------------------

START_YEAR: int = 2018
END_YEAR: int = 2024
YEARS: list[int] = list(range(START_YEAR, END_YEAR + 1))

# Date strings for yfinance / FRED
START_DATE: str = f"{START_YEAR}-01-01"
END_DATE: str = f"{END_YEAR}-12-31"

# ---------------------------------------------------------------------------
# SEC filing types
# ---------------------------------------------------------------------------

FILING_TYPES: list[str] = ["10-K", "10-Q", "8-K"]
DEFAULT_FILING_TYPES: list[str] = ["10-K", "10-Q"]

# ---------------------------------------------------------------------------
# FRED macroeconomic series
# ---------------------------------------------------------------------------

FRED_SERIES: dict[str, str] = {
    "DFF": "Federal Funds Effective Rate",
    "CPIAUCSL": "Consumer Price Index (All Urban Consumers)",
    "GDP": "Gross Domestic Product",
    "UNRATE": "Civilian Unemployment Rate",
    "T10Y2Y": "10-Year minus 2-Year Treasury Yield Spread",
    "DCOILWTICO": "Crude Oil Prices: West Texas Intermediate",
    "VIXCLS": "CBOE Volatility Index (VIX)",
}

FRED_API_KEY: str = os.getenv("FRED_API_KEY", "")

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

DATABASE_URL: str = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5432/financial_rag",
)

DB_POOL_MIN: int = 1
DB_POOL_MAX: int = 10

# ---------------------------------------------------------------------------
# Embedding model
# ---------------------------------------------------------------------------

EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM: int = 384
EMBEDDING_BATCH_SIZE: int = 64

# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

CHUNK_SIZE_TOKENS: int = 512
CHUNK_OVERLAP_TOKENS: int = 64

# ---------------------------------------------------------------------------
# Download settings
# ---------------------------------------------------------------------------

SEC_MAX_REQUESTS_PER_SECOND: int = 10
SEC_USER_AGENT: str = os.getenv(
    "SEC_USER_AGENT",
    "financial-rag-pipeline research@example.com",
)
SEC_DOWNLOAD_DIR: str = os.getenv("SEC_DOWNLOAD_DIR", "./data/sec_filings")

# sec-edgar-downloader names a filing directory after the CIK when it cannot
# resolve a ticker, which happens for delisted issuers: Pioneer Natural
# Resources was acquired by Exxon in 2024, so its filings landed under
# 0001038357. Map those directories back to the ticker the rest of the pipeline
# (and the companies table) uses.
# 中文：退市或并购公司可能只能按 CIK 落盘；该映射把它还原为项目内统一使用的 ticker。
SEC_TICKER_ALIASES: dict[str, str] = {
    "0001038357": "PXD",
}

REQUEST_TIMEOUT: int = 30          # seconds
REQUEST_DELAY_SECONDS: float = 1.0  # polite delay between web requests

# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

DB_BATCH_SIZE: int = 1_000

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE: str = os.getenv("LOG_FILE", "./pipeline.log")
