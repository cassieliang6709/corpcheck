"""
Market data downloader (yfinance)
===================================
Downloads daily OHLCV price history, quarterly income statement, balance
sheet, and cash-flow data for every ticker in the universe.

All data is returned as plain Python dicts / lists ready for insertion into
PostgreSQL via the db_loader.
"""

from __future__ import annotations

import logging
import time
from datetime import date
from typing import Any

import pandas as pd
import yfinance as yf

from corpcheck.ingestion.config import (
    ALL_TICKERS,
    END_DATE,
    START_DATE,
)
from corpcheck.ingestion.metadata import resolve_company_metadata

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
PriceRow = dict[str, Any]
FinancialsRow = dict[str, Any]
CompanyInfo = dict[str, Any]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_float(value: Any) -> float | None:
    """Convert a value to float, returning None on failure."""
    try:
        if pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> int | None:
    try:
        if pd.isna(value):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _period_label(period_end: pd.Timestamp) -> str:
    """
    Convert a period-end Timestamp to a quarter label such as 'Q1',
    'Q2', 'Q3', or 'Q4'.
    """
    month = period_end.month
    if month <= 3:
        return "Q1"
    elif month <= 6:
        return "Q2"
    elif month <= 9:
        return "Q3"
    return "Q4"


def _get_df_value(df: pd.DataFrame, row_keys: list[str], col: pd.Timestamp) -> float | None:
    """
    Try multiple possible row labels (to handle yfinance API variations)
    and return the first non-null float found for the given column.
    """
    for key in row_keys:
        if key in df.index:
            val = _safe_float(df.loc[key, col])
            if val is not None:
                return val
    return None


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

class MarketDownloader:
    """
    Wraps yfinance to download price history and fundamental data.

    Parameters
    ----------
    start_date / end_date:
        ISO date strings for the download window.
    request_delay:
        Seconds to sleep between tickers to avoid rate-limiting.
    """

    def __init__(
        self,
        start_date: str = START_DATE,
        end_date: str = END_DATE,
        request_delay: float = 0.5,
    ) -> None:
        self.start_date = start_date
        self.end_date = end_date
        self.request_delay = request_delay

    # ------------------------------------------------------------------
    # Company info
    # ------------------------------------------------------------------

    def get_company_info(self, ticker: str) -> CompanyInfo:
        """
        Return a dict suitable for inserting into the ``companies`` table.
        """
        try:
            tkr = yf.Ticker(ticker)
            info = tkr.info or {}
            metadata = resolve_company_metadata(ticker, info)
            return {
                "ticker": ticker,
                "name": metadata["name"],
                "sector": metadata["sector"],
                "industry": metadata["industry"],
                "market_cap": _safe_float(info.get("marketCap")),
                "description": metadata["description"],
            }
        except Exception as exc:
            logger.warning("Could not fetch info for %s: %s", ticker, exc)
            metadata = resolve_company_metadata(ticker, {})
            return {
                "ticker": ticker,
                "name": metadata["name"],
                "sector": metadata["sector"],
                "industry": metadata["industry"],
                "market_cap": None,
                "description": metadata["description"],
            }

    # ------------------------------------------------------------------
    # Price history
    # ------------------------------------------------------------------

    def get_price_history(self, ticker: str) -> list[PriceRow]:
        """
        Return daily OHLCV rows for *ticker* over the configured date range.
        Each row is a dict matching the ``market_data`` table schema.
        """
        rows: list[PriceRow] = []
        try:
            tkr = yf.Ticker(ticker)
            hist: pd.DataFrame = tkr.history(
                start=self.start_date,
                end=self.end_date,
                auto_adjust=True,
                actions=False,
            )
            if hist.empty:
                logger.warning("No price history for %s", ticker)
                return rows

            info = tkr.info or {}

            for ts, row in hist.iterrows():
                rows.append(
                    {
                        "ticker": ticker,
                        "date": ts.date(),
                        "open": _safe_float(row.get("Open")),
                        "high": _safe_float(row.get("High")),
                        "low": _safe_float(row.get("Low")),
                        "close": _safe_float(row.get("Close")),
                        "adj_close": _safe_float(row.get("Close")),  # already adjusted
                        "volume": _safe_int(row.get("Volume")),
                        "market_cap": _safe_float(info.get("marketCap")),
                        "pe_ratio": _safe_float(info.get("trailingPE")),
                        "pb_ratio": _safe_float(info.get("priceToBook")),
                        "ps_ratio": _safe_float(info.get("priceToSalesTrailing12Months")),
                        "dividend_yield": _safe_float(info.get("dividendYield")),
                        "beta": _safe_float(info.get("beta")),
                    }
                )
            logger.debug("Fetched %d price rows for %s", len(rows), ticker)
        except Exception as exc:
            logger.warning("Price history error for %s: %s", ticker, exc)
        return rows

    # ------------------------------------------------------------------
    # Fundamentals
    # ------------------------------------------------------------------

    def get_financials(self, ticker: str) -> list[FinancialsRow]:
        """
        Return quarterly financial rows for *ticker*.
        Merges income statement, balance sheet, and cash-flow data by
        period-end date.
        """
        rows: list[FinancialsRow] = []
        try:
            tkr = yf.Ticker(ticker)

            income_q = tkr.quarterly_income_stmt
            balance_q = tkr.quarterly_balance_sheet
            cashflow_q = tkr.quarterly_cashflow

            if income_q is None or income_q.empty:
                logger.warning("No quarterly income statement for %s", ticker)
                return rows

            for col in income_q.columns:
                period_end: pd.Timestamp = col
                fiscal_year = period_end.year
                period = _period_label(period_end)

                # --- income statement ---
                revenue = _get_df_value(
                    income_q,
                    ["Total Revenue", "Revenue", "Revenues"],
                    col,
                )
                gross_profit = _get_df_value(
                    income_q,
                    ["Gross Profit"],
                    col,
                )
                operating_income = _get_df_value(
                    income_q,
                    ["Operating Income", "EBIT"],
                    col,
                )
                net_income = _get_df_value(
                    income_q,
                    ["Net Income", "Net Income Common Stockholders"],
                    col,
                )
                eps_basic = _get_df_value(
                    income_q,
                    ["Basic EPS", "EPS Basic"],
                    col,
                )
                eps_diluted = _get_df_value(
                    income_q,
                    ["Diluted EPS", "EPS Diluted"],
                    col,
                )
                ebitda = _get_df_value(income_q, ["EBITDA"], col)

                # --- balance sheet ---
                total_assets = None
                total_liabilities = None
                total_debt = None
                shareholders_equity = None
                cash = None

                if balance_q is not None and not balance_q.empty and col in balance_q.columns:
                    total_assets = _get_df_value(balance_q, ["Total Assets"], col)
                    total_liabilities = _get_df_value(
                        balance_q, ["Total Liabilities Net Minority Interest", "Total Liabilities"], col
                    )
                    total_debt = _get_df_value(
                        balance_q, ["Total Debt", "Long Term Debt And Capital Lease Obligation"], col
                    )
                    shareholders_equity = _get_df_value(
                        balance_q,
                        ["Stockholders Equity", "Total Stockholders Equity", "Common Stock Equity"],
                        col,
                    )
                    cash = _get_df_value(
                        balance_q,
                        ["Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments"],
                        col,
                    )

                # --- cash flow ---
                operating_cf = None
                capex = None
                free_cf = None

                if cashflow_q is not None and not cashflow_q.empty and col in cashflow_q.columns:
                    operating_cf = _get_df_value(
                        cashflow_q,
                        ["Operating Cash Flow", "Cash Flow From Continuing Operating Activities"],
                        col,
                    )
                    capex = _get_df_value(
                        cashflow_q,
                        ["Capital Expenditure", "Purchase Of PPE"],
                        col,
                    )
                    free_cf = _get_df_value(
                        cashflow_q,
                        ["Free Cash Flow"],
                        col,
                    )
                    # Compute FCF if not directly available
                    if free_cf is None and operating_cf is not None and capex is not None:
                        free_cf = operating_cf + capex  # capex is usually negative

                rows.append(
                    {
                        "ticker": ticker,
                        "fiscal_year": fiscal_year,
                        "period": period,
                        "period_end_date": period_end.date(),
                        "revenue": revenue,
                        "gross_profit": gross_profit,
                        "operating_income": operating_income,
                        "net_income": net_income,
                        "eps_basic": eps_basic,
                        "eps_diluted": eps_diluted,
                        "ebitda": ebitda,
                        "total_assets": total_assets,
                        "total_liabilities": total_liabilities,
                        "total_debt": total_debt,
                        "shareholders_equity": shareholders_equity,
                        "cash_and_equivalents": cash,
                        "operating_cash_flow": operating_cf,
                        "capex": capex,
                        "free_cash_flow": free_cf,
                    }
                )

            logger.debug("Fetched %d financial rows for %s", len(rows), ticker)
        except Exception as exc:
            logger.warning("Financials error for %s: %s", ticker, exc)
        return rows

    # ------------------------------------------------------------------
    # Convenience: download everything for a list of tickers
    # ------------------------------------------------------------------

    def download_all(
        self,
        tickers: list[str] | None = None,
    ) -> tuple[list[CompanyInfo], list[PriceRow], list[FinancialsRow]]:
        """
        Download company info, price history, and financials for all
        *tickers* (default: ``ALL_TICKERS``).

        Returns
        -------
        (company_infos, price_rows, financial_rows)
        """
        tickers = tickers or ALL_TICKERS
        all_info: list[CompanyInfo] = []
        all_prices: list[PriceRow] = []
        all_financials: list[FinancialsRow] = []

        for i, ticker in enumerate(tickers, 1):
            logger.info("[%d/%d] Downloading market data for %s", i, len(tickers), ticker)
            all_info.append(self.get_company_info(ticker))
            all_prices.extend(self.get_price_history(ticker))
            all_financials.extend(self.get_financials(ticker))
            if self.request_delay > 0:
                time.sleep(self.request_delay)

        logger.info(
            "Market download complete: %d companies, %d price rows, %d financial rows",
            len(all_info),
            len(all_prices),
            len(all_financials),
        )
        return all_info, all_prices, all_financials


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------

def download_market_data(
    tickers: list[str] | None = None,
    start_date: str = START_DATE,
    end_date: str = END_DATE,
) -> tuple[list[CompanyInfo], list[PriceRow], list[FinancialsRow]]:
    """Shorthand for ``MarketDownloader().download_all()``."""
    dl = MarketDownloader(start_date=start_date, end_date=end_date)
    return dl.download_all(tickers)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    info, prices, fins = download_market_data(["AAPL"])
    print("Info:", info[0])
    print("Price rows:", len(prices))
    print("Financial rows:", len(fins))
