"""
Macroeconomic data downloader (FRED)
======================================
Downloads the configured FRED series for the 2019-2023 window and returns
rows ready for the ``macro_indicators`` table.  Different series have
different native frequencies (daily, monthly, quarterly); we forward-fill
to a common daily index so downstream consumers can join on date easily.

中文：不同频率的宏观指标会向前填充到日粒度，便于与交易日数据关联；原始发布频率不会在此
模块中被推断或修正。
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
from fredapi import Fred

from corpcheck.ingestion.config import (
    END_DATE,
    FRED_API_KEY,
    FRED_SERIES,
    START_DATE,
)

logger = logging.getLogger(__name__)

MacroRow = dict[str, Any]


# ---------------------------------------------------------------------------
# Downloader
# ---------------------------------------------------------------------------

class MacroDownloader:
    """
    Download FRED series and return rows for ``macro_indicators``.

    Parameters
    ----------
    api_key:
        FRED API key.  If empty the Fred client will attempt to read the
        ``FRED_API_KEY`` environment variable automatically.
    start_date / end_date:
        ISO date strings for the download window.
    fill_method:
        Pandas fill-forward method applied to non-daily series.
        'ffill' propagates the last observation; use 'bfill' for backward fill.

    中文：FRED 的下载适配器。输出是可直接入库的日频字典；填充只处理频率差异，不能补回
    数据供应商完全缺失的观测值。
    """

    def __init__(
        self,
        api_key: str = FRED_API_KEY,
        start_date: str = START_DATE,
        end_date: str = END_DATE,
        fill_method: str = "ffill",
    ) -> None:
        self.start_date = start_date
        self.end_date = end_date
        self.fill_method = fill_method
        self._fred = Fred(api_key=api_key) if api_key else Fred()

    # ------------------------------------------------------------------
    # Single series
    # ------------------------------------------------------------------

    def fetch_series(
        self,
        series_id: str,
        series_name: str,
    ) -> list[MacroRow]:
        """
        Download one FRED series and return a list of daily ``MacroRow``
        dicts suitable for bulk-inserting into ``macro_indicators``.

        Non-daily observations are forward-filled to daily frequency so
        the output always has one row per calendar day in the window.

        Parameters
        ----------
        series_id:
            FRED series identifier (e.g. ``"DFF"``).
        series_name:
            Human-readable label stored in the ``series_name`` column.

        中文：单个序列失败只记录告警并返回空列表，允许批量任务继续处理其他指标。
        """
        rows: list[MacroRow] = []
        try:
            raw: pd.Series = self._fred.get_series(
                series_id,
                observation_start=self.start_date,
                observation_end=self.end_date,
            )

            if raw is None or raw.empty:
                logger.warning("No data returned for FRED series %s", series_id)
                return rows

            # Ensure DatetimeIndex
            raw.index = pd.to_datetime(raw.index)
            raw.name = series_id

            # Create a complete daily date range and reindex
            daily_index = pd.date_range(
                start=self.start_date,
                end=self.end_date,
                freq="D",
            )
            reindexed = raw.reindex(daily_index)

            # Forward-fill gaps (weekend / holiday / low-frequency releases)
            if self.fill_method == "ffill":
                reindexed = reindexed.ffill()
            elif self.fill_method == "bfill":
                reindexed = reindexed.bfill()

            # Drop days where we still have no value (start of series)
            reindexed = reindexed.dropna()

            for ts, value in reindexed.items():
                rows.append(
                    {
                        "date": ts.date(),
                        "indicator_id": series_id,
                        "series_name": series_name,
                        "value": float(value),
                    }
                )

            logger.debug(
                "Fetched %d rows for FRED series %s (%s)",
                len(rows),
                series_id,
                series_name,
            )
        except Exception as exc:
            logger.warning("Error fetching FRED series %s: %s", series_id, exc)

        return rows

    # ------------------------------------------------------------------
    # All configured series
    # ------------------------------------------------------------------

    def download_all(
        self,
        series: dict[str, str] | None = None,
    ) -> list[MacroRow]:
        """
        Download all series in *series* (default: ``FRED_SERIES`` from
        config) and return a flat list of ``MacroRow`` dicts.

        中文：按配置顺序下载；返回扁平列表，而不是把来源数据帧泄漏给加载层。
        """
        series = series or FRED_SERIES
        all_rows: list[MacroRow] = []

        for series_id, series_name in series.items():
            logger.info("Downloading FRED series: %s (%s)", series_id, series_name)
            rows = self.fetch_series(series_id, series_name)
            all_rows.extend(rows)

        logger.info(
            "Macro download complete: %d total rows across %d series",
            len(all_rows),
            len(series),
        )
        return all_rows

    # ------------------------------------------------------------------
    # Pivot helper (optional, for analysis)
    # ------------------------------------------------------------------

    def as_dataframe(
        self,
        series: dict[str, str] | None = None,
    ) -> pd.DataFrame:
        """
        Return a wide-format DataFrame with one column per series,
        indexed by date.  Useful for exploratory analysis.

        中文：这是分析辅助接口，不参与主导入流程，也不会改变已下载的数据。
        """
        series = series or FRED_SERIES
        frames: dict[str, pd.Series] = {}

        for series_id, series_name in series.items():
            rows = self.fetch_series(series_id, series_name)
            if rows:
                s = pd.Series(
                    {r["date"]: r["value"] for r in rows},
                    name=series_id,
                )
                frames[series_id] = s

        if not frames:
            return pd.DataFrame()

        df = pd.DataFrame(frames)
        df.index = pd.to_datetime(df.index)
        df.index.name = "date"
        return df.sort_index()


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------

def download_macro_data(
    api_key: str = FRED_API_KEY,
    start_date: str = START_DATE,
    end_date: str = END_DATE,
) -> list[MacroRow]:
    """Shorthand for ``MacroDownloader().download_all()``.

    中文：为简单脚本保留的函数式入口；需要复用客户端时请直接使用 ``MacroDownloader``。
    """
    dl = MacroDownloader(api_key=api_key, start_date=start_date, end_date=end_date)
    return dl.download_all()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    rows = download_macro_data()
    if rows:
        print(f"Downloaded {len(rows)} macro rows")
        # Show one row per series
        seen: set[str] = set()
        for r in rows:
            if r["indicator_id"] not in seen:
                print(r)
                seen.add(r["indicator_id"])
