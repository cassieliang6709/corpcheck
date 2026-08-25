"""Query-time retrieval: candidate generation, fusion, and ranking.

中文：公开入口将自然语言问题转换为带溯源信息的 SEC 片段；各子模块分别负责解析、
搜索、融合和修订过滤。
"""

from corpcheck.retrieval.pipeline import retrieve
from corpcheck.retrieval.query_parse import load_known_tickers

__all__ = ["retrieve", "load_known_tickers"]
