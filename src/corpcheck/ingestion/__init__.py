"""
Financial Research RAG Data Pipeline
=====================================
End-to-end pipeline for downloading, processing, embedding, and loading
financial documents (SEC filings, market data, macro indicators, news,
and earnings call transcripts) into a PostgreSQL + pgvector database.

中文：离线数据管道的包入口。它把不同来源的金融资料处理成可检索的数据库记录；
具体下载、清洗和入库职责分别位于各个子模块。
"""

__version__ = "1.0.0"
__author__ = "NLP Course Project"
