"""Database loading adapters for the offline ingestion pipeline.

中文：加载器把已处理的记录批量写入 PostgreSQL/pgvector；其 SQL 事务边界是数据恢复策略的一部分。
"""
