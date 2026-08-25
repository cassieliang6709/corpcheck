"""Database access: connection pooling and schema.

中文：此包只导出共享连接池的生命周期函数；调用方不应自行创建临时连接池。
"""

from corpcheck.db.pool import close_pool, get_pool

__all__ = ["get_pool", "close_pool"]
