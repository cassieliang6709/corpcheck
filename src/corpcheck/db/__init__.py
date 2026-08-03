"""Database access: connection pooling and schema."""

from corpcheck.db.pool import close_pool, get_pool

__all__ = ["get_pool", "close_pool"]
