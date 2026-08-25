"""Own the process-wide asyncpg connection pool.

中文：查询服务复用一个进程级连接池，并在每条连接上注册 pgvector 支持。
"""

import asyncio
from typing import Optional

import asyncpg
from pgvector.asyncpg import register_vector
from corpcheck.settings import (
    DB_CONNECT_RETRIES,
    DB_CONNECT_RETRY_DELAY,
    DB_HOST,
    DB_PORT,
    DB_NAME,
    DB_USER,
    DB_PASSWORD,
    DB_POOL_MIN,
    DB_POOL_MAX,
)

_pool: Optional[asyncpg.Pool] = None


async def _init_connection(conn: asyncpg.Connection) -> None:
    """Enable pgvector and configure the vector-search probe count per connection.

    中文：连接创建时初始化扩展和 IVF 探测数；这是连接级设置，不能只在建池时设置一次。
    """
    await register_vector(conn)
    await conn.execute("SET ivfflat.probes = 10")


async def get_pool() -> asyncpg.Pool:
    """Return the lazily created shared pool, retrying transient database failures.

    中文：首次使用才连库，短暂连接错误按配置重试；成功后所有请求复用同一连接池。
    """
    global _pool
    if _pool is None:
        last_error: Exception | None = None
        for attempt in range(1, DB_CONNECT_RETRIES + 1):
            try:
                _pool = await asyncpg.create_pool(
                    host=DB_HOST,
                    port=DB_PORT,
                    database=DB_NAME,
                    user=DB_USER,
                    password=DB_PASSWORD,
                    min_size=DB_POOL_MIN,
                    max_size=DB_POOL_MAX,
                    init=_init_connection,
                )
                break
            except (
                ConnectionError,
                OSError,
                asyncpg.CannotConnectNowError,
                asyncpg.PostgresError,
            ) as exc:
                last_error = exc
                if attempt == DB_CONNECT_RETRIES:
                    raise
                await asyncio.sleep(DB_CONNECT_RETRY_DELAY)
        if _pool is None and last_error is not None:
            raise last_error
    return _pool


async def close_pool() -> None:
    """Close and clear the shared pool during application shutdown.

    中文：关闭后重置缓存引用，使下一次启动能够重新创建干净的连接池。
    """
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None
