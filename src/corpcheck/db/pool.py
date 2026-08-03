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
    await register_vector(conn)
    await conn.execute("SET ivfflat.probes = 10")


async def get_pool() -> asyncpg.Pool:
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
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None
