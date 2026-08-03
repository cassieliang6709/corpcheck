"""Service-layer settings for the retrieval API, LLM client, and evaluation harness.

Ingestion keeps its own configuration in :mod:`corpcheck.ingestion.config` — that
module owns the company universe, year ranges, and chunking hyperparameters. This
module owns everything the query-time path needs.

Shell environment variables always take precedence over the `.env` file.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()


def _env_float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


def _env_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

EMBEDDING_MODEL: str = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

DB_HOST: str = os.getenv("DB_HOST", "localhost")
DB_PORT: int = _env_int("DB_PORT", 5432)
DB_NAME: str = os.getenv("DB_NAME", "financial_rag")
DB_USER: str = os.getenv("DB_USER", "postgres")
DB_PASSWORD: str = os.getenv("DB_PASSWORD", "postgres")

DB_POOL_MIN: int = _env_int("DB_POOL_MIN", 1)
DB_POOL_MAX: int = _env_int("DB_POOL_MAX", 10)
DB_CONNECT_RETRIES: int = _env_int("DB_CONNECT_RETRIES", 10)
DB_CONNECT_RETRY_DELAY: float = _env_float("DB_CONNECT_RETRY_DELAY", 2.0)

# ---------------------------------------------------------------------------
# Retrieval defaults
# ---------------------------------------------------------------------------

DEFAULT_K: int = _env_int("DEFAULT_K", 5)
DEFAULT_ALPHA: float = _env_float("DEFAULT_ALPHA", 0.7)

# Multiplicative boosts applied inside the SQL ORDER BY so they influence which
# rows the database returns, not just how the returned rows are ordered.
# Set any of these to 1.0 to disable.
COMPANY_BOOST: float = _env_float("COMPANY_BOOST", 1.5)
FILING_TYPE_BOOST: float = _env_float("FILING_TYPE_BOOST", 1.3)
FISCAL_YEAR_BOOST: float = _env_float("FISCAL_YEAR_BOOST", 1.3)

# ---------------------------------------------------------------------------
# LLM (OpenAI-compatible SGLang endpoint, backend-only — never exposed to clients)
# ---------------------------------------------------------------------------

SGLANG_BASE_URL: str = os.getenv("SGLANG_BASE_URL", "")
SGLANG_MODEL: str = os.getenv("SGLANG_MODEL", "Qwen/Qwen3.5-397B-A17B-FP8")
SGLANG_API_KEY: str = os.getenv("SGLANG_API_KEY", "")
SGLANG_MAX_TOKENS: int = _env_int("SGLANG_MAX_TOKENS", 32768)

# ---------------------------------------------------------------------------
# API surface
# ---------------------------------------------------------------------------

# Public /chat endpoint auth — empty string disables auth entirely.
API_KEY: str = os.getenv("API_KEY", "")

# CORS — "*" or a comma-separated list of origins.
CORS_ORIGINS: list[str] = [
    o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",") if o.strip()
]
