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


def _env_bool(name: str, default: bool) -> bool:
    """Parse a boolean env var. Anything unrecognised falls back to ``default``."""
    raw = os.getenv(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


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

# When the question names a company, restrict the candidate pool to that issuer
# instead of merely boosting it. A 1.5x boost is a preference, and across ~470k
# chunks a strongly-worded question about a peer can still out-score the right
# issuer; a question about AMD should never be answered from Intel's 10-K.
# Applies only to a company the parser detected in the query text — an explicit
# `company` argument is already a hard filter. Falls back to the unscoped pool
# when the scoped search finds nothing, so a mis-detection degrades to the old
# behaviour rather than to an empty answer.
COMPANY_SCOPE_ENABLED: bool = _env_bool("COMPANY_SCOPE_ENABLED", True)

# Drop chunks from filings that a later amendment has superseded (a 10-K/A
# replaces the 10-K it amends). On by default: serving a figure the filer has
# since restated is the most damaging error this system can make. Exposed as a
# toggle so the IR suite can measure what the filter costs in recall.
REVISION_FILTER_ENABLED: bool = _env_bool("REVISION_FILTER_ENABLED", True)

# Hybrid search fusion strategy: "rrf" (Reciprocal Rank Fusion) or "minmax"
FUSION_STRATEGY: str = os.getenv("FUSION_STRATEGY", "rrf").lower()
RRF_K: int = _env_int("RRF_K", 60)

# Double-threshold abstain gating. These are floors on the raw, unboosted dense
# cosine similarity (ChunkResult.cos_sim) — NOT on the fused ranking score, which
# is min-max normalised and therefore always 1.0 at rank 1 regardless of
# relevance. See retrieval/abstain.py for why, and evaluation/calibrate_abstain.py
# for the measurement these defaults come from.
#
# Calibrated on all-MiniLM-L6-v2 over this corpus: in-domain queries bottom out
# at top1=0.566 / mean3=0.554, out-of-domain queries top out at 0.342 / 0.335.
# The defaults sit in that gap, nearer the out-of-domain edge. Re-run the
# calibration if the embedding model or the corpus changes.
ABSTAIN_TOP1_MIN: float = _env_float("ABSTAIN_TOP1_MIN", 0.42)
ABSTAIN_MEAN_TOP3_MIN: float = _env_float("ABSTAIN_MEAN_TOP3_MIN", 0.40)


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
