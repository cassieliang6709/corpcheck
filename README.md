# CorpCheck

Audit-grade fundamental research over SEC filings.

CorpCheck is not a generic RAG wrapper. It is built on the premise that generic
retrieval fails on financial disclosure because it treats a 10-K like prose:
it ignores that filings are *versioned* (a `10-K/A` supersedes the `10-K` it
amends), that management buries material risk in qualified language, and that a
confidently wrong number is worse than no number at all.

Three principles drive the design:

1. **Deterministic IR evaluation over vibes.** Retrieval quality is measured with
   `Recall@k` and `MRR` against gold evidence, not eyeballed.
2. **Strict provenance and version control.** Every chunk traces to a filing, and
   superseded filings are excluded from the evidence set.
3. **Abstain beats hallucinate.** Below a retrieval-confidence floor the system
   refuses deterministically, without consulting the LLM.

## Layout

```
src/corpcheck/
├── settings.py            Service-layer config (DB, embeddings, LLM, API)
├── models.py              Pydantic request/response contracts
├── db/
│   ├── pool.py            asyncpg pool with pgvector registration
│   └── schema.sql         Tables, indexes, and retrieval views
├── ingestion/             Offline: EDGAR download → clean → chunk → embed → load
│   ├── pipeline.py        Orchestrator + CLI
│   ├── downloaders/       EDGAR, market, macro, news, transcripts
│   ├── processors/        HTML cleaning, sectioning, chunking, embedding
│   └── loaders/           Postgres writes
├── retrieval/             Query-time
│   ├── pipeline.py        retrieve(): the single entry point
│   ├── query_parse.py     Company / filing-type / fiscal-year detection
│   ├── search.py          Dense (pgvector) + sparse (ts_rank) candidate generation
│   ├── fusion.py          Score fusion strategy
│   └── rerank.py          Evidence-form adjustments, citation titles
├── llm/chat.py            Grounded answer generation (OpenAI-compatible endpoint)
└── api/main.py            FastAPI: /retrieve, /chat, /filters, /health

evaluation/                Offline IR + end-to-end evaluation harness
tests/                     Unit tests
```

`retrieval.pipeline.retrieve()` is deliberately the only retrieval entry point,
so the HTTP API and the offline evaluation harness exercise identical code.

## Setup

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[ingestion,dev]"
```

Serving queries against an already-populated database needs only the base
dependencies; the `ingestion` extra is required to build the corpus.

```bash
cp .env.example .env
```

## Running the API

```bash
.venv/bin/uvicorn corpcheck.api.main:app --reload --port 8000
```

`/retrieve` needs only Postgres. `/chat` additionally needs `SGLANG_BASE_URL`
pointing at an OpenAI-compatible endpoint; it returns 503 when unset.

## Data

The corpus lives in PostgreSQL with pgvector. Schema is in
[schema.sql](src/corpcheck/db/schema.sql); the retrieval path reads the
`v_retrieval_chunks` view, which unions SEC filing chunks, news chunks, and
earnings-call transcript chunks behind one interface.

## Status

Ported from the prior `NLP-project` codebase with the service layer restructured
into a proper package. Behavior is unchanged from the port; the Phase 1 patches
(IR evaluation, revision-aware filtering, RRF fusion, abstain gating) are in
progress.
