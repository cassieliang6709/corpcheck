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

## Evaluation

Retrieval quality is measured deterministically against FinanceBench, in-process
against `retrieve()` — no HTTP server and no LLM, so a change in the numbers is
attributable to retrieval alone.

```bash
.venv/bin/python -m evaluation.ir_eval --label baseline
```

Reports `Recall@{1,3,5,10}`, `Hit@k`, and `MRR@10`, plus ablations and a
threshold sweep. Per-query detail lands in `evaluation/runs/<label>/`, so two
configurations can be diffed directly.

**How a hit is defined.** FinanceBench gives a gold evidence *span* — a page or
table lifted from the filing — not a chunk id. Our chunk boundaries differ, so
exact matching is impossible. A retrieved chunk counts as covering a gold span
when it contains at least `--threshold` (default 0.5) of that span's
content-bearing tokens, after normalisation, boilerplate removal, and stopword
removal. Numbers keep their thousands separators collapsed (`11,588` → `11588`)
because the exact figure is the most discriminative token in a filing table.

Two rules keep the measurement honest:

- **No gold metadata reaches the retriever.** Only the question text is passed —
  never the benchmark's company, period, or filing type. Supplying those as
  filters would benchmark a system that does not exist at serving time.
- **Provenance is checked before content.** A chunk must come from the filing the
  question is about before its overlap counts, because the right sentence from
  the wrong fiscal year is exactly the failure this project exists to prevent.
  `--no-doc-gate` shows what the gate costs.

### Validating the metric itself

A weak-supervision metric can read zero because retrieval is bad *or* because the
threshold is unreachable given the chunk size. To tell those apart:

```bash
.venv/bin/python -m evaluation.oracle
```

This computes the best overlap achievable by any chunk of the correct filing —
perfect oracle retrieval — and reports the resulting ceiling per threshold. Run
it whenever the corpus or the chunking strategy changes. Any `ir_eval` result
above the ceiling indicates a scoring bug.

## Status

Ported from the prior `NLP-project` codebase with the service layer restructured
into a proper package; behavior is unchanged from the port.

Phase 1 progress:

- [x] **Deterministic IR evaluation suite** — `Recall@k` / `MRR` with token-overlap
      weak supervision, provenance gating, and an oracle ceiling check.
- [ ] Revision-aware filtering (`10-K/A` supersedes `10-K`)
- [ ] Reciprocal Rank Fusion for hybrid search
- [ ] Strict abstain gating
