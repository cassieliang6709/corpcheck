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
├── api/main.py            FastAPI: /retrieve, /chat, /filters, /health
└── mcp/                   Model Context Protocol server (stdio)
    ├── server.py          Tools: check_answerable, search_filings, get_filing_context
    └── provenance.py      Accession lookup + version governance on direct lookup

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

## MCP server

The same retrieval stack is exposed over the [Model Context
Protocol](https://modelcontextprotocol.io) so any MCP client — Claude Code, Claude
Desktop, or a custom agent — can query the filings directly.

```bash
.venv/bin/pip install -e ".[mcp]"
.venv/bin/corpcheck-mcp          # or: .venv/bin/python -m corpcheck.mcp
```

stdio transport, so the client launches the process; it needs the same `.env` and
the same populated Postgres the HTTP API does.

### Registering it

Claude Code:

```bash
claude mcp add corpcheck -- /abs/path/to/corpcheck/.venv/bin/corpcheck-mcp
```

Any client that reads a JSON config (`claude_desktop_config.json`, `.mcp.json`, …):

```json
{
  "mcpServers": {
    "corpcheck": {
      "command": "/abs/path/to/corpcheck/.venv/bin/corpcheck-mcp",
      "env": { "DB_HOST": "localhost", "DB_PORT": "5432", "DB_NAME": "financial_rag" }
    }
  }
}
```

Use an absolute path to the venv's script: the server imports `sentence-transformers`
and `asyncpg`, so it must run on the project interpreter.

### Tools

| Tool | Purpose |
| --- | --- |
| `check_answerable` | Whether the corpus can support an answer — **no LLM is contacted** |
| `search_filings` | Evidence blocks with company / filing type / fiscal year / period / accession |
| `get_filing_context` | Untruncated source text around a chunk, or a filing opened by accession |

All three route through `retrieval.pipeline.retrieve()`. The MCP layer is a
protocol adapter and contains no search, ranking, or filtering logic of its own —
which is what lets the offline IR numbers describe what an agent actually gets.

`check_answerable` is the tool that matters. Everything else here is a retrieval
API with better metadata; this one lets a client ask *"can you answer this?"*
before it commits to answering, and get a deterministic reply computed from
measured cosine similarities rather than from a model's self-assessment. An agent
that calls it first has a defensible reason to say "the filings do not cover
this" — which is the whole thesis of the project, exported to any client.

Version governance applies on both paths: superseded chunks are dropped from the
candidate pool inside `retrieve()`, and `get_filing_context` runs the same
supersession check before returning text, so an agent holding a stale accession
number cannot route around the filter.

### A real session

Transcript from `mcp.ClientSession` over stdio against the live corpus
(469,874 chunks, 1,662 filings), abridged only where marked.

**`check_answerable`, in-domain:**

```json
→ {"query": "What was Apple's total net sales in fiscal 2022?", "k": 5}

← {
    "answerable": true,
    "gate_status": "pass",
    "reason": "Evidence passed both confidence floors.",
    "llm_consulted": false,
    "similarity": {
      "top1_cos_sim": 0.70896,
      "mean_top3_cos_sim": 0.6743513333333334,
      "top1_min": 0.42,
      "mean_top3_min": 0.4
    },
    "coverage": {
      "retrieved": 5, "with_dense_score": 5, "sparse_only": 0,
      "companies": ["AAPL"], "filing_types": ["10-K", "10-Q"],
      "fiscal_years": [2022], "source_types": ["sec"]
    },
    "governance": {"revision_filter_enabled": true, "note": "..."}
  }
```

**`check_answerable`, out-of-domain — the same call, refusing:**

```json
→ {"query": "What is the best recipe for sourdough bread?", "k": 5}

← {
    "answerable": false,
    "gate_status": "below_top1_floor",
    "reason": "The filings searched do not contain passages relevant enough to answer this question.",
    "llm_consulted": false,
    "similarity": {
      "top1_cos_sim": 0.295165,
      "mean_top3_cos_sim": 0.29333866666666664,
      "top1_min": 0.42,
      "mean_top3_min": 0.4
    },
    "coverage": {"retrieved": 5, "companies": ["MPC"], "fiscal_years": [2020, 2023, 2024, 2025]}
  }
```

Note that retrieval still returned five chunks — it always does. The refusal comes
from measuring them, not from an empty result set.

**`search_filings`** (top hit of three shown):

```json
→ {"query": "Apple total net sales fiscal 2022", "k": 3, "company": "AAPL"}

← {"results": [{
    "chunk_id": "39192",
    "source_type": "sec",
    "company": "AAPL",
    "filing_type": "10-K",
    "fiscal_year": 2022,
    "period": "annual",
    "filed_date": "2022-10-28",
    "accession_number": "0000320193-22-000108",
    "cik": "0000320193",
    "section": "Selected Financial Data",
    "chunk_index": 56,
    "source_url": "https://www.sec.gov/Archives/edgar/data/320193/000032019322000108/0000320193-22-000108-index.htm",
    "score": 1.097368,
    "cos_sim": 0.736293,
    "text": "Apple Inc. | 2022 Form 10-K | 19 … Fiscal 2022 Highlights \n Total net sales increased 8% or $28.5 billion during 2022 compared to 2021, driven primarily by higher net sales of iPhone, Services and Mac. …",
    "text_truncated": false
  }],
  "abstain": {"would_abstain": false, "top1_cos_sim": 0.736293, "mean_top3_cos_sim": 0.670791},
  "governance": {"revision_filter_enabled": true}}
```

**`get_filing_context`**, widening that hit (chunk texts abridged):

```json
→ {"chunk_id": "39192", "window": 1}

← {
    "filing": {
      "company": "AAPL", "company_name": "Apple Inc.", "filing_type": "10-K",
      "fiscal_year": 2022, "period": "annual", "filed_date": "2022-10-28",
      "period_of_report": "2022-09-24",
      "accession_number": "0000320193-22-000108", "cik": "0000320193"
    },
    "superseded": false,
    "anchor_chunk_id": "39192",
    "chunks": [
      {"chunk_id": "39191", "chunk_index": 55, "section": "Mine Safety Disclosures", "is_anchor": false, "text": "[TABLE] Table 14 …"},
      {"chunk_id": "39192", "chunk_index": 56, "section": "Selected Financial Data", "is_anchor": true,  "text": "… Total net sales increased 8% or $28.5 billion during 2022 …"},
      {"chunk_id": "39193", "chunk_index": 57, "section": "Selected Financial Data", "is_anchor": false, "text": "… During 2022, the Company repurchased $90.2 billion of its common stock …"}
    ]
  }
```

`get_filing_context` also accepts `accession_number` instead of `chunk_id`, which
opens the filing from `chunk_index` 0.

### Measured limitation

The revision filter is wired into all three tools, but the corpus currently loaded
contains **no amended filings** — 0 of 1,662 `filings` rows have a `filing_type`
ending in `/A`. So the supersession path is covered by unit tests
(`tests/test_mcp_server.py`) and by `tests/test_revision.py`, not by the live
session above. Nothing here has been demonstrated to suppress a real superseded
chunk, because there is not yet a real superseded chunk to suppress.

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

## Background and attribution

CorpCheck grew out of [`CS6120_finance_RAG`](https://github.com/cassieliang6709/CS6120_finance_RAG),
a four-person project built for CS6120 (Natural Language Processing) at
Northeastern University by
[@RobynJiang](https://github.com/RobynJiang),
[@zhiyul1998](https://github.com/zhiyul1998),
[@CodeBusher](https://github.com/CodeBusher),
and [@cassieliang6709](https://github.com/cassieliang6709).
That project established the original idea: ingest SEC filings, retrieve over
them, and ground generated answers in the retrieved evidence.

This repository is a solo continuation. It is a separate project rather than a
branch of the original, so that the coursework repository stays intact for the
team that built it.

The work here is a rewrite rather than an increment: the service layer was
restructured into an installable package, and the parts that make the system
defensible for financial use were designed and built from scratch — the
deterministic IR evaluation harness, revision-aware filtering, rank-based
fusion, and the abstain gate. Where the course project answered *can we build
a RAG system over 10-Ks*, CorpCheck asks the harder question: *can we prove the
retrieval is correct, and make the system refuse when it is not*.

Ideas, structure, and problem framing from the original team are gratefully
acknowledged.

## Status

Phase 1 progress:

- [x] **Deterministic IR evaluation suite** — `Recall@k` / `MRR` with token-overlap
      weak supervision, provenance gating, and an oracle ceiling check.
- [x] **Revision-aware filtering** — `10-K/A` supersedes the `10-K` it amends;
      resolved on the candidate pool before fusion.
- [x] **Reciprocal Rank Fusion** — rank-based hybrid fusion, A/B-switchable
      against the previous min-max blend.
- [x] **Strict abstain gating** — `/chat` refuses before contacting the LLM when
      the retrieved evidence is too weak. Gated on raw dense cosine rather than
      the fused score, with thresholds calibrated against the corpus
      (`evaluation/calibrate_abstain.py`).
- [x] **MCP server** — `check_answerable` / `search_filings` / `get_filing_context`
      over stdio, routed through the same `retrieve()` entry point. Exports the
      abstain gate as something a client can query *before* answering.
