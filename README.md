# CorpCheck

Audit-grade fundamental research over SEC filings.

CorpCheck is not a generic RAG wrapper. It is built on the premise that generic
retrieval fails on financial disclosure because it treats a 10-K like prose:
it ignores that filings are *versioned* (a `10-K/A` replaces the sections it
amends while unchanged sections may still live only in the original), that
management buries material risk in qualified language, and that a
confidently wrong number is worse than no number at all.

Three principles drive the design:

1. **Deterministic IR evaluation over vibes.** Retrieval quality is measured with
   `Recall@k` and `MRR` against gold evidence, not eyeballed.
2. **Strict provenance and version control.** Every chunk traces to a filing, and
   superseded sections are excluded without discarding unchanged disclosure.
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

Version governance applies on both paths: chunks from amended sections are
dropped from the candidate pool inside `retrieve()`, while unchanged sections
from the original remain available. `get_filing_context` runs the same
section-aware check before returning text, so an agent holding a stale chunk id
cannot route around the filter.

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
ending in `/A`. The repository now includes a fixture based on GameStop's real
March 2024 10-K/10-K/A pair. That amendment changes Item 5 only, so the tests
prove section-aware composition: the original Item 5 is suppressed, the amended
Item 5 survives, and unrelated original financial-statement sections remain
available. This is still a repository-contained test with real SEC metadata, not
a live-corpus demonstration.

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

Final-answer behavior has a separate deterministic scorer:

```bash
.venv/bin/python -m evaluation.answer_eval \
  --predictions predictions.jsonl \
  --gold gold.json \
  --output evaluation/runs/<label>/answer-eval.json
```

It measures answer correctness, correct abstention, citation presence and index
validity, a conservative supporting-chunk proxy, and an end-to-end pass rate.
It consumes recorded `/chat`-shaped outputs and never contacts a model or the
database while scoring. Supporting chunk ids measure provenance, not semantic
entailment.

A small no-filter seed is included for exercising the pre-generation path:

```bash
HF_HUB_OFFLINE=1 .venv/bin/python -m evaluation.collect_answer_predictions \
  --retrieval-only \
  --output evaluation/runs/answer-seed/predictions.jsonl
```

On its nine hand-selected records, the metadata coverage gate improved correct
answerability decisions from 7/9 to 9/9: all seven answerable questions remained
allowed, while a nonexistent Costco FY2099 request and an issuer absent from the
corpus were refused. This is a narrow regression seed, not a generation-quality
benchmark.

A subsequent local, no-cost generation run used Qwen 2.5 7B through Ollama with
an 8,192-token context. It answered 1/7 answerable questions correctly, included
valid citation indices on 5/7, correctly handled all nine answerability
decisions, and passed the combined end-to-end rule on 3/9 records (the one
correct cited answer plus both correct refusals). Manual failure analysis found
that six of seven answerable questions lacked the gold evidence in the retrieved
chunks, so this run primarily confirms that retrieval remains the bottleneck; it
is not a claim about larger hosted models.

The answer-evaluation set has since been expanded to 34 manually curated
FinanceBench records. One additional source record is excluded explicitly: its
published answer is `77.78`, while its own evidence says approximately $700
million total cost with 90% already incurred, implying about $70 million
remaining. On the 34 usable records, the same local 7B/8K setup produced this
pre-improvement baseline:

- answerability decisions: 33/34 correct;
- final-answer correctness: 1/34;
- present and in-range citation indices: 22/34;
- combined end-to-end pass: 1/34.

These numbers are intentionally reported before retrieval repair. Valid citation
indices do not establish that a citation supports the claim, and the 34 outputs
are still undergoing manual failure classification. The local artifacts live in
`evaluation/runs/R7_answer_financebench34/` and are gitignored.

Failure analysis of the current strict misses found that 19 of 30 reachable
misses are table-retrieval failures. A subsequent isolated experiment built
130,084 searchable row children from 29,570 benchmark-scoped table parents while
returning the existing parent chunk for citation. It improved loose Recall@10
from 0.4714 to 0.5143 but left strict Recall@10 unchanged at 0.1429 and increased
p95 latency from 461 ms to 2,374 ms. The arm therefore remains disabled by
default. Its parser, resumable indexer, feature flag, and full stop/continue
decision are retained for reproducibility; see
`evaluation/NEXT_RETRIEVAL_EXPERIMENT.md`.

A second experiment hard-filtered candidates to a query-derived company,
single fiscal year, and filing type. It was also rejected: strict Recall@10 did
not move from 0.1429, while warm p95 latency increased from 280 ms to 1,198 ms.
The rejected branch is not retained in production code.

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

## Current plan

The next work is evidence coverage, not a larger language model or more prompt
tuning.

1. **Do not ship the filing-local disjunctive sparse arm.** The evaluation-only
   prototype moved strict Recall@10 from 0.1429 to 0.2000 (5 to 7 hit queries),
   below the predeclared 0.2500 and five-recovery gates, while loose Recall@10
   fell from 0.4714 to 0.4429. Its six metric rules came from known failures,
   so the result is diagnostic rather than evidence of generalisation; the
   prototype was removed instead of entering production.
2. **Do not ship the generic filing-local hybrid either.** A second prototype
   used only normal query parsing, filing-local dense similarity, and a
   mechanically derived OR-sparse query. On the 35-query development set it
   moved strict Recall@10 only from 0.1429 to 0.1714, below the 0.2500 gate.
   Loose Recall@10 improved from 0.4714 to 0.5000 and zero-provenance queries
   fell from 4 to 3, but those secondary gains do not override the strict gate.
3. **Move the next retrieval experiment to representation and reranking.** The
   remaining reachable misses are dominated by large tables whose generic
   titles and distant headers weaken both sparse and dense ranking. Inspection
   of current gold parents confirmed the representation failure: relevant Amazon
   and Nike statement tables are stored as `Table 119` and `Table 117`, with
   `header_text = null`, even though their opening rows contain the reporting
   period and years. The next arm will first recover semantic table titles and
   `<td>`-encoded header/year rows, re-ingest a small isolated filing set, and
   measure whether the gold parents enter the candidate pool. Only if they enter
   the pool but still miss top 10 will a generic reranker be added.
4. **Create a real held-out contract before tuning that arm.** The current 35
   questions have informed multiple designs and are development data, not proof
   of generalisation. An audit of the official 150-record open-source
   FinanceBench file found 115 records outside this slice, but only two point to
   a supported filing already present in this corpus (`AMAZON_2017_10K`). Two
   questions from one issuer are not a credible held-out benchmark. The corpus
   therefore needs new issuers and filing years, followed by a predeclared split
   and gates, before representation or reranking gains can support a production
   claim. Benchmark PDFs may be used for parser diagnosis, but not as a substitute
   for validating the production SEC ingestion path.
5. **Repair corpus coverage separately.** The CVS FY2018 turnover evidence is
   absent from the indexed FY2018 filing chunks, so query/ranking changes cannot
   recover it. The cleaner now has a fail-closed path for a 10-K that explicitly
   incorporates an `ARS`/`EX-13*` annual report: it retains only recognised
   Financial Statements segments and ignores unrelated exhibits. Synthetic
   regression coverage includes the three missing CVS values; the exact
   accession still needs isolated redownload, re-ingestion, and chunk verification.
6. **Complete live amendment validation.** The isolated GameStop 2024
   10-K/10-K/A validator is implemented and fail-closed. Running it requires a
   real two-token SEC contact user-agent (`CorpCheck email@example.com`); it
   preflights an empty target before schema creation and will never write to the
   benchmark database.
7. **Rerun answer generation only after retrieval clears its gates.** Reuse the
   curated 34-record gold and compare against the 1/34 baseline only after strict
   Recall@10 improves on development data without a held-out regression. Build a
   small demo only after answer correctness and citation support are credible.

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
- [x] **Section-aware amendment composition** — amended sections replace their
      original counterparts before fusion, while unchanged original sections
      remain available.
- [x] **Reciprocal Rank Fusion** — rank-based hybrid fusion, A/B-switchable
      against the previous min-max blend.
- [x] **Strict abstain gating** — `/chat` refuses before contacting the LLM when
      the retrieved evidence is too weak. Gated on raw dense cosine rather than
      the fused score, with thresholds calibrated against the corpus
      (`evaluation/calibrate_abstain.py`). Explicit issuer and fiscal-year
      coverage mismatches are also refused before generation.
- [x] **MCP server** — `check_answerable` / `search_filings` / `get_filing_context`
      over stdio, routed through the same `retrieve()` entry point. Exports the
      abstain gate as something a client can query *before* answering.
