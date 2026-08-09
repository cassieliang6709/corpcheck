# Local database restore

Bring up a local Postgres + pgvector with the `financial_rag` corpus, matching
the corpus `evaluation/RESULTS.md` was measured against.

## What you need

- Docker
- The corpus dump as `financial_rag.dump` — **not in this repo** (too large).
  Get it from the project maintainer and place it in this directory.

## Start

```bash
# from this directory
docker compose up -d db
```

`init-db.sh` runs once on first start (empty volume) and `pg_restore`s the dump.
First restore takes a few minutes for the ~1.5 GB dump.

To force a fresh restore after a previous run:

```bash
docker compose down -v   # WARNING: drops the pgdata volume
docker compose up -d db
```

## Verify the corpus matches RESULTS

```bash
docker compose exec db psql -U postgres -d financial_rag -t -c "
SELECT 'filings', count(*) FROM filings
UNION ALL SELECT 'chunks', count(*) FROM chunks
UNION ALL SELECT 'amendments', count(*) FROM filings WHERE filing_type LIKE '%/A%';"
```

Must return:

```
 filings    |   1662
 chunks     | 469874
 amendments |      0
```

If any number differs, stop — the corpus does not match `RESULTS.md` and the
evaluation numbers are not reproducible from it.

## Then wire up CorpCheck

```bash
# from repo root
python3 -m venv .venv && .venv/bin/pip install -e ".[ingestion,dev]"
cp .env.example .env   # defaults already point at localhost:5432
```

First retrieval run downloads the embedding model
(`sentence-transformers/all-MiniLM-L6-v2`, ~90 MB from HuggingFace). Do **not**
set `HF_HUB_OFFLINE=1` on the first run; set it on subsequent runs to reuse the
cache.
