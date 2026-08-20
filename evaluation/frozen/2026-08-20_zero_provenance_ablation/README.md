# Frozen evidence · zero-provenance ablation (19 → 4)

This directory is the version-controlled record behind CorpCheck's most-quoted
result. `evaluation/runs/` is gitignored — it holds 5.2 GB of per-query traces —
so the summaries that the claim actually rests on were previously local-only.
They are frozen here on 2026-08-20.

## The claim these files support

> Across 35 FinanceBench questions, the number of queries where **none** of the
> top-10 retrieved chunks came from the filing the question is about fell from
> **19 to 4**.

It is a provenance-coverage measure. It is **not** answer accuracy, and it is
**not** Recall. It is threshold-independent: it does not use the token-overlap
proxy at all, which is why it is the most defensible number in the report.

## Measured values

Read from `runs/*/summary.json` → `gate_diagnostics`.

| Run | What changed | mean chunks passing gate (of 10) | queries with **zero** passing | unresolvable gold doc |
| --- | --- | --- | --- | --- |
| `R1_baseline` | — | 0.71 | **19 / 35** | 0 |
| `R2_yearfix` | year-notation parsing | 4.26 | 7 / 35 | 0 |
| `R3_shorthand` | company shorthand | 5.17 | 5 / 35 | 0 |
| `R4_current` | company scope | 5.40 | **4 / 35** | 0 |

Primary protocol, identical across all four runs
(`summary.json` → `primary.config`):

| | |
| --- | --- |
| threshold | 0.5 |
| variant | clean |
| doc_gate | on |
| k | 10 |
| queries scored | 35 |

Under that same strict protocol, Recall@10 is **0.1429** for `R4_current` and
**0.0000** for `R1_baseline`. Quote the absolute pair; with n = 35 one query is
worth 2.86 recall points, so no ratio is meaningful.

## Corpus snapshot

`corpus_manifest.json` was generated from the live database on the freeze date:

| | |
| --- | --- |
| filings | 1,662 (429 × 10-K, 1,233 × 10-Q) |
| chunks | 469,874 |
| issuers | 50 |
| fiscal years | 2017–2026 |
| filed dates | 2017-07-20 → 2026-04-01 |
| **amended filings** | **0** |

That last row is the one to volunteer rather than defend. The revision filter is
implemented and unit-tested against a real GameStop 10-K/10-K/A pair, but this
corpus contains no amendment, so revision handling is **proven in tests, not
demonstrated on the live corpus**.

## Reproducing

The ablation runs predate the current official runner. The runner that now owns
official benchmarks is:

```bash
.venv/bin/python -m evaluation.official.runner \
  --manifest evaluation/official/benchmark_manifest.yaml \
  --task financebench --label <label>
```

It writes to `evaluation/official/runs/`, which is deliberately **not**
gitignored, so future official results are version-controlled by default and do
not need a manual freeze like this one.

Regenerate the corpus manifest with:

```bash
docker exec nlp-project-db-1 psql -U postgres -d financial_rag -At \
  -c "select jsonb_pretty(jsonb_build_object(
        'filings',(select count(*) from filings),
        'chunks',(select count(*) from chunks),
        'issuers',(select count(distinct ticker) from filings)))"
```

## Caveats that must travel with these numbers

1. The evaluator passes **only the question text** to `retrieve()`. FinanceBench
   gold company/year/form are never handed to the retriever, because a live
   request would not have them.
2. `R3 → R4` (company scope) moves **zero** queries at threshold 0.5. It is kept
   because it is mechanically sound, not because the benchmark endorses it. Do
   not claim it improved Recall.
3. These four runs share one corpus build. They are comparable to each other and
   to nothing else.
