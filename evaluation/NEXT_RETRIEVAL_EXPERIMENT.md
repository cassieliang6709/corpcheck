# Next retrieval experiment: table-row child retrieval

## Decision

The next experiment should be **table-row child retrieval with parent-table
return**, not another metadata regex and not a post-hoc reranker.

The current failure is primarily candidate generation inside the correct filing:
the relevant full table exists, but its large embedding / full-text representation
does not rank in the candidate pool. A row-sized child containing the table title,
column headers, year labels, and one data row should be easier to retrieve; the
API should still return the existing parent table chunk so citation provenance
and the evaluation protocol remain unchanged.

## Quantified failure analysis

Source: `evaluation/runs/R4_current/queries.jsonl`, joined read-only to the 44
gold spans and the current `v_retrieval_chunks` corpus. A span is reachable when
the best chunk anywhere in the gold filing has clean token overlap >= 0.5.

At the strict 0.5 protocol:

| mutually exclusive category | spans | definition |
| --- | ---: | --- |
| Current hit | 5 | A returned in-scope chunk has overlap >= 0.5. |
| Query parsing / under-specified period | 3 | No in-scope chunk; the question omits the benchmark filing period or names a different year. IDs: `00206`, `00702`, `00283`. |
| Wrong document despite explicit metadata | 1 | `00394` says `2022 Q2`, but all ten results are JPM 2019 filings. This span is also metric-unreachable, so fixing provenance alone cannot raise strict recall for it. |
| Table candidate / context failure | 19 | An in-scope chunk is present, the span is reachable, and the oracle-best chunk is `content_kind=table`, but no returned chunk clears 0.5. |
| Correct-document narrative low-overlap | 8 | An in-scope chunk is present, the span is reachable, and the oracle-best chunk is narrative, but no returned chunk clears 0.5. |
| Other / metric-unreachable | 8 | Oracle-best overlap is below 0.5, excluding `00394` already counted above. Retrieval code alone cannot make these strict-protocol hits without changing chunking or the metric. |
| **Total** | **44** | |

Equivalent query-level provenance diagnostic: **4/35 questions return zero
chunks from the gold filing**. Three are under-specified/mismatched relative to
the benchmark document; one (`00394`) contains the correct year and quarter and
is a genuine wrong-document ranking failure.

Among the **30 reachable misses**, 19 (63%) are table failures, 8 (27%) are
narrative ranking failures, and 3 (10%) are query-period failures. This is why
table retrieval is the highest-leverage next experiment.

### Negative controls already checked

Two read-only candidate probes show that merely adding another search arm is
too weak:

- top-30 table-only dense + current `plainto_tsquery` sparse search recovered a
  >=0.5 candidate for only **3/20** reachable table-oracle misses (the twentieth
  is `00702`, categorized above as query-period failure);
- top-30 table-only OR-term sparse search recovered only **1/20**.

Therefore the experiment must change the searchable representation (row child
plus inherited headers), not only add `content_kind='table'` to the current
chunk search.

## Smallest implementation

Keep the existing parent chunks and response model unchanged.

1. Build an evaluation-only child index for SEC table chunks. Each child stores
   `parent_chunk_id` and text composed as:
   `table title + year/header rows + one data row`.
2. Embed and full-text-index those children using the existing embedding model
   and PostgreSQL text-search configuration.
3. For quantitative queries only, retrieve a fixed top-N child list under the
   same issuer, filing-type, year boosts, and revision filtering used today.
4. Promote each matched child's parent into the existing fusion pool; dedupe by
   `parent_chunk_id`. Return and score the parent chunk, never the synthetic
   child, so citations and the current overlap metric are comparable.
5. Put the arm behind one ablation flag (suggested name:
   `TABLE_CHILD_RETRIEVAL_ENABLED`) and do not tune N on the 35-question set;
   choose N=30 before the run.

This avoids a cross-encoder dependency and isolates one hypothesis: **does a
searchable row representation surface the correct financial statement table?**

## Before / after commands

Run both against the same corpus and commit, with `k=10`, `alpha=0.7`, RRF, and
the existing protocol cube unchanged:

```bash
HF_HUB_OFFLINE=1 TABLE_CHILD_RETRIEVAL_ENABLED=0 \
  .venv/bin/python -m evaluation.ir_eval --label R5_table_child_off

HF_HUB_OFFLINE=1 TABLE_CHILD_RETRIEVAL_ENABLED=1 \
  .venv/bin/python -m evaluation.ir_eval --label R5_table_child_on
```

Also emit a diagnostic in the after run: of the 19 currently reachable,
in-scope table misses, how many parent tables entered the pre-final candidate
pool and how many reached top 10.

## Acceptance threshold

Accept the experiment only if all of these hold:

- strict `0.5 / clean / doc_gate ON` Recall@10 rises from **0.1429 to at least
  0.2500** (roughly four additional full-query equivalents on n=35);
- at least **5 of the 19** currently reachable in-scope table misses become
  strict top-10 hits;
- loose `0.2 / clean / doc_gate ON` Recall@10 does not fall below the current
  **0.4714**;
- zero-in-scope-evidence questions do not increase above **4/35**;
- no regression in revision filtering tests, and p95 retrieval latency grows by
  no more than **50%**.

If the child arm gets the right parents into the candidate pool but fewer than
five reach top 10, the next experiment is reranking. If it cannot get five into
the candidate pool, stop and revisit table parsing/header inheritance rather
than tuning fusion weights.
