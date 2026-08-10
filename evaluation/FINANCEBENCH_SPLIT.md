# FinanceBench development and held-out contract

This split is frozen before the table-representation experiment is measured.
Selection uses document metadata only; answers and evidence text do not affect
membership.

## Source lock

- repository: `patronus-ai/financebench`
- repository commit: `cc39aeb4afdf33909ee1412188bf89035950c2eb`
- source path: `data/financebench_open_source.jsonl`
- Git blob: `4aef1d43a443474ba193f158f2baf70550ff528d`
- source SHA-256: `a5a2aa673e573e55675fc3c0f9aa38c1cf59d2abc91edb077534f71f10a71877`
- source rows: 150

## Development set

`evaluation/datasets/financebench_filtered.json` contains 35 questions and has
SHA-256
`1fdce65e8fe0ae03f5dcee732b8e70d9a22969d95981359c9505e105a9ca0f5f`.
It has informed several retrieval experiments and is development data only.

## Held-out selection

Starting from the other 115 official records, include a row only when:

1. its `company` does not occur in the 35-question development set; and
2. its `doc_name` matches `_<YYYY>_10K`.

The frozen result contains 80 questions, 45 10-K documents, and 21 issuers. It
excludes the two otherwise corpus-eligible `AMAZON_2017_10K` questions because
Amazon already occurs in development. It also excludes 10-Q, 8-K, and earnings
documents so the first held-out corpus expansion has one clear ingestion
contract.

Materialize the held-out dataset from the source lock above without changing
this rule:

```bash
.venv/bin/python -m evaluation.materialize_financebench_holdout \
  --source tmp/datasets/financebench_open_source.jsonl \
  --output evaluation/runs/R10_financebench_holdout/financebench_holdout.json \
  --manifest-output evaluation/runs/R10_financebench_holdout/ingestion_manifest.json
```

The materializer verifies both source hashes, derives `doc_type` and
`doc_period`, requires resolvable ticker provenance for every row, checks the
frozen counts, and refuses to overwrite different output. The locked output has
SHA-256
`f2d8ba8b3f1717166c862cc320c8c7a7678d19a4f3dd9c9438f1a74519ad5eae`.
The deduplicated 45-document corpus-requirements manifest has SHA-256
`2b17d7354f4a49f78f249422d49d3df9269d2c0977da0403ed878671f7aa4b10`.

The dataset is materialized locally but retrieval evaluation is not yet
runnable: the 45 filings still need to be added to an isolated corpus. The 21
issuer mappings are available for explicit ingestion but remain outside the
default 50-company universe.

This requirements manifest is not yet safe to drive downloads. Resolution must
use a versioned 21-issuer CIK lock, merge the SEC submissions `recent` and
historical-file arrays, accept exact `10-K` only, and confirm the requested
fiscal year through `dei.DocumentFiscalYearFocus`. This avoids guessing from
`reportDate.year`, which is wrong for some January/February retail fiscal years.
Missing or ambiguous confirmation fails unless the document has an explicit
reviewed accession lock. The historical edge locks are ATVI FY2019
`0000718877-20-000003`, Block FY2016 `0001628280-17-001754`, and Block FY2020
`0001512673-21-000008`; an accession prefix is not required to equal the
registrant CIK.

Each resolved document must freeze canonical ticker, CIK, exact accession,
filed date, period of report, fiscal-year end, archive URLs, and the exact
`full-submission.txt` SHA-256 and size. It must also freeze the ordered SGML
`<DOCUMENT>` components actually selected by the cleaner: ordinal, `TYPE`,
`FILENAME`, role, text SHA-256, and size, plus the selector/source fingerprint.
The current downloader's “largest HTML” heuristic is not part of this contract:
metadata comes from the full submission, and largest HTML is neither necessarily
the primary filing nor a stable tie-break. Missing, extra, amended, mismatched,
unsafe-path, or hash-drifted files fail closed.

The historical development database also lacks raw/component hashes, so it is
an identity reference rather than a proven matched representation baseline.
Both development and held-out gates must rebuild `baseline` (the four table
methods before `97c9fed`) and `candidate` from the same immutable full-submission
and selected-component byte set. Their run contracts bind distinct profile
fingerprints and a shared raw manifest, chunker, embedding model, and dimension.
Only those rebuilt databases may be compared by the paired gate.

## Acceptance gates

Compare old and new cleaners on separately built corpora with identical source
filings, embeddings, query order, and warm-up:

- development strict clean provenance-gated Recall@10 must be at least 0.2500;
- held-out strict clean provenance-gated Recall@10 must not regress;
- strict oracle reachability must not decrease on either split;
- zero-provenance queries must not increase; and
- paired warm p95 retrieval latency must not increase by more than 50%.

If the development gate fails, reject the representation arm without tuning on
held-out results. If it passes, evaluate held-out once for the stop/continue
decision. A failed held-out gate rejects the arm; it does not start another round
of held-out-driven rules. The held-out runner requires the accepted frozen
development report as `--development-report`; it records that report's SHA-256
and refuses to start without it.
