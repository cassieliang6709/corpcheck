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

This requirements manifest is not yet safe to drive downloads. Before corpus
construction, each document must be enriched and frozen with its canonical
ticker, CIK, exact accession, filed date, period of report, selected SEC raw
component, and SHA-256. The ingestion runner must reject missing, extra,
amended, or mismatched filings and prove that both representations consume the
same immutable raw-file set. This is especially important for delisted ATVI,
historical Square filings now resolved through Block/XYZ, and the eight selected
documents earlier than the default 2018 start year.

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
