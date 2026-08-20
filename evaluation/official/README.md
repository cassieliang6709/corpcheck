# Official benchmark runbook

This folder wraps CorpCheck’s “official benchmark” smoke + real-run paths.

## Benchmark map

`evaluation/official/benchmark_manifest.yaml`

- `financebench` → runner: `financebench`
- `finrank` → runner: `finrank`
- `averitec` → runner: `averitec`
- `corpcheck_claimcheckbench` → runner: `claimcheckbench`

## Input schema per task

### financebench
- `financebench_id` (string, optional)
- `question` (string)
- `doc_name` (string, optional)
- `company`, `doc_type`, `doc_period` (optional)
- `evidence` for official slices only

### finrank
- `question` / `query` (string)
- optional positive lists: `positive`, `positives`, `relevant`, `supports`, `label`, `gold`
- optional negative lists: `negative`, `negatives`, `hard_negative`, `hard_negatives`

### averitec / claimcheckbench
- `claim` / `claim_text` / `statement` (string)
- `label` or `verdict` (string)
- optional `id`

## Output artifact conventions

Each run writes:
- `evaluation/official/runs/results_<label>.json`
- `evaluation/official/runs/results.json` (backward-compatible pointer)

`results_<label>.json` contains:
- run-level metadata: `generated_at_utc`, `manifest_version`, `run_metadata`
- per-task `summary`, including:
  - `metadata` with:
    - `dataset_version` (file size + mtime fingerprint)
    - `commit` (git SHA)
    - `timestamp`
    - `hardware`
    - `seed`
    - `params` (k/alpha/threshold + task extras)

## Quick commands

```bash
cd /Users/cassie/Developer/01_Career_Portfolio/corpcheck

# Smoke run (no PostgreSQL needed)
./.venv/bin/python -m evaluation.official.runner \
  --manifest evaluation/official/benchmark_manifest.yaml \
  --task financebench \
  --smoke \
  --limit 3 \
  --label smoke-financebench

# Real run (requires DB + retrieval stack)
./.venv/bin/python -m evaluation.official.runner \
  --manifest evaluation/official/benchmark_manifest.yaml \
  --task financebench \
  --label financebench-offline-check

# All planned tasks with smoke output
./.venv/bin/python -m evaluation.official.runner \
  --manifest evaluation/official/benchmark_manifest.yaml \
  --only-planned \
  --smoke
```

## Smoke mode intent

`--smoke` is for
- CI sanity checks without DB
- demoing command/result shape in interviews
- verifying score/report structure before hitting expensive IR runs

Smoke results are **deterministic** for a fixed dataset + seed and include a `smoke` flag in summaries.
