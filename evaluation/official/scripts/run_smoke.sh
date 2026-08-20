#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-./.venv/bin/python}"
MANIFEST="${MANIFEST:-evaluation/official/benchmark_manifest.yaml}"
TASK="${TASK:-financebench}"
LIMIT="${LIMIT:-3}"
LABEL="${LABEL:-smoke-$(date +%Y%m%d_%H%M%S)}"
SEED="${SEED:-42}"

echo "[smoke-run] task=$TASK label=$LABEL seed=$SEED limit=$LIMIT"
"$PYTHON_BIN" -m evaluation.official.runner \
  --manifest "$MANIFEST" \
  --task "$TASK" \
  --smoke \
  --seed "$SEED" \
  --limit "$LIMIT" \
  --label "$LABEL"
