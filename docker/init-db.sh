#!/bin/bash
# Runs once on first start (empty pgdata volume) via the postgres entrypoint.
# Restores the dump placed next to this file as financial_rag.dump.
set -e
pg_restore \
  --username="$POSTGRES_USER" \
  --dbname="$POSTGRES_DB" \
  --no-owner \
  --no-privileges \
  /docker-entrypoint-initdb.d/financial_rag.dump
