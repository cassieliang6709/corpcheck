#!/usr/bin/env python3
"""Paired, read-only retrieval gate for two corpus representations.

The runner compares the production question-only retrieval path against two
explicit PostgreSQL databases.  It is evaluation-only: it does not ingest,
mutate schemas, or select benchmark metadata as retrieval filters.
"""

from __future__ import annotations

import os

# Set before importing sentence-transformers through the retrieval module.  A
# missing local model must fail instead of turning an evaluation into a download.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import argparse
import asyncio
import hashlib
import json
import math
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence
from urllib.parse import unquote, urlsplit

import asyncpg
from pgvector.asyncpg import register_vector

from corpcheck.retrieval import load_known_tickers, retrieve
from corpcheck.retrieval.search import get_model
from evaluation.financebench import gold_spans, parse_gold_doc
from evaluation.ir_eval import build_query_record, gate_diagnostics, score_records
from evaluation.metrics import token_overlap
from evaluation.oracle import _fetch_filing_chunks

K = 10
STRICT_THRESHOLD = 0.5
DATASET_CONTRACTS = {
    "development": {
        "sha256": "1fdce65e8fe0ae03f5dcee732b8e70d9a22969d95981359c9505e105a9ca0f5f",
        "queries": 35,
    },
    "heldout": {
        "sha256": "f2d8ba8b3f1717166c862cc320c8c7a7678d19a4f3dd9c9438f1a74519ad5eae",
        "queries": 80,
    },
}


class GateError(RuntimeError):
    """Fail-closed validation or evaluation error."""


@dataclass(frozen=True)
class CorpusSnapshot:
    companies: frozenset[tuple[str, str]]
    filings: frozenset[tuple[Any, ...]]
    chunk_count: int
    embedded_chunk_count: int


def _canonical_database_target(database_url: str) -> tuple[str, str, int, str]:
    parsed = urlsplit(database_url)
    if parsed.scheme not in {"postgres", "postgresql"} or not parsed.hostname:
        raise GateError("database URLs must be explicit postgresql:// URLs")
    database = unquote(parsed.path.lstrip("/"))
    if not database:
        raise GateError("database URL must name a database")
    return ("postgresql", parsed.hostname.lower(), parsed.port or 5432, database)


def validate_distinct_urls(old_url: str, new_url: str) -> None:
    if _canonical_database_target(old_url) == _canonical_database_target(new_url):
        raise GateError("old and new database URLs resolve to the same target")


async def _init_connection(connection: asyncpg.Connection) -> None:
    await register_vector(connection)
    await connection.execute("SET ivfflat.probes = 10")


async def create_pool(database_url: str) -> asyncpg.Pool:
    return await asyncpg.create_pool(
        dsn=database_url,
        min_size=1,
        max_size=1,
        init=_init_connection,
    )


async def runtime_database_identity(pool: Any) -> tuple[str, str, int]:
    row = await pool.fetchrow(
        "SELECT current_database() AS database, "
        "COALESCE(inet_server_addr()::text, 'local-socket') AS address, "
        "inet_server_port() AS port"
    )
    return (str(row["database"]), str(row["address"]), int(row["port"] or 0))


async def retrieval_environment_fingerprint(pool: Any) -> str:
    rows = await pool.fetch(
        """
        SELECT 'index' AS kind,
               schemaname || '.' || indexname AS name,
               indexdef AS definition
        FROM pg_indexes
        WHERE schemaname = current_schema()
          AND tablename IN ('companies', 'filings', 'chunks')
        UNION ALL
        SELECT 'view' AS kind,
               current_schema() || '.v_retrieval_chunks' AS name,
               pg_get_viewdef('v_retrieval_chunks'::regclass, true) AS definition
        ORDER BY kind, name
        """
    )
    payload = [
        [str(row["kind"]), str(row["name"]), str(row["definition"])] for row in rows
    ]
    if not payload:
        raise GateError("retrieval schema fingerprint is empty")
    encoded = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(encoded).hexdigest()


async def corpus_snapshot(pool: Any) -> CorpusSnapshot:
    company_rows = await pool.fetch("SELECT ticker, name FROM companies")
    filing_rows = await pool.fetch(
        """
        SELECT ticker, filing_type, fiscal_year, period, filed_date,
               period_of_report, accession_number, cik, source_url
        FROM filings
        """
    )
    chunk_count = await pool.fetchval("SELECT COUNT(*) FROM chunks")
    embedded_chunk_count = await pool.fetchval(
        "SELECT COUNT(*) FROM chunks WHERE embedding IS NOT NULL"
    )
    companies = frozenset((str(row["ticker"]), str(row["name"])) for row in company_rows)
    filings = frozenset(
        (
            str(row["ticker"]),
            str(row["filing_type"]),
            row["fiscal_year"],
            row["period"],
            row["filed_date"],
            row["period_of_report"],
            row["accession_number"],
            row["cik"],
            row["source_url"],
        )
        for row in filing_rows
    )
    return CorpusSnapshot(
        companies,
        filings,
        int(chunk_count),
        int(embedded_chunk_count),
    )


def assert_identity_sets_match(old: CorpusSnapshot, new: CorpusSnapshot) -> None:
    if old.companies != new.companies:
        raise GateError(
            "company identity sets differ "
            f"(old_only={len(old.companies - new.companies)}, "
            f"new_only={len(new.companies - old.companies)})"
        )
    if old.filings != new.filings:
        raise GateError(
            "filing identity sets differ "
            f"(old_only={len(old.filings - new.filings)}, "
            f"new_only={len(new.filings - old.filings)})"
        )


def assert_complete_embeddings(snapshot: CorpusSnapshot, *, label: str) -> None:
    if snapshot.chunk_count <= 0:
        raise GateError(f"{label} corpus contains no SEC chunks")
    if snapshot.embedded_chunk_count != snapshot.chunk_count:
        raise GateError(
            f"{label} corpus has missing embeddings "
            f"({snapshot.embedded_chunk_count}/{snapshot.chunk_count})"
        )


def assert_same_database_server(
    old: tuple[str, str, int], new: tuple[str, str, int]
) -> None:
    if old[0] == new[0]:
        raise GateError("old and new connections point to the same PostgreSQL database")
    if old[1:] != new[1:]:
        raise GateError(
            "old and new databases must run on the same PostgreSQL server for paired latency"
        )


def assert_same_retrieval_environment(old: str, new: str) -> None:
    if old != new:
        raise GateError("old and new retrieval schema/index fingerprints differ")


def identity_digest(snapshot: CorpusSnapshot) -> str:
    filing_rows = [
        json.dumps(
            [str(value) if value is not None else None for value in item],
            separators=(",", ":"),
        )
        for item in snapshot.filings
    ]
    payload = {
        "companies": sorted([list(item) for item in snapshot.companies]),
        "filings": sorted(filing_rows),
    }
    encoded = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def nearest_rank_p95(samples: Sequence[float]) -> float:
    if not samples:
        raise GateError("cannot compute p95 without measured samples")
    ordered = sorted(float(value) for value in samples)
    rank = max(1, math.ceil(0.95 * len(ordered)))
    return ordered[rank - 1]


async def oracle_summary(pool: Any, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute strict metric reachability against one explicit corpus pool."""
    best_overlaps: list[float] = []
    absent_docs: list[str] = []
    unresolvable_docs: list[str] = []
    async with pool.acquire() as connection:
        for row in rows:
            gold = parse_gold_doc(row)
            if not gold.is_resolvable:
                unresolvable_docs.append(gold.doc_name)
                continue
            chunks = await _fetch_filing_chunks(connection, gold)
            if not chunks:
                absent_docs.append(gold.doc_name)
                continue
            for span in gold_spans(row):
                best_overlaps.append(max(token_overlap(chunk, span) for chunk in chunks))

    reachable = sum(value >= STRICT_THRESHOLD for value in best_overlaps)
    measured = len(best_overlaps)
    return {
        "gold_spans_measured": measured,
        "strict_reachable_spans": reachable,
        "strict_ceiling_recall": round(reachable / measured, 4) if measured else 0.0,
        "gold_docs_absent": sorted(set(absent_docs)),
        "gold_docs_unresolvable": sorted(set(unresolvable_docs)),
    }


def load_dataset(path: Path) -> tuple[list[dict[str, Any]], str]:
    raw = path.read_bytes()
    try:
        rows = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise GateError(f"invalid dataset JSON: {exc}") from exc
    if not isinstance(rows, list) or not rows:
        raise GateError("dataset must be a non-empty JSON list")
    ids: list[str] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise GateError(f"dataset row {index} is not an object")
        question = str(row.get("question") or "").strip()
        case_id = str(row.get("financebench_id") or "").strip()
        if not question or not case_id:
            raise GateError(f"dataset row {index} lacks question or financebench_id")
        ids.append(case_id)
    if len(ids) != len(set(ids)):
        raise GateError("dataset contains duplicate financebench_id values")
    return rows, hashlib.sha256(raw).hexdigest()


def validate_dataset_contract(
    profile: str,
    rows: Sequence[dict[str, Any]],
    dataset_sha256: str,
) -> None:
    contract = DATASET_CONTRACTS.get(profile)
    if contract is None:
        raise GateError(f"unknown gate profile: {profile}")
    if dataset_sha256 != contract["sha256"] or len(rows) != contract["queries"]:
        raise GateError(
            f"{profile} gate requires frozen dataset "
            f"sha256={contract['sha256']} queries={contract['queries']}"
        )


def validate_development_prerequisite(
    profile: str,
    development_report_path: Optional[Path],
) -> Optional[str]:
    if profile == "development":
        if development_report_path is not None:
            raise GateError("development profile does not accept --development-report")
        return None
    if development_report_path is None:
        raise GateError("heldout profile requires an accepted --development-report")
    raw = development_report_path.read_bytes()
    try:
        report = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise GateError(f"invalid development report JSON: {exc}") from exc
    if not isinstance(report, dict):
        raise GateError("development report must be a JSON object")
    dataset = report.get("dataset") or {}
    protocol = report.get("protocol") or {}
    result = report.get("decision") or {}
    if (
        dataset.get("sha256") != DATASET_CONTRACTS["development"]["sha256"]
        or dataset.get("queries") != DATASET_CONTRACTS["development"]["queries"]
        or protocol.get("gate_profile") != "development"
        or result.get("profile") != "development"
        or result.get("accepted") is not True
    ):
        raise GateError("heldout gate requires an accepted frozen development report")
    return hashlib.sha256(raw).hexdigest()


def validate_oracle_coverage(
    rows: Sequence[dict[str, Any]],
    old: dict[str, Any],
    new: dict[str, Any],
) -> int:
    expected_spans = sum(len(gold_spans(row)) for row in rows)
    if expected_spans <= 0:
        raise GateError("frozen dataset contains no gold evidence spans")
    for label, summary in (("old", old), ("new", new)):
        absent = summary["gold_docs_absent"]
        unresolvable = summary["gold_docs_unresolvable"]
        measured = int(summary["gold_spans_measured"])
        if absent or unresolvable or measured != expected_spans:
            raise GateError(
                f"{label} oracle coverage incomplete: measured={measured}/"
                f"{expected_spans}, absent={absent}, unresolvable={unresolvable}"
            )
    return expected_spans


async def _retrieve_once(
    pool: Any,
    row: dict[str, Any],
    *,
    label: str,
    alpha: float,
) -> tuple[Any, float]:
    started = time.perf_counter()
    try:
        chunks = await retrieve(
            pool=pool,
            query=str(row["question"]),
            k=K,
            alpha=alpha,
            sector=None,
            company=None,
            filing_type=None,
            fiscal_year=None,
        )
    except Exception as exc:
        case_id = row.get("financebench_id")
        message = f"{label} retrieval failed for {case_id}: {type(exc).__name__}: {exc}"
        raise GateError(message) from exc
    latency_ms = (time.perf_counter() - started) * 1000
    return chunks, latency_ms


async def run_paired_measurements(
    old_pool: Any,
    new_pool: Any,
    rows: list[dict[str, Any]],
    *,
    alpha: float,
    warmup_rounds: int,
    measured_rounds: int,
) -> tuple[dict[str, list[list[Any]]], dict[str, list[float]]]:
    pools = {"old": old_pool, "new": new_pool}
    measured_records: dict[str, list[list[Any]]] = {"old": [], "new": []}
    latencies: dict[str, list[float]] = {"old": [], "new": []}
    pair_index = 0

    for round_index in range(warmup_rounds + measured_rounds):
        measured = round_index >= warmup_rounds
        round_records: dict[str, dict[str, Any]] = {"old": {}, "new": {}}
        for row in rows:
            order = ("old", "new") if pair_index % 2 == 0 else ("new", "old")
            pair_index += 1
            for label in order:
                chunks, latency_ms = await _retrieve_once(
                    pools[label], row, label=label, alpha=alpha
                )
                if not measured:
                    continue
                latencies[label].append(latency_ms)
                case_id = str(row["financebench_id"])
                round_records[label][case_id] = build_query_record(
                    row,
                    chunks,
                    round(latency_ms),
                    match_quarter=True,
                )
        if measured:
            for label in ("old", "new"):
                measured_records[label].append(
                    [round_records[label][str(row["financebench_id"])] for row in rows]
                )

    return measured_records, latencies


def summarize(rounds: list[list[Any]], latency_samples: list[float]) -> dict[str, Any]:
    round_summaries = []
    for round_index, records in enumerate(rounds, start=1):
        strict = score_records(
            records,
            threshold=STRICT_THRESHOLD,
            variant="clean",
            apply_gate=True,
            k=K,
        )
        diagnostics = gate_diagnostics(records)
        round_summaries.append(
            {
                "round": round_index,
                "strict": strict,
                "zero_provenance_queries": diagnostics["queries_with_zero_passing"],
            }
        )
    return {
        "rounds": round_summaries,
        "latency_ms": {
            "samples": len(latency_samples),
            "median": round(statistics.median(latency_samples), 3),
            "p95_nearest_rank": round(nearest_rank_p95(latency_samples), 3),
            "max": round(max(latency_samples), 3),
        },
    }


def decision(
    old: dict[str, Any],
    new: dict[str, Any],
    *,
    profile: str,
    oracle_non_regression: bool,
) -> dict[str, Any]:
    if profile not in {"development", "heldout"}:
        raise GateError(f"unknown gate profile: {profile}")
    old_p95 = float(old["latency_ms"]["p95_nearest_rank"])
    new_p95 = float(new["latency_ms"]["p95_nearest_rank"])
    paired_rounds = list(zip(old["rounds"], new["rounds"], strict=True))
    round_checks = [
        {
            "round": new_round["round"],
            "new_strict_recall_at_10_gte_0_25": (
                new_round["strict"]["recall_at_10"] >= 0.25
            ),
            "new_strict_recall_at_10_gte_old": (
                new_round["strict"]["recall_at_10"]
                >= old_round["strict"]["recall_at_10"]
            ),
            "zero_provenance_not_increased": (
                new_round["zero_provenance_queries"]
                <= old_round["zero_provenance_queries"]
            ),
        }
        for old_round, new_round in paired_rounds
    ]
    strict_profile_check = all(
        item["new_strict_recall_at_10_gte_old"]
        and (profile == "heldout" or item["new_strict_recall_at_10_gte_0_25"])
        for item in round_checks
    )
    checks = {
        f"every_round_passes_{profile}_strict_gate": strict_profile_check,
        "strict_oracle_reachability_not_decreased": oracle_non_regression,
        "every_round_zero_provenance_not_increased": all(
            item["zero_provenance_not_increased"] for item in round_checks
        ),
        "new_p95_lte_old_times_1_5": new_p95 <= old_p95 * 1.5,
    }
    return {
        "profile": profile,
        "accepted": all(checks.values()),
        "checks": checks,
        "rounds": round_checks,
    }


def write_report(path: Path, report: dict[str, Any]) -> None:
    rendered = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != rendered:
            raise GateError(f"refusing to overwrite different output: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rendered, encoding="utf-8")


async def run(args: argparse.Namespace) -> dict[str, Any]:
    validate_distinct_urls(args.old_db_url, args.new_db_url)
    rows, dataset_sha256 = load_dataset(args.dataset)
    validate_dataset_contract(args.gate_profile, rows, dataset_sha256)
    development_report_sha256 = validate_development_prerequisite(
        args.gate_profile,
        args.development_report,
    )
    old_pool = await create_pool(args.old_db_url)
    try:
        new_pool = await create_pool(args.new_db_url)
    except Exception:
        await old_pool.close()
        raise
    try:
        old_runtime, new_runtime = await asyncio.gather(
            runtime_database_identity(old_pool), runtime_database_identity(new_pool)
        )
        assert_same_database_server(old_runtime, new_runtime)

        old_snapshot, new_snapshot = await asyncio.gather(
            corpus_snapshot(old_pool), corpus_snapshot(new_pool)
        )
        old_environment, new_environment = await asyncio.gather(
            retrieval_environment_fingerprint(old_pool),
            retrieval_environment_fingerprint(new_pool),
        )
        assert_same_retrieval_environment(old_environment, new_environment)
        assert_identity_sets_match(old_snapshot, new_snapshot)
        assert_complete_embeddings(old_snapshot, label="old")
        assert_complete_embeddings(new_snapshot, label="new")
        old_oracle, new_oracle = await asyncio.gather(
            oracle_summary(old_pool, rows), oracle_summary(new_pool, rows)
        )
        expected_gold_spans = validate_oracle_coverage(rows, old_oracle, new_oracle)

        # The parser cache is process-global. Identical company identities make
        # the second load equivalent while ensuring aliases are ready before timing.
        await load_known_tickers(old_pool)
        await load_known_tickers(new_pool)
        get_model()

        records, latencies = await run_paired_measurements(
            old_pool,
            new_pool,
            rows,
            alpha=args.alpha,
            warmup_rounds=args.warmup_rounds,
            measured_rounds=args.measured_rounds,
        )
        old_summary = summarize(records["old"], latencies["old"])
        new_summary = summarize(records["new"], latencies["new"])
        report = {
            "dataset": {
                "path": str(args.dataset),
                "sha256": dataset_sha256,
                "queries": len(rows),
            },
            "protocol": {
                "k": K,
                "alpha": args.alpha,
                "threshold": STRICT_THRESHOLD,
                "variant": "clean",
                "provenance_gate": True,
                "warmup_rounds": args.warmup_rounds,
                "measured_rounds": args.measured_rounds,
                "order": "alternating old/new per question",
                "p95": "nearest-rank",
                "gate_profile": args.gate_profile,
                "development_report_sha256": development_report_sha256,
            },
            "corpus_identity": {
                "sha256": identity_digest(old_snapshot),
                "companies": len(old_snapshot.companies),
                "filings": len(old_snapshot.filings),
                "old_chunks": old_snapshot.chunk_count,
                "new_chunks": new_snapshot.chunk_count,
                "old_embedded_chunks": old_snapshot.embedded_chunk_count,
                "new_embedded_chunks": new_snapshot.embedded_chunk_count,
            },
            "databases": {"old": old_runtime[0], "new": new_runtime[0]},
            "retrieval_environment_sha256": old_environment,
            "oracle": {
                "expected_gold_spans": expected_gold_spans,
                "old": old_oracle,
                "new": new_oracle,
                "strict_reachability_non_regression": (
                    new_oracle["strict_reachable_spans"]
                    >= old_oracle["strict_reachable_spans"]
                ),
            },
            "old": old_summary,
            "new": new_summary,
            "decision": decision(
                old_summary,
                new_summary,
                profile=args.gate_profile,
                oracle_non_regression=(
                    new_oracle["strict_reachable_spans"]
                    >= old_oracle["strict_reachable_spans"]
                ),
            ),
        }
        write_report(args.output, report)
        return report
    finally:
        await asyncio.gather(old_pool.close(), new_pool.close())


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-db-url", required=True)
    parser.add_argument("--new-db-url", required=True)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--gate-profile",
        required=True,
        choices=("development", "heldout"),
    )
    parser.add_argument(
        "--development-report",
        type=Path,
        help="Required accepted frozen-development report for heldout evaluation",
    )
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--warmup-rounds", type=int, default=1)
    parser.add_argument("--measured-rounds", type=int, default=3)
    args = parser.parse_args(argv)
    if not 0.0 <= args.alpha <= 1.0:
        parser.error("--alpha must be between 0 and 1")
    if args.warmup_rounds < 0:
        parser.error("--warmup-rounds must be non-negative")
    if args.measured_rounds <= 0:
        parser.error("--measured-rounds must be positive")
    return args


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    try:
        report = asyncio.run(run(args))
    except (GateError, OSError, asyncpg.PostgresError) as exc:
        print(f"paired retrieval gate failed: {exc}")
        return 1
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["decision"]["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
