"""Backfill 10-Q chunks mislabelled by the duplicate-key bug in SECTION_MAP_10Q.

中文：这是一次有明确范围的历史数据修复。它依据现存 chunk 文本和位置重建标签，
并在无法证明修复安全时保留原值，而不是重新下载或推测源文件。

Background
----------
``SECTION_MAP_10Q`` defined "item 3" and "item 4" twice. Python keeps the last
value, so Part I Item 3 (Quantitative and Qualitative Disclosures about Market
Risk) was stored as "Defaults upon Senior Securities", and Part I Item 4
(Controls and Procedures) as "Mine Safety Disclosures". The cleaner is fixed;
this repairs the rows already in the database.

Why the labels are reconstructed rather than recomputed
-------------------------------------------------------
The correct move would be to re-run the fixed cleaner over the source filings.
Every row in ``filings`` has a ``local_path``, but none of those files still
exist on disk, and EDGAR is not reachable from where this was written. So the
label is reconstructed from what *is* still in the database: the chunk text and
its position in the document.

Classification
--------------
Chunks are grouped into *runs* — maximal consecutive ``chunk_index`` values in
one filing sharing an affected label, i.e. one occurrence of an Item section.
A page-footer table fragment has no topical signal by itself, but the run it
belongs to does, so the decision is made per run and applied to every chunk in
it.

Content decides first, structure breaks ties:

1. Run mentions genuine Part II subject matter (mine safety / senior securities)
   and no Part I vocabulary -> genuinely Part II, left alone.
2. Run mentions Part I vocabulary (market risk / disclosure controls) and no
   Part II subject matter -> relabel.
3. Both -> position decides: Part II always follows Part I in a 10-Q, so a run
   ending before the first unambiguous Part II section is Part I.
4. Neither -> position alone, and if there is no positional anchor either, the
   run is left untouched and reported.

Structure is *not* consulted first, even though it looks like the stronger
signal. The Part II anchor is itself a section label, and the cleaner sometimes
matches a table-of-contents row, which places the anchor near ``chunk_index=0``
and strands genuine Part I runs on the wrong side of it. An earlier draft of
this script ordered the signals the other way and wrongly preserved ~50
"Mine Safety Disclosures" rows whose text was plainly Controls and Procedures.

Safety
------
Dry run by default. Old values are copied to ``chunks_section_backfill_20260804``
before anything is written, and ``--rollback`` restores from it. The write runs
in one transaction. Updating ``section_name`` fires ``trig_chunks_tsv``, which
rebuilds ``content_tsv`` from ``section_name || content`` — desirable here,
since the sparse index currently carries the wrong section words. Embeddings are
derived from ``content`` alone and are unaffected.

Usage::

    python -m migrations.2026_08_04_fix_10q_part_sections            # dry run
    python -m migrations.2026_08_04_fix_10q_part_sections --apply
    python -m migrations.2026_08_04_fix_10q_part_sections --rollback
"""

from __future__ import annotations

import argparse
import asyncio
import re
from collections import Counter, defaultdict

import asyncpg

from corpcheck import settings as s

BACKUP_TABLE = "chunks_section_backfill_20260804"

# Wrong label -> the Part I label it should have had.
AFFECTED = {
    "Defaults upon Senior Securities":
        "Quantitative and Qualitative Disclosures about Market Risk",
    "Mine Safety Disclosures":
        "Controls and Procedures",
}

# Section names that exist only in Part II, usable to locate where Part II
# begins within a filing.
PART_II_ANCHORS = (
    "Legal Proceedings",
    "Risk Factors",
    "Unregistered Sales of Equity Securities",
)

_MARKET_RISK = re.compile(
    r"interest rate risk|commodity price|foreign currency|exchange rate|market risk"
    r"|value at risk|hedg|derivative|notional|sensitivity|basis point|trading portfolio"
    r"|marketable securities|available-for-sale|fair value|equity securities|impair",
    re.I,
)
_CONTROLS = re.compile(
    r"disclosure controls|internal control|principal executive|principal financial"
    r"|certification|exhibit 31|rule 13a-15|rule 15d-15|material weakness"
    r"|effectiveness of|evaluation of|chief executive officer|chief financial officer",
    re.I,
)
PART_I_VOCAB = {
    "Defaults upon Senior Securities": _MARKET_RISK,
    "Mine Safety Disclosures": _CONTROLS,
}
PART_II_VOCAB = {
    "Defaults upon Senior Securities": re.compile(
        r"default(s)? upon senior|dividend arrearage|senior securities", re.I),
    "Mine Safety Disclosures": re.compile(
        r"mine safety|mine act|msha|federal mine|dodd-frank.{0,40}mine", re.I),
}


def _runs(indices: list[int]) -> list[list[int]]:
    """Split a sorted index list into maximal consecutive runs."""
    out: list[list[int]] = []
    cur: list[int] = []
    for i in indices:
        if cur and i != cur[-1] + 1:
            out.append(cur)
            cur = []
        cur.append(i)
    if cur:
        out.append(cur)
    return out


async def connect() -> asyncpg.Connection:
    """Open the migration's explicit PostgreSQL connection.

    中文：连接参数沿用本迁移既有配置；连接失败必须阻止后续标签更新。
    """
    return await asyncpg.connect(
        host=s.DB_HOST, port=s.DB_PORT, database=s.DB_NAME,
        user=s.DB_USER, password=s.DB_PASSWORD, timeout=15,
    )


async def build_plan(conn: asyncpg.Connection):
    """Return (updates, keeps, unresolved, basis_counts) without writing."""
    filing_ids = [
        r["filing_id"] for r in await conn.fetch(
            """SELECT DISTINCT filing_id FROM chunks
               WHERE filing_type='10-Q' AND section_name = ANY($1::text[])""",
            list(AFFECTED),
        )
    ]

    updates: list[tuple[int, str, str]] = []   # (chunk_id, old, new)
    keeps: list[int] = []
    unresolved: list[dict] = []
    basis = Counter()

    for fid in filing_ids:
        rows = await conn.fetch(
            """SELECT id, chunk_index, section_name, content
               FROM chunks WHERE filing_id=$1 ORDER BY chunk_index""", fid)
        by_idx = {r["chunk_index"]: r for r in rows}
        anchors = [r["chunk_index"] for r in rows
                   if r["section_name"] in PART_II_ANCHORS]
        part_ii_start = min(anchors) if anchors else None

        grouped: dict[str, list[int]] = defaultdict(list)
        for r in rows:
            if r["section_name"] in AFFECTED:
                grouped[r["section_name"]].append(r["chunk_index"])

        for label, idxs in grouped.items():
            target = AFFECTED[label]
            for run in _runs(sorted(idxs)):
                text = " ".join(
                    " ".join(by_idx[i]["content"].split()) for i in run)
                ids = [by_idx[i]["id"] for i in run]

                has_p1 = bool(PART_I_VOCAB[label].search(text))
                has_p2 = bool(PART_II_VOCAB[label].search(text))

                if has_p2 and not has_p1:
                    why, is_part_i = "content:part-ii", False
                elif has_p1 and not has_p2:
                    why, is_part_i = "content:part-i", True
                elif has_p1 and has_p2:
                    if part_ii_start is None:
                        why, is_part_i = "mixed:no-anchor-default-part-i", True
                    else:
                        why = "mixed:position-tiebreak"
                        is_part_i = run[-1] < part_ii_start
                elif part_ii_start is not None and run[-1] < part_ii_start:
                    why, is_part_i = "position:before-part-ii", True
                elif part_ii_start is not None and run[0] >= part_ii_start:
                    why, is_part_i = "position:after-part-ii", False
                else:
                    unresolved.append(
                        {"filing_id": fid, "label": label, "chunk_ids": ids,
                         "sample": text[:120]})
                    basis["unresolved"] += 1
                    continue

                basis[why] += 1
                if is_part_i:
                    updates.extend((cid, label, target) for cid in ids)
                else:
                    keeps.extend(ids)

    return updates, keeps, unresolved, basis


async def apply(conn: asyncpg.Connection, updates) -> None:
    """Apply the precomputed, reviewed section-label updates in one transaction.

    中文：该函数不重新分类或扩大范围；数据库写入失败时由事务语义保留原有数据。
    """
    async with conn.transaction():
        await conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {BACKUP_TABLE} (
                chunk_id        BIGINT PRIMARY KEY,
                old_section_name TEXT NOT NULL,
                new_section_name TEXT NOT NULL,
                old_display_title TEXT,
                backfilled_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )""")
        await conn.execute(
            f"ALTER TABLE {BACKUP_TABLE} ADD COLUMN IF NOT EXISTS old_display_title TEXT")
        await conn.executemany(
            f"""INSERT INTO {BACKUP_TABLE} (chunk_id, old_section_name, new_section_name)
                VALUES ($1,$2,$3) ON CONFLICT (chunk_id) DO NOTHING""",
            updates,
        )
        # Guard: only ever move a row off one of the two known-bad labels.
        await conn.executemany(
            """UPDATE chunks SET section_name=$3
               WHERE id=$1 AND section_name=$2""",
            updates,
        )

        # display_title carries the same wrong string. The chunker sets it to
        # the section name for narrative segments, so it has to move too —
        # otherwise the API keeps citing "Mine Safety Disclosures" over what is
        # really Controls and Procedures text. Table segments have their own
        # caption in display_title and must not be touched, so the update is
        # conditioned on display_title still equalling the *old* section name
        # rather than on content_kind.
        await conn.execute(f"""
            UPDATE {BACKUP_TABLE} b SET old_display_title = c.display_title
            FROM chunks c
            WHERE c.id = b.chunk_id
              AND b.old_display_title IS NULL
              AND c.display_title = b.old_section_name""")
        n = await conn.fetchval(f"""
            WITH moved AS (
                UPDATE chunks c SET display_title = b.new_section_name
                FROM {BACKUP_TABLE} b
                WHERE c.id = b.chunk_id
                  AND c.display_title = b.old_section_name
                RETURNING 1)
            SELECT count(*) FROM moved""")
        print(f"   display_title also updated on {n} narrative chunks")


async def rollback(conn: asyncpg.Connection) -> int:
    """Restore only rows covered by this migration's recorded reversal rules.

    中文：回滚范围与修复范围同样受限，连接或 SQL 失败不会被静默忽略。
    """
    exists = await conn.fetchval("SELECT to_regclass($1)", BACKUP_TABLE)
    if not exists:
        print(f"no backup table {BACKUP_TABLE}; nothing to roll back")
        return 0
    async with conn.transaction():
        await conn.execute(f"""
            UPDATE chunks c SET display_title = b.old_display_title
            FROM {BACKUP_TABLE} b
            WHERE c.id = b.chunk_id
              AND b.old_display_title IS NOT NULL
              AND c.display_title = b.new_section_name""")
        n = await conn.fetchval(f"""
            WITH restored AS (
                UPDATE chunks c SET section_name = b.old_section_name
                FROM {BACKUP_TABLE} b
                WHERE c.id = b.chunk_id AND c.section_name = b.new_section_name
                RETURNING 1)
            SELECT count(*) FROM restored""")
    return n


async def report(conn: asyncpg.Connection, title: str) -> None:
    """Print a read-only summary of affected section labels.

    中文：报告用于人工确认迁移前后状态，不改变任何 chunk 或 filing 数据。
    """
    print(f"\n-- {title} --")
    rows = await conn.fetch("""
        SELECT section_name, count(*) AS n FROM chunks
        WHERE filing_type='10-Q' AND section_name = ANY($1::text[])
        GROUP BY 1 ORDER BY 1""",
        list(AFFECTED) + list(AFFECTED.values()))
    for r in rows:
        print(f"   {r['section_name']:60s} {r['n']:6d}")


async def main() -> None:
    """Select the existing migration action and run its guarded workflow.

    中文：入口保留原有交互与失败边界；未确认的修复不应继续写入数据库。
    """
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true", help="write the changes")
    ap.add_argument("--rollback", action="store_true", help="restore from backup")
    args = ap.parse_args()

    conn = await connect()
    try:
        if args.rollback:
            await report(conn, "before rollback")
            n = await rollback(conn)
            print(f"\nrestored {n} rows")
            await report(conn, "after rollback")
            return

        updates, keeps, unresolved, basis = await build_plan(conn)

        print("=== classification basis (per run) ===")
        for k, v in sorted(basis.items()):
            print(f"   {k:32s} {v}")
        print(f"\n   chunks to relabel : {len(updates)}")
        print(f"   left as Part II   : {len(keeps)}")
        print(f"   unresolved runs   : {len(unresolved)}")
        for u in unresolved[:5]:
            print(f"      filing {u['filing_id']} {u['label'][:26]}: {u['sample'][:70]}")

        targets = Counter(new for _, _, new in updates)
        print("\n=== target labels ===")
        for k, v in targets.items():
            print(f"   -> {k}: {v}")

        await report(conn, "current")
        if not args.apply:
            print("\nDRY RUN — pass --apply to write.")
            return

        await apply(conn, updates)
        await report(conn, "after backfill")
        n = await conn.fetchval(f"SELECT count(*) FROM {BACKUP_TABLE}")
        print(f"\nbackup rows in {BACKUP_TABLE}: {n}")
        print("rollback with: --rollback")
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
