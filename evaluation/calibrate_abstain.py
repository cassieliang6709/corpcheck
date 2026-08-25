"""Calibrate the abstain thresholds against the live corpus.

中文：从当前语料的可回答与不可回答问题分布校准拒答阈值。阈值依赖具体嵌入模型
和语料版本，因此结果用于人工配置决策，而不会由脚本自动写回生产设置。

Measures the raw (unboosted) dense cosine distribution for questions the corpus
can genuinely answer versus questions it cannot, so ABSTAIN_TOP1_MIN and
ABSTAIN_MEAN_TOP3_MIN are set from data rather than guessed.

The thresholds belong between the two populations. They are a property of the
embedding model paired with this corpus, not universal constants — re-run this
whenever either changes::

    .venv/bin/python -m evaluation.calibrate_abstain
"""

import asyncio
import statistics

import asyncpg

from corpcheck import settings as s
from corpcheck.retrieval.search import embed_query

# Queries a user of this system would plausibly ask, and which the corpus
# (50 large-cap tickers, 2018-2025 10-K/10-Q) genuinely can answer.
IN_DOMAIN = [
    "What were JPMorgan's net interest income drivers in 2023?",
    "Apple revenue by product segment",
    "How does Tesla describe supply chain risk?",
    "NVIDIA data center segment growth",
    "Goldman Sachs trading VaR",
    "What are Pfizer's patent expiration risks?",
    "Exxon capital expenditure guidance",
    "Microsoft cloud gross margin",
    "Walmart inventory shrinkage",
    "UnitedHealth medical cost ratio",
    "effects of rising interest rates on the loan portfolio",
    "goodwill impairment charge",
    "material weakness in internal control over financial reporting",
    "share repurchase program authorization",
    "legal proceedings related to antitrust",
]

# Nothing in a 10-K can answer these. The gate must fire.
OUT_OF_DOMAIN = [
    "how do I bake sourdough bread",
    "best hiking trails in Patagonia",
    "who won the 2014 FIFA World Cup",
    "translate good morning into Japanese",
    "symptoms of vitamin D deficiency",
    "how to change a bicycle tire",
    "python asyncio event loop tutorial",
    "lyrics to a pop song about summer",
    "what is the capital of Mongolia",
    "how long to boil an egg",
]

SQL = """
    SELECT (1 - (embedding <=> $1::vector)) AS cos
    FROM v_retrieval_chunks
    WHERE embedding IS NOT NULL
    ORDER BY embedding <=> $1::vector
    LIMIT 10
"""


async def probe(conn, q):
    """Measure top retrieval similarities for one calibration query.

    中文：返回原始余弦统计供人工设定阈值；连接或嵌入失败应由调用方可见。
    """
    vec = embed_query(q)
    rows = await conn.fetch(SQL, str(vec))
    cos = [r["cos"] for r in rows]
    return cos[0], statistics.fmean(cos[:3]), statistics.fmean(cos)


async def main():
    """Print live-corpus calibration observations without changing settings.

    中文：该脚本只报告分布，绝不自动写回拒答阈值；数据库不可用时应直接失败。
    """
    conn = await asyncpg.connect(
        host=s.DB_HOST, port=s.DB_PORT, database=s.DB_NAME,
        user=s.DB_USER, password=s.DB_PASSWORD, timeout=10,
    )
    await conn.execute("SET LOCAL ivfflat.probes = 10")
    for label, qs in (("IN-DOMAIN", IN_DOMAIN), ("OUT-OF-DOMAIN", OUT_OF_DOMAIN)):
        print(f"\n===== {label} =====")
        print(f"{'top1':>7} {'mean3':>7} {'mean10':>7}   query")
        stats = []
        for q in qs:
            t1, m3, m10 = await probe(conn, q)
            stats.append((t1, m3))
            print(f"{t1:7.4f} {m3:7.4f} {m10:7.4f}   {q[:52]}")
        t1s = sorted(x[0] for x in stats)
        m3s = sorted(x[1] for x in stats)
        print(f"  top1  min={t1s[0]:.4f}  p50={statistics.median(t1s):.4f}  max={t1s[-1]:.4f}")
        print(f"  mean3 min={m3s[0]:.4f}  p50={statistics.median(m3s):.4f}  max={m3s[-1]:.4f}")
    print(
        "\nPut the thresholds in the gap between the two blocks above, nearer the "
        "out-of-domain edge to avoid refusing marginal but answerable questions."
    )
    await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
