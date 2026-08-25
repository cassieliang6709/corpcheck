# CorpCheck

[English](README.md) · [简体中文](README.zh-CN.md)

面向 SEC 公司申报文件的审计级基本面研究系统。

CorpCheck 不是一个通用 RAG 外壳。它从金融披露的真实约束出发：10-K 不是普通长文本，申报文件有版本关系，管理层会把重大风险藏在限定语句和表格中，而一个自信但错误的数字往往比不回答更危险。

[在线演示](https://corpcheck.liangyue.site/) · [英文页面](https://corpcheck.liangyue.site/en/)

> 当前公开页面提供检索证据演示。未配置公开 API 时，它会回退到明确标记的录制状态；自定义问题不会被伪装成实时结果。

## 核心原则

1. **用确定性评测代替主观观感。** 检索质量通过 `Recall@k`、`MRR` 和逐查询证据轨迹评估，而不是挑几个看起来不错的回答。
2. **严格保存来源与版本关系。** 每个 chunk 都能追溯到公司、申报类型、财年、提交日期、accession number 和原始 SEC 页面。
3. **证据不足时拒绝回答。** 检索置信度低于校准阈值时，系统会在调用 LLM 之前确定性地拒绝，而不是让模型自行判断自己是否知道。

## 为什么金融检索更难

通用语义搜索容易在以下场景失败：

- 找到正确数字，但来自错误公司或错误财年；
- `10-K/A` 只修订原文件的部分 section，不能简单地整份覆盖或整份保留；
- 表格标题、单位、年份和具体数值被切到不同 chunk；
- 返回了固定数量的结果，但这些结果并不足以支持答案；
- 引用编号格式正确，却没有真正支持回答中的主张。

CorpCheck 把这些问题拆成查询解析、混合检索、版本治理、证据门控和可复现评测，而不是把责任全部交给生成模型。

## 系统架构

```text
SEC / 新闻 / 财报电话会
          │
          ▼
下载 → 清洗 → 分段 → 向量化 → PostgreSQL + pgvector
                                  │
用户问题 → 公司/年份/文件类型解析 │
          │                       │
          └── Dense + Sparse 候选 ┘
                       │
                  RRF 融合与重排
                       │
               修订版本与来源过滤
                       │
                  可回答性门控
                  │          │
              证据不足      证据通过
                  │          │
                拒答     API / MCP / LLM
```

HTTP API、MCP 服务和离线评测都调用同一个 `retrieval.pipeline.retrieve()`。协议层不各自实现检索，因此离线指标描述的就是实际客户端收到的结果。

## 功能概览

### 混合检索

- 使用 pgvector 生成 dense candidates；
- 使用 PostgreSQL 全文检索生成 sparse candidates；
- 默认通过 Reciprocal Rank Fusion（RRF）合并排名；
- 识别公司简称、财年写法和申报类型；
- 对显式公司和财年约束执行来源覆盖检查。

### 版本治理

修订文件不是整份替换原文件。CorpCheck 在 section 层面组合 `10-K` 与 `10-K/A`：

- 被修订的原 section 从候选集中移除；
- amendment 中的新 section 保留；
- 原文件中未被修订的 section 继续可用；
- 直接通过 chunk id 获取上下文时也会执行同一检查，不能绕过治理规则。

仓库包含基于 GameStop 2024 年真实 `10-K` / `10-K/A` 元数据的回归 fixture，用于验证 section-aware composition。当前基准语料库本身没有 amendment，因此这是仓库内测试，不是线上语料实证。

### 确定性拒答

`/answerability` 会返回：

- `answerable` 与具体 `gate_status`；
- top-1 及 top-3 平均余弦相似度；
- 当前阈值；
- 检索到的公司、申报类型、财年与来源覆盖；
- 原始证据 chunks；
- `llm_consulted: false`。

如果门控失败，`/chat` 会在联系 LLM 前拒绝。即使检索仍返回 5 个 chunk，也不代表其中任何一个足以支撑答案；拒答依据是证据强度与覆盖，而不是结果列表是否为空。

### Claim 级验证

API 还提供主张拆解与验证入口：

| Endpoint | 用途 |
| --- | --- |
| `POST /claims/extract` | 从输入文本中提取可核查主张 |
| `POST /claims/verify` | 检索证据并给出 verified、refuted、conflicting 或 evidence-insufficient 等判断 |

输出会把支持证据、反对证据、缺失条件和 reason code 分开，避免把“找到相关段落”等同于“主张已证实”。

## 当前状态

| 模块 | 状态 |
| --- | --- |
| SEC / 新闻 /电话会摄取管线 | 已实现 |
| Dense + sparse 混合检索与 RRF | 已实现 |
| 公司、财年和申报类型解析 | 已实现 |
| section-aware amendment governance | 已实现并有 fixture 测试 |
| 可回答性门控与生成前拒答 | 已实现 |
| FastAPI 服务 | 已实现 |
| 三个 STDIO MCP 工具 | 已实现 |
| 确定性 IR 与答案评测 | 已实现 |
| 中英文 Evidence Console | 已实现 |
| 自训练 CorpCheck embedding / reranker | 尚未发布 |
| 公开实时 API | 尚未配置 |

## 快速开始

### 环境要求

- Python 3.12+
- PostgreSQL 与 pgvector
- 约 90 MB 的 `sentence-transformers/all-MiniLM-L6-v2` 首次下载
- 可选：Docker，用于恢复项目语料库
- 可选：OpenAI-compatible 推理服务，仅 `/chat` 需要

### 安装

```bash
git clone https://github.com/cassieliang6709/corpcheck.git
cd corpcheck

python3 -m venv .venv
.venv/bin/pip install -e ".[ingestion,dev,mcp]"
cp .env.example .env
```

只对已有数据库提供检索服务时，基础依赖已经足够；`ingestion` extra 用于构建语料，`mcp` extra 用于运行 MCP server。

### 准备数据库

检索需要一个已经填充的 PostgreSQL + pgvector 数据库。用于复现实验的 `financial_rag.dump` 约 1.5 GB，**不包含在 Git 仓库中**；需要从项目维护者处获取并放入 `docker/` 目录。

```bash
cd docker
docker compose up -d db
```

首次启动会执行 `pg_restore`。恢复完成后验证：

```bash
docker compose exec db psql -U postgres -d financial_rag -t -c "
SELECT 'filings', count(*) FROM filings
UNION ALL SELECT 'chunks', count(*) FROM chunks
UNION ALL SELECT 'amendments', count(*) FROM filings WHERE filing_type LIKE '%/A%';"
```

与当前评测快照一致的结果应为：

```text
filings    |   1662
chunks     | 469874
amendments |      0
```

如果数量不同，就不能直接把本 README 中的评测数字归因于当前数据库。完整恢复说明见 [docker/README.md](docker/README.md)。

### 启动 API

```bash
.venv/bin/uvicorn corpcheck.api.main:app --reload --port 8000
```

服务默认运行在 `http://127.0.0.1:8000`：

| Endpoint | 是否需要 LLM | 用途 |
| --- | --- | --- |
| `GET /health` | 否 | 健康检查 |
| `GET /filters` | 否 | 返回可用公司、文件类型和财年 |
| `POST /retrieve` | 否 | 返回混合检索结果 |
| `POST /answerability` | 否 | 判断证据是否足以回答 |
| `POST /claims/extract` | 否 | 拆解待验证主张 |
| `POST /claims/verify` | 否 | 对主张进行证据验证 |
| `POST /chat` | 是 | 通过门控后生成带引用回答 |

服务会在启动时预加载 embedding model，让模型加载错误在首个请求之前暴露。`/chat` 需要在 `.env` 中设置 `SGLANG_BASE_URL`；未设置时返回 503，其他检索接口仍可使用。

如果配置了 `API_KEY`，调用 `/chat` 时需要发送匹配的 `X-API-Key`。空值表示关闭该端点的 key 校验。

## Evidence Console

启动 API 后，在另一个终端运行静态页面：

```bash
python3 -m http.server 4173 --directory landing
```

打开 `http://127.0.0.1:4173/#demo-console`。本地页面会调用 `http://127.0.0.1:8000/answerability`。

线上静态页在没有公开 API origin 时会展示两个明确标记的录制证据状态。该模式下，自定义问题不会被显示为实时查询。演示的生成能力保持关闭，直到检索门槛和 34 条答案评测达到预设标准。

## MCP 服务

CorpCheck 通过 STDIO 暴露与 HTTP API 相同的检索栈：

```bash
.venv/bin/corpcheck-mcp
# 或
.venv/bin/python -m corpcheck.mcp
```

### 注册到客户端

Claude Code：

```bash
claude mcp add corpcheck -- /absolute/path/to/corpcheck/.venv/bin/corpcheck-mcp
```

通用 MCP JSON 配置：

```json
{
  "mcpServers": {
    "corpcheck": {
      "command": "/absolute/path/to/corpcheck/.venv/bin/corpcheck-mcp",
      "env": {
        "DB_HOST": "localhost",
        "DB_PORT": "5432",
        "DB_NAME": "financial_rag"
      }
    }
  }
}
```

必须使用项目虚拟环境中的绝对路径，因为服务依赖 `sentence-transformers`、`asyncpg` 和项目自己的包。

### MCP 工具

| 工具 | 用途 |
| --- | --- |
| `check_answerable` | 在不调用 LLM 的情况下判断当前语料能否支撑回答 |
| `search_filings` | 返回带公司、文件类型、财年、期间和 accession 的证据块 |
| `get_filing_context` | 展开 chunk 周围未截断的上下文，或通过 accession 打开申报文件 |

`check_answerable` 是最关键的产品能力：Agent 可以先问“这个问题能回答吗”，再决定回答或拒绝，而不是依赖语言模型对自身知识的主观判断。

## 数据与配置

主要环境变量见 `.env.example`：

| 配置 | 说明 |
| --- | --- |
| `DB_HOST` / `DB_PORT` / `DB_NAME` | 查询服务使用的数据库 |
| `DATABASE_URL` | 摄取管线使用的连接串 |
| `EMBEDDING_MODEL` | 默认 `all-MiniLM-L6-v2` |
| `DEFAULT_K` / `DEFAULT_ALPHA` | 检索数量与混合参数 |
| `SGLANG_BASE_URL` | 可选 OpenAI-compatible 生成端点 |
| `API_KEY` | 可选 `/chat` 访问密钥 |
| `SEC_USER_AGENT` | SEC 下载所需的产品名和真实联系邮箱 |
| `SEC_DOWNLOAD_DIR` | SEC 原始文件目录 |

SEC 要求可识别的 User-Agent。不要使用示例邮箱进行真实下载，应设置为包含产品名与真实联系邮箱的两段值。

## 可复现评测

### 检索评测

```bash
HF_HUB_OFFLINE=1 .venv/bin/python -m evaluation.ir_eval --label baseline
```

评测进程内直接调用 `retrieve()`，不经过 HTTP，也不调用 LLM。它输出 `Recall@{1,3,5,10}`、`Hit@k`、`MRR@10`、阈值扫描和逐查询明细。

当前公开结果来自 2026-08-06 的固定快照：

| 项目 | 数值 |
| --- | ---: |
| Filings | 1,662 |
| Chunks | 469,874 |
| Issuers | 50 |
| FinanceBench 问题 | 35 |
| Gold evidence spans | 44 |
| Amendment | 0 |

主协议要求证据来自正确公司、申报类型和财年，并覆盖至少 50% 的有效 gold tokens：

| 配置 | Strict R@10 | Loose R@10 | 零正确来源问题 |
| --- | ---: | ---: | ---: |
| R1 baseline | 0.0000 | 0.0857 | 19 / 35 |
| R4 current | **0.1429** | **0.4714** | **4 / 35** |

“零正确来源问题”从 19 降到 4 是最稳健的改进，因为它不依赖 token-overlap 阈值；但 strict Recall@10 仍然只有 0.1429，说明当前系统离可靠金融问答还有明显距离。

完整协议、ceiling、ablation 和逐查询结果见 [evaluation/RESULTS.md](evaluation/RESULTS.md)。

### 指标上限

FinanceBench 提供的是原文 evidence span，不是项目内部 chunk id。使用 token overlap 只能做弱监督，因此还需要计算在现有切分下理论上可达到的上限：

```bash
.venv/bin/python -m evaluation.oracle
```

任何检索结果超过 oracle ceiling 都表示评测代码存在错误。语料或 chunk 策略变化后必须重新计算。

### 最终答案评测

```bash
.venv/bin/python -m evaluation.answer_eval +  --predictions predictions.jsonl +  --gold gold.json +  --output evaluation/runs/baseline/answer-eval.json
```

当前 34 条人工整理的 FinanceBench 基线：

| 指标 | 结果 |
| --- | ---: |
| 可回答性判断正确 | 33 / 34 |
| 最终答案正确 | 1 / 34 |
| 引用编号存在且合法 | 22 / 34 |
| 端到端通过 | 1 / 34 |

引用编号合法不代表引用内容真正支持结论。失败分析显示，主要瓶颈仍是检索不到 gold evidence，而不是生成模型规模。基于这个结果，项目没有把当前 Demo 包装成“可用的金融问答产品”。

### 已拒绝的实验

一个 table-row child 检索实验构建了 130,084 个可搜索行子块：

- loose Recall@10 从 0.4714 提升到 0.5143；
- strict Recall@10 仍为 0.1429；
- p95 延迟从 461 ms 增至 2,374 ms。

由于没有达到预设 strict gate 且延迟显著恶化，该方案默认关闭。另一个硬过滤实验也没有改善 strict Recall@10，因此同样未进入生产路径。

## 验证

```bash
.venv/bin/pytest
.venv/bin/ruff check src evaluation tests
```

检索或切分逻辑变化后，还应运行：

```bash
.venv/bin/python -m evaluation.oracle
.venv/bin/python -m evaluation.ir_eval --label your-change
```

不要在不同语料快照之间直接比较指标，也不要把 benchmark 的 gold metadata 作为过滤条件传给 retriever；那会评测一个线上并不存在的系统。

## 项目结构

```text
src/corpcheck/
├── api/          FastAPI、answerability、claims 与 chat
├── claims/       主张规范化和证据验证
├── db/           asyncpg pool、pgvector schema
├── ingestion/    下载、清洗、切分、向量化和入库
├── retrieval/    查询解析、混合搜索、融合、重排和拒答
├── llm/          基于检索证据的回答生成
├── mcp/          STDIO MCP 协议适配层
├── models.py     Pydantic 请求与响应模型
└── settings.py   服务配置

evaluation/       IR、oracle、ablation、答案与配对评测
tests/            单元、集成与真实元数据 fixture 测试
landing/          中英文静态 Evidence Console
docker/           本地 pgvector 数据库恢复
docs/             Demo、发布边界和升级计划
```

## 模型边界

当前 embedding encoder 是公开模型 [`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)。

CorpCheck 自己实现并负责摄取、查询解析、混合检索、版本治理、拒答门控、API/MCP 适配和评测；**目前不声称拥有自训练模型权重**。只有当自训练 reranker 或 embedding adapter 通过 held-out 检索与拒答门槛后，才会发布 CorpCheck 模型仓库。

## 项目来源

CorpCheck 起源于东北大学 CS6120（Natural Language Processing）课程项目 [`CS6120_finance_RAG`](https://github.com/cassieliang6709/CS6120_finance_RAG)，最初由 [@RobynJiang](https://github.com/RobynJiang)、[@zhiyul1998](https://github.com/zhiyul1998)、[@CodeBusher](https://github.com/CodeBusher) 和 [@cassieliang6709](https://github.com/cassieliang6709) 四人共同完成。

本仓库是 Cassie 的个人延续项目，不是原课程仓库的分支。它重新组织了可安装的服务层，并新增了确定性 IR 评测、section-aware 修订治理、RRF 混合检索和严格拒答门控。原课程项目回答“能否在 10-K 上构建 RAG”，CorpCheck 继续追问：“能否证明检索是对的，并在证据不足时让系统拒绝回答？”

更完整的实验记录、真实 MCP 会话和下一步检索计划见[英文 README](README.md)。
