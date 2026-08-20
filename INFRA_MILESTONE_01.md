# INFRA_MILESTONE_01

> 目标：把 AI Infra（GSM8K 轨道）跑出 interview-ready 的基线对照表。

## 当前进度

- 状态：GSM8K 执行脚手架已就绪，配置加载与产物落盘可复现（`evals/results_<label>.json`）。
- **尚无任何真实训练或评测结果。** `--smoke` 只验证接线，产物里所有指标写 `null`、
  `status: scaffolding_only`、`measured: false`，不要把它当成基线。

## 预期产出

| 阶段 | 命令/入口 | 关键指标 | 说明 |
|---|---|---|---|
| baseline | `scripts/run_gsm8k.py --mode baseline --config benchmarks/official/GSM8K/baseline_config.yaml` | pass@1 / 平均时长 / 显存 | 基线，不含 LoRA |
| LoRA SFT | `scripts/run_gsm8k.py --mode lora --config benchmarks/official/GSM8K/lora_config.yaml` | pass@1 / 稳定性 / 成本 | 固定 seed=42 / batch/steps |
| GRPO | `scripts/run_gsm8k.py --mode grpo --config benchmarks/official/GSM8K/grpo_config.yaml` | pass@1 / 方差 / 复现性 | 官方路线复现尝试 |

## 本次复现记录

> 三个阶段都还没有真实跑过。等拿到真实结果再填下表，并且必须记录：命令、
> `evals/` 产物路径、硬件、seed。填表前对照 `evaluation/RESULTS.md` §6 的
> 「什么能写进简历」规则。

| 阶段 | 日期 | pass@1 | 平均 latency | VRAM | 产物 |
|---|---|---|---|---|---|
| baseline | — | 未测 | 未测 | 未测 | — |
| LoRA | — | 未测 | 未测 | 未测 | — |
| GRPO | — | 未测 | 未测 | 未测 | — |

## 成果落盘

- 文件命名：`evals/results_<timestamp>.json`
- 统一字段：`dataset_version` / `commit` / `timestamp` / `hardware` / `seed` / `params`
