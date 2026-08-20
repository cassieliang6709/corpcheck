# GSM8K 官方执行清单（先闭环）

## 统一环境

```bash
. ./.venv/bin/activate
# AI Infra 依赖（按需补齐）：
# pip install torch transformers datasets evaluate trl peft accelerate bitsandbytes
```

## Day 1（AI Infra）

- Baseline（不含 LoRA）
```bash
python scripts/run_gsm8k.py --mode baseline \
  --config benchmarks/official/GSM8K/baseline_config.yaml
```

- LoRA SFT（固定 seed/batch/steps）
```bash
python scripts/run_gsm8k.py --mode lora \
  --config benchmarks/official/GSM8K/lora_config.yaml
```

- 快速打通（离线）：
```bash
python scripts/run_gsm8k.py --mode baseline \
  --config benchmarks/official/GSM8K/baseline_config.yaml \
  --smoke --label gsm8k-baseline-smoke
```

## Day 2（AI Infra）

- GRPO（固定种子+步数）
```bash
python scripts/run_gsm8k.py --mode grpo \
  --config benchmarks/official/GSM8K/grpo_config.yaml
```

- 开始真实训练前先替换执行器（当前为 placeholder）：
```bash
python scripts/run_gsm8k.py --mode grpo \
  --config benchmarks/official/GSM8K/grpo_config.yaml --execute
```

> `--smoke` 会生成 `evals/results_<label>.json`（含 dataset_version/commit/hardware/seed/params）用于面试材料；真实训练请改造 `scripts/run_placeholder_pipeline.py` 或替换成你的实际 trainer 命令。

## 对外汇报产物

- 生成 `INFRA_MILESTONE_01.md`：写 baseline / LoRA / GRPO 的 pass@1 与稳定性（粗粒度成本）
- 每条结果命名为 `evals/results_<timestamp>.json`
