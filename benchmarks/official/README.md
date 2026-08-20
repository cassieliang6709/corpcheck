# Official benchmark plan (AI Infra + CorpCheck)

- 目标：把 Plan A（AI Infra）和 CorpCheck 官方任务放在同一份执行入口。
- 结果统一落库到 `evals/results_*.json`。
- 关键约束：
  - 不注入任务外泄漏字段（company/year/filing）到检索器。
  - 每次跑完写 `dataset_version / commit / timestamp / hardware / seed / params`。

## 执行入口

- AI Infra（Tiny-Reasoning）：见 `GSM8K/README.md`
- CorpCheck 官方：见 `../..` 项目的 `evaluation/official/benchmark_manifest.yaml`。

