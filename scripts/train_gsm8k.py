#!/usr/bin/env python3
"""GSM8K trainer entrypoint for AI Infra experiments.

This module provides a real baseline execution path (HF Trainer) for
`baseline` / `lora`, with smoke-compatible metadata output and explicit failure
when optional dependencies for requested modes are missing.

`grpo` currently requires TRL and is intentionally explicit about its
dependency gate so the project doesn't pretend to run unsupported stacks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import re
import statistics
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = ROOT / "evals"


@dataclass(frozen=True)
class RunResult:
    task: str
    status: str
    timestamp_utc: str
    elapsed_seconds: float
    summary: dict[str, Any]
    config: dict[str, Any]
    error: str | None = None


_RE_GSM8K_ANS = re.compile(r"####\s*([-+]?\d[\d,]*(?:\.\d+)?)")
_RE_ANY_NUM = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")


def _read_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must be a YAML mapping")
    return payload


def _coerce_int(value: Any, *, default: int) -> int:
    try:
        value_int = int(value)
        return value_int if value_int > 0 else default
    except Exception:
        return default


def _coerce_float(value: Any, *, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _coerce_bool(value: Any, *, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "y"}
    return bool(value) if value is not None else default


def _set_deterministic(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def _git_sha() -> str | None:
    try:
        return (
            subprocess.check_output(
                ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
                text=True,
            )
            .strip()
        )
    except Exception:
        return None


def _hardware_fingerprint() -> str:
    return "|".join(
        filter(
            bool,
            [
                platform.system(),
                platform.release(),
                platform.machine(),
                f"python={sys.version.split()[0]}",
                f"cpu={os.cpu_count()}",
            ],
        )
    )


def _resolve_output_path(value: str | Path) -> Path:
    output = Path(value)
    if output.is_absolute():
        return output
    return (ROOT / output).resolve()


def _normalize_num(value: str | int | float | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        return None
    cleaned = value.replace(",", "").strip()
    try:
        return float(cleaned)
    except Exception:
        return None


def _extract_gsm8k_answer(text: str) -> float | None:
    if not isinstance(text, str):
        return None
    match = _RE_GSM8K_ANS.search(text)
    if match:
        return _normalize_num(match.group(1))
    fallback = _RE_ANY_NUM.search(text)
    if fallback:
        return _normalize_num(fallback.group(0))
    return None


def _format_prompt(question: str, tokenizer=None) -> str:
    question = (question or "").strip()
    return (
        "You are a strong math assistant. Solve the problem with steps and end with"
        " only the final numeric answer.\n\n"
        f"Question: {question}\n\nAnswer:"
    )


def _dataset_info(dataset_cfg: dict[str, Any] | None) -> tuple[str, str]:
    ds_name = "gsm8k"
    ds_split = "main"
    if isinstance(dataset_cfg, dict):
        ds_name = str(dataset_cfg.get("name", ds_name))
        ds_split = str(dataset_cfg.get("split", ds_split))
    return ds_name, ds_split


def _load_gsm8k_dataset(dataset_name: str, split: str, *, seed: int) -> list[dict[str, Any]]:
    from datasets import load_dataset

    # gsm8k commonly publishes split by name: "main" and partitions train/test.
    dataset_id = "gsm8k" if dataset_name.lower() == "gsm8k" else dataset_name
    split_map = {
        "main": "train",
        "train": "train",
        "test": "test",
    }

    hf_split = split_map.get(split, split)
    raw = load_dataset(dataset_id, "main", trust_remote_code=True)
    candidates = []
    if hf_split in raw:
        candidates = raw[hf_split]
    elif split == "main":
        candidates = raw["train"]
    else:
        raise ValueError(f"Split '{split}' not found in {dataset_id}")

    rows: list[dict[str, Any]] = [
        {
            "question": str(item.get("question", "")).strip(),
            "answer": str(item.get("answer", "")).strip(),
        }
        for item in candidates
        if isinstance(item, dict)
    ]

    # make train subset deterministic so repeated runs are comparable
    rnd = random.Random(seed)
    rnd.shuffle(rows)
    return rows


def _tokenize_batch(tokenizer, texts: list[str], max_len: int) -> dict[str, Any]:
    outputs = tokenizer(
        texts,
        truncation=True,
        padding=False,
        max_length=max_len,
        add_special_tokens=True,
    )
    outputs["labels"] = [ids[:] for ids in outputs["input_ids"]]
    return outputs


def _eval_pass_at1(
    model,
    tokenizer,
    rows: list[dict[str, Any]],
    *,
    eval_limit: int,
    max_new_tokens: int,
    max_length: int,
) -> dict[str, Any]:
    import torch

    selected = rows[:eval_limit] if eval_limit > 0 else rows
    if not selected:
        return {
            "pass_at_1": 0.0,
            "samples": 0,
            "correct": 0,
            "avg_latency_ms": 0.0,
            "latencies_ms": [],
            "errors": 0,
            "coverage": 0.0,
        }

    prompts = [_format_prompt(row["question"]) for row in selected]
    targets = [_extract_gsm8k_answer(row["answer"]) for row in selected]

    if hasattr(model, "device"):
        device = next(model.parameters()).device
    else:  # pragma: no cover
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

    pred_count = 0
    correct = 0
    errors = 0
    latencies: list[float] = []
    sample_limit = min(len(selected), len(prompts))

    for idx in range(sample_limit):
        start = datetime.now(tz=UTC)
        try:
            enc = tokenizer(
                prompts[idx],
                return_tensors="pt",
                truncation=True,
                padding=False,
                max_length=max_length,
            )
            for k in ("input_ids", "attention_mask"):
                enc[k] = enc[k].to(device)

            prompt_len = int(enc["input_ids"].shape[-1])
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=enc["input_ids"],
                    attention_mask=enc["attention_mask"],
                    do_sample=False,
                    max_new_tokens=max_new_tokens,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.eos_token_id,
                )

            decoded = tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True)
            pred = _extract_gsm8k_answer(decoded)
            target = targets[idx]
            if pred is not None and target is not None:
                pred_count += 1
                if abs(pred - target) < 1e-6:
                    correct += 1
        except Exception:
            errors += 1

        latencies.append((datetime.now(tz=UTC) - start).total_seconds() * 1000)

    samples = len(selected)
    evaluated = max(1, pred_count)
    return {
        "pass_at_1": round(correct / samples, 4) if samples else 0.0,
        "samples": samples,
        "correct": correct,
        "coverage": round(pred_count / evaluated * 100.0 if pred_count else 0.0, 2),
        "avg_latency_ms": round(statistics.mean(latencies), 2) if latencies else 0.0,
        "latencies_ms": latencies,
        "errors": errors,
        "raw_targets": targets[: min(len(targets), len(latencies))],
    }


def _build_payload(
    cfg: dict[str, Any],
    mode: str,
    *,
    seed: int,
    repeat: int,
    duration_seconds: float,
    eval_summary: dict[str, Any],
    extra: dict[str, Any],
) -> dict[str, Any]:
    return {
        "generated_at_utc": datetime.now(tz=UTC).isoformat(),
        "manifest_version": "2026-08-20",
        "run_metadata": {
            "dataset_version": (
                f"{cfg.get('dataset', {}).get('name', 'gsm8k')}/"
                f"{cfg.get('dataset', {}).get('split', 'main')}"
            ),
            "commit": _git_sha(),
            "timestamp": datetime.now(tz=UTC).isoformat(),
            "hardware": _hardware_fingerprint(),
            "seed": seed,
            "params": {
                **(cfg.get("training", {}) or {}),
                **extra,
            },
            "repeat_runs": repeat,
        },
        "results": [
            {
                "task": cfg.get("name", f"gsm8k_{mode}"),
                "status": "completed",
                "timestamp_utc": datetime.now(tz=UTC).isoformat(),
                "elapsed_seconds": round(duration_seconds, 4),
                "summary": {
                    "smoke": False,
                    "metrics": eval_summary,
                },
                "config": {
                    "mode": mode,
                    "dataset": cfg.get("dataset", {}),
                    "evaluation": cfg.get("evaluation", {}),
                    "training": cfg.get("training", {}),
                    "seed": seed,
                },
            }
        ],
    }


def _write_payload(output_root: Path, label: str, payload: dict[str, Any]) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    out = output_root / f"results_{label}.json"
    latest = output_root / "results.json"
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    out.write_text(rendered, encoding="utf-8")
    latest.write_text(rendered, encoding="utf-8")
    return out


def _run_sft_variant(
    cfg: dict[str, Any],
    *,
    mode: str,
    seed: int,
    eval_limit: int,
    max_steps: int | None,
    batch_size: int | None,
    output_root: Path,
    label: str,
) -> RunResult:
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )

    start = datetime.now(tz=UTC)
    _set_deterministic(seed)

    dataset_name, split = _dataset_info(cfg.get("dataset"))
    rows = _load_gsm8k_dataset(dataset_name, split, seed=seed)

    train_cfg = cfg.get("training", {}) or {}
    model_cfg = cfg.get("model", {}) or {}
    eval_cfg = cfg.get("evaluation", {}) or {}

    model_id = str(model_cfg.get("id", "Qwen/Qwen2.5-0.5B-Instruct"))
    max_len = _coerce_int(model_cfg.get("max_seq_length"), default=1024)

    output_dir = _resolve_output_path(train_cfg.get("output_dir", "output/ai_infra/gsm8k-run"))

    eval_count = _coerce_int(eval_cfg.get("repeats", 3), default=3)
    eval_rows = rows[-200:] if split != "train" else rows[:200]
    if len(eval_rows) < eval_count:
        eval_rows = rows[:]

    train_batch = _coerce_int(batch_size, default=_coerce_int(train_cfg.get("per_device_train_batch_size"), default=8))
    grad_acc = _coerce_int(train_cfg.get("gradient_accumulation_steps"), default=1)
    eval_batch = _coerce_int(train_cfg.get("per_device_eval_batch_size"), default=max(1, train_batch))
    warmup_ratio = _coerce_float(train_cfg.get("warmup_ratio"), default=0.05)
    lr = _coerce_float(train_cfg.get("learning_rate"), default=2e-5)
    max_steps = _coerce_int(max_steps, default=_coerce_int(train_cfg.get("max_steps"), default=50))
    num_epochs = _coerce_int(train_cfg.get("num_train_epochs"), default=1)
    grad_ckpt = _coerce_bool(train_cfg.get("gradient_checkpointing", False), default=False)
    logging_steps = _coerce_int(train_cfg.get("logging_steps"), default=20)
    save_steps = _coerce_int(train_cfg.get("save_steps"), default=0)

    prompts = [_format_prompt(row["question"]) for row in rows]
    answers = [f" {row['answer']}" for row in rows]
    full_texts = [prompt + ans for prompt, ans in zip(prompts, answers)]

    base = int(max_steps * train_batch * max(1, grad_acc))
    if base > 0:
        full_texts = full_texts[: min(len(full_texts), base)]

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto",
    )

    if mode == "lora":
        from peft import LoraConfig, TaskType, get_peft_model

        lora_cfg = train_cfg.get("lora", {}) or {}
        target_modules = train_cfg.get("target_modules", None)
        if target_modules is None:
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
        peft_cfg = LoraConfig(
            r=_coerce_int(lora_cfg.get("r"), default=16),
            lora_alpha=_coerce_int(lora_cfg.get("alpha"), default=32),
            lora_dropout=_coerce_float(lora_cfg.get("dropout"), default=0.05),
            target_modules=target_modules,
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, peft_cfg)

    from datasets import Dataset

    train_data = tokenizer(
        full_texts,
        truncation=True,
        max_length=max_len,
        padding=False,
    )
    train_data["labels"] = [ids[:] for ids in train_data["input_ids"]]
    train_dataset = Dataset.from_dict(train_data)

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    args_kwargs = dict(
        output_dir=str(output_dir),
        overwrite_output_dir=True,
        per_device_train_batch_size=train_batch,
        per_device_eval_batch_size=eval_batch,
        gradient_accumulation_steps=grad_acc,
        learning_rate=lr,
        warmup_ratio=warmup_ratio,
        logging_steps=logging_steps,
        report_to=[],
        gradient_checkpointing=grad_ckpt,
        max_steps=max_steps,
        bf16=False,
        fp16=False,
        dataloader_pin_memory=False,
    )
    if save_steps > 0:
        args_kwargs["save_steps"] = save_steps
    if grad_ckpt:
        args_kwargs["gradient_checkpointing"] = True
    if not max_steps:
        args_kwargs.pop("max_steps", None)
        args_kwargs["num_train_epochs"] = num_epochs

    training_args = TrainingArguments(**args_kwargs)

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collator,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model(output_dir)

    eval_summary = _eval_pass_at1(
        model,
        tokenizer,
        eval_rows,
        eval_limit=max(1, int(eval_count)),
        max_new_tokens=_coerce_int(eval_cfg.get("max_new_tokens"), default=64),
        max_length=max_len,
    )

    elapsed = (datetime.now(tz=UTC) - start).total_seconds()
    payload = _build_payload(
        cfg,
        mode=mode,
        seed=seed,
        repeat=_coerce_int(eval_cfg.get("repeats"), default=3),
        duration_seconds=elapsed,
        eval_summary=eval_summary,
        extra={
            "mode": mode,
            "resolved_output_dir": str(output_dir),
            "max_steps": max_steps,
            "eval_limit": eval_count,
            "smoke": False,
        },
    )
    output = _write_payload(output_root=output_root, label=label, payload=payload)

    return RunResult(
        task=str(cfg.get("name", f"gsm8k_{mode}")),
        status="completed",
        timestamp_utc=datetime.now(tz=UTC).isoformat(),
        elapsed_seconds=elapsed,
        summary={
            "output_file": str(output),
            "pass_at_1": eval_summary.get("pass_at_1"),
            "metrics": eval_summary,
            "smoke": False,
        },
        config={
            "mode": mode,
            "dataset": cfg.get("dataset", {}),
            "training": cfg.get("training", {}),
            "evaluation": cfg.get("evaluation", {}),
            "seed": seed,
            "output_dir": str(output_dir),
            "label": label,
            "eval_limit": eval_count,
        },
    )


def _run_grpo(
    cfg: dict[str, Any],
    *,
    seed: int,
    eval_limit: int,
    output_root: Path,
    label: str,
) -> RunResult:
    start = datetime.now(tz=UTC)

    try:
        import trl  # noqa: F401
    except Exception as exc:
        return RunResult(
            task=str(cfg.get("name", "gsm8k_grpo")),
            status="failed",
            timestamp_utc=datetime.now(tz=UTC).isoformat(),
            elapsed_seconds=(datetime.now(tz=UTC) - start).total_seconds(),
            summary={
                "smoke": False,
                "reason": "trl dependency not installed",
                "details": str(exc),
            },
            config={
                "mode": "grpo",
                "seed": seed,
                "eval_limit": eval_limit,
            },
            error="trl dependency is required for GRPO path",
        )

    return RunResult(
        task=str(cfg.get("name", "gsm8k_grpo")),
        status="failed",
        timestamp_utc=datetime.now(tz=UTC).isoformat(),
        elapsed_seconds=(datetime.now(tz=UTC) - start).total_seconds(),
        summary={
            "smoke": False,
            "reason": "GRPO execution is not implemented in this branch",
            "note": "Install trl and extend _run_grpo() before enabling real GRPO runs",
        },
        config={
            "mode": "grpo",
            "seed": seed,
            "eval_limit": eval_limit,
        },
        error="GRPO execution path is intentionally explicit and currently disabled",
    )


def _format_missing_dependencies(mode: str, missing: list[str]) -> None:
    print(f"[{mode}] Missing required dependencies: {', '.join(missing)}")
    if mode == "baseline":
        print("Install with: python -m pip install torch transformers datasets accelerate")
    elif mode == "lora":
        print("Install with: python -m pip install torch transformers datasets peft accelerate bitsandbytes")
    elif mode == "grpo":
        print("Install with: python -m pip install torch transformers datasets accelerate trl")


def _check_dependencies(mode: str) -> list[str]:
    deps: list[str] = []
    required: dict[str, list[str]] = {
        "baseline": ["torch", "transformers", "datasets", "accelerate"],
        "lora": ["torch", "transformers", "datasets", "peft", "accelerate", "bitsandbytes"],
        "grpo": ["torch", "transformers", "datasets", "trl", "accelerate"],
    }
    for dep in required.get(mode, []):
        try:
            __import__(dep)
        except Exception:
            deps.append(dep)
    return deps


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=["baseline", "lora", "grpo"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--dataset")
    parser.add_argument("--output-dir")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--lora-r", type=int)
    parser.add_argument("--eval-limit", type=int)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--label", help="result label")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print(f"Config not found: {cfg_path}")
        return 2

    cfg = _read_yaml(cfg_path)
    mode = args.mode

    train_cfg = cfg.get("training", {})
    seed = _coerce_int(
        args.seed if args.seed is not None else train_cfg.get("seed"),
        default=42,
    )
    if args.output_dir:
        train_cfg["output_dir"] = args.output_dir
    if args.lora_r is not None and mode == "lora":
        lora_cfg = train_cfg.get("lora", {})
        if not isinstance(lora_cfg, dict):
            lora_cfg = {}
        lora_cfg["r"] = args.lora_r
        train_cfg["lora"] = lora_cfg
    train_cfg["seed"] = seed

    eval_cfg = cfg.get("evaluation", {}) or {}
    eval_limit = _coerce_int(args.eval_limit, default=_coerce_int(eval_cfg.get("repeats"), default=3))

    label_seed = hashlib.sha256(f"{mode}|{seed}|{datetime.now(tz=UTC).isoformat()}".encode("utf-8")).hexdigest()[:10]
    label = args.label or f"{mode}-{label_seed}"
    output_root = Path(args.output_root)

    _set_deterministic(seed)

    missing = _check_dependencies(mode)
    if missing:
        _format_missing_dependencies(mode, missing)
        return 2

    if mode in {"baseline", "lora"}:
        result = _run_sft_variant(
            cfg,
            mode=mode,
            seed=seed,
            eval_limit=eval_limit,
            max_steps=args.max_steps,
            batch_size=args.batch_size,
            output_root=output_root,
            label=label,
        )
    else:
        result = _run_grpo(
            cfg,
            seed=seed,
            eval_limit=eval_limit,
            output_root=output_root,
            label=label,
        )

    print(f"[mode] {mode}")
    print(f"[seed] {seed}")
    print(f"[status] {result.status}")
    print(f"[output-root] {output_root}")
    print(f"[label] {label}")
    if result.summary.get("pass_at_1") is not None:
        print(f"[pass@1] {result.summary.get('pass_at_1')}")
    if result.error:
        print(f"[error] {result.error}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
