#!/usr/bin/env python3
"""GSM8K execution helper for AI Infra runbook.

The script keeps a reproducible offline path (``--smoke``) and a concrete command
path for real training scaffolding.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = ROOT / "evals"


Mode = str

DEPENDENCIES: dict[Mode, list[str]] = {
    "baseline": ["torch", "transformers", "datasets", "accelerate"],
    "lora": ["torch", "transformers", "datasets", "peft", "accelerate", "bitsandbytes"],
    "grpo": ["torch", "transformers", "datasets", "trl", "accelerate"],
}

TRAINER_SCRIPT = ROOT / "scripts" / "train_gsm8k.py"


@dataclass(frozen=True)
class SmokeMeta:
    status: str
    bucket: str


def _read_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must be YAML mapping")
    return payload


def _coerce_int(value: Any, *, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _hardware() -> str:
    cpu = os.cpu_count() or "unknown"
    return "|".join(
        filter(
            bool,
            [
                platform.system(),
                platform.release(),
                platform.machine(),
                f"python={sys.version.split()[0]}",
                f"cpu={cpu}",
            ],
        )
    )


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


def _missing_dependencies(mode: Mode) -> list[str]:
    missing: list[str] = []
    for pkg in DEPENDENCIES.get(mode, []):
        try:
            importlib.import_module(pkg)
        except ModuleNotFoundError:
            missing.append(pkg)
    return missing


def _render_command(
    mode: Mode,
    cfg: dict[str, Any],
    *,
    seed: int,
    max_steps: int | None,
    batch_size: int | None,
) -> list[str]:
    cmd = [
        sys.executable,
        str(TRAINER_SCRIPT),
        "--mode",
        mode,
        "--config",
        str(cfg["__path"]),
        "--seed",
        str(seed),
        "--label",
        f"{mode}-{cfg.get('name', 'gsm8k')}",
        "--output-root",
        str(ROOT / "evals"),
    ]

    if mode == "grpo":
        cmd.extend(["--eval-limit", str(cfg.get("evaluation", {}).get("repeats", 3))])

    if mode == "lora":
        lora_r = _coerce_int(cfg.get("training", {}).get("lora", {}).get("r", 16), default=16)
        cmd.extend(["--lora-r", str(lora_r)])

    if max_steps is not None:
        cmd.extend(["--max-steps", str(max_steps)])
    if batch_size is not None:
        cmd.extend(["--batch-size", str(batch_size)])

    return cmd


def _smoke_summary() -> SmokeMeta:
    """A smoke run has no quality to report, so it does not claim one."""
    return SmokeMeta(status="scaffolding", bucket="not-measured")


def _write_smoke_record(
    *,
    mode: Mode,
    cfg: dict[str, Any],
    seed: int,
    repeats: int,
    label: str,
    output_root: Path,
) -> Path:
    # A smoke run exercises config loading and artifact writing without touching
    # a GPU. Earlier this synthesised plausible-looking metrics from a hash of the
    # config, which is worse than reporting nothing: 0.869 pass@1 reads like a
    # result and gets quoted like one. Emit nulls so a missing measurement is
    # visibly missing.
    metrics = {
        "pass_at_1": None,
        "repeats": repeats,
        "avg_latency_ms": None,
        "vram_gb": None,
        "throughput_tokens_per_s": None,
    }

    payload = {
        "generated_at_utc": datetime.now(tz=UTC).isoformat(),
        "manifest_version": "2026-08-20",
        "run_metadata": {
            "dataset_version": (
                f"{cfg.get('dataset', {}).get('name', 'gsm8k')}/"
                f"{cfg.get('dataset', {}).get('split', 'main')}"
            ),
            "commit": _git_sha(),
            "timestamp": datetime.now(tz=UTC).isoformat(),
            "hardware": _hardware(),
            "seed": seed,
            "params": {
                "mode": mode,
                **(cfg.get("training", {}) or {}),
            },
        },
        "results": [
            {
                "task": cfg.get("name", f"gsm8k_{mode}"),
                "status": "scaffolding_only",
                "timestamp_utc": datetime.now(tz=UTC).isoformat(),
                "elapsed_seconds": None,
                "summary": {
                    "smoke": True,
                    "measured": False,
                    **asdict(_smoke_summary()),
                    "metadata": {
                        "pass_at_1": None,
                        "metrics": metrics,
                    },
                },
                "config": {
                    "seed": seed,
                    "dataset": cfg.get("dataset", {}),
                    "evaluation": cfg.get("evaluation", {}),
                    "training": cfg.get("training", {}),
                },
            }
        ],
    }

    output_root.mkdir(parents=True, exist_ok=True)
    target = output_root / f"results_{label}.json"
    latest = output_root / "results.json"
    serialized = json.dumps(payload, ensure_ascii=False, indent=2)
    target.write_text(serialized, encoding="utf-8")
    latest.write_text(serialized, encoding="utf-8")
    return target


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=["baseline", "lora", "grpo"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="run trainer command now (scripts/train_gsm8k.py)",
    )
    parser.add_argument("--max-steps", type=int, help="override max steps")
    parser.add_argument("--batch-size", type=int, help="override train batch size")
    parser.add_argument("--label", help="result label; defaults to timestamp")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print(f"Config not found: {cfg_path}")
        return 2

    cfg = _read_yaml(cfg_path)
    cfg["__path"] = cfg_path

    if args.seed is not None:
        cfg.setdefault("training", {})["seed"] = args.seed

    mode = args.mode
    seed = _coerce_int(cfg.get("training", {}).get("seed"), default=42)
    repeats = _coerce_int(cfg.get("evaluation", {}).get("repeats"), default=3)
    label = args.label or f"{mode}-{datetime.now(tz=UTC).strftime('%Y%m%d_%H%M%S')}"
    output_root = Path(args.output_root)

    print(f"[mode] {mode}")
    print(f"[seed] {seed}")
    print(f"[output] {cfg.get('training', {}).get('output_dir', 'output/ai_infra')}")
    print(f"[project] {ROOT}")

    if args.smoke:
        out = _write_smoke_record(
            mode=mode,
            cfg=cfg,
            seed=seed,
            repeats=repeats,
            label=label,
            output_root=output_root,
        )
        print(f"\n[smoke] wrote {out}")
        return 0

    cmd = _render_command(
        mode,
        cfg,
        seed=seed,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
    )

    print("\nSuggested command:")
    print("  ".join(cmd))

    missing = _missing_dependencies(mode)
    if missing:
        print("\nMissing optional dependencies for real run:")
        print("  " + ", ".join(missing))
        print(f"Install first: python -m pip install {' '.join(DEPENDENCIES[mode])}")

    if not args.execute:
        print("\nTip: add --execute to run the scaffold now.")
        return 0

    if missing:
        return 2

    try:
        proc = subprocess.run(cmd, check=True, cwd=ROOT)
        return int(proc.returncode)
    except FileNotFoundError as exc:
        print(f"Failed to run scaffold: {exc}")
        return 2
    except subprocess.CalledProcessError as exc:
        print(f"Scaffold failed with code {exc.returncode}")
        return int(exc.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
