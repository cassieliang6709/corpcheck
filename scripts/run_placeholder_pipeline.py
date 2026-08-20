#!/usr/bin/env python3
"""Placeholder runner for AI Infra workflow.

This keeps `run_gsm8k.py` executable in offline mode while you are wiring your
preferred training stack (TRL/PEFT/HF). Do not use this file as your production
trainer.
"""

from __future__ import annotations

import argparse


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=["baseline", "lora", "grpo"])
    parser.add_argument("--dataset", default="gsm8k")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--lora-r", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--batch-size", type=int)
    args = parser.parse_args()

    print("[ai_infra_placeholder] mode=", args.mode)
    print("[ai_infra_placeholder] dataset=", args.dataset)
    print("[ai_infra_placeholder] seed=", args.seed)
    print("[ai_infra_placeholder] output_dir=", args.output_dir)
    if args.lora_r is not None:
        print("[ai_infra_placeholder] lora_r=", args.lora_r)

    print("\nThis is a placeholder. Replace with your real training command before interview demo.")
    if args.max_steps is not None or args.batch_size is not None:
        print(f"[ai_infra_placeholder] max_steps={args.max_steps} batch_size={args.batch_size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
