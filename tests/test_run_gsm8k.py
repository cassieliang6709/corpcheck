import json
from pathlib import Path

from scripts import run_gsm8k


def test_render_command_for_lora_includes_overrides_and_output_root():
    cfg = {
        "__path": Path("benchmarks/official/GSM8K/lora_config.yaml"),
        "dataset": {"name": "gsm8k"},
        "training": {
            "seed": 42,
            "output_dir": "output/ai_infra/gsm8k-lora",
            "lora": {"r": 16},
        },
    }

    cmd = run_gsm8k._render_command(
        "lora",
        cfg,
        seed=42,
        max_steps=222,
        batch_size=12,
    )

    assert "train_gsm8k.py" in cmd[1]
    assert str(cmd).count("--mode") == 1
    assert "--config" in cmd
    assert "--max-steps" in cmd
    assert "--batch-size" in cmd


def test_smoke_write_records_are_written_and_loadable(tmp_path: Path):
    cfg = {
        "dataset": {"name": "gsm8k", "split": "main"},
        "training": {"seed": 42, "output_dir": "output/ai_infra/gsm8k-baseline"},
        "evaluation": {"metric": "pass@1", "repeats": 4},
        "name": "gsm8k_baseline",
    }

    out = run_gsm8k._write_smoke_record(
        mode="baseline",
        cfg=cfg,
        seed=42,
        repeats=4,
        label="unit-smoke",
        output_root=tmp_path / "evals",
    )

    stamped = out.read_text(encoding="utf-8")
    latest = (tmp_path / "evals" / "results.json").read_text(encoding="utf-8")

    assert json.loads(stamped)["manifest_version"] == "2026-08-20"
    assert json.loads(stamped)["results"][0]["summary"]["smoke"] is True
    assert stamped == latest
    assert (tmp_path / "evals" / "results_unit-smoke.json").exists()
