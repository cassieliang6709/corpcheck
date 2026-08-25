"""Manifest parsing for official benchmark orchestration.

中文：把正式评测清单解析为受约束的计划对象；这里保留状态、输入与默认参数的显式
声明，缺失或无效字段必须在运行前暴露。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Optional

try:
    import yaml
except Exception:  # pragma: no cover - optional; used only in CLI execution path
    yaml = None


@dataclass(frozen=True)
class BenchmarkPlan:
    """Single benchmark entry from ``benchmark_manifest.yaml``.

    中文：一个受 manifest 约束的 benchmark 声明；其状态和所需输入决定能否安全执行。
    """

    name: str
    status: str
    source_repo: str
    notes: str = ""
    dataset: Optional[str] = None
    runner: str = "generic"
    required_files: list[str] | None = None
    default_args: dict[str, Any] | None = None

    @property
    def is_planned(self) -> bool:
        """Whether this plan is intentionally not executable yet.

        中文：统一识别清单中的待办状态，调用方据此避免把尚未准备的评测当作已运行。
        """
        return self.status.lower() in {"planned", "todo", "pending"}


@dataclass(frozen=True)
class BenchmarkResult:
    """Structured run result persisted by ``runner.py``.

    中文：正式运行的可审计结果载体，保留配置、耗时和失败信息而非只保存单一分数。
    """

    task: str
    status: str
    timestamp_utc: str
    elapsed_seconds: float
    summary: dict[str, Any]
    config: dict[str, Any]
    error: str | None = None


@dataclass(frozen=True)
class OfficialManifest:
    """Top-level manifest object with benchmark definitions.

    中文：正式评测计划的唯一解析结果；runner 不应根据目录内容自行发现替代任务。
    """

    version: str
    created_at_utc: str
    runs: list[BenchmarkPlan]

    def get(self, name: str) -> Optional[BenchmarkPlan]:
        """Return the exact named plan, or ``None`` when the manifest lacks it.

        中文：名称查询不做模糊匹配，调用方可据此把拼写或清单遗漏当成显式失败。
        """
        for run in self.runs:
            if run.name == name:
                return run
        return None


def _required_args(payload: dict[str, Any]) -> dict[str, Any] | None:
    raw = payload.get("default_args")
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError(f"benchmark default_args must be a dict, got {type(raw).__name__}")
    return raw


def _required_files(payload: dict[str, Any]) -> list[str] | None:
    raw = payload.get("required_files")
    if raw is None:
        return None
    if not isinstance(raw, list):
        raise ValueError(f"benchmark required_files must be a list, got {type(raw).__name__}")
    fixed: list[str] = []
    for item in raw:
        if not isinstance(item, str):
            raise ValueError("benchmark required_files must contain only strings")
        fixed.append(item)
    return fixed


def _normalize_plan(item: dict[str, Any], index: int) -> BenchmarkPlan:
    name = item.get("name")
    status = item.get("status")
    source_repo = item.get("source_repo")
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"run[{index}].name must be a non-empty string")
    if not isinstance(status, str) or not status.strip():
        raise ValueError(f"run[{index}].status must be a non-empty string")
    if not isinstance(source_repo, str) or not source_repo.strip():
        raise ValueError(f"run[{index}].source_repo must be a non-empty string")

    return BenchmarkPlan(
        name=name.strip(),
        status=status.strip(),
        source_repo=source_repo.strip(),
        notes=str(item.get("notes") or ""),
        dataset=item.get("dataset"),
        runner=item.get("runner", "generic"),
        required_files=_required_files(item),
        default_args=_required_args(item),
    )


def load_manifest(path: Path) -> OfficialManifest:
    """Load ``benchmark_manifest.yaml`` or any YAML-compatible dict payload."""
    if not path.exists():
        raise FileNotFoundError(f"manifest not found: {path}")

    raw_text = path.read_text(encoding="utf-8")
    if yaml is None:
        raise ImportError("PyYAML is required to parse evaluation/official manifest files")

    payload = yaml.safe_load(raw_text)
    if not isinstance(payload, dict):
        raise ValueError("manifest must be a YAML mapping")

    version = payload.get("version")
    created_at = payload.get("created_at_utc")
    runs = payload.get("runs")

    if not isinstance(version, int | float | str | datetime | date) or str(
        version
    ).strip() == "":
        raise ValueError("manifest.version must be set")
    if isinstance(created_at, datetime):
        created_at = created_at.isoformat()
    elif isinstance(created_at, date):
        created_at = datetime(
            created_at.year,
            created_at.month,
            created_at.day,
            tzinfo=UTC,
        ).isoformat()
    elif not isinstance(created_at, str) or not created_at.strip():
        created_at = datetime.now(tz=UTC).isoformat()
    if not isinstance(runs, list):
        raise ValueError("manifest.runs must be a list")

    normalized = [
        _normalize_plan(item, index=i) for i, item in enumerate(runs) if isinstance(item, dict)
    ]
    if not normalized:
        raise ValueError("manifest.runs cannot be empty")
    return OfficialManifest(
        version=str(version), created_at_utc=created_at.strip(), runs=normalized
    )
