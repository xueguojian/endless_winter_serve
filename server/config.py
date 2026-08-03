from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config.yaml"
TASK_DEFAULTS_PATH = ROOT / "task_defaults.yaml"


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"配置文件格式错误: {path}")
    return data


def load_server_config() -> dict[str, Any]:
    return load_yaml(CONFIG_PATH)


def load_task_defaults() -> dict[str, Any]:
    return load_yaml(TASK_DEFAULTS_PATH)


def merge_task_config(task_id: str, override: dict[str, Any] | None = None) -> dict[str, Any]:
    defaults = load_task_defaults()
    common = dict(defaults.get("common") or {})
    base = dict(defaults.get(task_id) or {})
    # common 作为缺省，不覆盖任务已有字段
    merged = {**common, **base}
    if override:
        for key, value in override.items():
            if key == "coords" and isinstance(value, dict):
                coords = dict(merged.get("coords") or {})
                coords.update(value)
                merged["coords"] = coords
            else:
                merged[key] = value
    return merged
