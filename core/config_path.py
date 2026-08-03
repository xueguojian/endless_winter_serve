"""配置文件路径解析（多开实例 config_5555.yaml / config_5557.yaml 等）。"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
EXAMPLE_CONFIG_PATH = ROOT / "config.example.yaml"
# 未指定 --config 时的默认实例（与 run_gui_5555.bat 一致）
PRIMARY_CONFIG_PATH = ROOT / "config_5555.yaml"
# 兼容旧引用
DEFAULT_CONFIG_PATH = PRIMARY_CONFIG_PATH

INSTANCE_CONFIG_GLOB = "config_555*.yaml"


def resolve_config_path(raw: str | Path | None) -> Path:
    if raw is None or (isinstance(raw, str) and not str(raw).strip()):
        return PRIMARY_CONFIG_PATH
    path = Path(raw)
    if not path.is_absolute():
        path = ROOT / path
    return path


def infer_adb_port_from_stem(stem: str) -> int | None:
    """从 config_5557.yaml 等文件名推断 ADB 端口。"""
    for match in re.finditer(r"\d+", stem):
        port = int(match.group())
        if 5554 <= port <= 5600:
            return port
    return None


def list_instance_config_paths() -> list[Path]:
    """本机已存在的多开实例配置文件（config_5555.yaml …）。"""
    return sorted(ROOT.glob(INSTANCE_CONFIG_GLOB))


def default_instance_name(cfg: dict, config_path: Path) -> str:
    """GUI 窗口标题用的实例名；未设置时按端口或配置文件名推断。"""
    gui = cfg.get("gui") or {}
    name = str(gui.get("instance_name", "")).strip()
    if name:
        return name
    dev = cfg.get("device") or {}
    port = dev.get("adb_port")
    if port is not None:
        return f"模拟器 {port}"
    inferred = infer_adb_port_from_stem(config_path.stem)
    if inferred is not None:
        return f"模拟器 {inferred}"
    return config_path.stem


def _apply_port_from_stem(path: Path, cfg: dict) -> dict:
    port = infer_adb_port_from_stem(path.stem)
    if port is None:
        return cfg
    dev = cfg.setdefault("device", {})
    dev["adb_host"] = dev.get("adb_host", "127.0.0.1")
    dev["adb_port"] = port
    gui = cfg.setdefault("gui", {})
    if not str(gui.get("instance_name", "")).strip():
        gui["instance_name"] = f"模拟器 {port}"
    return cfg


def ensure_config_file(path: Path) -> Path:
    """实例配置不存在或为空时，从 config.example.yaml 复制并按文件名写入 device.adb_port。"""
    if path.is_file() and path.stat().st_size > 0:
        return path
    if not EXAMPLE_CONFIG_PATH.is_file():
        raise FileNotFoundError(
            f"无法创建 {path.name}：缺少模板 {EXAMPLE_CONFIG_PATH.name}"
        )

    shutil.copy2(EXAMPLE_CONFIG_PATH, path)

    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    cfg = _apply_port_from_stem(path, cfg)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, allow_unicode=True, sort_keys=False)

    return path


def add_config_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-c",
        "--config",
        dest="config",
        default=None,
        metavar="FILE",
        help=(
            "实例配置文件（默认 config_5555.yaml）。"
            "多开示例：--config config_5557.yaml"
        ),
    )


def parse_config_from_args(
    argv: list[str] | None = None,
) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    add_config_arg(parser)
    return parser.parse_known_args(argv)
