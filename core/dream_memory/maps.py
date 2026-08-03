"""寻梦记忆地图配置（物品名 → 点击坐标）。"""

from __future__ import annotations

import ast
import stat
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from loguru import logger

from core.dream_memory.config import (
    CURRENT_MAP_PERIOD,
    DEFAULT_MAP_PERIOD,
    MAPS_DIR,
    PREVIEWS_DIR,
)


def _ensure_writable(path: Path) -> None:
    """Windows 下从别处复制的 YAML 常带只读属性，写入前先清除。"""
    if not path.exists():
        return
    mode = path.stat().st_mode
    if not mode & stat.S_IWRITE:
        path.chmod(mode | stat.S_IWRITE)


def _parse_coord(value) -> list[int] | None:
    """解析 YAML 坐标或误写入 aliases 的坐标字符串。"""
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        if isinstance(value[0], (list, tuple)):
            first = _parse_coord(value[0])
            return first
        try:
            return [int(value[0]), int(value[1])]
        except (TypeError, ValueError):
            return None
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("["):
            try:
                parsed = ast.literal_eval(text)
            except (SyntaxError, ValueError):
                return None
            if isinstance(parsed, (list, tuple)) and len(parsed) >= 2:
                try:
                    return [int(parsed[0]), int(parsed[1])]
                except (TypeError, ValueError):
                    return None
    return None


def _sanitize_aliases(
    aliases: dict[str, str],
    items: dict[str, list[int]],
) -> dict[str, str]:
    """仅保留「误识别名 → 已有物品名」别名，丢弃坐标等脏数据。"""
    clean: dict[str, str] = {}
    for alias, canonical in aliases.items():
        key = str(alias).strip()
        target = str(canonical).strip()
        if not key or not target:
            continue
        if _parse_coord(canonical) is not None:
            continue
        if target not in items:
            continue
        if key in items:
            continue
        clean[key] = target
    return clean


def _load_aliases_raw(
    aliases_raw: dict,
    items: dict[str, list[int]],
) -> dict[str, str]:
    """加载 aliases；若误把坐标写进 aliases，自动恢复到 items。"""
    aliases: dict[str, str] = {}
    if not isinstance(aliases_raw, dict):
        return aliases

    for alias, canonical in aliases_raw.items():
        key = str(alias).strip()
        if not key:
            continue

        coord = _parse_coord(canonical)
        if coord is not None:
            if key not in items:
                items[key] = coord
                logger.warning(
                    f"地图 aliases 中误存坐标，已恢复到 items: {key!r} -> {coord}"
                )
            continue

        target = str(canonical).strip()
        if target:
            aliases[key] = target

    return _sanitize_aliases(aliases, items)


@dataclass
class DreamMemoryMap:
    map_id: str
    name: str
    items: dict[str, list[int]] = field(default_factory=dict)
    aliases: dict[str, str] = field(default_factory=dict)
    period: int = 4
    source_path: Path | None = None

    def resolve_label(self, label: str) -> str:
        from core.dream_memory.vision import normalize_item_name

        key = normalize_item_name(label)
        if not key:
            return ""
        if key in self.items:
            return key
        canonical = self.aliases.get(key)
        if canonical and canonical in self.items:
            return canonical
        return key

    def lookup(self, label: str) -> tuple[int, int] | None:
        from core.dream_memory.vision import normalize_item_name

        key = normalize_item_name(label)
        if not key:
            return None

        direct = self.resolve_label(key)
        if direct in self.items:
            coord = self.items[direct]
            if len(coord) >= 2:
                return int(coord[0]), int(coord[1])

        for name, coord in self.items.items():
            norm = normalize_item_name(name)
            if norm == key:
                if len(coord) >= 2:
                    return int(coord[0]), int(coord[1])
            elif len(key) > len(norm) and norm in key:
                if len(coord) >= 2:
                    return int(coord[0]), int(coord[1])
        return None

    def lookup_strict(self, label: str) -> tuple[int, int] | None:
        """仅精确/别名命中，不做前缀模糊（PK 防误点）。"""
        from core.dream_memory.vision import normalize_item_name

        key = normalize_item_name(label)
        if not key:
            return None
        direct = self.resolve_label(key)
        if direct in self.items:
            coord = self.items[direct]
            if len(coord) >= 2:
                return int(coord[0]), int(coord[1])
        return None


def list_maps(
    maps_dir: Path | None = None,
    *,
    period: int | None = None,
) -> list[DreamMemoryMap]:
    root = maps_dir or MAPS_DIR
    if not root.is_dir():
        return []
    maps: list[DreamMemoryMap] = []
    for path in sorted(root.glob("*.yaml")):
        if path.name.startswith("_"):
            continue
        try:
            dream_map = load_map(path)
        except (OSError, yaml.YAMLError, ValueError):
            continue
        if period is not None and dream_map.period != int(period):
            continue
        maps.append(dream_map)
    maps.sort(key=lambda m: (m.period, m.name, m.map_id))
    return maps


def list_map_periods(maps_dir: Path | None = None) -> list[int]:
    """已有地图出现过的期数（升序）；至少包含当前活动期。"""
    periods = {m.period for m in list_maps(maps_dir)}
    periods.add(CURRENT_MAP_PERIOD)
    periods.add(DEFAULT_MAP_PERIOD)
    return sorted(periods)


def load_map(map_id_or_path: str | Path, *, maps_dir: Path | None = None) -> DreamMemoryMap:
    root = maps_dir or MAPS_DIR
    path = Path(map_id_or_path)
    if path.is_file():
        target = path
    else:
        map_id = str(map_id_or_path).removesuffix(".yaml")
        target = root / f"{map_id}.yaml"
        if not target.is_file():
            raise FileNotFoundError(f"地图不存在: {target}")

    with target.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    map_id = str(raw.get("id") or target.stem)
    name = str(raw.get("name") or map_id)
    try:
        period = int(raw.get("period", DEFAULT_MAP_PERIOD))
    except (TypeError, ValueError):
        period = DEFAULT_MAP_PERIOD
    if period < 1:
        period = DEFAULT_MAP_PERIOD

    items_raw = raw.get("items") or {}
    items: dict[str, list[int]] = {}
    for label, coord in items_raw.items():
        parsed = _parse_coord(coord)
        if parsed:
            items[str(label)] = parsed

    aliases_raw = raw.get("aliases") or {}
    aliases = _load_aliases_raw(aliases_raw, items)

    return DreamMemoryMap(
        map_id=map_id,
        name=name,
        items=items,
        aliases=aliases,
        period=period,
        source_path=target,
    )


def format_map_choice(dream_map: DreamMemoryMap) -> str:
    """下拉框展示文案：有独立显示名时「名称 (id)」，否则仅 id。"""
    if dream_map.name and dream_map.name != dream_map.map_id:
        return f"{dream_map.name} ({dream_map.map_id})"
    return dream_map.map_id


def format_period_choice(period: int) -> str:
    return f"第 {int(period)} 期"


def parse_period_choice(label: str, *, default: int = CURRENT_MAP_PERIOD) -> int:
    text = (label or "").strip()
    if not text:
        return int(default)
    digits = "".join(ch for ch in text if ch.isdigit())
    if digits:
        try:
            value = int(digits)
            if value >= 1:
                return value
        except ValueError:
            pass
    return int(default)


def save_map(
    dream_map: DreamMemoryMap,
    *,
    maps_dir: Path | None = None,
) -> Path:
    root = maps_dir or MAPS_DIR
    root.mkdir(parents=True, exist_ok=True)
    out = root / f"{dream_map.map_id}.yaml"
    dream_map.aliases = _sanitize_aliases(dream_map.aliases, dream_map.items)
    period = max(1, int(dream_map.period or DEFAULT_MAP_PERIOD))
    dream_map.period = period
    payload = {
        "id": dream_map.map_id,
        "name": dream_map.name,
        "period": period,
        "items": dict(dream_map.items),
    }
    if dream_map.aliases:
        payload["aliases"] = dict(sorted(dream_map.aliases.items()))
    _ensure_writable(out)
    with out.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(payload, fh, allow_unicode=True, sort_keys=False, width=88)
    dream_map.source_path = out
    return out


def rename_map_name(
    map_id: str,
    new_name: str,
    *,
    maps_dir: Path | None = None,
) -> Path:
    """修改地图显示名称（YAML 内 name 字段，不改 id/文件名）。"""
    name = new_name.strip()
    if not name:
        raise ValueError("地图名称不能为空")
    dream_map = load_map(map_id, maps_dir=maps_dir)
    dream_map.name = name
    return save_map(dream_map, maps_dir=maps_dir)


def delete_map(
    map_id: str,
    *,
    maps_dir: Path | None = None,
    previews_dir: Path | None = None,
) -> None:
    """删除整张地图配置文件及预览图。"""
    dream_map = load_map(map_id, maps_dir=maps_dir)
    path = dream_map.source_path
    if path is None or not path.is_file():
        root = maps_dir or MAPS_DIR
        path = root / f"{map_id}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"地图不存在: {map_id}")
    _ensure_writable(path)
    path.unlink()
    delete_map_preview(map_id, previews_dir=previews_dir)


PREVIEW_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


def find_map_preview(map_id: str, *, previews_dir: Path | None = None) -> Path | None:
    root = previews_dir or PREVIEWS_DIR
    if not root.is_dir():
        return None
    for ext in PREVIEW_EXTENSIONS:
        path = root / f"{map_id}{ext}"
        if path.is_file():
            return path
    return None


def has_map_preview(map_id: str, *, previews_dir: Path | None = None) -> bool:
    return find_map_preview(map_id, previews_dir=previews_dir) is not None


def delete_map_preview(map_id: str, *, previews_dir: Path | None = None) -> None:
    root = previews_dir or PREVIEWS_DIR
    if not root.is_dir():
        return
    for ext in PREVIEW_EXTENSIONS:
        path = root / f"{map_id}{ext}"
        if path.is_file():
            path.unlink()


def save_map_preview(
    map_id: str,
    source: Path | str,
    *,
    previews_dir: Path | None = None,
) -> Path:
    """上传/替换地图预览图，统一保存为 PNG。"""
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("需要安装 Pillow 才能保存预览图") from exc

    root = previews_dir or PREVIEWS_DIR
    root.mkdir(parents=True, exist_ok=True)
    delete_map_preview(map_id, previews_dir=root)

    src = Path(source)
    if not src.is_file():
        raise FileNotFoundError(f"图片不存在: {src}")

    out = root / f"{map_id}.png"
    with Image.open(src) as img:
        img.convert("RGB").save(out, format="PNG")
    return out
