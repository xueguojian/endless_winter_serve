"""多任务共享的出征 / 搜索选项（GUI 与 yaml 读写）。"""

from __future__ import annotations

from typing import Any

# 搜索面板 tab：野兽(84,914) → 巨兽(244,918)，步径 160（活动期左侧多图标时按 index*步径右移）
DEFAULT_BEAST_TAB = (84, 914)
DEFAULT_ICE_BEAST_TAB = (244, 918)
DEFAULT_SEARCH_TAB_STEP = DEFAULT_ICE_BEAST_TAB[0] - DEFAULT_BEAST_TAB[0]  # 160
DEFAULT_BEAST_ICON_INDEX = 0


def _section_bool(section: dict[str, Any], key: str, default: bool) -> bool | None:
    if key not in section:
        return None
    return bool(section[key])


def resolve_beast_icon_index(tasks: dict[str, Any] | None = None, raw: Any = None) -> int:
    """野兽图标位置（活动期左侧插入的图标个数），默认 0。"""
    if raw is not None:
        try:
            return max(0, int(raw))
        except (TypeError, ValueError):
            return DEFAULT_BEAST_ICON_INDEX
    tasks = tasks or {}
    common = tasks.get("common") or {}
    if "beast_icon_index" in common:
        try:
            return max(0, int(common["beast_icon_index"]))
        except (TypeError, ValueError):
            return DEFAULT_BEAST_ICON_INDEX
    for key in ("hunt_monster", "hunt_ice_beast"):
        sec = tasks.get(key) or {}
        if "beast_icon_index" in sec:
            try:
                return max(0, int(sec["beast_icon_index"]))
            except (TypeError, ValueError):
                return DEFAULT_BEAST_ICON_INDEX
    return DEFAULT_BEAST_ICON_INDEX


def resolve_search_tab_step(coords: dict[str, Any] | None = None) -> int:
    """从野兽/巨兽基准坐标推步径；缺省用 160。"""
    coords = coords or {}
    try:
        beast_x = int((coords.get("beast_tab") or DEFAULT_BEAST_TAB)[0])
        ice_x = int((coords.get("ice_beast_tab") or DEFAULT_ICE_BEAST_TAB)[0])
        step = abs(ice_x - beast_x)
        if step > 0:
            return step
    except (TypeError, ValueError, IndexError):
        pass
    return DEFAULT_SEARCH_TAB_STEP


def shift_search_tab_xy(
    x: int,
    y: int,
    *,
    beast_icon_index: int = 0,
    step: int | None = None,
) -> tuple[int, int]:
    """活动期 tab 右移：x' = x + index * step，y 不变。"""
    index = max(0, int(beast_icon_index))
    tab_step = DEFAULT_SEARCH_TAB_STEP if step is None else max(1, int(step))
    return int(x) + index * tab_step, int(y)


def resolve_common_options(tasks: dict[str, Any]) -> dict[str, Any]:
    """从 tasks.common 与各任务段合并读取通用选项。"""
    common = tasks.get("common") or {}
    ice = tasks.get("hunt_ice_beast") or {}
    monster = tasks.get("hunt_monster") or {}
    mining = tasks.get("auto_mining") or {}
    lighthouse = tasks.get("auto_lighthouse") or {}

    def _pick_bool(
        key: str,
        *,
        default: bool,
        legacy_keys: tuple[str, ...] = (),
        sections: tuple[dict[str, Any], ...],
    ) -> bool:
        val = _section_bool(common, key, default)
        if val is not None:
            return val
        for sec in sections:
            val = _section_bool(sec, key, default)
            if val is not None:
                return val
        for legacy in legacy_keys:
            for sec in sections:
                val = _section_bool(sec, legacy, default)
                if val is not None:
                    return val
        return default

    def _pick_int(key: str, *, default: int, sections: tuple[dict[str, Any], ...]) -> int:
        if key in common:
            return int(common[key])
        for sec in sections:
            if key in sec:
                return int(sec[key])
        return default

    march_sections = (ice, monster, lighthouse)
    return {
        "use_formation": _pick_bool(
            "use_formation",
            default=True,
            legacy_keys=("check_march_heroes",),
            sections=march_sections,
        ),
        "adjust_level": _pick_bool(
            "adjust_level",
            default=False,
            sections=(common, ice, monster, mining),
        ),
        "use_stamina": _pick_bool(
            "use_stamina",
            default=False,
            sections=march_sections,
        ),
        "stamina_can_limit": _pick_int(
            "stamina_can_limit",
            default=800,
            sections=march_sections,
        ),
        "beast_icon_index": resolve_beast_icon_index(tasks),
    }


def apply_common_options(tasks: dict[str, Any], opts: dict[str, Any]) -> None:
    """写入 tasks.common，并同步到相关任务段（兼容旧配置结构）。"""
    common = tasks.setdefault("common", {})
    use_formation = bool(opts["use_formation"])
    adjust_level = bool(opts["adjust_level"])
    use_stamina = bool(opts["use_stamina"])
    stamina_can_limit = int(opts["stamina_can_limit"])
    beast_icon_index = resolve_beast_icon_index(raw=opts.get("beast_icon_index", 0))

    common["use_formation"] = use_formation
    common["adjust_level"] = adjust_level
    common["use_stamina"] = use_stamina
    common["stamina_can_limit"] = stamina_can_limit
    common["beast_icon_index"] = beast_icon_index

    for key in ("hunt_ice_beast", "hunt_monster"):
        sec = tasks.setdefault(key, {})
        sec["use_formation"] = use_formation
        sec["use_stamina"] = use_stamina
        sec["stamina_can_limit"] = stamina_can_limit
        sec["adjust_level"] = adjust_level
        sec["beast_icon_index"] = beast_icon_index
        sec.pop("check_march_heroes", None)

    mining = tasks.setdefault("auto_mining", {})
    mining["adjust_level"] = adjust_level

    lighthouse = tasks.setdefault("auto_lighthouse", {})
    lighthouse["use_formation"] = use_formation
    lighthouse["use_stamina"] = use_stamina
    lighthouse["stamina_can_limit"] = stamina_can_limit
    lighthouse.pop("check_march_heroes", None)


def resolve_use_formation(cfg: dict[str, Any]) -> bool:
    """单任务段读取「启用编队」（含旧 check_march_heroes 兼容）。"""
    if "use_formation" in cfg:
        return bool(cfg["use_formation"])
    if "check_march_heroes" in cfg:
        return bool(cfg["check_march_heroes"])
    return True
