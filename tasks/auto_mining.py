"""自动采集资源：生肉 → 木头 → 煤矿 → 铁矿。"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

from loguru import logger

from core.adb_client import AdbClient
from core.capture_roi import (
    MARCH_BTN_CAPTURE_ROI,
    SEARCH_CONFIRM_CAPTURE_ROI,
    SEARCH_LEVEL_CAPTURE_ROI,
    SEARCH_PANEL_CAPTURE_ROI,
    capture_screen,
)
from core.deploy_march import find_march_button
from core.navigation import WildernessNavigator
from core.search_level import (
    adjust_search_level,
    ensure_full_resource_checked,
    read_search_level,
)
from core.vision import MatchResult, Vision

StatusCallback = Callable[[str], None]

TEMPLATE_DIR = Path(__file__).parent.parent / "assets" / "templates"

SEARCH_ICON = "search_icon.png"
SEARCH_CONFIRM_BTN = "search_confirm_btn.png"
SEARCH_PANEL_MARKER = "beast_tab.png"
ICE_BEAST_TAB_TEMPLATE = "ice_beast_tab.png"
ICE_BEAST_TAB_SELECTED = "ice_beast_tab_selected.png"

SEARCH_PANEL_TEMPLATES = (
    SEARCH_PANEL_MARKER,
    SEARCH_CONFIRM_BTN,
    ICE_BEAST_TAB_TEMPLATE,
    ICE_BEAST_TAB_SELECTED,
)

SEARCH_ICON_ROI = (0, 780, 160, 960)
SEARCH_TAB_ROI = (0, 850, 720, 980)
SEARCH_BTN_ROI = (100, 1130, 530, 1240)

TAB_BAR_FIRST_SCROLL_SWIPES = 3
TAB_BAR_LATER_SCROLL_SWIPES = 1
TAB_BAR_SWIPE_DURATION_MS = 400

MINING_HERO_MATCH_THRESHOLD = 0.68
DEFAULT_HERO_ROI = (98, 308, 246, 570)

DEFAULT_COORDS: dict[str, list[int]] = {
    "world_map": [630, 1220],
    "search_open": [55, 880],
    "meat_tab": [154, 914],
    "wood_tab": [320, 914],
    "coal_tab": [474, 914],
    "iron_mine_tab": [622, 912],
    "search_confirm": [360, 1175],
    "gather_btn": [354, 628],
    "hero_remove_2": [424, 318],
    "hero_remove_3": [618, 318],
    "level_minus": [66, 1050],
    "level_plus": [482, 1048],
    "dialog_cancel": [250, 780],
    "full_resource_check": [214, 1136],
    "march": [560, 1200],
}

DEFAULT_STEP_DELAY = 1.5


@dataclass(frozen=True)
class MiningResource:
    resource_id: str
    label: str
    tab_key: str
    hero_template: str


# 采集顺序：生肉 → 木头 → 煤矿 → 铁矿
MINING_RESOURCES: tuple[MiningResource, ...] = (
    MiningResource("meat", "生肉", "meat_tab", "mining/mining_hero_meat_face.png"),
    MiningResource("wood", "木头", "wood_tab", "mining/mining_hero_wood_face.png"),
    MiningResource("coal", "煤矿", "coal_tab", "mining/mining_hero_coal_face.png"),
    MiningResource("iron", "铁矿", "iron_mine_tab", "mining/mining_hero_iron_face.png"),
)


class MiningHeroMismatchError(Exception):
    """首位英雄不是对应资源的采矿英雄。"""


class MiningNotSupportedError(Exception):
    """当前配置不支持非采矿英雄模式。"""


def merge_task_config(cfg: dict) -> dict:
    coords = {**cfg.get("coords", {}), **DEFAULT_COORDS}
    raw_roi = cfg.get("hero_roi", list(DEFAULT_HERO_ROI))
    if len(raw_roi) == 4:
        hero_roi = tuple(int(v) for v in raw_roi)
    else:
        hero_roi = DEFAULT_HERO_ROI
    return {
        "step_delay": cfg.get("step_delay", DEFAULT_STEP_DELAY),
        "coords": coords,
        "hero_roi": hero_roi,
        "hero_match_threshold": float(
            cfg.get("hero_match_threshold", MINING_HERO_MATCH_THRESHOLD)
        ),
    }


class AutoMiningTask:
    """按顺序采集生肉、木头、煤矿、铁矿。"""

    def __init__(
        self,
        adb: AdbClient,
        coords: dict[str, list[int]] | None = None,
        interval: float = 3600.0,
        level_min: int = 8,
        level_max: int = 8,
        use_mining_hero: bool = True,
        skip_hour: int = -1,
        step_delay: float = DEFAULT_STEP_DELAY,
        hero_match_threshold: float = MINING_HERO_MATCH_THRESHOLD,
        adjust_level: bool = False,
        on_status: StatusCallback | None = None,
    ):
        merged = merge_task_config(
            {
                "coords": coords or {},
                "step_delay": step_delay,
                "hero_match_threshold": hero_match_threshold,
            }
        )
        self.adb = adb
        self.coords = merged["coords"]
        self.interval = interval
        self.level_min = level_min
        self.level_max = level_max
        self.mine_level = level_max
        self.use_mining_hero = use_mining_hero
        self.skip_hour = skip_hour
        self.step_delay = merged["step_delay"]
        self.hero_roi = merged["hero_roi"]
        self.hero_match_threshold = merged["hero_match_threshold"]
        self.adjust_level = adjust_level
        self.on_status = on_status
        self._last_run = 0.0
        self._stop_event = threading.Event()
        self.vision = Vision(TEMPLATE_DIR, threshold=0.70)
        self._wilderness = WildernessNavigator.from_task(
            self, is_overlay_open=self._is_search_panel_visible
        )

    @property
    def name(self) -> str:
        return "自动采集"

    def stop(self) -> None:
        self._stop_event.set()

    def reset_stop(self) -> None:
        self._stop_event.clear()

    def _emit(self, message: str) -> None:
        logger.info(f"[{self.name}] {message}")
        if self.on_status:
            self.on_status(message)

    def _interrupted(self) -> bool:
        return self._stop_event.is_set()

    def should_run(self) -> bool:
        if self.skip_hour >= 0 and datetime.now().hour == self.skip_hour:
            return False
        return time.time() - self._last_run >= self.interval

    def _tap_xy(self, x: int, y: int, delay: float | None = None) -> None:
        if self._interrupted():
            raise InterruptedError("任务已停止")
        self.adb.tap(x, y)
        time.sleep(delay if delay is not None else self.step_delay)

    def _swipe_xy(
        self, x1: int, y1: int, x2: int, y2: int, duration_ms: int = 300
    ) -> None:
        if self._interrupted():
            raise InterruptedError("任务已停止")
        self.adb.swipe(x1, y1, x2, y2, duration_ms)

    def _tap(self, key: str, delay: float | None = None) -> None:
        x, y = self.coords[key]
        logger.debug(f"[{self.name}] 点击 {key} ({x}, {y})")
        self._tap_xy(x, y, delay)

    def _back(self, times: int = 1) -> None:
        for _ in range(times):
            if self._interrupted():
                raise InterruptedError("任务已停止")
            self.adb.back()
            time.sleep(0.5)

    def _dismiss_dialog(self) -> None:
        if "dialog_cancel" in self.coords:
            self.adb.tap(*self.coords["dialog_cancel"])
            time.sleep(0.5)

    def _return_to_wilderness(self) -> None:
        self._wilderness.try_return_to_wilderness()

    def _match_in_roi(
        self, screen, template: str, roi: tuple[int, int, int, int]
    ) -> MatchResult:
        x1, y1, x2, y2 = roi
        result = self.vision.match_template(screen[y1:y2, x1:x2], template)
        if not result.found:
            return result
        cx, cy = result.center
        return MatchResult(
            found=True,
            confidence=result.confidence,
            center=(x1 + cx, y1 + cy),
            top_left=(x1 + result.top_left[0], y1 + result.top_left[1]),
            size=result.size,
        )

    def _is_search_panel_visible(self, screen=None) -> bool:
        """以搜索确认按钮为准，避免野外地图误匹配 tab 模板而跳过点放大镜。"""
        if screen is None:
            screen = capture_screen(self.adb, SEARCH_PANEL_CAPTURE_ROI)
        if self._match_in_roi(screen, SEARCH_CONFIRM_BTN, SEARCH_BTN_ROI).found:
            return True
        # 兼容：确认按钮模板失效时，再看 tab 区标记
        for template in (SEARCH_PANEL_MARKER, ICE_BEAST_TAB_TEMPLATE, ICE_BEAST_TAB_SELECTED):
            if self._match_in_roi(screen, template, SEARCH_TAB_ROI).found:
                return True
        return False

    def _ensure_clean_ui(self) -> None:
        self._emit("清理界面…")
        if self._wilderness.return_to_wilderness():
            self._emit("界面已就绪")
        else:
            self._emit("已尝试清理界面，继续执行")

    def _ensure_wilderness(self) -> None:
        self._wilderness.ensure_wilderness()

    def _open_search_panel(self) -> None:
        """与巨兽一致：野外固定点放大镜打开搜索，避免误判已打开而直接拖地图。"""
        self._ensure_wilderness()
        screen = capture_screen(self.adb, SEARCH_PANEL_CAPTURE_ROI)
        if self._is_search_panel_visible(screen):
            self._emit("搜索面板已打开")
            return
        sx, sy = self.coords["search_open"]
        self._emit(f"打开搜索面板 @ ({int(sx)},{int(sy)})")
        self._tap("search_open", delay=1.5)
        if not self._is_search_panel_visible():
            # 模板偶发失败时再点一次坐标
            self._emit("未识别到搜索面板，再点一次放大镜")
            self._tap("search_open", delay=1.5)
        if not self._is_search_panel_visible():
            raise RuntimeError("搜索面板未打开")
        self._emit("已打开搜索面板")

    def _scroll_search_tab_bar(self, swipes: int) -> None:
        """采集：向左滑 tab 栏，露出右侧资源图标。"""
        if swipes <= 0:
            return
        x1, y1, x2, y2 = SEARCH_TAB_ROI
        cy = (y1 + y2) // 2
        swipe_start_x = x2 - 60
        swipe_end_x = x1 + 60
        self._emit(f"滚动搜索 tab 栏（{swipes} 次）")
        for _ in range(swipes):
            self._swipe_xy(
                swipe_start_x,
                cy,
                swipe_end_x,
                cy,
                duration_ms=TAB_BAR_SWIPE_DURATION_MS,
            )
            time.sleep(0.25)

    def _select_resource_tab(self, resource: MiningResource) -> None:
        tx, ty = self.coords[resource.tab_key]
        self._emit(f"选择{resource.label} ({tx}, {ty})")
        self._tap_xy(tx, ty, delay=1.0)

    def _set_mine_level(self) -> None:
        """每种资源 tab 各校验/调整一次等级；未勾选配置则跳过。

        当前等级在 [level_min, level_max] 内不改；低于最低则调到最低；
        高于最高则调到最高。允许 min == max。
        """
        if not self.adjust_level:
            self._emit("未启用修改等级，跳过")
            return
        current = read_search_level(
            capture_screen(self.adb, SEARCH_LEVEL_CAPTURE_ROI)
        )
        if current is None:
            raise RuntimeError("无法识别当前搜索等级，已中止以免按错误等级搜索")
        if self.level_min <= current <= self.level_max:
            self._emit(
                f"当前等级 {current} 已在 {self.level_min}~{self.level_max} 内，跳过"
            )
            return
        target = self.level_min if current < self.level_min else self.level_max
        adjust_search_level(
            self.adb,
            target,
            emit=self._emit,
            interrupted=self._interrupted,
        )

    def _ensure_full_resource_option(self) -> None:
        ensure_full_resource_checked(
            self.adb,
            emit=self._emit,
            interrupted=self._interrupted,
        )

    def _tap_search_confirm(self) -> None:
        screen = capture_screen(self.adb, SEARCH_CONFIRM_CAPTURE_ROI)
        result = self._match_in_roi(screen, SEARCH_CONFIRM_BTN, SEARCH_BTN_ROI)
        if result.found:
            self._tap_xy(*result.center, delay=2.5)
        else:
            self._tap("search_confirm", delay=2.5)

    def _tap_gather_button(self) -> None:
        self._emit("点击采集")
        self._tap("gather_btn", delay=2.0)

    def _verify_mining_hero(self, resource: MiningResource) -> None:
        time.sleep(0.5)
        x1, y1, x2, y2 = self.hero_roi
        screen = capture_screen(self.adb, (x1, y1, x2, y2))
        roi = screen[y1:y2, x1:x2]
        hero_vision = Vision(TEMPLATE_DIR, threshold=self.hero_match_threshold)
        result = hero_vision.match_template_multiscale(roi, resource.hero_template)
        self._emit(
            f"{resource.label}英雄识别 ({x1},{y1})-({x2},{y2})，"
            f"匹配度 {result.confidence:.2f}"
        )
        if not result.found:
            raise MiningHeroMismatchError(
                f"{resource.label}首位英雄不匹配（{result.confidence:.2f}）"
            )
        self._emit(f"{resource.label}采矿英雄匹配成功")

    def _configure_mining_heroes(self, resource: MiningResource) -> None:
        self._verify_mining_hero(resource)
        self._emit("移除第 2、3 位英雄")
        self._tap("hero_remove_2", delay=1.2)
        self._tap("hero_remove_3", delay=1.2)
        # 卸英雄后界面刷新较慢，等稳再出征
        time.sleep(1.5)

    def _resolve_march_tap(self) -> tuple[int, int, float, bool]:
        fallback = tuple(self.coords.get("march", [560, 1200]))
        result = find_march_button(capture_screen(self.adb, MARCH_BTN_CAPTURE_ROI))
        if result.found:
            return result.center[0], result.center[1], result.confidence, True
        return fallback[0], fallback[1], result.confidence, False

    def _tap_march(self) -> None:
        """点击出征；优先固定坐标，点完确认已离开出征界面。"""
        last_conf = 0.0
        for attempt in range(3):
            if self._interrupted():
                raise InterruptedError("任务已停止")
            # 坐标本身无问题：优先用配置坐标，匹配仅作日志参考
            cx, cy = tuple(self.coords.get("march", [560, 1200]))
            matched_cx, matched_cy, conf, matched = self._resolve_march_tap()
            last_conf = conf
            if matched:
                self._emit(
                    f"出征 @ ({cx},{cy})（配置坐标；按钮匹配 {conf:.2f} @ "
                    f"({matched_cx},{matched_cy})）"
                )
            else:
                self._emit(f"出征 @ ({cx},{cy})（配置坐标）")
            self._tap_xy(cx, cy, delay=2.0)

            time.sleep(1.2)
            still = find_march_button(capture_screen(self.adb, MARCH_BTN_CAPTURE_ROI))
            if not still.found:
                return
            self._emit(
                f"出征未生效（仍在出征界面，匹配 {still.confidence:.2f}），"
                f"重试 {attempt + 1}/3"
            )
            time.sleep(1.0)
        raise RuntimeError(
            f"点击出征后仍停留在出征界面（最高匹配 {last_conf:.2f}），请检查出征按钮"
        )

    def _run_single_resource(self, resource: MiningResource, *, is_first: bool) -> bool:
        if is_first:
            self._open_search_panel()
            self._scroll_search_tab_bar(TAB_BAR_FIRST_SCROLL_SWIPES)
        else:
            self._open_search_panel()
            self._scroll_search_tab_bar(TAB_BAR_LATER_SCROLL_SWIPES)

        self._emit(f"开始采集{resource.label}（{self.level_min}~{self.level_max} 级）")
        self._select_resource_tab(resource)
        self._ensure_full_resource_option()
        self._set_mine_level()
        self._tap_search_confirm()
        self._tap_gather_button()
        try:
            self._configure_mining_heroes(resource)
        except MiningHeroMismatchError as exc:
            self._emit(str(exc))
            self._emit(f"跳过{resource.label}，继续下一项")
            self._back(1)
            time.sleep(0.5)
            return False

        self._tap_march()
        self._emit(f"{resource.label}采集部队已出征")
        return True

    def run_mining_cycle(self) -> tuple[list[str], list[str]]:
        if not self.use_mining_hero:
            raise MiningNotSupportedError("未启用采矿英雄，当前仅支持采矿英雄模式")

        self._ensure_clean_ui()
        succeeded: list[str] = []
        skipped: list[str] = []
        for index, resource in enumerate(MINING_RESOURCES):
            if self._interrupted():
                raise InterruptedError("任务已停止")
            if self._run_single_resource(resource, is_first=(index == 0)):
                succeeded.append(resource.label)
            else:
                skipped.append(resource.label)
        self._return_to_wilderness()
        return succeeded, skipped

    def run_once(self, *, force: bool = False) -> bool:
        if not force and not self.should_run():
            return False

        self._last_run = time.time()
        try:
            succeeded, skipped = self.run_mining_cycle()
            if skipped:
                done = "、".join(succeeded) if succeeded else "无"
                miss = "、".join(skipped)
                self._emit(
                    f"本轮完成：已采集 {done}；跳过 {miss}，"
                    f"{int(self.interval // 60)} 分钟后再次采集"
                )
            else:
                labels = "、".join(succeeded)
                self._emit(f"本轮完成（{labels}），{int(self.interval // 60)} 分钟后再次采集")
            return True
        except InterruptedError:
            self._emit("任务已停止")
            raise
        except MiningNotSupportedError as exc:
            self._emit(str(exc))
            self._return_to_wilderness()
            return False
        except Exception as exc:
            logger.exception(f"[{self.name}] 执行失败")
            self._emit(f"执行失败：{exc}")
            self._return_to_wilderness()
            return False
