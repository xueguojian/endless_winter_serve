"""自动打野怪任务。"""

from __future__ import annotations

import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

from loguru import logger

from core.adb_client import AdbBusyError, AdbClient, AdbUnavailableError
from core.deploy_march import (
    DEPLOY_WAIT_SETTLE_SEC,
    find_march_button,
)
from core.navigation import WildernessNavigator
from core.search_level import adjust_search_level
from core.stamina_use import (
    DEFAULT_STAMINA_CAN_LIMIT,
    MARCH_OUTCOME_DELAY_SEC,
    STAMINA_USE_XY,
    StaminaCanBudget,
    StaminaCanLimitReached,
    is_stamina_get_more_title,
    use_stamina_cans_batch,
)
from core.vision import MatchResult, Vision
from core.common_task_opts import (
    resolve_search_tab_step,
    shift_search_tab_xy,
)
from tasks.hunt_ice_beast import (
    BTN_TOWN_LABEL,
    BTN_WILDERNESS_LABEL,
    FORMATION_SLOTS,
    MARCH_CENTER,
    MarchHeroCheckError,
    NoStaminaError,
    SCENE_TOGGLE_ROI,
    SEARCH_CONFIRM_BTN,
    SEARCH_PANEL_MARKER,
    SEARCH_TAB_ROI,
    STAMINA_MATCH_THRESHOLD,
    STAMINA_USE_BTN,
    STAMINA_USE_ROW_ROI,
    STAMINA_USE_Y_MAX,
    STAMINA_USE_Y_MIN,
    TAB_BAR_SCROLL_SWIPES,
    TAB_BAR_SWIPE_DURATION_MS,
    HuntIceBeastTask,
)

StatusCallback = Callable[[str], None]

TEMPLATE_DIR = Path(__file__).parent.parent / "assets" / "templates"

BEAST_TAB_TEMPLATE = "beast_tab.png"
SEARCH_PANEL_TEMPLATES = (
    SEARCH_PANEL_MARKER,
    SEARCH_CONFIRM_BTN,
    BEAST_TAB_TEMPLATE,
)

DEFAULT_BEAST_TAB = (84, 914)
DEFAULT_SEARCH_CONFIRM = (366, 1216)
DEFAULT_TARGET_TAP = (360, 640)

DEFAULT_COORDS: dict[str, list[int]] = {
    "dialog_cancel": [250, 780],
    "search_open": [42, 872],
    "beast_tab": list(DEFAULT_BEAST_TAB),
    "search_confirm": list(DEFAULT_SEARCH_CONFIRM),
    "target_tap": list(DEFAULT_TARGET_TAP),
    "march": [560, 1200],
    "stamina_use": [576, 522],
    "level_minus": [66, 1050],
    "level_plus": [482, 1048],
}


class HuntMonsterTask:
    """野外 → 搜索野兽 → 攻击出征 → 回野外。"""

    def __init__(
        self,
        adb: AdbClient,
        coords: dict[str, list[int]],
        interval: float = 300.0,
        monster_level: int = 30,
        formation_name: str = "7",
        skip_hour: int = -1,
        step_delay: float = 1.5,
        use_stamina: bool = True,
        stamina_can_limit: int = DEFAULT_STAMINA_CAN_LIMIT,
        use_formation: bool = True,
        adjust_level: bool = False,
        beast_icon_index: int = 0,
        tab_bar_already_scrolled: bool = False,
        level_already_adjusted: bool = False,
        on_status: StatusCallback | None = None,
    ):
        self.adb = adb
        self.coords = {**DEFAULT_COORDS, **coords}
        self.interval = interval
        self.monster_level = monster_level
        self.formation_name = str(formation_name).strip()
        if self.formation_name == "0" or (
            self.formation_name.isdigit() and int(self.formation_name) == 0
        ):
            self.use_formation = False
            self.formation_name = "0"
        else:
            self.use_formation = bool(use_formation)
        self.skip_hour = skip_hour
        self.step_delay = step_delay
        self.use_stamina = use_stamina
        self.stamina_budget = StaminaCanBudget(
            enabled=use_stamina, limit=stamina_can_limit
        )
        self.adjust_level = adjust_level
        self.beast_icon_index = max(0, int(beast_icon_index))
        self._search_tab_step = resolve_search_tab_step(self.coords)
        self.on_status = on_status
        self.vision = Vision(TEMPLATE_DIR, threshold=0.72)

        self._last_run = 0.0
        self._stop_event = threading.Event()
        self._running = False
        self._tab_bar_already_scrolled = bool(tab_bar_already_scrolled)
        self._level_already_adjusted = bool(level_already_adjusted)
        self._wilderness = WildernessNavigator.from_task(
            self, is_overlay_open=self._is_search_panel_visible
        )

    @property
    def name(self) -> str:
        return "自动打野怪"

    @property
    def is_running(self) -> bool:
        return self._running

    def stop(self) -> None:
        self._stop_event.set()
        self._emit("正在停止…")

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

    def wait_reason(self) -> str:
        if self.skip_hour >= 0 and datetime.now().hour == self.skip_hour:
            return f"跳过小时 {self.skip_hour}:00–{self.skip_hour}:59，整点后继续"
        remain = self.interval - (time.time() - self._last_run)
        if remain <= 0:
            return "即将开始下一轮"
        if remain >= 60:
            return f"约 {int(remain // 60)} 分 {int(remain % 60)} 秒后再次搜索"
        return f"约 {int(remain)} 秒后再次搜索"

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
        self._tap_xy(int(x), int(y), delay)

    def _back(self, times: int = 1) -> None:
        for _ in range(times):
            if self._interrupted():
                raise InterruptedError("任务已停止")
            self.adb.back()
            time.sleep(0.5)

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

    def _is_in_wilderness(self, screen) -> bool:
        return self._match_in_roi(screen, BTN_TOWN_LABEL, SCENE_TOGGLE_ROI).found

    def _is_in_town(self, screen) -> bool:
        return self._match_in_roi(screen, BTN_WILDERNESS_LABEL, SCENE_TOGGLE_ROI).found

    def _has_scene_templates(self) -> bool:
        return (TEMPLATE_DIR / BTN_TOWN_LABEL).is_file() or (
            TEMPLATE_DIR / BTN_WILDERNESS_LABEL
        ).is_file()

    def _is_search_panel_visible(self, screen=None) -> bool:
        if screen is None:
            screen = self.adb.screenshot()
        for template in SEARCH_PANEL_TEMPLATES:
            if self.vision.match_template(screen, template).found:
                return True
        return False

    def _is_on_main_map(self, screen) -> bool:
        if not self._has_scene_templates():
            return not self._is_search_panel_visible(screen)
        return self._is_in_wilderness(screen) or self._is_in_town(screen)

    def _ensure_clean_ui(self) -> None:
        self._emit("清理界面…")
        if self._wilderness.return_to_wilderness():
            self._emit("界面已就绪")
            time.sleep(0.5)
        else:
            self._emit("已尝试清理界面，继续执行")

    def _ensure_wilderness(self) -> None:
        self._wilderness.ensure_wilderness()

    def _open_search_panel(self) -> None:
        """在野外点击左下角放大镜，打开搜索面板（固定坐标，不截图匹配）。"""
        self._ensure_wilderness()
        sx, sy = self.coords["search_open"]
        self._emit(f"打开搜索面板 @ ({int(sx)},{int(sy)})")
        self._tap("search_open", delay=1.5)

    def _scroll_search_tab_bar_to_rightmost(self) -> None:
        x1, y1, x2, y2 = SEARCH_TAB_ROI
        cy = (y1 + y2) // 2
        swipe_start_x = x1 + 60
        swipe_end_x = x2 - 60
        self._emit("滚动搜索 tab 栏到最右端")
        for _ in range(TAB_BAR_SCROLL_SWIPES):
            self._swipe_xy(
                swipe_start_x,
                cy,
                swipe_end_x,
                cy,
                duration_ms=TAB_BAR_SWIPE_DURATION_MS,
            )
            time.sleep(0.25)

    def _select_beast_tab(self) -> None:
        """滚到 tab 栏最右端后，点击「野兽」tab（按野兽图标位置右移）。"""
        if self._tab_bar_already_scrolled:
            self._emit("上次已是野兽任务，跳过 tab 栏拖动")
        else:
            self._scroll_search_tab_bar_to_rightmost()
            self._tab_bar_already_scrolled = True

        bx, by = self.coords["beast_tab"]
        tx, ty = shift_search_tab_xy(
            int(bx),
            int(by),
            beast_icon_index=self.beast_icon_index,
            step=self._search_tab_step,
        )
        if self.beast_icon_index:
            self._emit(
                f"选中野兽 tab @ ({tx},{ty})"
                f"（基准 {int(bx)},{int(by)} + 位置{self.beast_icon_index}"
                f"×步径{self._search_tab_step}）"
            )
        else:
            self._emit(f"选中野兽 tab @ ({tx},{ty})")
        self._tap_xy(tx, ty, delay=1.0)

    def _tap_search_confirm(self) -> None:
        tx, ty = self.coords["search_confirm"]
        self._emit(f"点击搜索 @ ({tx},{ty})")
        self._tap_xy(int(tx), int(ty), delay=2.5)
        self._emit("正在搜索野兽…")

    def _parse_formation_slot(self, name: str) -> int | None:
        if name.isdigit():
            slot = int(name)
            if 1 <= slot <= 8:
                return slot
        return None

    def _require_formation_slot(self) -> int:
        slot = self._parse_formation_slot(self.formation_name)
        if slot is None:
            raise RuntimeError(
                f"编队槽位「{self.formation_name}」无效，请在 GUI 中填写 1~8"
            )
        return slot

    def _tap_formation_slot(self) -> None:
        if not self.use_formation:
            self._emit("未启用编队，跳过编队选择")
            return
        if not self.formation_name:
            self._emit("未配置编队槽位，沿用当前编队")
            return
        slot = self._require_formation_slot()
        sx, sy = FORMATION_SLOTS[slot]
        self._emit(f"点击编队槽位 {slot} @ ({sx},{sy})")
        self._tap_xy(sx, sy, delay=1.0)
        self._emit(f"已选择编队槽位 {slot}")

    def _analyze_deploy_and_resolve_march(self, screen) -> tuple[int, int, float]:
        if self.use_formation:
            self._emit("检查出征英雄…")
            empty_slots = HuntIceBeastTask._find_empty_march_hero_slots(screen)
            if empty_slots:
                slots_text = "、".join(str(i) for i in empty_slots)
                raise MarchHeroCheckError(f"第 {slots_text} 个英雄位为空，跳过本轮")
            self._emit("出征英雄已配满（3/3）")

        result = find_march_button(screen)
        if not result.found:
            raise RuntimeError(
                "未检测到出征界面（右下角「出征」按钮）。"
                f"最高匹配 {result.confidence:.2f}。请确认上一步已定位目标。"
            )
        cx, cy = result.center
        self._emit(f"出征按钮就绪（{result.confidence:.2f} @ ({cx},{cy})）")
        return cx, cy, result.confidence

    def _march_tap_xy(self) -> tuple[int, int]:
        if "march" in self.coords:
            return tuple(self.coords["march"])
        return MARCH_CENTER

    def _click_march_at(self, cx: int, cy: int, *, match_conf: float | None = None) -> None:
        if match_conf is not None:
            self._emit(f"点击出征 @ ({cx},{cy})（匹配 {match_conf:.2f}）")
        else:
            self._emit(f"点击出征 @ ({cx},{cy})（固定坐标）")
        self._tap_xy(cx, cy, delay=0.6)

    def _prepare_and_tap_march(self):
        """定位目标后：延迟 → 点编队(可选) → 截一张验英雄+出征按钮 → 点出征并判结果。"""
        time.sleep(DEPLOY_WAIT_SETTLE_SEC)
        if self._interrupted():
            raise InterruptedError("任务已停止")

        self._tap_formation_slot()

        screen = self.adb.screenshot()
        cx, cy, match_conf = self._analyze_deploy_and_resolve_march(screen)
        self._click_march_at(cx, cy, match_conf=match_conf)
        time.sleep(0.8)

        is_popup, outcome = self._wait_march_outcome()
        if not is_popup:
            self._emit("队伍已出征")
            return outcome

        self._emit("检测到体力不足弹窗，出征未成功")
        if not self.use_stamina:
            raise NoStaminaError("体力不足，任务结束")
        self._emit("自动使用领主体力道具…")
        self._use_stamina_items()
        mx, my = self._march_tap_xy()
        self._click_march_at(mx, my)
        time.sleep(0.8)
        is_popup, outcome = self._wait_march_outcome()
        if is_popup:
            self._emit("使用体力后仍出现体力不足弹窗，已停止")
            raise NoStaminaError("没有体力，已停止")
        self._emit("队伍已出征")
        return outcome

    def _find_stamina_use_button(self, screen) -> MatchResult:
        if not (TEMPLATE_DIR / STAMINA_USE_BTN).exists():
            return MatchResult(found=False)
        x1, y1, x2, y2 = STAMINA_USE_ROW_ROI
        crop = screen[y1:y2, x1:x2]
        stamina_vision = Vision(TEMPLATE_DIR, threshold=STAMINA_MATCH_THRESHOLD)
        result = stamina_vision.match_template_multiscale(crop, STAMINA_USE_BTN)
        if not result.found:
            result = stamina_vision.match_template(crop, STAMINA_USE_BTN)
        if result.found:
            cx, cy = result.center
            global_center = (x1 + cx, y1 + cy)
            if not (STAMINA_USE_Y_MIN <= global_center[1] <= STAMINA_USE_Y_MAX):
                return MatchResult(found=False, confidence=result.confidence)
            return MatchResult(
                found=True,
                confidence=result.confidence,
                center=global_center,
                top_left=(x1 + result.top_left[0], y1 + result.top_left[1]),
                size=result.size,
            )
        return MatchResult(found=False, confidence=result.confidence)

    def _is_stamina_popup(self, screen) -> bool:
        return is_stamina_get_more_title(screen)

    def _wait_march_outcome(self, delay: float = MARCH_OUTCOME_DELAY_SEC):
        """出征后延迟，再 OCR 一次。返回 (是否弹窗, 截图)。"""
        self._emit("正在检测出征结果…")
        time.sleep(delay)
        if self._interrupted():
            raise InterruptedError("任务已停止")
        screen = self.adb.screenshot()
        return self._is_stamina_popup(screen), screen

    def _use_stamina_items(self) -> None:
        use_stamina_cans_batch(
            self.adb,
            self.stamina_budget,
            tap_xy=STAMINA_USE_XY,
            emit=self._emit,
            interrupted=self._interrupted,
            close_with_back=True,
        )

    def _finish_after_march(self, screen=None) -> None:
        """优先复用出征结果截图判断主界面，避免再截一张。"""
        if screen is None:
            time.sleep(1.0)
            screen = self.adb.screenshot()
        if self._is_on_main_map(screen) and not self._is_search_panel_visible(screen):
            if self._is_in_wilderness(screen):
                self._emit("出征完成，已在野外主界面")
            elif self._is_in_town(screen):
                self._emit("出征完成，已在城镇主界面")
            else:
                self._emit("出征完成，已在主界面")
            return
        self._emit("清理出征后的残留界面…")
        self._wilderness.return_to_wilderness()

    def run_hunt_cycle(self) -> None:
        self._ensure_clean_ui()
        self._emit(f"开始搜索 {self.monster_level} 级野兽")

        self._open_search_panel()
        self._select_beast_tab()
        self._set_monster_level()

        self._tap_search_confirm()

        self._tap("target_tap", delay=2.0)
        self._emit("已定位目标")

        march_screen = self._prepare_and_tap_march()
        self._finish_after_march(march_screen)

    def _set_monster_level(self) -> None:
        if not self.adjust_level:
            self._emit("未启用修改等级，跳过")
            return
        if self._level_already_adjusted:
            self._emit("本轮调度已调整过等级，跳过")
            return
        adjust_search_level(
            self.adb,
            self.monster_level,
            emit=self._emit,
            interrupted=self._interrupted,
        )
        self._level_already_adjusted = True

    def run_once(self, *, force: bool = False) -> bool:
        if not force and not self.should_run():
            return False

        self._last_run = time.time()
        try:
            self.run_hunt_cycle()
            self._emit(f"本轮完成，{int(self.interval // 60)} 分钟后再次攻击")
            return True
        except InterruptedError:
            self._emit("任务已停止")
            raise
        except NoStaminaError:
            if self.use_stamina:
                self._emit("体力不足但已启用自动使用，跳过本轮继续循环")
                self._wilderness.try_return_to_wilderness()
                return False
            self._stop_event.set()
            raise InterruptedError("没有体力，已停止")
        except StaminaCanLimitReached as exc:
            self._emit(str(exc))
            self._stop_event.set()
            raise InterruptedError(str(exc)) from exc
        except MarchHeroCheckError as exc:
            self._emit(str(exc))
            self._emit("退回野外，等待下次循环")
            self._wilderness.try_return_to_wilderness()
            return False
        except AdbBusyError as exc:
            logger.warning(f"[{self.name}] {exc}")
            self._emit(f"ADB 正忙，跳过本轮（{exc}）")
            return False
        except AdbUnavailableError as exc:
            logger.exception(f"[{self.name}] ADB 不可用，延长等待后继续循环")
            self._emit(f"本轮异常：{exc}，将等待后重试")
            # 双开过久常出现 0xc0000142，短暂退避给 adb.exe 喘息
            time.sleep(30)
            return False
        except Exception as exc:
            logger.exception(f"[{self.name}] 执行失败，恢复界面后继续循环")
            self._emit(f"本轮异常：{exc}，恢复界面后继续")
            self._wilderness.try_return_to_wilderness()
            return False

    def run_loop(self) -> None:
        self._running = True
        self.reset_stop()
        self._emit("=== 自动打野怪已启动 ===")
        formation_desc = (
            f"编队槽位 {self.formation_name}"
            if self.use_formation
            else "不启用编队，直接出征"
        )
        self._emit(
            f"间隔 {int(self.interval // 60)} 分钟，{formation_desc}"
        )
        try:
            while not self._interrupted():
                try:
                    self.run_once()
                except InterruptedError:
                    break
                for _ in range(30):
                    if self._interrupted():
                        break
                    time.sleep(1)
        finally:
            self._running = False
            self._emit("自动打野怪已停止")
