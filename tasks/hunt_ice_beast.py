"""自动打冰原巨兽（集结）任务。"""

from __future__ import annotations

import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

import cv2
from loguru import logger

from core.adb_client import AdbBusyError, AdbClient, AdbUnavailableError
from core.capture_roi import (
    BOTTOM_UI_CAPTURE_ROI,
    DEPLOY_CAPTURE_ROI,
    SEARCH_PANEL_CAPTURE_ROI,
    STAMINA_POPUP_CAPTURE_ROI,
    capture_screen,
)
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
    DEFAULT_ICE_BEAST_TAB,
    resolve_search_tab_step,
    shift_search_tab_xy,
)

StatusCallback = Callable[[str], None]

TEMPLATE_DIR = Path(__file__).parent.parent / "assets" / "templates"

# 右下角显示「城镇」→ 实际在野外；显示「野外」→ 实际在城镇内
BTN_TOWN_LABEL = "btn_town_label.png"
BTN_WILDERNESS_LABEL = "btn_wilderness_label.png"
SEARCH_PANEL_MARKER = "beast_tab.png"
SEARCH_CONFIRM_BTN = "search_confirm_btn.png"
RALLY_BTN = "rally_btn.png"
RALLY_CONFIRM_BTN = "rally_confirm_btn.png"
ICE_BEAST_TAB_TEMPLATE = "ice_beast_tab.png"
ICE_BEAST_TAB_SELECTED = "ice_beast_tab_selected.png"
SEARCH_PANEL_TEMPLATES = (
    SEARCH_PANEL_MARKER,
    SEARCH_CONFIRM_BTN,
    ICE_BEAST_TAB_TEMPLATE,
    ICE_BEAST_TAB_SELECTED,
)
UI_CLEANUP_ATTEMPTS = 6
MARCH_BTN = "march_btn.png"  # 仅 deploy_march.find_march_button 使用其上半「出征」区域
STAMINA_USE_BTN = "stamina_use_btn.png"
STAMINA_GET_MORE_TITLE = "stamina_get_more_title.png"

# 「获取更多」弹窗 (720×1280)；标题 ROI 见 core.stamina_use.STAMINA_TITLE_ROI
STAMINA_USE_ROW_ROI = (500, 520, 710, 660)
STAMINA_USE_CENTER = (576, 522)
STAMINA_USE_Y_MIN = 520
STAMINA_USE_Y_MAX = 660
STAMINA_MATCH_THRESHOLD = 0.58
STAMINA_TITLE_THRESHOLD = 0.60

# 出征界面顶部编队栏 Y 范围（标定脚本用）
FORMATION_BAR_Y1 = 95
FORMATION_BAR_Y2 = 220
MARCH_CENTER = (560, 1200)

# 720×1280 竖屏下 1~8 号编队槽位中心（实机标定）
FORMATION_SLOTS: dict[int, tuple[int, int]] = {
    1: (44, 129),
    2: (121, 129),
    3: (198, 129),
    4: (275, 129),
    5: (352, 129),
    6: (429, 129),
    7: (506, 129),
    8: (583, 129),
}

# 右下角场景切换按钮区域 (720×1280)
SCENE_TOGGLE_ROI = (500, 1150, 720, 1280)
# 搜索面板内目标 tab 图标行
SEARCH_TAB_ROI = (0, 850, 720, 980)
# tab 栏横向滚动：先滑到最右端，再从左起选第 2 个（冰原巨兽）
TAB_BAR_SCROLL_SWIPES = 3
TAB_BAR_SWIPE_DURATION_MS = 400
ICE_BEAST_TAB_SLOT = 2
TAB_FIRST_CENTER_X = 90
TAB_ICON_SPACING = 115
# 等级数字白框（仅裁白框内数字，勿含圆角边框）
RALLY_MATCH_THRESHOLD = 0.68
RALLY_RETRY_COUNT = 6
RALLY_RETRY_INTERVAL = 1.0
RALLY_CONFIRM_THRESHOLD = 0.72
RALLY_CONFIRM_MIN_Y = 780
RALLY_CONFIRM_MAX_DX = 90
# 发起集结弹窗底部确认按钮固定中心（720×1280）
RALLY_CONFIRM_CENTER = (360, 882)
# 发起集结弹窗内时长选项（720×1280 竖屏固定坐标）
RALLY_TIME_COORDS: dict[int, tuple[int, int]] = {
    5: (220, 462),
    15: (500, 462),
    30: (220, 518),
    60: (500, 518),
}
RALLY_DURATION_OPTIONS = tuple(RALLY_TIME_COORDS.keys())
DEFAULT_RALLY_DURATION = 5

# 出征界面英雄栏（720×1280）：三个头像框区域
MARCH_HERO_ROI = (90, 304, 634, 580)
MARCH_HERO_SLOT_COUNT = 3
MARCH_HERO_SLOT_INSET = 0.08
EMPTY_HERO_SLOT_TEMPLATE = "march_empty_hero_slot.png"
# 空槽中心区域蓝色占比高于此值视为未上阵英雄
EMPTY_SLOT_BLUE_RATIO = 0.40
EMPTY_SLOT_TM_THRESHOLD = 0.55


class NoStaminaError(RuntimeError):
    """体力不足且未启用自动使用体力。"""


class MarchHeroCheckError(RuntimeError):
    """出征英雄栏存在空槽位。"""


class HuntIceBeastTask:
    """搜索冰原巨兽并发起联盟集结。"""

    def __init__(
        self,
        adb: AdbClient,
        coords: dict[str, list[int]],
        interval: float = 900.0,
        beast_level: int = 8,
        default_beast_level: int = 1,
        formation_name: str = "7",
        rally_duration_minutes: int = DEFAULT_RALLY_DURATION,
        skip_hour: int = -1,
        step_delay: float = 1.5,
        use_stamina: bool = True,
        stamina_can_limit: int = DEFAULT_STAMINA_CAN_LIMIT,
        use_formation: bool = True,
        adjust_level: bool = False,
        beast_icon_index: int = 0,
        on_status: StatusCallback | None = None,
    ):
        self.adb = adb
        self.coords = coords
        self.interval = interval
        self.beast_level = beast_level
        self.default_beast_level = default_beast_level
        self.formation_name = str(formation_name).strip()
        self.use_formation = use_formation
        self.adjust_level = adjust_level
        self.beast_icon_index = max(0, int(beast_icon_index))
        self._search_tab_step = resolve_search_tab_step(coords)
        if rally_duration_minutes not in RALLY_TIME_COORDS:
            raise ValueError(
                f"集结时长仅支持 {list(RALLY_DURATION_OPTIONS)} 分钟，"
                f"收到 {rally_duration_minutes}"
            )
        self.rally_duration_minutes = rally_duration_minutes
        self.skip_hour = skip_hour
        self.step_delay = step_delay
        self.use_stamina = use_stamina
        self.stamina_budget = StaminaCanBudget(
            enabled=use_stamina, limit=stamina_can_limit
        )
        self.on_status = on_status
        self.vision = Vision(TEMPLATE_DIR, threshold=0.72)

        self._last_run = 0.0
        self._stop_event = threading.Event()
        self._running = False
        # 连续执行冰原巨兽时，搜索 tab 栏仍停留在上次滚动的位置
        self._tab_bar_already_scrolled = False
        self._level_already_adjusted = False
        self._wilderness = WildernessNavigator.from_task(
            self, is_overlay_open=self._is_search_panel_visible
        )

    @property
    def name(self) -> str:
        return "冰原巨兽集结"

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
        """循环等待时的可读原因（给 GUI 心跳用）。"""
        if self.skip_hour >= 0 and datetime.now().hour == self.skip_hour:
            return f"跳过小时 {self.skip_hour}:00–{self.skip_hour}:59，整点后继续"
        remain = self.interval - (time.time() - self._last_run)
        if remain <= 0:
            return "即将开始下一轮"
        if remain >= 60:
            return f"约 {int(remain // 60)} 分 {int(remain % 60)} 秒后再次集结"
        return f"约 {int(remain)} 秒后再次集结"

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
        """右下角显示「城镇」说明当前在野外地图。"""
        return self._match_in_roi(screen, BTN_TOWN_LABEL, SCENE_TOGGLE_ROI).found

    def _is_in_town(self, screen) -> bool:
        """右下角显示「野外」说明当前在城镇内部。"""
        return self._match_in_roi(screen, BTN_WILDERNESS_LABEL, SCENE_TOGGLE_ROI).found

    def _has_scene_templates(self) -> bool:
        return (TEMPLATE_DIR / BTN_TOWN_LABEL).is_file() or (
            TEMPLATE_DIR / BTN_WILDERNESS_LABEL
        ).is_file()

    def _dismiss_exit_dialog(self) -> None:
        """取消 ESC 触发的退出确认弹窗，避免反复按键死循环。"""
        self._dismiss_dialog()

    def _close_search_panel(self) -> None:
        if self._interrupted():
            raise InterruptedError("任务已停止")
        self._back(1)

    def _is_search_panel_visible(self, screen=None) -> bool:
        """搜索面板会遮挡底部导航与放大镜，需先关闭再操作。"""
        if screen is None:
            screen = capture_screen(self.adb, SEARCH_PANEL_CAPTURE_ROI)
        for template in SEARCH_PANEL_TEMPLATES:
            if self.vision.match_template(screen, template).found:
                return True
        x1, y1, x2, y2 = SEARCH_TAB_ROI
        roi = screen[y1:y2, x1:x2]
        tab_vision = Vision(TEMPLATE_DIR, threshold=0.68)
        for template in (ICE_BEAST_TAB_TEMPLATE, ICE_BEAST_TAB_SELECTED):
            if tab_vision.match_template(roi, template).found:
                return True
        return False

    def _is_on_main_map(self, screen) -> bool:
        if not self._has_scene_templates():
            return not self._is_search_panel_visible(screen)
        return self._is_in_wilderness(screen) or self._is_in_town(screen)

    def _ensure_clean_ui(self) -> None:
        """启动前回到野外主界面（关闭搜索面板等遮挡）。"""
        self._emit("清理界面…")
        if self._wilderness.return_to_wilderness():
            self._emit("界面已就绪")
            time.sleep(0.5)
        else:
            self._emit("已尝试清理界面，继续执行")

    def _close_overlays(self) -> None:
        self._wilderness.return_to_wilderness()

    def _ensure_wilderness(self) -> None:
        """确保处于野外场景（右下角显示「城镇」）。"""
        self._wilderness.ensure_wilderness()

    def _open_search_panel(self) -> None:
        """在野外点击左下角放大镜，打开搜索面板（固定坐标，不截图匹配）。"""
        self._ensure_wilderness()
        sx, sy = self.coords["search_open"]
        self._emit(f"打开搜索面板 @ ({int(sx)},{int(sy)})")
        self._tap("search_open", delay=1.5)

    def _scroll_search_tab_bar_to_rightmost(self) -> None:
        """将搜索目标 tab 栏滚到最右端，露出左侧图标（野兽、冰原巨兽等）。"""
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

    def _select_ice_beast_tab(self) -> None:
        """滚到 tab 栏最右端后，点击「冰原巨兽」tab（按野兽图标位置右移）。"""
        if self._tab_bar_already_scrolled:
            self._emit("上次已是冰原巨兽任务，跳过 tab 栏拖动")
        else:
            self._scroll_search_tab_bar_to_rightmost()
            self._tab_bar_already_scrolled = True

        if "ice_beast_tab" in self.coords:
            bx, by = self.coords["ice_beast_tab"]
        else:
            bx, by = DEFAULT_ICE_BEAST_TAB
        tx, ty = shift_search_tab_xy(
            int(bx),
            int(by),
            beast_icon_index=self.beast_icon_index,
            step=self._search_tab_step,
        )
        if self.beast_icon_index:
            self._emit(
                f"选中冰原巨兽 tab @ ({tx},{ty})"
                f"（基准 {int(bx)},{int(by)} + 位置{self.beast_icon_index}"
                f"×步径{self._search_tab_step}）"
            )
        else:
            self._emit(f"选中冰原巨兽 tab @ ({tx},{ty})")
        self._tap_xy(tx, ty, delay=1.0)

    def _set_beast_level(self) -> None:
        """按配置调整搜索等级；仅大循环首次执行（游戏会记忆等级）。"""
        if not self.adjust_level:
            self._emit("未启用修改等级，跳过")
            return
        if self._level_already_adjusted:
            self._emit("本轮调度已调整过等级，跳过")
            return

        adjust_search_level(
            self.adb,
            self.beast_level,
            emit=self._emit,
            interrupted=self._interrupted,
        )
        self._level_already_adjusted = True

    def _tap_search_confirm(self) -> None:
        """点击底部「搜索」按钮（固定坐标，不截图匹配）。"""
        tx, ty = self.coords["search_confirm"]
        self._emit(f"点击搜索 @ ({int(tx)},{int(ty)})")
        self._tap_xy(int(tx), int(ty), delay=2.5)
        self._emit("正在搜索冰原巨兽…")

    def _tap_rally_button(self) -> None:
        """全屏匹配橙色「集结」按钮并点击（弹窗位置不固定）。"""
        template_path = TEMPLATE_DIR / RALLY_BTN
        if not template_path.exists():
            logger.warning("集结按钮模板缺失，使用配置坐标")
            self._tap("rally", delay=2.0)
            self._emit("已开启集结")
            return

        rally_vision = Vision(TEMPLATE_DIR, threshold=RALLY_MATCH_THRESHOLD)
        best_conf = 0.0

        for attempt in range(RALLY_RETRY_COUNT):
            screen = self.adb.screenshot()
            result = rally_vision.match_template_multiscale(screen, RALLY_BTN)
            best_conf = max(best_conf, result.confidence)

            if result.found:
                tx, ty = result.center
                logger.debug(
                    f"模板匹配集结按钮 → ({tx}, {ty})，置信度 {result.confidence:.2f}"
                )
                self._tap_xy(tx, ty, delay=2.0)
                self._emit("已开启集结")
                return

            time.sleep(RALLY_RETRY_INTERVAL)
            logger.debug(
                f"集结按钮匹配重试 {attempt + 1}/{RALLY_RETRY_COUNT}（最高 {result.confidence:.2f}）"
            )

        raise RuntimeError(
            f"未找到「集结」按钮（最高置信度 {best_conf:.2f}）。"
            f"请确认已打开巨兽详情弹窗，或更新 assets/templates/{RALLY_BTN}"
        )

    def _get_rally_time_coord(self, minutes: int) -> tuple[int, int]:
        key = f"rally_time_{minutes}"
        if key in self.coords:
            return tuple(self.coords[key])
        return RALLY_TIME_COORDS[minutes]

    def _find_rally_confirm_button(self, screen) -> MatchResult:
        """匹配弹窗底部「发起集结」按钮，忽略出征界面等场景的误匹配。"""
        confirm_vision = Vision(TEMPLATE_DIR, threshold=RALLY_CONFIRM_THRESHOLD)
        confirm_x, _ = RALLY_CONFIRM_CENTER
        candidates = [
            confirm_vision.match_template_multiscale(screen, RALLY_CONFIRM_BTN),
            confirm_vision.match_template(screen, RALLY_CONFIRM_BTN),
        ]
        best = MatchResult(found=False)
        for result in candidates:
            if result.confidence <= best.confidence:
                continue
            if result.found and result.center[1] < RALLY_CONFIRM_MIN_Y:
                logger.debug(
                    f"忽略误匹配 {RALLY_CONFIRM_BTN} @ {result.center} "
                    f"(y<{RALLY_CONFIRM_MIN_Y})，置信度 {result.confidence:.2f}"
                )
                continue
            if result.found and abs(result.center[0] - confirm_x) > RALLY_CONFIRM_MAX_DX:
                logger.debug(
                    f"忽略误匹配 {RALLY_CONFIRM_BTN} @ {result.center} "
                    f"(x 偏离弹窗按钮 {confirm_x}±{RALLY_CONFIRM_MAX_DX})，"
                    f"置信度 {result.confidence:.2f}"
                )
                continue
            best = result
        return best

    def _confirm_rally_popup(self) -> None:
        """在「发起集结」弹窗中选择时长并点击底部确认按钮。

        延迟后只截图识别一次确认按钮；点完不再二次校验弹窗/出征页
        （出征页由后续编队/出征一步截图负责）。
        """
        time.sleep(RALLY_RETRY_INTERVAL)
        if self._interrupted():
            raise InterruptedError("任务已停止")

        screen = self.adb.screenshot()
        result = self._find_rally_confirm_button(screen)
        cx, cy = (
            tuple(self.coords["rally_confirm"])
            if "rally_confirm" in self.coords
            else RALLY_CONFIRM_CENTER
        )
        if result.found:
            cx, cy = result.center
            logger.debug(
                f"发起集结确认按钮 → ({cx}, {cy})，置信度 {result.confidence:.2f}"
            )
        else:
            logger.warning(
                f"发起集结确认按钮未匹配（最高 {result.confidence:.2f}），"
                f"使用固定坐标 ({cx}, {cy})"
            )

        duration = self.rally_duration_minutes
        if duration != DEFAULT_RALLY_DURATION:
            tx, ty = self._get_rally_time_coord(duration)
            logger.debug(f"选择集结时长 {duration} 分钟 → ({tx}, {ty})")
            self._tap_xy(tx, ty, delay=0.5)
        else:
            logger.debug(f"集结时长 {DEFAULT_RALLY_DURATION} 分钟（默认已选中）")

        self._tap_xy(cx, cy, delay=1.0)
        self._emit(f"已确认集结（{duration} 分钟）")

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
        """按配置点击编队槽位（不截图等待）。未启用编队则跳过。"""
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

    def _analyze_deploy_and_resolve_march(
        self, screen
    ) -> tuple[int, int, float]:
        """同一张出征截图：可选英雄检查 + 匹配出征按钮。返回 (x, y, confidence)。"""
        if self.use_formation:
            self._emit("检查出征英雄…")
            empty_slots = self._find_empty_march_hero_slots(screen)
            if empty_slots:
                slots_text = "、".join(str(i) for i in empty_slots)
                raise MarchHeroCheckError(f"第 {slots_text} 个英雄位为空，跳过本轮")
            self._emit("出征英雄已配满（3/3）")

        result = find_march_button(screen)
        if not result.found:
            raise RuntimeError(
                "未检测到出征界面（右下角「出征」按钮）。"
                f"最高匹配 {result.confidence:.2f}。请确认上一步已成功发起集结。"
            )
        cx, cy = result.center
        self._emit(f"出征按钮就绪（{result.confidence:.2f} @ ({cx},{cy})）")
        return cx, cy, result.confidence

    def _march_tap_xy(self) -> tuple[int, int]:
        """配置或默认出征坐标（体力弹窗关闭后二次出征用）。"""
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
        """确认集结后：延迟 → 点编队(可选) → 截一张验英雄+出征按钮 → 点出征并判结果。"""
        time.sleep(DEPLOY_WAIT_SETTLE_SEC)
        if self._interrupted():
            raise InterruptedError("任务已停止")

        self._tap_formation_slot()

        screen = capture_screen(self.adb, DEPLOY_CAPTURE_ROI)
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

    @staticmethod
    def _march_hero_slot_rois(
        hero_roi: tuple[int, int, int, int] = MARCH_HERO_ROI,
        slot_count: int = MARCH_HERO_SLOT_COUNT,
    ) -> list[tuple[int, int, int, int]]:
        x1, y1, x2, y2 = hero_roi
        width = x2 - x1
        slot_w = width / slot_count
        inset = MARCH_HERO_SLOT_INSET
        rois: list[tuple[int, int, int, int]] = []
        for index in range(slot_count):
            sx1 = int(x1 + index * slot_w + slot_w * inset)
            sx2 = int(x1 + (index + 1) * slot_w - slot_w * inset)
            rois.append((sx1, y1, sx2, y2))
        return rois

    @staticmethod
    def _empty_hero_slot_blue_ratio(crop) -> float:
        """空槽为蓝色底+加号，中心区域蓝色像素占比高。"""
        height, width = crop.shape[:2]
        y1, y2 = int(height * 0.18), int(height * 0.82)
        x1, x2 = int(width * 0.18), int(width * 0.82)
        center = crop[y1:y2, x1:x2]
        if center.size == 0:
            return 0.0
        hsv = cv2.cvtColor(center, cv2.COLOR_BGR2HSV)
        hue, saturation, value = cv2.split(hsv)
        blue_mask = (hue >= 90) & (hue <= 130) & (saturation >= 35) & (value >= 35)
        return float(blue_mask.mean())

    @staticmethod
    def _empty_hero_slot_template_score(crop) -> float:
        template_path = TEMPLATE_DIR / EMPTY_HERO_SLOT_TEMPLATE
        if not template_path.exists() or crop.shape[0] < 20 or crop.shape[1] < 20:
            return 0.0
        template = cv2.imread(str(template_path))
        if template is None:
            return 0.0
        th, tw = template.shape[:2]
        ch, cw = crop.shape[:2]
        if th > ch or tw > cw:
            scale = min(ch / th, cw / tw) * 0.95
            template = cv2.resize(
                template,
                (max(1, int(tw * scale)), max(1, int(th * scale))),
                interpolation=cv2.INTER_AREA,
            )
            th, tw = template.shape[:2]
        if th > ch or tw > cw:
            return 0.0
        result = cv2.matchTemplate(crop, template, cv2.TM_CCOEFF_NORMED)
        return float(result.max())

    @staticmethod
    def _is_empty_hero_slot(crop) -> tuple[bool, float, float]:
        blue_ratio = HuntIceBeastTask._empty_hero_slot_blue_ratio(crop)
        template_score = HuntIceBeastTask._empty_hero_slot_template_score(crop)
        is_empty = blue_ratio >= EMPTY_SLOT_BLUE_RATIO
        if not is_empty and template_score >= EMPTY_SLOT_TM_THRESHOLD:
            is_empty = True
        return is_empty, blue_ratio, template_score

    @staticmethod
    def _find_empty_march_hero_slots(screen) -> list[int]:
        empty_slots: list[int] = []
        for slot_index, roi in enumerate(
            HuntIceBeastTask._march_hero_slot_rois(), start=1
        ):
            x1, y1, x2, y2 = roi
            crop = screen[y1:y2, x1:x2]
            is_empty, blue_ratio, template_score = HuntIceBeastTask._is_empty_hero_slot(crop)
            logger.info(
                f"出征英雄位 {slot_index} ROI=({x1},{y1},{x2},{y2}) "
                f"blue={blue_ratio:.3f} tmpl={template_score:.3f} empty={is_empty}"
            )
            if is_empty:
                empty_slots.append(slot_index)
        return empty_slots

    def _find_stamina_use_button(self, screen) -> MatchResult:
        """定位领主体力道具行的绿色「使用」按钮（仅第三行区域）。"""
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
        """是否出现「获取更多」体力不足弹窗（ROI OCR，出现即表示出征未成功）。"""
        return is_stamina_get_more_title(screen)

    def _get_stamina_use_tap_point(self, screen) -> tuple[int, int]:
        """优先用当前画面模板匹配定位，配置坐标仅作兜底。"""
        btn = self._find_stamina_use_button(screen)
        if btn.found:
            return btn.center
        if "stamina_use" in self.coords:
            return tuple(self.coords["stamina_use"])
        return STAMINA_USE_CENTER

    def _use_stamina_items(self) -> None:
        """弹窗内连点领主体力（默认 20 次）。固定用实测坐标，勿用 stamina_use 配置误点。"""
        use_stamina_cans_batch(
            self.adb,
            self.stamina_budget,
            tap_xy=STAMINA_USE_XY,
            emit=self._emit,
            interrupted=self._interrupted,
            close_with_back=True,
        )

    def _wait_march_outcome(
        self, delay: float = MARCH_OUTCOME_DELAY_SEC
    ) -> tuple[bool, object]:
        """等待出征结果。

        返回 (是否体力不足弹窗, 本次截图)。
        无弹窗时截图可复用给出征收尾，避免再截一张。
        """
        self._emit("正在检测出征结果…")
        logger.info(f"开始检测出征结果（延迟 {delay:.1f}s 后 OCR 一次）")
        time.sleep(delay)
        if self._interrupted():
            raise InterruptedError("任务已停止")

        screen = capture_screen(self.adb, STAMINA_POPUP_CAPTURE_ROI)
        is_popup = self._is_stamina_popup(screen)
        if is_popup:
            logger.info("检测到体力不足弹窗（OCR）")
            return True, screen
        logger.info("未检测到体力不足弹窗，视为出征成功")
        # 体力 ROI 不含底部导航，成功时不复用该帧
        return False, None

    def _finish_after_march(self, screen=None) -> None:
        """出征成功后确认主界面；优先复用出征结果截图，避免再截一张。

        冰原集结出征成功后会直接回到野外地图（右下角显示「城镇」），
        不存在单独的「返回城镇」按钮，因此只确认主界面就绪即可。
        """
        if screen is None:
            time.sleep(1.0)
            screen = capture_screen(self.adb, BOTTOM_UI_CAPTURE_ROI)
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
        self._emit(f"开始搜索 {self.beast_level} 级冰原巨兽")

        self._open_search_panel()
        self._select_ice_beast_tab()
        self._set_beast_level()

        self._tap_search_confirm()

        self._tap("target_tap", delay=2.0)
        self._emit("已定位目标")

        self._tap_rally_button()

        self._confirm_rally_popup()

        march_screen = self._prepare_and_tap_march()
        self._finish_after_march(march_screen)

    def run_once(self, *, force: bool = False) -> bool:
        if not force and not self.should_run():
            return False

        self._last_run = time.time()
        try:
            self.run_hunt_cycle()
            self._emit(f"本轮完成，{int(self.interval // 60)} 分钟后再次集结")
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
        self._emit("=== 冰原巨兽自动集结已启动 ===")
        formation_desc = (
            f"编队槽位 {self.formation_name}"
            if self.use_formation
            else "不启用编队，直接出征"
        )
        self._emit(
            f"目标 {self.beast_level} 级冰原巨兽，间隔 {int(self.interval // 60)} 分钟，"
            f"{formation_desc}"
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
            self._emit("自动集结已停止")
