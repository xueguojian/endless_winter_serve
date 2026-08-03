"""游戏场景导航：回到主界面 / 野外。"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

import numpy as np
from loguru import logger

from core.adb_client import AdbClient
from core.capture_roi import BOTTOM_UI_CAPTURE_ROI, capture_screen
from core.vision import Vision

TEMPLATE_DIR = Path(__file__).parent.parent / "assets" / "templates"
SCENE_TOGGLE_ROI = (500, 1150, 720, 1280)
BTN_TOWN_LABEL = "btn_town_label.png"
BTN_WILDERNESS_LABEL = "btn_wilderness_label.png"
SEARCH_TEMPLATES = ("beast_tab.png", "search_confirm_btn.png")

MAX_ATTEMPTS = 14
WILDERNESS_NAV_MAX_ATTEMPTS = 16
TOWN_NAV_MAX_ATTEMPTS = 16
FALLBACK_BACKS = 12
SCENE_THRESHOLD = 0.70
BACK_STEP_DELAY = 0.8
WILDERNESS_SWITCH_DELAY = 2.0
DEFAULT_DIALOG_CANCEL = (250, 780)
TOWN_SWITCH_MAX_ATTEMPTS = 6


def parse_dialog_cancel(coords: dict | None) -> tuple[int, int]:
    """从任务 coords 解析退出弹窗取消按钮；缺省使用全局默认坐标。"""
    if coords and "dialog_cancel" in coords:
        raw = coords["dialog_cancel"]
        return int(raw[0]), int(raw[1])
    return DEFAULT_DIALOG_CANCEL


def _tap_town_switch(adb: AdbClient, vision: Vision, screen: np.ndarray) -> bool:
    center = _match_center_in_roi(vision, screen, BTN_TOWN_LABEL, SCENE_TOGGLE_ROI)
    if center is None:
        return False
    adb.tap(*center)
    time.sleep(WILDERNESS_SWITCH_DELAY)
    return True


def _get_vision(vision: Vision | None) -> Vision:
    return vision or Vision(TEMPLATE_DIR, threshold=SCENE_THRESHOLD)


def _nav_capture(
    adb: AdbClient,
    *,
    need_full: bool = False,
) -> np.ndarray:
    """导航截图：无灯塔情报检测时只上传底部 UI。"""
    if need_full:
        return capture_screen(adb, None)
    return capture_screen(adb, BOTTOM_UI_CAPTURE_ROI)


def _match_in_roi(
    vision: Vision, screen: np.ndarray, template: str, roi: tuple[int, int, int, int]
) -> bool:
    x1, y1, x2, y2 = roi
    return vision.match_template(screen[y1:y2, x1:x2], template).found


def _match_center_in_roi(
    vision: Vision, screen: np.ndarray, template: str, roi: tuple[int, int, int, int]
) -> tuple[int, int] | None:
    x1, y1, x2, y2 = roi
    result = vision.match_template(screen[y1:y2, x1:x2], template)
    if not result.found:
        return None
    cx, cy = result.center
    return x1 + cx, y1 + cy


def is_in_wilderness(vision: Vision, screen: np.ndarray) -> bool:
    """右下角显示「城镇」说明当前在野外地图。"""
    return _match_in_roi(vision, screen, BTN_TOWN_LABEL, SCENE_TOGGLE_ROI)


def is_in_town(vision: Vision, screen: np.ndarray) -> bool:
    """右下角显示「野外」说明当前在城镇内部。"""
    return _match_in_roi(vision, screen, BTN_WILDERNESS_LABEL, SCENE_TOGGLE_ROI)


def is_on_main_map(vision: Vision, screen: np.ndarray) -> bool:
    return is_in_wilderness(vision, screen) or is_in_town(vision, screen)


def _has_scene_templates() -> bool:
    return (TEMPLATE_DIR / BTN_TOWN_LABEL).is_file() or (
        TEMPLATE_DIR / BTN_WILDERNESS_LABEL
    ).is_file()


def _overlay_open(vision: Vision, screen: np.ndarray) -> bool:
    return any(vision.match_template(screen, tpl).found for tpl in SEARCH_TEMPLATES)


def _tap_dialog_cancel(adb: AdbClient, dialog_cancel: tuple[int, int] | None) -> bool:
    if not dialog_cancel:
        return False
    adb.tap(*dialog_cancel)
    time.sleep(0.5)
    return True


def _tap_wilderness_switch(
    adb: AdbClient, vision: Vision, screen: np.ndarray
) -> bool:
    center = _match_center_in_roi(
        vision, screen, BTN_WILDERNESS_LABEL, SCENE_TOGGLE_ROI
    )
    if center is None:
        return False
    adb.tap(*center)
    time.sleep(WILDERNESS_SWITCH_DELAY)
    return True


def _is_known_subscreen(
    screen: np.ndarray,
    *,
    is_lighthouse_intel: Callable[[np.ndarray], bool] | None = None,
    is_deploy_screen: Callable[[np.ndarray], bool] | None = None,
    is_overlay_open: Callable[[np.ndarray], bool] | None = None,
) -> bool:
    if is_overlay_open and is_overlay_open(screen):
        return True
    if is_deploy_screen and is_deploy_screen(screen):
        return True
    if is_lighthouse_intel and is_lighthouse_intel(screen):
        return True
    return False


def _maybe_recover_exit_dialog(
    adb: AdbClient,
    vision: Vision,
    screen: np.ndarray,
    dialog_cancel: tuple[int, int] | None,
    *,
    is_lighthouse_intel: Callable[[np.ndarray], bool] | None = None,
    is_deploy_screen: Callable[[np.ndarray], bool] | None = None,
    is_overlay_open: Callable[[np.ndarray], bool] | None = None,
) -> bool:
    """疑似误触「退出游戏」弹窗时点取消。返回 True 表示已尝试取消。"""
    if not dialog_cancel:
        return False
    if is_on_main_map(vision, screen):
        return False
    if _is_known_subscreen(
        screen,
        is_lighthouse_intel=is_lighthouse_intel,
        is_deploy_screen=is_deploy_screen,
        is_overlay_open=is_overlay_open,
    ):
        return False
    if _overlay_open(vision, screen):
        return False
    logger.debug("疑似退出游戏弹窗，点击取消")
    return _tap_dialog_cancel(adb, dialog_cancel)


def return_to_wilderness_screen(
    adb: AdbClient,
    *,
    vision: Vision | None = None,
    dialog_cancel: tuple[int, int] | None = None,
    on_status: Callable[[str], None] | None = None,
    interrupted: Callable[[], bool] | None = None,
    is_lighthouse_intel: Callable[[np.ndarray], bool] | None = None,
    is_deploy_screen: Callable[[np.ndarray], bool] | None = None,
    is_overlay_open: Callable[[np.ndarray], bool] | None = None,
    max_attempts: int = WILDERNESS_NAV_MAX_ATTEMPTS,
) -> bool:
    """状态机式返回野外主界面（右下角出现「城镇」按钮）。

    每步截图确认当前界面，仅在子界面/弹层时按返回；已在主界面时绝不连按 back，
    避免弹出「退出游戏」。返回 True 表示已确认在野外。
    """
    vision = _get_vision(vision)

    if not _has_scene_templates():
        logger.debug("未找到场景模板，使用固定次数返回")
        for _ in range(FALLBACK_BACKS):
            if interrupted and interrupted():
                raise InterruptedError("任务已停止")
            adb.back()
            time.sleep(BACK_STEP_DELAY)
        if on_status:
            on_status("已尝试返回野外（未确认场景）")
        return False

    saw_main_map = False
    need_full = is_lighthouse_intel is not None

    for attempt in range(max_attempts):
        if interrupted and interrupted():
            raise InterruptedError("任务已停止")

        screen = _nav_capture(adb, need_full=need_full)

        if is_in_wilderness(vision, screen):
            if on_status:
                on_status("已在野外")
            return True

        if is_in_town(vision, screen):
            saw_main_map = True
            if on_status:
                on_status("当前在城镇，切换到野外…")
            if _tap_wilderness_switch(adb, vision, screen):
                continue
            logger.warning("未能点击野外切换按钮")
            adb.back()
            time.sleep(BACK_STEP_DELAY)
            continue

        if saw_main_map:
            if _maybe_recover_exit_dialog(
                adb,
                vision,
                screen,
                dialog_cancel,
                is_lighthouse_intel=is_lighthouse_intel,
                is_deploy_screen=is_deploy_screen,
                is_overlay_open=is_overlay_open,
            ):
                saw_main_map = False
                continue

        saw_main_map = False

        if is_overlay_open and is_overlay_open(screen):
            adb.back()
            time.sleep(BACK_STEP_DELAY)
            continue

        if is_deploy_screen and is_deploy_screen(screen):
            adb.back()
            time.sleep(BACK_STEP_DELAY)
            continue

        if is_lighthouse_intel and is_lighthouse_intel(screen):
            adb.back()
            time.sleep(BACK_STEP_DELAY)
            continue

        if _overlay_open(vision, screen):
            adb.back()
            time.sleep(BACK_STEP_DELAY)
            continue

        adb.back()
        time.sleep(BACK_STEP_DELAY)

        screen_after = _nav_capture(adb, need_full=need_full)
        if is_in_wilderness(vision, screen_after):
            if on_status:
                on_status("已在野外")
            return True
        if is_in_town(vision, screen_after):
            saw_main_map = True
            continue

        _maybe_recover_exit_dialog(
            adb,
            vision,
            screen_after,
            dialog_cancel,
            is_lighthouse_intel=is_lighthouse_intel,
            is_deploy_screen=is_deploy_screen,
            is_overlay_open=is_overlay_open,
        )

    if on_status:
        on_status("已尝试返回野外（未确认场景）")
    logger.warning("return_to_wilderness_screen: 未能在限定步数内确认野外")
    return False


def return_to_town_screen(
    adb: AdbClient,
    *,
    vision: Vision | None = None,
    dialog_cancel: tuple[int, int] | None = None,
    on_status: Callable[[str], None] | None = None,
    interrupted: Callable[[], bool] | None = None,
    is_lighthouse_intel: Callable[[np.ndarray], bool] | None = None,
    is_deploy_screen: Callable[[np.ndarray], bool] | None = None,
    is_overlay_open: Callable[[np.ndarray], bool] | None = None,
    max_attempts: int = TOWN_NAV_MAX_ATTEMPTS,
) -> bool:
    """状态机式返回城镇主界面（右下角出现「野外」按钮）。

    已在城镇子界面（兵营、状态面板等）时逐步返回，不绕道野外；
    若当前在野外则点击右下角「城镇」进入。
    """
    vision = _get_vision(vision)

    if not _has_scene_templates():
        logger.debug("未找到场景模板，使用固定次数返回")
        for _ in range(FALLBACK_BACKS):
            if interrupted and interrupted():
                raise InterruptedError("任务已停止")
            adb.back()
            time.sleep(BACK_STEP_DELAY)
        if on_status:
            on_status("已尝试返回城镇（未确认场景）")
        return False

    saw_main_map = False
    need_full = is_lighthouse_intel is not None

    for attempt in range(max_attempts):
        if interrupted and interrupted():
            raise InterruptedError("任务已停止")

        screen = _nav_capture(adb, need_full=need_full)

        if is_in_town(vision, screen):
            if on_status:
                on_status("已在城镇")
            return True

        if is_in_wilderness(vision, screen):
            saw_main_map = True
            if on_status:
                on_status("当前在野外，切换到城镇…")
            if _tap_town_switch(adb, vision, screen):
                continue
            logger.warning("未能点击城镇切换按钮")
            adb.back()
            time.sleep(BACK_STEP_DELAY)
            continue

        if saw_main_map:
            if _maybe_recover_exit_dialog(
                adb,
                vision,
                screen,
                dialog_cancel,
                is_lighthouse_intel=is_lighthouse_intel,
                is_deploy_screen=is_deploy_screen,
                is_overlay_open=is_overlay_open,
            ):
                saw_main_map = False
                continue

        saw_main_map = False

        if is_overlay_open and is_overlay_open(screen):
            adb.back()
            time.sleep(BACK_STEP_DELAY)
            continue

        if is_deploy_screen and is_deploy_screen(screen):
            adb.back()
            time.sleep(BACK_STEP_DELAY)
            continue

        if is_lighthouse_intel and is_lighthouse_intel(screen):
            adb.back()
            time.sleep(BACK_STEP_DELAY)
            continue

        if _overlay_open(vision, screen):
            adb.back()
            time.sleep(BACK_STEP_DELAY)
            continue

        adb.back()
        time.sleep(BACK_STEP_DELAY)

        screen_after = _nav_capture(adb, need_full=need_full)
        if is_in_town(vision, screen_after):
            if on_status:
                on_status("已在城镇")
            return True
        if is_in_wilderness(vision, screen_after):
            saw_main_map = True
            continue

        _maybe_recover_exit_dialog(
            adb,
            vision,
            screen_after,
            dialog_cancel,
            is_lighthouse_intel=is_lighthouse_intel,
            is_deploy_screen=is_deploy_screen,
            is_overlay_open=is_overlay_open,
        )

    if on_status:
        on_status("已尝试返回城镇（未确认场景）")
    logger.warning("return_to_town_screen: 未能在限定步数内确认城镇")
    return False


def switch_to_town_screen(
    adb: AdbClient,
    *,
    vision: Vision | None = None,
    dialog_cancel: tuple[int, int] | None = None,
    on_status: Callable[[str], None] | None = None,
    interrupted: Callable[[], bool] | None = None,
    is_lighthouse_intel: Callable[[np.ndarray], bool] | None = None,
    is_deploy_screen: Callable[[np.ndarray], bool] | None = None,
    is_overlay_open: Callable[[np.ndarray], bool] | None = None,
    max_attempts: int = TOWN_SWITCH_MAX_ATTEMPTS,
) -> bool:
    """确保处于城镇主界面（右下角出现「野外」按钮）。"""
    return return_to_town_screen(
        adb,
        vision=vision,
        dialog_cancel=dialog_cancel,
        on_status=on_status,
        interrupted=interrupted,
        is_lighthouse_intel=is_lighthouse_intel,
        is_deploy_screen=is_deploy_screen,
        is_overlay_open=is_overlay_open,
        max_attempts=max_attempts,
    )


class WildernessNavigator:
    """各任务统一的野外场景导航入口。"""

    def __init__(
        self,
        adb: AdbClient,
        *,
        vision: Vision | None = None,
        dialog_cancel: tuple[int, int] | None = None,
        on_status: Callable[[str], None] | None = None,
        interrupted: Callable[[], bool] | None = None,
        is_lighthouse_intel: Callable[[np.ndarray], bool] | None = None,
        is_deploy_screen: Callable[[np.ndarray], bool] | None = None,
        is_overlay_open: Callable[[np.ndarray], bool] | None = None,
    ):
        self.adb = adb
        self.vision = vision
        self.dialog_cancel = dialog_cancel
        self.on_status = on_status
        self.interrupted = interrupted
        self.is_lighthouse_intel = is_lighthouse_intel
        self.is_deploy_screen = is_deploy_screen
        self.is_overlay_open = is_overlay_open

    @classmethod
    def from_task(
        cls,
        task: object,
        *,
        vision: Vision | None = None,
        is_overlay_open: Callable[[np.ndarray], bool] | None = None,
        is_lighthouse_intel: Callable[[np.ndarray], bool] | None = None,
        is_deploy_screen: Callable[[np.ndarray], bool] | None = None,
    ) -> WildernessNavigator:
        coords = getattr(task, "coords", None)
        interrupted = getattr(task, "_interrupted", None)
        return cls(
            getattr(task, "adb"),
            vision=vision or getattr(task, "vision", None),
            dialog_cancel=parse_dialog_cancel(coords if isinstance(coords, dict) else None),
            on_status=getattr(task, "on_status", None),
            interrupted=interrupted if callable(interrupted) else None,
            is_overlay_open=is_overlay_open,
            is_lighthouse_intel=is_lighthouse_intel,
            is_deploy_screen=is_deploy_screen,
        )

    def _kwargs(self) -> dict:
        return {
            "vision": self.vision,
            "dialog_cancel": self.dialog_cancel,
            "on_status": self.on_status,
            "interrupted": self.interrupted,
            "is_lighthouse_intel": self.is_lighthouse_intel,
            "is_deploy_screen": self.is_deploy_screen,
            "is_overlay_open": self.is_overlay_open,
        }

    def return_to_wilderness(self) -> bool:
        return return_to_wilderness_screen(self.adb, **self._kwargs())

    def ensure_wilderness(self) -> None:
        if not self.return_to_wilderness():
            raise RuntimeError("无法进入野外地图")

    def try_return_to_wilderness(self) -> None:
        try:
            if not self.return_to_wilderness():
                logger.warning("未能确认已回到野外")
        except Exception as exc:
            logger.warning(f"返回野外失败: {exc}")

    def return_to_town(self) -> bool:
        return return_to_town_screen(self.adb, **self._kwargs())

    def ensure_town(self) -> None:
        if not self.return_to_town():
            raise RuntimeError("无法进入城镇主界面")

    def try_return_to_town(self) -> None:
        try:
            if not self.return_to_town():
                logger.warning("未能确认已回到城镇")
        except Exception as exc:
            logger.warning(f"返回城镇失败: {exc}")

    def switch_to_town(self) -> None:
        self.ensure_town()


def return_to_main_screen(
    adb: AdbClient,
    on_status: Callable[[str], None] | None = None,
    max_attempts: int = MAX_ATTEMPTS,
    *,
    vision: Vision | None = None,
    dialog_cancel: tuple[int, int] | None = None,
    interrupted: Callable[[], bool] | None = None,
) -> None:
    """关闭子界面/弹窗，直到右下角出现「城镇」或「野外」切换按钮。"""
    vision = _get_vision(vision)

    if not _has_scene_templates():
        logger.debug("未找到场景模板，使用固定次数返回")
        for _ in range(FALLBACK_BACKS):
            if interrupted and interrupted():
                raise InterruptedError("任务已停止")
            adb.back()
            time.sleep(BACK_STEP_DELAY)
        if on_status:
            on_status("已尝试返回主界面")
        return

    for _ in range(max_attempts):
        if interrupted and interrupted():
            raise InterruptedError("任务已停止")

        screen = _nav_capture(adb, need_full=False)
        if is_on_main_map(vision, screen):
            if on_status:
                on_status("已回到主界面")
            return

        if _overlay_open(vision, screen):
            adb.back()
            time.sleep(BACK_STEP_DELAY)
            continue

        adb.back()
        time.sleep(BACK_STEP_DELAY)

        screen_after = _nav_capture(adb, need_full=False)
        if is_on_main_map(vision, screen_after):
            if on_status:
                on_status("已回到主界面")
            return

        _maybe_recover_exit_dialog(
            adb, vision, screen_after, dialog_cancel
        )

    if on_status:
        on_status("已尝试返回主界面（未确认场景）")
    logger.warning("return_to_main_screen: 未能确认主界面")
