"""出征界面：编队选择、出征、体力不足处理。"""

from __future__ import annotations

import time
from functools import lru_cache
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from loguru import logger

from core.adb_client import AdbClient
from core.capture_roi import MARCH_BTN_CAPTURE_ROI, capture_screen
from core.stamina_use import (
    DEFAULT_STAMINA_CAN_LIMIT,
    MARCH_OUTCOME_DELAY_SEC,
    STAMINA_POPUP_CAPTURE_ROI,
    STAMINA_USE_XY,
    StaminaCanBudget,
    is_stamina_get_more_title,
    use_stamina_cans_batch,
)
from core.vision import MatchResult, Vision

TEMPLATE_DIR = Path(__file__).parent.parent / "assets" / "templates"

MARCH_BTN = "march_btn.png"
MARCH_BTN_LABEL = "march_btn_label.png"
STAMINA_USE_BTN = "stamina_use_btn.png"

STAMINA_USE_ROW_ROI = (500, 520, 710, 660)
STAMINA_USE_CENTER = (576, 522)
STAMINA_USE_Y_MIN = 520
STAMINA_USE_Y_MAX = 660
STAMINA_MATCH_THRESHOLD = 0.58

# 用「出征」标签模板；完整 march_btn 含体力数字/行军时间，换编队后极易掉置信度
MARCH_MATCH_THRESHOLD = 0.62
MARCH_MIN_Y = 1050
MARCH_CENTER = (560, 1200)
MARCH_BTN_ROI = (340, 1080, 720, 1280)
# 无独立标签图时，从完整模板截掉顶栏时间与底部费用数字
MARCH_LABEL_Y0_RATIO = 0.20
MARCH_LABEL_Y1_RATIO = 0.50
MARCH_LABEL_X0_RATIO = 0.18
MARCH_LABEL_X1_RATIO = 0.82
MARCH_SCALES = (0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.25)

# 双开减负：出征页等待 / 点出征 截图间隔拉长、次数减少
DEPLOY_WAIT_TIMEOUT = 10.0
DEPLOY_WAIT_SETTLE_SEC = 0.8
DEPLOY_WAIT_POLL_SEC = 0.9
MARCH_RESOLVE_ATTEMPTS = 4
MARCH_RESOLVE_POLL_SEC = 0.9


@lru_cache(maxsize=1)
def _march_label_template() -> np.ndarray | None:
    label_path = TEMPLATE_DIR / MARCH_BTN_LABEL
    if label_path.exists():
        label = cv2.imread(str(label_path), cv2.IMREAD_GRAYSCALE)
        if label is not None and label.size > 0:
            return label

    full_path = TEMPLATE_DIR / MARCH_BTN
    if not full_path.exists():
        return None
    full = cv2.imread(str(full_path), cv2.IMREAD_GRAYSCALE)
    if full is None or full.size == 0:
        return None
    h, w = full.shape[:2]
    y1 = int(h * MARCH_LABEL_Y0_RATIO)
    y2 = int(h * MARCH_LABEL_Y1_RATIO)
    x1 = int(w * MARCH_LABEL_X0_RATIO)
    x2 = int(w * MARCH_LABEL_X1_RATIO)
    crop = full[y1:y2, x1:x2]
    return crop.copy() if crop.size else None


def find_march_button(
    screen: np.ndarray,
    *,
    threshold: float = MARCH_MATCH_THRESHOLD,
) -> MatchResult:
    """在右下角 ROI 匹配「出征」文字（忽略按钮上会变的体力数字）。"""
    label = _march_label_template()
    if label is None:
        return MatchResult(found=False)

    x1, y1, x2, y2 = MARCH_BTN_ROI
    h, w = screen.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return MatchResult(found=False)

    crop = screen[y1:y2, x1:x2]
    vision = Vision(TEMPLATE_DIR, threshold=threshold)
    result = vision.match_gray_multiscale(
        crop, label, scales=MARCH_SCALES, offset=(x1, y1)
    )
    if not result.found or result.center[1] < MARCH_MIN_Y:
        return MatchResult(found=False, confidence=result.confidence)

    # 标签中心偏上，点击点略下移到按钮主体
    cx, cy = result.center
    tap_cy = min(h - 8, cy + max(18, result.size[1]))
    return MatchResult(
        found=True,
        confidence=result.confidence,
        center=(cx, tap_cy),
        top_left=result.top_left,
        size=result.size,
    )

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


class StaminaInsufficientError(RuntimeError):
    """体力不足且未启用自动使用体力。"""


class DeployMarchHelper:
    def __init__(
        self,
        adb: AdbClient,
        *,
        formation_slot: int,
        use_stamina: bool,
        stamina_can_limit: int = DEFAULT_STAMINA_CAN_LIMIT,
        coords: dict[str, list[int]] | None = None,
        on_status: Callable[[str], None] | None = None,
        interrupted: Callable[[], bool] | None = None,
        step_delay: float = 1.5,
        stamina_budget: StaminaCanBudget | None = None,
    ):
        self.adb = adb
        self.formation_slot = formation_slot
        self.use_stamina = use_stamina
        self.coords = coords or {}
        self.on_status = on_status
        self._interrupted = interrupted or (lambda: False)
        self.step_delay = step_delay
        self.stamina_budget = stamina_budget or StaminaCanBudget(
            enabled=use_stamina, limit=stamina_can_limit
        )

    def _emit(self, message: str) -> None:
        logger.info(message)
        if self.on_status:
            self.on_status(message)

    def _tap_xy(self, x: int, y: int, delay: float | None = None) -> None:
        if self._interrupted():
            raise InterruptedError("任务已停止")
        self.adb.tap(x, y)
        time.sleep(delay if delay is not None else self.step_delay)

    def is_stamina_popup(self, screen) -> bool:
        """是否出现「获取更多」体力不足弹窗（ROI OCR）。"""
        return is_stamina_get_more_title(screen)

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

    def _get_stamina_use_tap_point(self, screen) -> tuple[int, int]:
        btn = self._find_stamina_use_button(screen)
        if btn.found:
            return btn.center
        if "stamina_use" in self.coords:
            return tuple(self.coords["stamina_use"])
        return STAMINA_USE_CENTER

    def use_stamina_items(self) -> None:
        # 领主体力行固定坐标（与冰原巨兽实测一致）；勿用错误的 stamina_use 兜底点
        use_stamina_cans_batch(
            self.adb,
            self.stamina_budget,
            tap_xy=STAMINA_USE_XY,
            emit=self._emit,
            interrupted=self._interrupted,
            close_with_back=True,
        )

    def is_deploy_screen(self, screen) -> bool:
        return find_march_button(screen).found

    def wait_for_deploy_screen(self, timeout: float = DEPLOY_WAIT_TIMEOUT) -> None:
        # 过场空等不占 ADB，再稀疏截图确认
        time.sleep(DEPLOY_WAIT_SETTLE_SEC)
        deadline = time.time() + timeout
        best_conf = 0.0
        while time.time() < deadline:
            if self._interrupted():
                raise InterruptedError("任务已停止")
            result = find_march_button(
                capture_screen(self.adb, MARCH_BTN_CAPTURE_ROI)
            )
            best_conf = max(best_conf, result.confidence)
            if result.found:
                return
            time.sleep(DEPLOY_WAIT_POLL_SEC)
        raise RuntimeError(
            f"未检测到出征界面（出征匹配最高 {best_conf:.2f}，阈值 {MARCH_MATCH_THRESHOLD}）"
        )

    def select_formation(self) -> None:
        slot = self.formation_slot
        if not (1 <= slot <= 8):
            raise RuntimeError(f"编队槽位 {slot} 无效，请填写 1~8")
        self._emit(f"选择编队槽位 {slot}…")
        self.wait_for_deploy_screen()
        sx, sy = FORMATION_SLOTS[slot]
        self._emit(f"点击编队槽位 {slot} @ ({sx},{sy})")
        self._tap_xy(sx, sy, delay=1.0)

    def _resolve_march_button(self) -> tuple[int, int, float, bool]:
        cx, cy = (
            tuple(self.coords["march"])
            if "march" in self.coords
            else MARCH_CENTER
        )
        matched = False
        match_conf = 0.0

        for attempt in range(MARCH_RESOLVE_ATTEMPTS):
            result = find_march_button(
                capture_screen(self.adb, MARCH_BTN_CAPTURE_ROI)
            )
            if result.found:
                cx, cy = result.center
                matched = True
                match_conf = result.confidence
                break
            time.sleep(MARCH_RESOLVE_POLL_SEC)
            logger.debug(
                f"等待出征按钮 {attempt + 1}/{MARCH_RESOLVE_ATTEMPTS}"
                f"（当前最高 {result.confidence:.2f}）"
            )

        return cx, cy, match_conf, matched

    def _click_march_button(self) -> None:
        cx, cy, match_conf, matched = self._resolve_march_button()
        if matched:
            self._emit(f"点击出征 @ ({cx},{cy})（匹配 {match_conf:.2f}）")
        else:
            self._emit(f"点击出征 @ ({cx},{cy})（固定坐标）")
        self._tap_xy(cx, cy, delay=0.6)

    def _wait_march_stamina_popup(self, delay: float = MARCH_OUTCOME_DELAY_SEC) -> bool:
        time.sleep(delay)
        if self._interrupted():
            raise InterruptedError("任务已停止")
        return self.is_stamina_popup(
            capture_screen(self.adb, STAMINA_POPUP_CAPTURE_ROI)
        )

    def tap_march(self) -> None:
        """出征；体力不足时按配置处理。"""
        self._click_march_button()
        time.sleep(0.8)
        if not self._wait_march_stamina_popup():
            self._emit("队伍已出征")
            return

        self._emit("检测到体力不足弹窗")
        if not self.use_stamina:
            raise StaminaInsufficientError("体力不足，任务结束")

        self._emit("自动使用领主体力道具…")
        self.use_stamina_items()
        time.sleep(0.8)
        self.wait_for_deploy_screen(timeout=DEPLOY_WAIT_TIMEOUT)
        self._click_march_button()
        time.sleep(0.8)
        if self._wait_march_stamina_popup():
            raise StaminaInsufficientError("使用体力后仍体力不足，任务结束")
        self._emit("队伍已出征")

    def handle_stamina_popup_if_any(self) -> bool:
        """检测体力弹窗。返回 True=已处理并应重新扫描；False=无弹窗。

        未启用体力时抛出 StaminaInsufficientError。
        """
        time.sleep(0.8)
        if not self.is_stamina_popup(
            capture_screen(self.adb, STAMINA_POPUP_CAPTURE_ROI)
        ):
            return False
        self._emit("检测到体力不足弹窗")
        if not self.use_stamina:
            raise StaminaInsufficientError("体力不足，任务结束")
        self.use_stamina_items()
        return True
