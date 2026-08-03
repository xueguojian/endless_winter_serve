"""领主体力道具：批量点击 + 罐头用量上限；体力不足弹窗标题 OCR。"""

from __future__ import annotations

import re
import time
from typing import Callable

import cv2
import numpy as np
from loguru import logger

from core.adb_client import AdbClient
from core.capture_roi import STAMINA_POPUP_CAPTURE_ROI, capture_screen
from core.dream_memory.ocr_engine import ocr_chip_text

# 体力不足弹窗一次连点次数（减少反复弹窗）
STAMINA_BATCH_CLICKS = 20
DEFAULT_STAMINA_CAN_LIMIT = 800
STAMINA_USE_XY = (576, 522)
STAMINA_TAP_INTERVAL = 0.12
STAMINA_AFTER_BATCH_DELAY = 0.5

# 「获取更多」标题区域（720×1280）；出征后延迟再 OCR 一次
STAMINA_TITLE_ROI = (242, 100, 494, 162)
STAMINA_TITLE_KEYWORDS = ("获取更多",)
MARCH_OUTCOME_DELAY_SEC = 2.5


def _normalize_stamina_title(text: str) -> str:
    return re.sub(r"\s+", "", (text or "").strip())


def is_stamina_get_more_title(screen: np.ndarray) -> bool:
    """ROI 内 OCR：识别到「获取更多」则判定为体力不足弹窗。"""
    x1, y1, x2, y2 = STAMINA_TITLE_ROI
    h, w = screen.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        logger.info(
            f"体力弹窗 OCR: ROI 无效 ({x1},{y1},{x2},{y2}) size={w}x{h}"
        )
        return False

    crop = screen[y1:y2, x1:x2]
    big = cv2.resize(crop, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
    text, engine = ocr_chip_text(big)
    normalized = _normalize_stamina_title(text)
    hit = any(key in normalized for key in STAMINA_TITLE_KEYWORDS) or (
        "获取" in normalized and "更多" in normalized
    )
    logger.info(
        f"体力弹窗 OCR: text={text!r} normalized={normalized!r} "
        f"engine={engine} hit={hit} ROI=({x1},{y1},{x2},{y2})"
    )
    return hit


class StaminaCanLimitReached(InterruptedError):
    """已用完配置的体力罐头数量，应终止任务循环。"""


class StaminaCanBudget:
    """勾选「使用体力」后生效：累计点击次数达到上限则停止。"""

    def __init__(self, *, enabled: bool, limit: int = DEFAULT_STAMINA_CAN_LIMIT):
        self.enabled = bool(enabled)
        self.limit = max(1, int(limit))
        self.used = 0

    @property
    def remaining(self) -> int:
        if not self.enabled:
            return STAMINA_BATCH_CLICKS
        return max(0, self.limit - self.used)

    def record_click(self) -> None:
        if not self.enabled:
            return
        self.used += 1
        if self.used >= self.limit:
            raise StaminaCanLimitReached(
                f"体力罐头已达上限 {self.used}/{self.limit}，停止循环"
            )


def use_stamina_cans_batch(
    adb: AdbClient,
    budget: StaminaCanBudget,
    *,
    tap_xy: tuple[int, int] = STAMINA_USE_XY,
    emit: Callable[[str], None] | None = None,
    interrupted: Callable[[], bool] | None = None,
    close_with_back: bool = True,
) -> int:
    """在体力弹窗内连点使用道具。返回本次实际点击次数。"""
    if interrupted and interrupted():
        raise InterruptedError("任务已停止")

    clicks = min(STAMINA_BATCH_CLICKS, budget.remaining)
    if clicks <= 0:
        raise StaminaCanLimitReached(
            f"体力罐头已达上限 {budget.used}/{budget.limit}，停止循环"
        )

    cx, cy = tap_xy
    msg = (
        f"使用领主体力 ×{clicks} @ ({cx},{cy})"
        f"（本次前累计 {budget.used}/{budget.limit if budget.enabled else '∞'}）"
    )
    logger.info(msg)
    if emit:
        emit(msg)

    for index in range(clicks):
        if interrupted and interrupted():
            raise InterruptedError("任务已停止")
        adb.tap(cx, cy)
        budget.record_click()
        time.sleep(STAMINA_TAP_INTERVAL)
        if index == clicks - 1:
            break

    time.sleep(STAMINA_AFTER_BATCH_DELAY)
    if close_with_back:
        if emit:
            emit("按返回键关闭弹窗")
        adb.back()
        time.sleep(0.8)
    return clicks
