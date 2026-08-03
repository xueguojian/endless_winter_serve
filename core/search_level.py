"""搜索面板：等级 OCR、固定 +/-、采矿「已满资源点」勾选。"""

from __future__ import annotations

import re
import time
from typing import Callable

import cv2
import numpy as np
from loguru import logger

from core.adb_client import AdbClient
from core.capture_roi import (
    FULL_RESOURCE_CAPTURE_ROI,
    SEARCH_LEVEL_CAPTURE_ROI,
    capture_screen,
)
from core.dream_memory.ocr_engine import ocr_chip_text

# 等级数字白框（720×1280）
LEVEL_NUM_ROI = (584, 1030, 634, 1072)
LEVEL_MINUS = (66, 1050)
LEVEL_PLUS = (482, 1048)
LEVEL_MIN = 1
LEVEL_MAX = 30

# 「搜索资源为满的资源点」多选框中心
FULL_RESOURCE_CHECK_CENTER = (214, 1136)
FULL_RESOURCE_CHECK_HALF = 12


def read_search_level(screen: np.ndarray) -> int | None:
    """OCR 读取搜索面板右侧白框中的等级数字。"""
    x1, y1, x2, y2 = LEVEL_NUM_ROI
    h, w = screen.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return None

    crop = screen[y1:y2, x1:x2]
    # 放大 + 对比度增强，便于识别小号白底黑字
    big = cv2.resize(crop, None, fx=4.0, fy=4.0, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # OCR 吃 BGR；白底黑字更常见
    if binary.mean() < 127:
        binary = 255 - binary
    chip = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    text, engine = ocr_chip_text(chip)
    digits = re.findall(r"\d+", text or "")
    logger.debug(f"搜索等级 OCR({engine}) raw={text!r} digits={digits}")
    if not digits:
        # 再试原色彩图放大
        text2, engine2 = ocr_chip_text(big)
        digits = re.findall(r"\d+", text2 or "")
        logger.debug(f"搜索等级 OCR 回退({engine2}) raw={text2!r} digits={digits}")
    if not digits:
        return None

    level = int(digits[0])
    if LEVEL_MIN <= level <= LEVEL_MAX:
        return level
    return None


def tap_level_minus(
    adb: AdbClient,
    *,
    delay: float = 0.35,
    interrupted: Callable[[], bool] | None = None,
) -> None:
    if interrupted and interrupted():
        raise InterruptedError("任务已停止")
    adb.tap(*LEVEL_MINUS)
    time.sleep(delay)


def tap_level_plus(
    adb: AdbClient,
    *,
    delay: float = 0.35,
    interrupted: Callable[[], bool] | None = None,
) -> None:
    if interrupted and interrupted():
        raise InterruptedError("任务已停止")
    adb.tap(*LEVEL_PLUS)
    time.sleep(delay)


def adjust_search_level(
    adb: AdbClient,
    target: int,
    *,
    emit: Callable[[str], None] | None = None,
    interrupted: Callable[[], bool] | None = None,
) -> None:
    """将搜索面板等级调到 target；识别失败则抛错，避免默默用错误等级出征。"""
    target = max(LEVEL_MIN, min(LEVEL_MAX, int(target)))

    def _emit(msg: str) -> None:
        logger.info(msg)
        if emit:
            emit(msg)

    screen = capture_screen(adb, SEARCH_LEVEL_CAPTURE_ROI)
    current = read_search_level(screen)
    if current == target:
        _emit(f"等级已是 {target}，无需调整")
        return
    if current is None:
        raise RuntimeError(
            f"无法识别当前搜索等级（ROI={LEVEL_NUM_ROI}），已中止以免按错误等级搜索"
        )

    diff = target - current
    _emit(f"调整等级：{current} → {target}")
    for _ in range(abs(diff)):
        if interrupted and interrupted():
            raise InterruptedError("任务已停止")
        if diff > 0:
            tap_level_plus(adb, interrupted=interrupted)
        else:
            tap_level_minus(adb, interrupted=interrupted)
        now = read_search_level(capture_screen(adb, SEARCH_LEVEL_CAPTURE_ROI))
        if now == target:
            _emit(f"已调整到 {target} 级")
            return
        if now is None:
            logger.warning("调等级后 OCR 暂时失败，继续点击")

    final = read_search_level(capture_screen(adb, SEARCH_LEVEL_CAPTURE_ROI))
    if final != target:
        raise RuntimeError(
            f"未能将等级调到 {target}（当前识别为 {final}），请检查 +/- 坐标或 OCR"
        )
    _emit(f"已调整到 {target} 级")


def _full_resource_checkbox_roi(
    screen: np.ndarray,
) -> tuple[int, int, int, int]:
    cx, cy = FULL_RESOURCE_CHECK_CENTER
    half = FULL_RESOURCE_CHECK_HALF
    h, w = screen.shape[:2]
    return (
        max(0, cx - half),
        max(0, cy - half),
        min(w, cx + half),
        min(h, cy + half),
    )


def is_full_resource_checked(screen: np.ndarray) -> bool:
    """判断「搜索资源为满的资源点」是否已勾选。

    未勾选：深蓝空框；勾选：框内出现亮绿色对勾。
    """
    x1, y1, x2, y2 = _full_resource_checkbox_roi(screen)
    crop = screen[y1:y2, x1:x2]
    if crop.size == 0:
        return False
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    # 绿色对勾：H≈35–95，较高饱和度与亮度
    green = cv2.inRange(hsv, (35, 80, 80), (95, 255, 255))
    green_ratio = float(green.mean()) / 255.0
    checked = green_ratio >= 0.05
    logger.debug(f"满资源勾选 green={green_ratio:.3f} → {checked}")
    return checked


def ensure_full_resource_checked(
    adb: AdbClient,
    *,
    emit: Callable[[str], None] | None = None,
    interrupted: Callable[[], bool] | None = None,
) -> None:
    """未勾选「搜索资源为满的资源点」时点击勾选。"""
    if interrupted and interrupted():
        raise InterruptedError("任务已停止")
    screen = capture_screen(adb, FULL_RESOURCE_CAPTURE_ROI)
    if is_full_resource_checked(screen):
        if emit:
            emit("已勾选「搜索已满资源点」")
        return
    cx, cy = FULL_RESOURCE_CHECK_CENTER
    if emit:
        emit(f"勾选「搜索已满资源点」@ ({cx},{cy})")
    adb.tap(cx, cy)
    time.sleep(0.4)
