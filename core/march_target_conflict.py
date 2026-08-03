"""出征同目标冲突弹窗：他人已打同一巨兽时的确认框。"""

from __future__ import annotations

import re

import cv2
import numpy as np
from loguru import logger

from core.dream_memory.ocr_engine import ocr_chip_text

# 弹窗正文区域（720×1280），用户标定
SAME_TARGET_TEXT_ROI = (86, 544, 630, 676)
# 左侧橙色「取消」按钮大致中心
SAME_TARGET_CANCEL_XY = (220, 800)
# 云控截图：覆盖体力标题 + 冲突正文 + 取消按钮
SAME_TARGET_CAPTURE_ROI = (50, 80, 720, 880)

SAME_TARGET_KEYWORDS = (
    "其他队伍与您的出征目标相同",
    "出征目标相同",
    "依然要发兵",
)


class SameTargetConflictError(RuntimeError):
    """出征时出现「其他队伍目标相同」弹窗，已点取消，应立刻重开一轮搜索。"""


def _normalize(text: str) -> str:
    return re.sub(r"\s+", "", (text or "").strip())


def is_same_target_conflict_dialog(screen: np.ndarray) -> bool:
    """ROI OCR：识别到同目标冲突文案则返回 True。"""
    x1, y1, x2, y2 = SAME_TARGET_TEXT_ROI
    h, w = screen.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        logger.info(
            f"同目标冲突 OCR: ROI 无效 ({x1},{y1},{x2},{y2}) size={w}x{h}"
        )
        return False

    crop = screen[y1:y2, x1:x2]
    big = cv2.resize(crop, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    text, engine = ocr_chip_text(big)
    normalized = _normalize(text)
    hit = any(key in normalized for key in SAME_TARGET_KEYWORDS) or (
        "其他队伍" in normalized and "出征目标" in normalized
    ) or ("依然" in normalized and "发兵" in normalized)
    logger.info(
        f"同目标冲突 OCR: text={text!r} normalized={normalized!r} "
        f"engine={engine} hit={hit} ROI=({x1},{y1},{x2},{y2})"
    )
    return hit
