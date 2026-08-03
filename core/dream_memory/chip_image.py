"""底栏槽位图像归一化（供 OCR 与模板匹配共用）。"""

from __future__ import annotations

import cv2
import numpy as np


def normalize_chip(chip_bgr: np.ndarray, *, width: int = 210, height: int = 54) -> np.ndarray:
    """将槽位裁切归一化为白字黑底二值图。"""
    if chip_bgr.size == 0:
        return np.zeros((height, width), dtype=np.uint8)
    h, w = chip_bgr.shape[:2]
    margin_x = max(2, int(w * 0.06))
    cropped = chip_bgr[:, margin_x : max(margin_x + 1, w - margin_x)]
    resized = cv2.resize(cropped, (width, height), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0]
    _, binary = cv2.threshold(l_channel, 170, 255, cv2.THRESH_BINARY)
    return binary
