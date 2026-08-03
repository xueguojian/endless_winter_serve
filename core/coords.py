"""屏幕坐标约定（720×1280 竖屏）。

雷电「物理分辨率」wm size 常为 1280×720（窗口横屏），但 ADB 截图与 input tap
均使用游戏竖屏坐标系：宽 720、高 1280，即 numpy 数组 shape 为 (1280, 720)。

标定模板时：
- 一律在 adb.screenshot() 返回的竖屏图上，用同一套坐标裁剪
- 不要使用横屏截图的 x/y，也不要对截图做 l2t 之类的横竖屏换算
"""

from __future__ import annotations

import numpy as np

PORTRAIT_WIDTH = 720
PORTRAIT_HEIGHT = 1280


def crop_rect(screen: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    """按竖屏坐标 [x1,y1,x2,y2] 裁剪（右下为开区间）。"""
    h, w = screen.shape[:2]
    x1, x2 = max(0, x1), min(w, x2)
    y1, y2 = max(0, y1), min(h, y2)
    return screen[y1:y2, x1:x2].copy()


def crop_center(screen: np.ndarray, cx: int, cy: int, width: int, height: int) -> np.ndarray:
    """以中心点 (cx,cy) 裁剪 width×height 区域。"""
    x1 = cx - width // 2
    y1 = cy - height // 2
    return crop_rect(screen, x1, y1, x1 + width, y1 + height)


def center_of_box(x1: int, y1: int, x2: int, y2: int) -> tuple[int, int]:
    return (x1 + x2) // 2, (y1 + y2) // 2
