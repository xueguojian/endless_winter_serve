"""OpenCV 图像识别工具。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from loguru import logger


@dataclass
class MatchResult:
    found: bool
    confidence: float = 0.0
    center: tuple[int, int] = (0, 0)
    top_left: tuple[int, int] = (0, 0)
    size: tuple[int, int] = (0, 0)


class Vision:
    """基于模板匹配的界面识别。"""

    def __init__(self, template_dir: str | Path, threshold: float = 0.80):
        self.template_dir = Path(template_dir)
        self.threshold = threshold

    def _load_gray(self, path: Path) -> np.ndarray:
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"无法读取模板图片: {path}")
        return image

    def match_template(self, screen: np.ndarray, template_name: str) -> MatchResult:
        template_path = self.template_dir / template_name
        if not template_path.exists():
            logger.warning(f"模板不存在: {template_path}")
            return MatchResult(found=False)

        screen_gray = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
        template = self._load_gray(template_path)
        h, w = template.shape[:2]

        result = cv2.matchTemplate(screen_gray, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        if max_val < self.threshold:
            return MatchResult(found=False, confidence=float(max_val))

        x, y = max_loc
        return MatchResult(
            found=True,
            confidence=float(max_val),
            center=(x + w // 2, y + h // 2),
            top_left=(x, y),
            size=(w, h),
        )

    def find_any(self, screen: np.ndarray, template_names: list[str]) -> tuple[str, MatchResult] | None:
        best_name = ""
        best_result = MatchResult(found=False)

        for name in template_names:
            result = self.match_template(screen, name)
            if result.found and result.confidence > best_result.confidence:
                best_name = name
                best_result = result

        if best_result.found:
            return best_name, best_result
        return None

    def match_template_multiscale(
        self,
        screen: np.ndarray,
        template_name: str,
        scales: tuple[float, ...] = (0.85, 0.95, 1.0, 1.1, 1.25, 1.4, 1.55),
    ) -> MatchResult:
        """多尺度模板匹配，用于弹窗按钮等尺寸可能略有偏差的场景。"""
        template_path = self.template_dir / template_name
        if not template_path.exists():
            logger.warning(f"模板不存在: {template_path}")
            return MatchResult(found=False)

        return self.match_gray_multiscale(
            screen, self._load_gray(template_path), scales=scales
        )

    def match_gray_multiscale(
        self,
        screen: np.ndarray,
        template_gray: np.ndarray,
        scales: tuple[float, ...] = (0.85, 0.95, 1.0, 1.1, 1.25, 1.4, 1.55),
        *,
        offset: tuple[int, int] = (0, 0),
    ) -> MatchResult:
        """对灰度模板做多尺度匹配；offset 用于 ROI 裁图还原全图坐标。"""
        if template_gray is None or template_gray.size == 0:
            return MatchResult(found=False)

        if len(screen.shape) == 2:
            screen_gray = screen
        else:
            screen_gray = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
        ox, oy = offset
        best = MatchResult(found=False)

        for scale in scales:
            if scale == 1.0:
                template = template_gray
            else:
                template = cv2.resize(
                    template_gray,
                    None,
                    fx=scale,
                    fy=scale,
                    interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR,
                )
            th, tw = template.shape[:2]
            if th > screen_gray.shape[0] or tw > screen_gray.shape[1]:
                continue

            result = cv2.matchTemplate(screen_gray, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)

            if max_val > best.confidence:
                x, y = max_loc
                best = MatchResult(
                    found=max_val >= self.threshold,
                    confidence=float(max_val),
                    center=(ox + x + tw // 2, oy + y + th // 2),
                    top_left=(ox + x, oy + y),
                    size=(tw, th),
                )

        return best
