"""寻梦记忆：伪随机误点（约每 N 次正常点击插入一次）。"""

from __future__ import annotations

import random


class PseudoRandomMisclickScheduler:
    """计数式误点调度：两次误点之间至少间隔 interval_min 次正常点击。"""

    def __init__(
        self,
        *,
        interval_min: int = 8,
        interval_max: int = 12,
        center_x: int = 360,
        center_y: int = 640,
        radius_x: int = 90,
        radius_y: int = 120,
    ) -> None:
        self.interval_min = max(1, interval_min)
        self.interval_max = max(self.interval_min, interval_max)
        self.center_x = center_x
        self.center_y = center_y
        self.radius_x = max(1, radius_x)
        self.radius_y = max(1, radius_y)
        self._clicks_since_misclick = 0
        self._next_interval = self._roll_interval()

    def _roll_interval(self) -> int:
        return random.randint(self.interval_min, self.interval_max)

    def on_normal_click(self) -> bool:
        """记录一次正常点击；若返回 True，应在此时插入一次误点。"""
        self._clicks_since_misclick += 1
        if self._clicks_since_misclick < self._next_interval:
            return False
        self._clicks_since_misclick = 0
        self._next_interval = self._roll_interval()
        return True

    def register_normal_clicks(self, count: int) -> bool:
        """记录多次正常点击，返回是否应在本批结束后误点一次。"""
        due = False
        for _ in range(max(0, count)):
            if self.on_normal_click():
                due = True
        return due

    def sample_point(self) -> tuple[int, int]:
        dx = random.randint(-self.radius_x, self.radius_x)
        dy = random.randint(-self.radius_y, self.radius_y)
        return self.center_x + dx, self.center_y + dy
