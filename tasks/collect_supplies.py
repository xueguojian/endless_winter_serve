"""循环领取探险挂机物资。"""

from __future__ import annotations

import threading
import time
from typing import Callable

from loguru import logger

from core.adb_client import AdbClient
from core.navigation import WildernessNavigator

StatusCallback = Callable[[str], None]

DEFAULT_COORDS: dict[str, list[int]] = {
    "explore_open": [90, 1220],
    "idle_reward_open": [615, 875],
    "idle_reward_confirm": [360, 920],
    "dialog_cancel": [250, 780],
}

DEFAULT_STEP_DELAY = 1.5
DEFAULT_INTERVAL = 5 * 3600  # 5 小时


def merge_task_config(cfg: dict) -> dict:
    coords = {**DEFAULT_COORDS, **cfg.get("coords", {})}
    return {
        "step_delay": cfg.get("step_delay", DEFAULT_STEP_DELAY),
        "coords": coords,
    }


class CollectSuppliesTask:
    """野外 → 探险 → 领取挂机收益 → 回野外。"""

    def __init__(
        self,
        adb: AdbClient,
        coords: dict[str, list[int]] | None = None,
        interval: float = DEFAULT_INTERVAL,
        step_delay: float = DEFAULT_STEP_DELAY,
        on_status: StatusCallback | None = None,
    ):
        merged = merge_task_config({"coords": coords or {}, "step_delay": step_delay})
        self.adb = adb
        self.coords = merged["coords"]
        self.interval = interval
        self.step_delay = merged["step_delay"]
        self.on_status = on_status
        self._last_run = 0.0
        self._stop_event = threading.Event()
        self._wilderness = WildernessNavigator.from_task(self)

    @property
    def name(self) -> str:
        return "一键领取探险物资"

    def stop(self) -> None:
        self._stop_event.set()

    def reset_stop(self) -> None:
        self._stop_event.clear()

    def _emit(self, message: str) -> None:
        logger.info(f"[{self.name}] {message}")
        if self.on_status:
            self.on_status(message)

    def _interrupted(self) -> bool:
        return self._stop_event.is_set()

    def should_run(self) -> bool:
        return time.time() - self._last_run >= self.interval

    def _tap(self, key: str, delay: float | None = None) -> None:
        if self._interrupted():
            raise InterruptedError("任务已停止")
        if key not in self.coords:
            raise KeyError(f"缺少坐标配置: {key}")
        x, y = self.coords[key]
        logger.debug(f"[{self.name}] 点击 {key} ({x}, {y})")
        self.adb.tap(x, y)
        time.sleep(delay if delay is not None else self.step_delay)

    def _return_to_wilderness(self) -> None:
        self._wilderness.try_return_to_wilderness()

    def execute(self) -> None:
        """野外 → 探险 → 领取挂机收益。"""
        self._emit("确保在野外主界面")
        self._wilderness.ensure_wilderness()

        self._emit("打开探险界面")
        self._tap("explore_open", delay=2.0)

        self._emit("打开挂机收益")
        self._tap("idle_reward_open", delay=2.0)

        self._emit("确认领取")
        self._tap("idle_reward_confirm", delay=1.5)
        self._emit("领取完成")

    def run_once(self, *, force: bool = False) -> bool:
        if not force and not self.should_run():
            return False

        self._last_run = time.time()
        try:
            self.execute()
            self._return_to_wilderness()
            hours = max(1, int(self.interval // 3600))
            self._emit(f"本轮完成，{hours} 小时后再次领取")
            return True
        except InterruptedError:
            self._emit("任务已停止")
            raise
        except Exception as exc:
            logger.exception(f"[{self.name}] 执行失败")
            self._emit(f"执行失败：{exc}")
            self._return_to_wilderness()
            return False
