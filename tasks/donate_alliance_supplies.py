"""循环捐献联盟物资（暂时仅支持联盟永续）。"""

from __future__ import annotations

import threading
import time
from datetime import datetime
from typing import Callable

from loguru import logger

from core.adb_client import AdbClient
from core.navigation import WildernessNavigator

StatusCallback = Callable[[str], None]

# 720×1280 竖屏固定坐标
DEFAULT_COORDS: dict[str, list[int]] = {
    "alliance_open": [522, 1220],   # 主界面底部「联盟」按钮（城镇按钮左侧）
    "alliance_tech": [546, 938],    # 联盟主页「联盟科技」
    "perpetual_tech": [350, 1040],  # 联盟科技页「联盟永续」
    "donate_btn": [510, 1040],      # 捐献弹窗右下角蓝色「捐献」按钮（生肉 10000）
    "popup_close": [660, 195],      # 捐献弹窗右上角关闭
    "dialog_cancel": [250, 780],
}

DEFAULT_STEP_DELAY = 1.5
DEFAULT_DONATE_TIMES = 25
DEFAULT_DONATE_CLICK_DELAY = 0.35


def merge_task_config(cfg: dict) -> dict:
    # 代码默认坐标优先，避免实例 yaml 中陈旧坐标覆盖修复后的值
    coords = {**cfg.get("coords", {}), **DEFAULT_COORDS}
    return {
        "step_delay": cfg.get("step_delay", DEFAULT_STEP_DELAY),
        "donate_times": int(cfg.get("donate_times", DEFAULT_DONATE_TIMES)),
        "donate_click_delay": float(
            cfg.get("donate_click_delay", DEFAULT_DONATE_CLICK_DELAY)
        ),
        "coords": coords,
    }


class DonateAllianceSuppliesTask:
    """野外 → 联盟 → 联盟科技 → 联盟永续 → 捐献 N 次 → 回野外。"""

    def __init__(
        self,
        adb: AdbClient,
        coords: dict[str, list[int]] | None = None,
        interval: float = 3600.0,
        donate_times: int = DEFAULT_DONATE_TIMES,
        donate_click_delay: float = DEFAULT_DONATE_CLICK_DELAY,
        skip_hour: int = -1,
        step_delay: float = DEFAULT_STEP_DELAY,
        on_status: StatusCallback | None = None,
    ):
        merged = merge_task_config(
            {
                "coords": coords or {},
                "step_delay": step_delay,
                "donate_times": donate_times,
                "donate_click_delay": donate_click_delay,
            }
        )
        self.adb = adb
        self.coords = merged["coords"]
        self.interval = interval
        self.donate_times = merged["donate_times"]
        self.donate_click_delay = merged["donate_click_delay"]
        self.skip_hour = skip_hour
        self.step_delay = merged["step_delay"]
        self.on_status = on_status
        self._last_run = 0.0
        self._stop_event = threading.Event()
        self._wilderness = WildernessNavigator.from_task(self)

    @property
    def name(self) -> str:
        return "捐献联盟物资"

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
        if self.skip_hour >= 0 and datetime.now().hour == self.skip_hour:
            return False
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

    def _back(self, times: int = 1) -> None:
        for _ in range(times):
            if self._interrupted():
                raise InterruptedError("任务已停止")
            self.adb.back()
            time.sleep(0.5)

    def _ensure_wilderness(self) -> None:
        self._emit("确保在野外主界面…")
        self._wilderness.ensure_wilderness()

    def _return_to_wilderness(self) -> None:
        self._wilderness.try_return_to_wilderness()

    def run_donate_cycle(self) -> None:
        """野外 → 联盟 → 联盟科技 → 联盟永续 → 捐献 N 次 → 回野外。"""
        self._ensure_wilderness()

        self._emit("打开联盟")
        self._tap("alliance_open", delay=2.5)

        self._emit("进入联盟科技")
        tx, ty = self.coords["alliance_tech"]
        self._emit(f"点击联盟科技 ({tx}, {ty})")
        self._tap("alliance_tech", delay=2.0)

        self._emit("选择联盟永续")
        px, py = self.coords["perpetual_tech"]
        self._emit(f"点击联盟永续 ({px}, {py})")
        self._tap("perpetual_tech", delay=2.0)

        dx, dy = self.coords["donate_btn"]
        self._emit(f"开始捐献（共 {self.donate_times} 次），点击 ({dx}, {dy})")
        for i in range(1, self.donate_times + 1):
            if self._interrupted():
                raise InterruptedError("任务已停止")
            self._tap("donate_btn", delay=self.donate_click_delay)
            if i % 5 == 0 or i == self.donate_times:
                self._emit(f"已捐献 {i}/{self.donate_times} 次")

        self._emit("关闭捐献弹窗")
        if "popup_close" in self.coords:
            self._tap("popup_close", delay=0.8)
        else:
            self._back()

        self._return_to_wilderness()
        self._emit("捐献完成，已回到野外")

    def run_once(self, *, force: bool = False) -> bool:
        if not force and not self.should_run():
            return False

        self._last_run = time.time()
        try:
            self.run_donate_cycle()
            self._emit(f"本轮完成，{int(self.interval // 60)} 分钟后再次捐献")
            return True
        except InterruptedError:
            self._emit("任务已停止")
            raise
        except Exception as exc:
            logger.exception(f"[{self.name}] 执行失败")
            self._emit(f"执行失败：{exc}")
            self._return_to_wilderness()
            return False
