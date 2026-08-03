"""供任务线程使用的虚拟 ADB：动作在客户端执行，服务端只做识别/决策。"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from core.coords import PORTRAIT_HEIGHT, PORTRAIT_WIDTH


@dataclass
class ClientCommand:
    actions: list[dict[str, Any]] = field(default_factory=list)
    need_screenshot: bool = False
    # 非空时客户端只上传该 ROI（x1,y1,x2,y2），服务端再贴回整屏画布
    screenshot_roi: list[int] | None = None
    status: str = ""
    running: bool = True
    error: str | None = None
    done: bool = False


class ProxyAdb:
    """与 AdbClient 关键接口对齐。

    tap/swipe/back 先入队；sleep/screenshot 时冲刷并等待客户端完成本轮。
    """

    def __init__(self, touch_width: int = 720, touch_height: int = 1280):
        self.touch_width = touch_width
        self.touch_height = touch_height
        self.address = "proxy://client"
        self._cv = threading.Condition()
        self._pending: list[dict[str, Any]] = []
        self._frame: np.ndarray | None = None
        self._frame_gen = 0
        self._out_cmd: ClientCommand | None = None
        self._cmd_seq = 0
        self._acked_seq = 0
        self._closed = False
        self._status = ""
        self._error: str | None = None
        self._awaiting_roi: tuple[int, int, int, int] | None = None
        self._next_roi: tuple[int, int, int, int] | None = None

    def set_status(self, text: str) -> None:
        with self._cv:
            self._status = text or ""

    def close(self, error: str | None = None) -> None:
        with self._cv:
            self._closed = True
            self._error = error
            if self._out_cmd is None:
                self._out_cmd = ClientCommand(
                    actions=list(self._pending),
                    need_screenshot=False,
                    status=self._status,
                    running=False,
                    error=error,
                    done=True,
                )
                self._pending.clear()
            self._cv.notify_all()

    def mark_done(self) -> None:
        with self._cv:
            self._closed = True
            self._out_cmd = ClientCommand(
                actions=list(self._pending),
                need_screenshot=False,
                status=self._status or "本轮完成",
                running=False,
                error=self._error,
                done=True,
            )
            self._pending.clear()
            self._cv.notify_all()

    def _paste_roi_frame(self, crop: np.ndarray) -> np.ndarray:
        roi = self._awaiting_roi
        if roi is None:
            return crop
        x1, y1, x2, y2 = roi
        canvas = np.zeros((PORTRAIT_HEIGHT, PORTRAIT_WIDTH, 3), dtype=np.uint8)
        ch, cw = crop.shape[:2]
        # 允许 JPEG 轻微尺寸偏差，按 ROI 框贴入
        rh, rw = max(1, y2 - y1), max(1, x2 - x1)
        if (ch, cw) != (rh, rw):
            import cv2

            crop = cv2.resize(crop, (rw, rh), interpolation=cv2.INTER_AREA)
            ch, cw = crop.shape[:2]
        y2e = min(PORTRAIT_HEIGHT, y1 + ch)
        x2e = min(PORTRAIT_WIDTH, x1 + cw)
        canvas[y1:y2e, x1:x2e] = crop[: y2e - y1, : x2e - x1]
        return canvas

    def push_client_result(self, frame: np.ndarray | None) -> ClientCommand:
        """客户端 tick：提交截图或 ACK，取回下一批动作。"""
        with self._cv:
            if frame is not None:
                if self._awaiting_roi is not None:
                    self._frame = self._paste_roi_frame(frame)
                    self._awaiting_roi = None
                else:
                    # 整图：若尺寸不对则尽量缩放
                    h, w = frame.shape[:2]
                    if (h, w) != (PORTRAIT_HEIGHT, PORTRAIT_WIDTH):
                        import cv2

                        frame = cv2.resize(
                            frame,
                            (PORTRAIT_WIDTH, PORTRAIT_HEIGHT),
                            interpolation=cv2.INTER_AREA,
                        )
                    self._frame = frame
                self._frame_gen += 1
            self._acked_seq = self._cmd_seq
            self._cv.notify_all()

            deadline = time.time() + 180.0
            while self._out_cmd is None:
                if self._closed:
                    return ClientCommand(
                        status=self._status,
                        running=False,
                        error=self._error,
                        done=True,
                    )
                remaining = deadline - time.time()
                if remaining <= 0:
                    return ClientCommand(
                        status=self._status,
                        running=True,
                        error="等待服务端指令超时",
                        done=False,
                    )
                self._cv.wait(timeout=min(1.0, remaining))

            cmd = self._out_cmd
            self._out_cmd = None
            return cmd

    def _flush(self, *, need_screenshot: bool) -> None:
        with self._cv:
            if self._closed:
                raise RuntimeError(self._error or "会话已结束")

            start_gen = self._frame_gen
            self._cmd_seq += 1
            my_seq = self._cmd_seq
            roi = self._next_roi if need_screenshot else None
            self._next_roi = None
            if need_screenshot:
                self._awaiting_roi = roi
            self._out_cmd = ClientCommand(
                actions=list(self._pending),
                need_screenshot=need_screenshot,
                screenshot_roi=list(roi) if roi else None,
                status=self._status,
                running=True,
                error=None,
                done=False,
            )
            self._pending.clear()
            self._cv.notify_all()

            while self._acked_seq < my_seq and not self._closed:
                self._cv.wait(timeout=1.0)
            if self._closed and self._acked_seq < my_seq:
                raise RuntimeError(self._error or "会话已结束")

            if need_screenshot:
                while self._frame_gen <= start_gen and not self._closed:
                    self._cv.wait(timeout=1.0)
                if self._frame is None:
                    raise RuntimeError("未收到客户端截图")

    def screenshot(
        self, roi: tuple[int, int, int, int] | None = None
    ) -> np.ndarray:
        """截图。roi=(x1,y1,x2,y2) 时只让客户端上传该区域。"""
        with self._cv:
            self._next_roi = tuple(int(v) for v in roi) if roi else None
        self._flush(need_screenshot=True)
        with self._cv:
            assert self._frame is not None
            return self._frame.copy()

    def tap(self, x: int, y: int) -> None:
        x = max(0, min(int(x), self.touch_width - 1))
        y = max(0, min(int(y), self.touch_height - 1))
        with self._cv:
            if self._closed:
                raise RuntimeError(self._error or "会话已结束")
            self._pending.append({"op": "tap", "x": x, "y": y})

    def swipe(
        self, x1: int, y1: int, x2: int, y2: int, duration_ms: int = 300
    ) -> None:
        with self._cv:
            if self._closed:
                raise RuntimeError(self._error or "会话已结束")
            self._pending.append(
                {
                    "op": "swipe",
                    "x1": int(x1),
                    "y1": int(y1),
                    "x2": int(x2),
                    "y2": int(y2),
                    "ms": int(duration_ms),
                }
            )

    def back(self) -> None:
        with self._cv:
            if self._closed:
                raise RuntimeError(self._error or "会话已结束")
            self._pending.append({"op": "back"})

    def sleep(self, seconds: float) -> None:
        ms = max(0, int(float(seconds) * 1000))
        with self._cv:
            if self._closed:
                raise RuntimeError(self._error or "会话已结束")
            if ms > 0:
                self._pending.append({"op": "sleep", "ms": ms})
        self._flush(need_screenshot=False)
