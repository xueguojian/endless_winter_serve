"""供任务线程使用的虚拟 ADB：动作在客户端执行，服务端只做识别/决策。"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from core.coords import PORTRAIT_HEIGHT, PORTRAIT_WIDTH

# 与客户端 tick HTTP 超时对齐；服务端持有连接等待下一指令时上限
_TICK_WAIT_SEC = 45.0


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
    # True：客户端应终止整个任务大循环（如体力不足未勾选自动补体）
    abort_loop: bool = False


class ProxyAdb:
    """与 AdbClient 关键接口对齐。

    tap/swipe/back 先入队；sleep/screenshot 时冲刷并等待客户端完成本轮。

    客户端时序：上传帧 → 取回动作 → 执行 →（若 need_screenshot）再截图上传。
    ACK 必须在「取走动作」之后，截图结果必须是动作之后的新帧。
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
        self._abort_loop = False
        self._awaiting_roi: tuple[int, int, int, int] | None = None
        self._next_roi: tuple[int, int, int, int] | None = None

    def set_status(self, text: str) -> None:
        with self._cv:
            self._status = text or ""

    def close(self, error: str | None = None, *, abort_loop: bool = False) -> None:
        with self._cv:
            self._closed = True
            self._error = error
            if abort_loop:
                self._abort_loop = True
            if self._out_cmd is None:
                self._out_cmd = ClientCommand(
                    actions=list(self._pending),
                    need_screenshot=False,
                    status=self._status,
                    running=False,
                    error=error,
                    done=True,
                    abort_loop=self._abort_loop,
                )
                self._pending.clear()
            self._cv.notify_all()

    def mark_done(self, *, abort_loop: bool = False) -> None:
        with self._cv:
            self._closed = True
            if abort_loop:
                self._abort_loop = True
            self._out_cmd = ClientCommand(
                actions=list(self._pending),
                need_screenshot=False,
                status=self._status or "本轮完成",
                running=False,
                error=self._error,
                done=True,
                abort_loop=self._abort_loop,
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

    def _accept_frame(self, frame: np.ndarray) -> None:
        if self._awaiting_roi is not None:
            self._frame = self._paste_roi_frame(frame)
            self._awaiting_roi = None
        else:
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

    def push_client_result(self, frame: np.ndarray | None) -> ClientCommand:
        """客户端 tick：先提交截图，再取走下一批动作（取走后才 ACK）。"""
        with self._cv:
            if frame is not None:
                self._accept_frame(frame)
                self._cv.notify_all()

            deadline = time.time() + _TICK_WAIT_SEC
            while self._out_cmd is None:
                if self._closed:
                    return ClientCommand(
                        status=self._status,
                        running=False,
                        error=self._error,
                        done=True,
                        abort_loop=self._abort_loop,
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
            # 必须在取走指令后 ACK，否则下一轮 flush 会以为已 ACK 而死锁
            self._acked_seq = self._cmd_seq
            self._cv.notify_all()
            return cmd

    def _flush(self, *, need_screenshot: bool) -> None:
        """冲刷动作队列；需要截图时只接受「动作执行之后」上传的新帧。"""
        with self._cv:
            if self._closed:
                raise RuntimeError(self._error or "会话已结束")

            self._cmd_seq += 1
            my_seq = self._cmd_seq
            roi = self._next_roi if need_screenshot else None
            self._next_roi = None
            # 等客户端取走动作后再挂 ROI，避免取指令同一次带来的旧帧被误贴
            self._awaiting_roi = None
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
                self._awaiting_roi = roi
                # 取指令之前/之时的帧都是动作前画面，必须再等执行后的新帧
                ack_gen = self._frame_gen
                deadline = time.time() + _TICK_WAIT_SEC
                while self._frame_gen <= ack_gen and not self._closed:
                    remaining = deadline - time.time()
                    if remaining <= 0:
                        break
                    self._cv.wait(timeout=min(1.0, remaining))
                if self._frame is None or self._frame_gen <= ack_gen:
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

    def queue_sleep(self, seconds: float) -> None:
        """仅入队 sleep，不 flush；与后续 tap/sleep 合并成一批下发。"""
        ms = max(0, int(float(seconds) * 1000))
        with self._cv:
            if self._closed:
                raise RuntimeError(self._error or "会话已结束")
            if ms > 0:
                self._pending.append({"op": "sleep", "ms": ms})

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
