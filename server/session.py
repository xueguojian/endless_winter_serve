"""用户任务会话：在服务端跑状态机，动作用 ProxyAdb 下发。"""

from __future__ import annotations

import threading
import time
import traceback
from typing import Any, Callable

import cv2
import numpy as np
from loguru import logger

from server.config import merge_task_config
from server.proxy_adb import ClientCommand, ProxyAdb

SUPPORTED_TASKS = ("hunt_ice_beast", "hunt_monster", "auto_lighthouse", "auto_mining")
# 共用搜索面板 tab 拖动记忆
_SEARCH_TAB_TASKS = frozenset({"hunt_ice_beast", "hunt_monster"})


class UserSession:
    def __init__(self, username: str, device_id: str):
        self.username = username
        self.device_id = device_id
        self._lock = threading.RLock()
        self._proxy: ProxyAdb | None = None
        self._task = None
        self._worker: threading.Thread | None = None
        self._task_id: str | None = None
        self._running = False
        # 巨兽跨轮次状态（云控每轮新建任务实例，需会话级记忆）
        self._hunt_tab_bar_scrolled = False
        self._hunt_level_adjusted = False

    @property
    def busy(self) -> bool:
        return self._running

    def start_task(
        self,
        task_id: str,
        override: dict[str, Any] | None = None,
        on_status: Callable[[str], None] | None = None,
    ) -> None:
        if task_id not in SUPPORTED_TASKS:
            raise ValueError(f"不支持的任务: {task_id}")
        with self._lock:
            if self._running:
                raise RuntimeError("已有任务在运行，请先停止")
            # 中途切到非搜索类任务后，游戏 UI 可能已复位，下次搜索需重新拖 tab
            if task_id not in _SEARCH_TAB_TASKS:
                self._hunt_tab_bar_scrolled = False
                self._hunt_level_adjusted = False
            cfg = merge_task_config(task_id, override)
            proxy = ProxyAdb()
            task = self._build_task(task_id, cfg, proxy, on_status)
            self._proxy = proxy
            self._task = task
            self._task_id = task_id
            self._running = True
            self._worker = threading.Thread(
                target=self._run_task_thread,
                args=(task, proxy),
                name=f"task-{self.username}-{task_id}",
                daemon=True,
            )
            self._worker.start()

    def stop_task(self) -> None:
        with self._lock:
            task = self._task
            proxy = self._proxy
        if task is not None:
            try:
                task.stop()
            except Exception:
                pass
        if proxy is not None:
            proxy.close(error="用户停止")
        if self._worker is not None and self._worker.is_alive():
            self._worker.join(timeout=3.0)
        with self._lock:
            self._running = False

    def tick(self, jpeg_bytes: bytes | None) -> dict[str, Any]:
        proxy = self._proxy
        if proxy is None:
            return {
                "running": False,
                "done": True,
                "need_screenshot": False,
                "actions": [],
                "status": "无任务",
                "error": None,
                "abort_loop": False,
            }
        frame = None
        if jpeg_bytes:
            frame = self._decode_jpeg(jpeg_bytes)
        cmd: ClientCommand = proxy.push_client_result(frame)
        return {
            "running": cmd.running and not cmd.done,
            "done": cmd.done,
            "need_screenshot": cmd.need_screenshot,
            "screenshot_roi": cmd.screenshot_roi,
            "actions": cmd.actions,
            "status": cmd.status,
            "error": cmd.error,
            "abort_loop": bool(cmd.abort_loop),
            "task_id": self._task_id,
        }

    def _run_task_thread(self, task: Any, proxy: ProxyAdb) -> None:
        import time as time_mod

        original_sleep = time_mod.sleep
        task_ident = threading.get_ident()

        def patched_sleep(seconds: float) -> None:
            if threading.get_ident() != task_ident:
                original_sleep(seconds)
                return
            try:
                proxy.sleep(float(seconds))
            except RuntimeError:
                return

        time_mod.sleep = patched_sleep
        try:
            proxy.set_status(f"开始执行 {getattr(task, 'name', self._task_id)}")
            ok = bool(task.run_once(force=True))
            proxy.set_status("完成" if ok else "本轮未成功")
            proxy.mark_done()
        except InterruptedError as exc:
            # 体力不足 / 罐头上限 / 主动停止：干净结束，并通知客户端终止大循环
            msg = str(exc).strip() or "任务已停止"
            logger.info("任务中断: {}", msg)
            proxy.set_status(msg)
            proxy.mark_done(abort_loop=True)
        except Exception as exc:
            logger.exception("任务执行异常: {}", exc)
            proxy.set_status("任务异常")
            proxy.close(error=f"{exc}\n{traceback.format_exc(limit=3)}")
        finally:
            time_mod.sleep = original_sleep
            with self._lock:
                # 无论成败，记下本轮是否已拖过 tab / 调过等级
                if self._task_id in _SEARCH_TAB_TASKS and task is not None:
                    self._hunt_tab_bar_scrolled = bool(
                        getattr(task, "_tab_bar_already_scrolled", False)
                    )
                    self._hunt_level_adjusted = bool(
                        getattr(task, "_level_already_adjusted", False)
                    )
                self._running = False

    @staticmethod
    def _decode_jpeg(data: bytes) -> np.ndarray:
        arr = np.frombuffer(data, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("截图解码失败")
        return frame

    def _build_task(
        self,
        task_id: str,
        cfg: dict[str, Any],
        proxy: ProxyAdb,
        on_status: Callable[[str], None] | None,
    ) -> Any:
        def _status(msg: str) -> None:
            proxy.set_status(msg)
            if on_status:
                on_status(msg)

        coords = dict(cfg.get("coords") or {})
        if task_id == "hunt_ice_beast":
            from core.common_task_opts import resolve_formation_slot
            from tasks.hunt_ice_beast import HuntIceBeastTask

            use_formation, formation_slot = resolve_formation_slot(
                cfg.get("formation_name", cfg.get("formation_slot", 6)),
                default_slot=6,
            )
            use_formation, formation_slot = resolve_formation_slot(
                cfg.get("formation_name", cfg.get("formation_slot", 6)),
                default_slot=6,
            )
            return HuntIceBeastTask(
                adb=proxy,
                coords=coords,
                interval=float(cfg.get("interval") or 120),
                beast_level=int(cfg.get("beast_level") or 8),
                default_beast_level=int(cfg.get("default_beast_level") or 1),
                formation_name=str(formation_slot),
                rally_duration_minutes=int(cfg.get("rally_duration_minutes") or 5),
                skip_hour=int(cfg.get("skip_hour") or -1),
                step_delay=float(cfg.get("step_delay") or 1.5),
                use_stamina=bool(cfg.get("use_stamina", False)),
                stamina_can_limit=int(cfg.get("stamina_can_limit") or 800),
                use_formation=use_formation,
                adjust_level=bool(cfg.get("adjust_level", False)),
                beast_icon_index=int(cfg.get("beast_icon_index") or 0),
                tab_bar_already_scrolled=self._hunt_tab_bar_scrolled,
                level_already_adjusted=self._hunt_level_adjusted,
                on_status=_status,
            )
        if task_id == "hunt_monster":
            from core.common_task_opts import resolve_formation_slot
            from tasks.hunt_monster import HuntMonsterTask

            use_formation, formation_slot = resolve_formation_slot(
                cfg.get("formation_name", cfg.get("formation_slot", 6)),
                default_slot=6,
            )
            return HuntMonsterTask(
                adb=proxy,
                coords=coords,
                interval=float(cfg.get("interval") or 60),
                monster_level=int(cfg.get("monster_level") or 30),
                formation_name=str(formation_slot),
                skip_hour=int(cfg.get("skip_hour") or -1),
                step_delay=float(cfg.get("step_delay") or 1.5),
                use_stamina=bool(cfg.get("use_stamina", False)),
                stamina_can_limit=int(cfg.get("stamina_can_limit") or 800),
                use_formation=use_formation,
                adjust_level=bool(cfg.get("adjust_level", False)),
                beast_icon_index=int(cfg.get("beast_icon_index") or 0),
                tab_bar_already_scrolled=self._hunt_tab_bar_scrolled,
                level_already_adjusted=self._hunt_level_adjusted,
                on_status=_status,
            )
        if task_id == "hunt_monster":
            from core.common_task_opts import resolve_formation_slot
            from tasks.hunt_monster import HuntMonsterTask

            use_formation, formation_slot = resolve_formation_slot(
                cfg.get("formation_name", cfg.get("formation_slot", 6)),
                default_slot=6,
            )
            return HuntMonsterTask(
                adb=proxy,
                coords=coords,
                interval=float(cfg.get("interval") or 60),
                monster_level=int(cfg.get("monster_level") or 30),
                formation_name=str(formation_slot),
                skip_hour=int(cfg.get("skip_hour") or -1),
                step_delay=float(cfg.get("step_delay") or 1.5),
                use_stamina=bool(cfg.get("use_stamina", False)),
                stamina_can_limit=int(cfg.get("stamina_can_limit") or 800),
                use_formation=use_formation,
                adjust_level=bool(cfg.get("adjust_level", False)),
                beast_icon_index=int(cfg.get("beast_icon_index") or 0),
                tab_bar_already_scrolled=self._hunt_tab_bar_scrolled,
                level_already_adjusted=self._hunt_level_adjusted,
                on_status=_status,
            )
        if task_id == "auto_lighthouse":
            from core.common_task_opts import resolve_formation_slot
            from tasks.auto_lighthouse import AutoLighthouseTask

            use_formation, formation_slot = resolve_formation_slot(
                cfg.get("formation_slot", 8),
                default_slot=8,
            )
            return AutoLighthouseTask(
                adb=proxy,
                coords=coords,
                interval=float(cfg.get("interval") or 60),
                formation_slot=formation_slot,
                use_stamina=bool(cfg.get("use_stamina", False)),
                stamina_can_limit=int(cfg.get("stamina_can_limit") or 800),
                use_formation=use_formation,
                event_period=bool(cfg.get("event_period", False)),
                monster_cooldown=float(cfg.get("monster_cooldown") or 60),
                step_delay=float(cfg.get("step_delay") or 1.5),
                on_status=_status,
            )
        if task_id == "auto_mining":
            from tasks.auto_mining import AutoMiningTask

            return AutoMiningTask(
                adb=proxy,
                coords=coords,
                interval=float(cfg.get("interval") or 3600),
                level_min=int(cfg.get("level_min") or 8),
                level_max=int(cfg.get("level_max") or cfg.get("level_min") or 8),
                use_mining_hero=bool(cfg.get("use_mining_hero", True)),
                skip_hour=int(cfg.get("skip_hour") or -1),
                step_delay=float(cfg.get("step_delay") or 1.5),
                hero_match_threshold=float(cfg.get("hero_match_threshold") or 0.68),
                adjust_level=bool(cfg.get("adjust_level", False)),
                on_status=_status,
            )
        raise ValueError(task_id)


class SessionHub:
    """token -> UserSession；被挤下线时旧会话停任务。"""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._sessions: dict[str, UserSession] = {}

    def bind(self, token: str, username: str, device_id: str) -> UserSession:
        with self._lock:
            session = UserSession(username, device_id)
            self._sessions[token] = session
            return session

    def unbind(self, token: str) -> None:
        with self._lock:
            session = self._sessions.pop(token, None)
        if session is not None:
            session.stop_task()

    def get(self, token: str) -> UserSession | None:
        with self._lock:
            return self._sessions.get(token)

    def drop_conflicts(self, *, username: str, device_id: str) -> None:
        with self._lock:
            victims = [
                token
                for token, session in self._sessions.items()
                if session.username == username or session.device_id == device_id
            ]
        for token in victims:
            self.unbind(token)
