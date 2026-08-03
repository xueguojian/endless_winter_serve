"""ADB 设备连接与操作封装。"""

from __future__ import annotations

import io
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from loguru import logger
from PIL import Image

from core.adb_lock import (
    DEFAULT_ADB_LOCK_TIMEOUT_SEC,
    AdbBusyError,
    hold_adb_lock,
)

_WIN_SUBPROCESS_FLAGS = 0
if sys.platform == "win32":
    _WIN_SUBPROCESS_FLAGS = getattr(subprocess, "CREATE_NO_WINDOW", 0)

_ADB_RETRY_COUNT = 4
_ADB_RETRY_BASE_DELAY = 0.6


class AdbUnavailableError(RuntimeError):
    """ADB 进程/连接不可用（含长时间双开后的 0xc0000142）。"""


# 兼容旧 import：from core.adb_client import AdbBusyError
__all__ = [
    "AdbBusyError",
    "AdbClient",
    "AdbUnavailableError",
    "DEFAULT_ADB_LOCK_TIMEOUT_SEC",
]


class AdbClient:
    """通过 ADB 控制雷电模拟器。"""

    LDPLAYER_PROBE_PORTS = tuple(range(5555, 5571, 2))

    LDPLAYER_ADB_CANDIDATES = [
        # 雷电 14 / 新版常见路径
        r"C:\leidian\LDPlayer14\adb.exe",
        r"D:\leidian\LDPlayer14\adb.exe",
        r"C:\LDPlayer\LDPlayer14\adb.exe",
        r"D:\LDPlayer\LDPlayer14\adb.exe",
        r"C:\leidian\LDPlayer\adb.exe",
        r"D:\leidian\LDPlayer\adb.exe",
        # 雷电 9
        r"C:\leidian\LDPlayer9\adb.exe",
        r"D:\leidian\LDPlayer9\adb.exe",
        r"C:\Program Files\LDPlayer\LDPlayer9\adb.exe",
        r"D:\Program Files\LDPlayer\LDPlayer9\adb.exe",
        # 雷电 4
        r"C:\leidian\LDPlayer4\adb.exe",
        r"D:\leidian\LDPlayer4\adb.exe",
    ]

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 5555,
        adb_path: str = "",
        touch_width: int = 720,
        touch_height: int = 1280,
    ):
        self.address = f"{host}:{port}"
        self.adb = self._resolve_adb(adb_path)
        self.touch_width = touch_width
        self.touch_height = touch_height
        self._fail_streak = 0
        self._screenshot_count = 0

    @classmethod
    def resolve_adb_path(cls, adb_path: str = "") -> str:
        if adb_path and Path(adb_path).is_file():
            return adb_path

        for candidate in cls.LDPLAYER_ADB_CANDIDATES:
            if Path(candidate).is_file():
                logger.info(f"找到雷电 ADB: {candidate}")
                return candidate

        raise FileNotFoundError(
            "未找到 adb.exe。请在实例 config_555x.yaml 的 device.adb_path 中填写雷电 adb.exe 路径。"
        )

    @staticmethod
    def parse_address(serial: str) -> tuple[str, int]:
        """解析 adb devices 中的序列号，如 127.0.0.1:5555、emulator-5554。"""
        serial = serial.strip()
        if not serial:
            raise ValueError("设备地址不能为空")

        if serial.startswith("emulator-"):
            port = int(serial.rsplit("-", 1)[1])
            return "127.0.0.1", port

        if ":" in serial:
            host, port_text = serial.rsplit(":", 1)
            return host, int(port_text)

        raise ValueError(f"无法解析设备地址：{serial}")

    @classmethod
    def format_address(cls, host: str, port: int) -> str:
        return f"{host}:{int(port)}"

    def _resolve_adb(self, adb_path: str) -> str:
        return self.resolve_adb_path(adb_path)

    def _parse_devices_output(self, output: str) -> list[str]:
        devices: list[str] = []
        for line in output.splitlines():
            line = line.strip()
            if not line or line.lower().startswith("list of devices"):
                continue
            parts = line.split()
            if len(parts) >= 2 and parts[1] == "device":
                devices.append(parts[0])
        return devices

    @staticmethod
    def normalize_device_serial(serial: str) -> str:
        """统一设备显示地址，合并 emulator-5554 与 127.0.0.1:5555 等等价项。"""
        serial = serial.strip()
        if serial.startswith("emulator-"):
            console_port = int(serial.rsplit("-", 1)[1])
            return f"127.0.0.1:{console_port + 1}"

        host, port = AdbClient.parse_address(serial)
        if host in {"127.0.0.1", "localhost"}:
            return f"127.0.0.1:{port}"
        return f"{host}:{port}"

    @classmethod
    def dedupe_device_serials(cls, devices: list[str]) -> list[str]:
        seen: set[str] = set()
        ordered: list[str] = []
        for serial in devices:
            try:
                normalized = cls.normalize_device_serial(serial)
            except ValueError:
                continue
            if normalized in seen:
                continue
            seen.add(normalized)
            ordered.append(normalized)
        return ordered

    @classmethod
    def list_connected_devices(
        cls,
        adb_path: str = "",
        *,
        probe_ldplayer: bool = True,
    ) -> list[str]:
        """列出当前已连接的 ADB 设备；probe_ldplayer 时会尝试连接雷电多开端口。"""
        adb_bin = cls.resolve_adb_path(adb_path)
        runner = cls(host="127.0.0.1", port=5555, adb_path=adb_bin)

        result = runner._run("devices")
        devices = runner._parse_devices_output(result.stdout)

        if probe_ldplayer:
            known = {cls.normalize_device_serial(item) for item in devices}
            for port in cls.LDPLAYER_PROBE_PORTS:
                address = f"127.0.0.1:{port}"
                if address in known:
                    continue
                connect = runner._run("connect", address, timeout=4)
                output = (connect.stdout + connect.stderr).lower()
                if "connected" in output and "cannot" not in output:
                    check = runner._run("devices")
                    devices = runner._parse_devices_output(check.stdout)
                    known = {cls.normalize_device_serial(item) for item in devices}

        devices = cls.dedupe_device_serials(devices)
        logger.info(f"ADB 已连接设备: {devices or '无'}")
        return devices

    def _spawn(
        self,
        args: list[str],
        *,
        timeout: int,
        binary: bool = False,
    ) -> subprocess.CompletedProcess:
        kwargs: dict = {
            "capture_output": True,
            "timeout": timeout,
            "creationflags": _WIN_SUBPROCESS_FLAGS,
            "stdin": subprocess.DEVNULL,
        }
        if not binary:
            kwargs.update(text=True, encoding="utf-8", errors="replace")
        return subprocess.run([self.adb, *args], **kwargs)

    def _recover_connection(self) -> None:
        """尽量恢复连接，不 kill-server（会打断另一开脚本）。"""
        try:
            self._spawn(["start-server"], timeout=10)
        except Exception as exc:
            logger.warning(f"ADB start-server 失败: {exc}")
        try:
            self._spawn(["connect", self.address], timeout=8)
        except Exception as exc:
            logger.warning(f"ADB reconnect 失败: {exc}")

    def _run_once(self, *args: str, timeout: int = 30) -> subprocess.CompletedProcess[str]:
        cmd_preview = " ".join([self.adb, *args])
        logger.debug(f"ADB: {cmd_preview}")
        return self._spawn(list(args), timeout=timeout, binary=False)

    def _run(self, *args: str, timeout: int = 30) -> subprocess.CompletedProcess[str]:
        last_error: Exception | None = None
        last_result: subprocess.CompletedProcess[str] | None = None

        with hold_adb_lock(who=f"run:{self.address}"):
            for attempt in range(1, _ADB_RETRY_COUNT + 1):
                try:
                    result = self._run_once(*args, timeout=timeout)
                    last_result = result
                    if result.returncode == 0:
                        self._fail_streak = 0
                        return result
                    logger.warning(
                        f"ADB 返回码 {result.returncode} "
                        f"({attempt}/{_ADB_RETRY_COUNT}): "
                        f"{(result.stderr or result.stdout).strip()[:200]}"
                    )
                except (OSError, subprocess.TimeoutExpired) as exc:
                    last_error = exc
                    logger.warning(
                        f"ADB 进程异常 ({attempt}/{_ADB_RETRY_COUNT}): {exc}"
                    )

                self._fail_streak += 1
                if attempt < _ADB_RETRY_COUNT:
                    self._recover_connection()
                    time.sleep(_ADB_RETRY_BASE_DELAY * attempt)

        if last_result is not None:
            return last_result
        raise AdbUnavailableError(
            f"ADB 无法启动（可能双开过久导致 adb.exe 0xc0000142）: {last_error}"
        )

    def connect(self) -> bool:
        result = self._run("connect", self.address)
        output = (result.stdout + result.stderr).strip()
        logger.info(output or "ADB connect 完成")
        return "connected" in output.lower() or self.address in self._run("devices").stdout

    def wait_for_device(self, retries: int = 30, interval: float = 2.0) -> bool:
        quick = self._run("-s", self.address, "shell", "echo", "ok", timeout=5)
        if quick.returncode == 0 and "ok" in quick.stdout:
            logger.debug(f"设备已就绪(快速): {self.address}")
            return True
        for attempt in range(1, retries + 1):
            if self.connect():
                check = self._run("-s", self.address, "shell", "echo", "ok")
                if check.returncode == 0 and "ok" in check.stdout:
                    logger.info(f"设备已就绪: {self.address}")
                    return True
            logger.warning(f"等待设备连接 ({attempt}/{retries})...")
            time.sleep(interval)
        return False

    def shell(self, command: str) -> str:
        result = self._run("-s", self.address, "shell", command)
        if result.returncode != 0:
            raise AdbUnavailableError(result.stderr.strip() or "ADB shell 命令失败")
        return result.stdout.strip()

    def get_screen_size(self) -> tuple[int, int]:
        """返回截图宽高 (width, height)，应与 touch_width/touch_height 一致。"""
        img = self.screenshot()
        h, w = img.shape[:2]
        return w, h

    def screenshot(self) -> np.ndarray:
        """截取屏幕，返回 BGR 数组，shape=(height, width)，与 input tap 同一竖屏坐标系。"""
        last_error: Exception | None = None

        with hold_adb_lock(who=f"screenshot:{self.address}"):
            self._screenshot_count += 1
            shot_no = self._screenshot_count
            for attempt in range(1, _ADB_RETRY_COUNT + 1):
                try:
                    cmd_preview = (
                        f"{self.adb} -s {self.address} exec-out screencap -p"
                    )
                    if attempt == 1:
                        logger.debug(f"ADB: {cmd_preview}  # screenshot #{shot_no}")
                    else:
                        logger.debug(
                            f"ADB: {cmd_preview}  # screenshot #{shot_no}"
                            f" retry {attempt}/{_ADB_RETRY_COUNT}"
                        )
                    proc = self._spawn(
                        ["-s", self.address, "exec-out", "screencap", "-p"],
                        timeout=15,
                        binary=True,
                    )
                    if proc.returncode == 0 and proc.stdout:
                        image = Image.open(io.BytesIO(proc.stdout)).convert("RGB")
                        rgb = np.array(image)
                        self._fail_streak = 0
                        return rgb[:, :, ::-1].copy()
                    last_error = RuntimeError(
                        f"returncode={proc.returncode}, bytes={len(proc.stdout or b'')}"
                    )
                    logger.warning(
                        f"截图失败 ({attempt}/{_ADB_RETRY_COUNT}): {last_error}"
                    )
                except (OSError, subprocess.TimeoutExpired) as exc:
                    last_error = exc
                    logger.warning(
                        f"截图进程异常 ({attempt}/{_ADB_RETRY_COUNT}): {exc}"
                    )
                except Exception as exc:
                    last_error = exc
                    logger.warning(
                        f"截图解析失败 ({attempt}/{_ADB_RETRY_COUNT}): {exc}"
                    )

                self._fail_streak += 1
                if attempt < _ADB_RETRY_COUNT:
                    self._recover_connection()
                    time.sleep(_ADB_RETRY_BASE_DELAY * attempt)

        raise AdbUnavailableError(
            "截图失败，请确认模拟器已启动且 ADB 已连接"
            "（双开长时间运行若弹出 adb.exe 0xc0000142，请完全退出两侧脚本后重开，"
            "或在任务管理器结束多余 adb.exe）"
            + (f"；详情: {last_error}" if last_error else "")
        )

    def tap(self, x: int, y: int) -> None:
        x = max(0, min(x, self.touch_width - 1))
        y = max(0, min(y, self.touch_height - 1))
        self._run("-s", self.address, "shell", "input", "tap", str(x), str(y))

    def swipe(self, x1: int, y1: int, x2: int, y2: int, duration_ms: int = 300) -> None:
        self._run(
            "-s",
            self.address,
            "shell",
            "input",
            "swipe",
            str(x1),
            str(y1),
            str(x2),
            str(y2),
            str(duration_ms),
        )

    def back(self) -> None:
        self._run("-s", self.address, "shell", "input", "keyevent", "4")

    def escape(self) -> None:
        """发送 ESC 键（Android KEYCODE_ESCAPE）。"""
        self._run("-s", self.address, "shell", "input", "keyevent", "111")

    def launch_game(self, package: str, activity: str) -> None:
        self._run("-s", self.address, "shell", "am", "start", "-n", activity)
        logger.info(f"已启动游戏: {package}")

    def get_touch_size(self) -> tuple[int, int]:
        return self.touch_width, self.touch_height
