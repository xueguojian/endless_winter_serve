"""跨进程 ADB 全局锁：短超时拿不到则跳过，避免三开互相堵死。"""

from __future__ import annotations

import sys
import threading
from contextlib import contextmanager
from typing import Iterator

from loguru import logger

# 短超时：够等完一次 tap/短截图尾声，又不会把另一开卡很久
DEFAULT_ADB_LOCK_TIMEOUT_SEC = 1.5
_MUTEX_NAME = "Local\\EndlessWinterAdbLock"

# WaitForSingleObject
_WAIT_OBJECT_0 = 0x00000000
_WAIT_ABANDONED = 0x00000080
_WAIT_TIMEOUT = 0x00000102


class AdbBusyError(RuntimeError):
    """短超时内未拿到 ADB 全局锁，本次操作可跳过。"""


class _WinNamedMutex:
    """Windows 命名互斥量（跨进程；同线程可重入）。"""

    def __init__(self, name: str) -> None:
        import ctypes
        from ctypes import wintypes

        self._k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self._k32.CreateMutexW.argtypes = [
            wintypes.LPVOID,
            wintypes.BOOL,
            wintypes.LPCWSTR,
        ]
        self._k32.CreateMutexW.restype = wintypes.HANDLE
        self._k32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        self._k32.WaitForSingleObject.restype = wintypes.DWORD
        self._k32.ReleaseMutex.argtypes = [wintypes.HANDLE]
        self._k32.ReleaseMutex.restype = wintypes.BOOL

        handle = self._k32.CreateMutexW(None, False, name)
        if not handle:
            raise OSError(f"CreateMutexW 失败: {ctypes.get_last_error()}")
        self._handle = handle

    def acquire(self, timeout_sec: float) -> bool:
        ms = max(0, int(timeout_sec * 1000))
        code = self._k32.WaitForSingleObject(self._handle, ms)
        if code in (_WAIT_OBJECT_0, _WAIT_ABANDONED):
            return True
        if code == _WAIT_TIMEOUT:
            return False
        logger.warning(f"WaitForSingleObject 异常返回码: {code}")
        return False

    def release(self) -> None:
        if not self._k32.ReleaseMutex(self._handle):
            logger.warning("ReleaseMutex 失败")


class _ThreadRLock:
    """同进程回退锁（非 Windows 或不支持命名互斥时）。"""

    def __init__(self) -> None:
        self._lock = threading.RLock()

    def acquire(self, timeout_sec: float) -> bool:
        return bool(self._lock.acquire(timeout=max(0.0, timeout_sec)))

    def release(self) -> None:
        self._lock.release()


def _build_lock() -> _WinNamedMutex | _ThreadRLock:
    if sys.platform == "win32":
        try:
            return _WinNamedMutex(_MUTEX_NAME)
        except OSError as exc:
            logger.warning(f"跨进程 ADB 锁不可用，回退线程锁: {exc}")
    return _ThreadRLock()


_ADB_LOCK = _build_lock()


@contextmanager
def hold_adb_lock(
    timeout_sec: float = DEFAULT_ADB_LOCK_TIMEOUT_SEC,
    *,
    who: str = "adb",
) -> Iterator[None]:
    """获取全局 ADB 锁；超时则抛 AdbBusyError（调用方可跳过本次操作）。"""
    ok = _ADB_LOCK.acquire(timeout_sec)
    if not ok:
        raise AdbBusyError(
            f"ADB 正忙（{who} 等待 {timeout_sec:.1f}s 未拿到锁），跳过本次操作"
        )
    try:
        yield
    finally:
        _ADB_LOCK.release()
