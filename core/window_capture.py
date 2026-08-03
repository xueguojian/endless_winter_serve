"""Windows 窗口区域抓取：优先 DXGI(dxcam)，其次 mss，再次 BitBlt。

用于快反应小游戏：本机盯雷电窗口一小块 → 分析 → ADB 点击。
返回 BGR numpy，与 OpenCV 一致。
"""

from __future__ import annotations

import ctypes
import ctypes.wintypes as wintypes
from dataclasses import dataclass

import numpy as np

user32 = ctypes.windll.user32
gdi32 = ctypes.windll.gdi32

user32.GetClientRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
user32.GetClientRect.restype = wintypes.BOOL
user32.ClientToScreen.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.POINT)]
user32.ClientToScreen.restype = wintypes.BOOL
user32.IsWindow.argtypes = [wintypes.HWND]
user32.IsWindow.restype = wintypes.BOOL
user32.IsWindowVisible.argtypes = [wintypes.HWND]
user32.IsWindowVisible.restype = wintypes.BOOL
user32.GetWindowTextLengthW.argtypes = [wintypes.HWND]
user32.GetWindowTextLengthW.restype = ctypes.c_int
user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
user32.GetWindowTextW.restype = ctypes.c_int
user32.GetClassNameW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
user32.GetClassNameW.restype = ctypes.c_int
user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
user32.PrintWindow.argtypes = [wintypes.HWND, wintypes.HDC, ctypes.c_uint]
user32.PrintWindow.restype = wintypes.BOOL

PW_RENDERFULLCONTENT = 0x00000002
SRCCOPY = 0x00CC0020

_HAS_DXCAM = False
_HAS_MSS = False
try:
    import dxcam

    _HAS_DXCAM = True
except Exception:
    dxcam = None  # type: ignore

try:
    import mss

    _HAS_MSS = True
except Exception:
    mss = None  # type: ignore


@dataclass(frozen=True)
class WindowInfo:
    hwnd: int
    title: str
    class_name: str

    @property
    def label(self) -> str:
        title = self.title or "(无标题)"
        return f"{title}  [{self.hwnd}]"


@dataclass(frozen=True)
class ClientRect:
    """窗口客户区在屏幕上的位置与尺寸。"""

    left: int
    top: int
    width: int
    height: int

    @property
    def right(self) -> int:
        return self.left + self.width

    @property
    def bottom(self) -> int:
        return self.top + self.height


def list_visible_windows(*, title_keywords: tuple[str, ...] = ()) -> list[WindowInfo]:
    """枚举可见顶级窗口；可按标题关键字过滤（不区分大小写）。"""
    keywords = tuple(k.lower() for k in title_keywords if k)
    found: list[WindowInfo] = []

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def _enum(hwnd: int, _lparam: int) -> bool:
        if not user32.IsWindowVisible(hwnd):
            return True
        length = user32.GetWindowTextLengthW(hwnd)
        if length <= 0:
            return True
        buf = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buf, length + 1)
        title = buf.value.strip()
        if not title:
            return True
        if keywords and not any(k in title.lower() for k in keywords):
            return True
        cls_buf = ctypes.create_unicode_buffer(256)
        user32.GetClassNameW(hwnd, cls_buf, 256)
        found.append(WindowInfo(hwnd=int(hwnd), title=title, class_name=cls_buf.value))
        return True

    user32.EnumWindows(_enum, 0)
    found.sort(key=lambda w: w.title.lower())
    return found


def list_ldplayer_windows() -> list[WindowInfo]:
    """优先列出标题含雷电/LDPlayer 的窗口；没有则返回全部可见窗口。"""
    preferred = list_visible_windows(
        title_keywords=("雷电", "ldplayer", "leidian", "change", "dnplayer")
    )
    if preferred:
        return preferred
    return list_visible_windows()


def get_client_rect_on_screen(hwnd: int) -> ClientRect:
    if not user32.IsWindow(hwnd):
        raise RuntimeError("窗口句柄无效")
    rect = wintypes.RECT()
    if not user32.GetClientRect(hwnd, ctypes.byref(rect)):
        raise RuntimeError("GetClientRect 失败")
    width = int(rect.right - rect.left)
    height = int(rect.bottom - rect.top)
    if width <= 0 or height <= 0:
        raise RuntimeError("窗口客户区尺寸无效（是否最小化？）")
    pt = wintypes.POINT(0, 0)
    if not user32.ClientToScreen(hwnd, ctypes.byref(pt)):
        raise RuntimeError("ClientToScreen 失败")
    return ClientRect(left=int(pt.x), top=int(pt.y), width=width, height=height)


def client_to_touch(
    x: float,
    y: float,
    *,
    client_w: int,
    client_h: int,
    touch_w: int = 720,
    touch_h: int = 1280,
) -> tuple[int, int]:
    """窗口客户区像素 → 720×1280 触控坐标。"""
    if client_w <= 0 or client_h <= 0:
        raise ValueError("客户区尺寸无效")
    tx = int(round(x / client_w * touch_w))
    ty = int(round(y / client_h * touch_h))
    tx = max(0, min(touch_w - 1, tx))
    ty = max(0, min(touch_h - 1, ty))
    return tx, ty


def touch_to_client(
    x: float,
    y: float,
    *,
    client_w: int,
    client_h: int,
    touch_w: int = 720,
    touch_h: int = 1280,
) -> tuple[int, int]:
    """720×1280 触控坐标 → 窗口客户区像素。"""
    if client_w <= 0 or client_h <= 0:
        raise ValueError("客户区尺寸无效")
    if touch_w <= 0 or touch_h <= 0:
        raise ValueError("触控尺寸无效")
    cx = int(round(x / touch_w * client_w))
    cy = int(round(y / touch_h * client_h))
    cx = max(0, min(client_w - 1, cx))
    cy = max(0, min(client_h - 1, cy))
    return cx, cy


def touch_roi_to_client_xywh(
    roi: tuple[int, int, int, int],
    *,
    client_w: int,
    client_h: int,
    touch_w: int = 720,
    touch_h: int = 1280,
) -> tuple[int, int, int, int]:
    """触控 ROI (x1,y1,x2,y2) → 客户区 (x,y,w,h)。"""
    x1, y1, x2, y2 = roi
    cx1, cy1 = touch_to_client(
        x1, y1, client_w=client_w, client_h=client_h, touch_w=touch_w, touch_h=touch_h
    )
    cx2, cy2 = touch_to_client(
        x2, y2, client_w=client_w, client_h=client_h, touch_w=touch_w, touch_h=touch_h
    )
    left, top = min(cx1, cx2), min(cy1, cy2)
    right, bottom = max(cx1, cx2), max(cy1, cy2)
    return left, top, max(1, right - left), max(1, bottom - top)


def _bgr_from_rgb(rgb: np.ndarray) -> np.ndarray:
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        raise ValueError("截图格式无效")
    return rgb[:, :, :3][:, :, ::-1].copy()


def _grab_bitblt_client(hwnd: int) -> np.ndarray:
    """BitBlt/PrintWindow 抓取整个客户区，返回 BGR。"""
    crect = get_client_rect_on_screen(hwnd)
    w, h = crect.width, crect.height

    hwnd_dc = user32.GetDC(hwnd)
    if not hwnd_dc:
        raise RuntimeError("GetDC 失败")
    mem_dc = gdi32.CreateCompatibleDC(hwnd_dc)
    bmp = gdi32.CreateCompatibleBitmap(hwnd_dc, w, h)
    old = gdi32.SelectObject(mem_dc, bmp)
    try:
        # PrintWindow 在窗口被部分遮挡时更稳；失败再 BitBlt
        ok = user32.PrintWindow(hwnd, mem_dc, PW_RENDERFULLCONTENT)
        if not ok:
            ok = gdi32.BitBlt(mem_dc, 0, 0, w, h, hwnd_dc, 0, 0, SRCCOPY)
        if not ok:
            raise RuntimeError("PrintWindow/BitBlt 失败")

        class BITMAPINFOHEADER(ctypes.Structure):
            _fields_ = [
                ("biSize", wintypes.DWORD),
                ("biWidth", wintypes.LONG),
                ("biHeight", wintypes.LONG),
                ("biPlanes", wintypes.WORD),
                ("biBitCount", wintypes.WORD),
                ("biCompression", wintypes.DWORD),
                ("biSizeImage", wintypes.DWORD),
                ("biXPelsPerMeter", wintypes.LONG),
                ("biYPelsPerMeter", wintypes.LONG),
                ("biClrUsed", wintypes.DWORD),
                ("biClrImportant", wintypes.DWORD),
            ]

        class BITMAPINFO(ctypes.Structure):
            _fields_ = [("bmiHeader", BITMAPINFOHEADER), ("bmiColors", wintypes.DWORD * 3)]

        bmi = BITMAPINFO()
        bmi.bmiHeader.biSize = ctypes.sizeof(BITMAPINFOHEADER)
        bmi.bmiHeader.biWidth = w
        bmi.bmiHeader.biHeight = -h  # top-down
        bmi.bmiHeader.biPlanes = 1
        bmi.bmiHeader.biBitCount = 32
        bmi.bmiHeader.biCompression = 0

        buf_len = w * h * 4
        buf = (ctypes.c_ubyte * buf_len)()
        got = gdi32.GetDIBits(mem_dc, bmp, 0, h, buf, ctypes.byref(bmi), 0)
        if got == 0:
            raise RuntimeError("GetDIBits 失败")
        arr = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)
        return arr[:, :, :3].copy()  # BGRX → BGR
    finally:
        gdi32.SelectObject(mem_dc, old)
        gdi32.DeleteObject(bmp)
        gdi32.DeleteDC(mem_dc)
        user32.ReleaseDC(hwnd, hwnd_dc)


class WindowCapture:
    """抓取指定窗口客户区或其子区域。优先 DXGI，自动回退。"""

    def __init__(self) -> None:
        self._hwnd: int | None = None
        self._backend = "none"
        self._dxcam = None
        self._mss = None
        self._init_backend()

    @property
    def backend(self) -> str:
        return self._backend

    @property
    def hwnd(self) -> int | None:
        return self._hwnd

    def _init_backend(self) -> None:
        if _HAS_DXCAM:
            try:
                self._dxcam = dxcam.create(output_color="BGR")
                if self._dxcam is not None:
                    self._backend = "dxcam"
                    return
            except Exception:
                self._dxcam = None
        if _HAS_MSS:
            try:
                self._mss = mss.mss()
                self._backend = "mss"
                return
            except Exception:
                self._mss = None
        self._backend = "bitblt"

    def set_window(self, hwnd: int) -> ClientRect:
        if not user32.IsWindow(hwnd):
            raise RuntimeError("窗口句柄无效")
        self._hwnd = int(hwnd)
        return get_client_rect_on_screen(self._hwnd)

    def client_rect(self) -> ClientRect:
        if self._hwnd is None:
            raise RuntimeError("尚未选择窗口")
        return get_client_rect_on_screen(self._hwnd)

    def grab_region(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
    ) -> np.ndarray:
        """抓取客户区内相对坐标区域，返回 BGR。"""
        if self._hwnd is None:
            raise RuntimeError("尚未选择窗口")
        if width <= 0 or height <= 0:
            raise ValueError("区域尺寸无效")

        crect = get_client_rect_on_screen(self._hwnd)
        x = max(0, min(crect.width - 1, int(x)))
        y = max(0, min(crect.height - 1, int(y)))
        width = max(1, min(crect.width - x, int(width)))
        height = max(1, min(crect.height - y, int(height)))

        left = crect.left + x
        top = crect.top + y
        right = left + width
        bottom = top + height

        if self._backend == "dxcam" and self._dxcam is not None:
            frame = self._dxcam.grab(region=(left, top, right, bottom))
            if frame is not None:
                return np.ascontiguousarray(frame)
            # 偶发 None（无变化帧），回退一次
        if self._backend in {"dxcam", "mss"} and self._mss is None and _HAS_MSS:
            self._mss = mss.mss()
        if self._mss is not None:
            shot = self._mss.grab(
                {"left": left, "top": top, "width": width, "height": height}
            )
            # mss: BGRA
            arr = np.asarray(shot, dtype=np.uint8)
            return arr[:, :, :3].copy()

        full = _grab_bitblt_client(self._hwnd)
        return full[y : y + height, x : x + width].copy()

    def grab_client(self) -> np.ndarray:
        if self._hwnd is None:
            raise RuntimeError("尚未选择窗口")
        crect = get_client_rect_on_screen(self._hwnd)
        return self.grab_region(0, 0, crect.width, crect.height)

    def close(self) -> None:
        if self._dxcam is not None:
            try:
                self._dxcam.release()
            except Exception:
                pass
            self._dxcam = None
        if self._mss is not None:
            try:
                self._mss.close()
            except Exception:
                pass
            self._mss = None
