"""云控截图 ROI：本地可整屏截，上传只传需要的区域。"""

from __future__ import annotations

from typing import Any

# —— 出征 / 场景 ——
MARCH_BTN_CAPTURE_ROI = (340, 1080, 720, 1280)
SCENE_TOGGLE_CAPTURE_ROI = (500, 1150, 720, 1280)
# 底部 UI：搜索面板 + 出征 + 城镇/野外（不含灯塔顶栏）
BOTTOM_UI_CAPTURE_ROI = (0, 780, 720, 1280)
# 出征页：英雄栏 + 出征按钮（编队后验英雄与按钮）
DEPLOY_CAPTURE_ROI = (90, 304, 720, 1280)

# —— 搜索面板 ——
SEARCH_ICON_CAPTURE_ROI = (0, 780, 160, 960)
SEARCH_TAB_CAPTURE_ROI = (0, 850, 720, 980)
SEARCH_CONFIRM_CAPTURE_ROI = (100, 1130, 530, 1240)
SEARCH_LEVEL_CAPTURE_ROI = (560, 1010, 650, 1090)
FULL_RESOURCE_CAPTURE_ROI = (180, 1100, 260, 1180)
# 打开/判断搜索面板：图标+tab+确认+等级条
SEARCH_PANEL_CAPTURE_ROI = (0, 780, 720, 1280)

# —— 体力（与 stamina_use 一致）——
STAMINA_POPUP_CAPTURE_ROI = (180, 80, 720, 700)
# 出征结果：体力标题 + 同目标冲突弹窗（含取消按钮）
MARCH_OUTCOME_CAPTURE_ROI = (50, 80, 720, 880)


def capture_screen(adb: Any, roi: tuple[int, int, int, int] | None = None):
    """截图；ProxyAdb 支持 roi= 只上传该区域，普通 AdbClient 自动降级整图。"""
    try:
        return adb.screenshot(roi=roi)
    except TypeError:
        return adb.screenshot()
