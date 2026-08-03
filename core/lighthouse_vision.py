"""灯塔任务图标识别。"""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from core.coords import PORTRAIT_HEIGHT, PORTRAIT_WIDTH
from core.dream_memory.ocr_engine import ocr_chip_text
from loguru import logger

from core.vision import MatchResult, Vision

TEMPLATE_DIR = Path(__file__).parent.parent / "assets" / "templates"

# 情报页中间地图区域，排除顶部刷新条与底部等级栏
LIGHTHOUSE_SCAN_ROI = (20, 130, 700, 1020)
# 情报页背景：平常 / 活动期间（含红色晶簇）
LIGHTHOUSE_MAP_BG_NORMAL = "lighthouse/lighthouse_map_bg.png"
LIGHTHOUSE_MAP_BG_EVENT = "lighthouse/lighthouse_map_bg_1.png"
_active_map_bg_name: str = LIGHTHOUSE_MAP_BG_NORMAL
# 情报页顶部刷新条，用于区分「情报页」与「野外大地图」
LIGHTHOUSE_HEADER_ROI = (0, 155, 720, 210)
LIGHTHOUSE_HEADER_MEAN_DIFF_MAX = 18.0
# 与背景图差分：仅保留相对背景新出现且颜色鲜艳的图钉
BG_DIFF_THRESHOLD = 28
BG_DIFF_VIVID_MIN = 0.10
BG_DIFF_PIN_SUPPORT_MIN = 0.15
BG_MAX_PINS_PER_DIFF_BLOB = 2
BG_LARGE_BLOB_AREA = 3500
BG_LARGE_BLOB_PIN_SPACING = 24
BG_PEAK_MIN_DISTANCE = 4.0
BG_PIN_AREA_MIN = 60
BG_PIN_AREA_MAX = 1800
BG_DIFF_BLOB_MIN_AREA = 800
BG_TEARDROP_TOP_BOTTOM_MIN = 1.05
BG_TEARDROP_ASPECT_MIN = 0.85
BG_TEARDROP_ASPECT_MAX = 2.8
SUPER_BOSS_Y_MAX = 320
UI_EXCLUDE_Y_MAX = 230
UI_EXCLUDE_X_MAX = 120
PLANE_EXCLUDE_Y_MIN = 820
PLANE_EXCLUDE_X_MIN = 480
PLANE_TEMPLATE = "lighthouse/small_plane.png"
PLANE_PATCH_HALF = 42
PLANE_ROTATE_MATCH_MIN = 0.45
PLANE_ROTATE_ANGLES = tuple(range(0, 360, 30))
PLANE_ORANGE_RATIO_MIN = 0.18
PLANE_COCKPIT_PALE_MIN = 0.20
PLANE_COLOR_TMPL_MIN = 0.38
BASE_EXCLUDE_VIVID_KEEP_MIN = 0.35
BG_PIN_DIFF_SEARCH_RADIUS = 50
LIGHTHOUSE_PIN_PATCH_HALF = 36
LIGHTHOUSE_SLOT_MERGE_DISTANCE = 18
LIGHTHOUSE_MAX_PIN_CANDIDATES = 18
PIN_BLOB_OPEN_KERNEL = 5
PIN_LARGE_BLOB_MIN_AREA = 900
PIN_BLOB_MAX_WHITE_RATIO = 0.28
PIN_BLOB_TENT_MAX_WHITE_RATIO = 0.85
TENT_PIN_BLOB_MIN_AREA = 80
TENT_PIN_BLOB_MAX_AREA = 1200
MISSION_PIN_MIN_AREA = 100
MISSION_PIN_MAX_AREA = 700
MISSION_PIN_MIN_CIRCULARITY = 0.20
MISSION_PIN_MAX_CIRCULARITY = 0.95
MISSION_PIN_MIN_EXTENT = 0.20
MISSION_PIN_MAX_EXTENT = 0.85
MISSION_PIN_MIN_ASPECT = 0.9
MISSION_PIN_MAX_ASPECT = 2.2
MISSION_PIN_MERGE_DISTANCE = 18
MAP_PIN_MIN_SCREEN_Y = 250
MAP_PIN_MAX_SCREEN_Y = 980
MAP_PIN_MIN_SCREEN_X = 50
MAP_PIN_MAX_SCREEN_X = 670
BOSS_ZONE_Y_MAX = 400
BOSS_ZONE_X = (160, 560)
BOSS_ZONE_MIN_AREA = 480
PLANE_ZONE_Y_MIN = 640
PLANE_ZONE_X_MAX = 220
BASE_EXCLUDE_CENTER = (360, 530)
BASE_EXCLUDE_RADIUS = 95

# 任务详情页（点击「立即查看」后）：底部行动按钮与标题副文案
MISSION_DETAIL_ACTION_ROI = (238, 592, 482, 666)
MISSION_DETAIL_SUBTITLE_ROI = (272, 380, 452, 412)
MISSION_BTN_MARCH = "lighthouse/mission_btn_march.png"
MISSION_BTN_ADVENTURE = "lighthouse/mission_btn_adventure.png"
MISSION_BTN_RESCUE = "lighthouse/mission_btn_rescue.png"
BOUNTY_MASTER_LABEL = "lighthouse/bounty_master_label.png"
BOUNTY_KEYWORD_DASHI_XUANSHANG = "lighthouse/bounty_keyword_dashi_xuanshang.png"
# 仅匹配「大师/宗师悬赏」专属词，不用单独的「悬赏：」（易与等级怪副标题混淆）
BOUNTY_SUBTITLE_TEMPLATES: tuple[tuple[str, float], ...] = (
    (BOUNTY_KEYWORD_DASHI_XUANSHANG, 0.48),
    (BOUNTY_MASTER_LABEL, 0.45),
)
MISSION_DETAIL_BOUNTY_THRESHOLD = 0.45
BOUNTY_SUBTITLE_SEARCH_WIDTH_RATIO = 0.40
BOUNTY_SUBTITLE_MAX_MATCH_X_RATIO = 0.35
BOUNTY_SUBTITLE_SCALES = (0.82, 0.90, 0.96, 1.0, 1.06, 1.14, 1.22)
MONSTER_KEYWORD_LEVEL = "lighthouse/monster_keyword_level.png"
BEAST_KEYWORD_HAO = "lighthouse/beast_keyword_hao.png"
MONSTER_LEVEL_SUBTITLE_MIN = 0.48
MONSTER_LEVEL_MAX_MATCH_X_RATIO = 0.18
BEAST_HAO_SUBTITLE_MIN = 0.70
BEAST_HAO_SEARCH_WIDTH_RATIO = 0.58
BEAST_HAO_MIN_MATCH_X_RATIO = 0.55
LEVEL_OVERRIDE_HAO_MIN = 0.55
BEAST_PIN_TEMPLATE = "lighthouse/small_monster_beast.png"
BEAST_PIN_MATCH_MIN = 0.48
SKIP_MISSION_KINDS = frozenset({"small_monster_beast"})
MISSION_DETAIL_BTN_THRESHOLD = 0.65
MISSION_DETAIL_BTN_ACCEPT_SCORE = 0.60
MISSION_DETAIL_BTN_MIN_MARGIN = 0.04
MISSION_DETAIL_ABSOLUTE_MIN = 0.55
MISSION_DETAIL_COLOR_BACKUP_MIN_TEMPLATE = 0.32
MISSION_DETAIL_COLOR_STRONG_RATIO = 0.22
MISSION_DETAIL_COLOR_STRONG_MIN_TEMPLATE = 0.30
MISSION_DETAIL_BTN_SCALES = (0.88, 0.94, 1.0, 1.06, 1.12)
MISSION_DETAIL_COLOR_MIN_RATIO = 0.14
MISSION_DETAIL_COLOR_MIN_MARGIN = 0.04
MISSION_DETAIL_ACTION_OCR_CONF = 0.90
# 详情页行动按钮文字 → 任务类型
DETAIL_ACTION_TEXT_KINDS: tuple[tuple[str, str, str], ...] = (
    ("营救", "tent", "营救"),
    ("探险", "hero_journey", "探险"),
    ("出征", "small_monster", "出征"),
)
DETAIL_ACTION_BUTTON_TEMPLATES: tuple[tuple[str, str, str], ...] = (
    ("small_monster", "出征", MISSION_BTN_MARCH),
    ("hero_journey", "探险", MISSION_BTN_ADVENTURE),
    ("tent", "营救", MISSION_BTN_RESCUE),
)
# 主 ROI 未命中时，在更大底部区域再试一次（部分机型/任务弹窗略有偏移）
DETAIL_ACTION_ROI_FALLBACK = (200, 560, 520, 690)
# 灯塔小怪详情弹窗按钮可能更靠下/偏宽，再扫一遍屏幕下沿
DETAIL_ACTION_BOTTOM_BAND = (140, 520, 580, 740)
DETAIL_ACTION_RELAXED_ACCEPT_SCORE = 0.50
DETAIL_ACTION_RELAXED_MIN_MARGIN = 0.05
DETAIL_ORANGE_ONLY_MIN_RATIO = 0.16

_ACTION_BTN_TEMPLATE_CACHE: dict[str, tuple[np.ndarray, np.ndarray]] = {}

# HSV 行动按钮主色（橙=出征，蓝=探险，绿=营救）
_DETAIL_BTN_HSV: tuple[tuple[str, str, str, tuple[int, int, int], tuple[int, int, int]], ...] = (
    ("small_monster", "出征", "orange", (5, 90, 130), (30, 255, 255)),
    ("hero_journey", "探险", "blue", (95, 70, 110), (125, 255, 255)),
    ("tent", "营救", "green", (32, 55, 70), (88, 255, 255)),
)


@dataclass(frozen=True)
class MissionDetailClassification:
    """详情页行动按钮 + 副标题联合分类结果。"""

    kind: str
    label: str
    confidence: float = 0.0
    action_center: tuple[int, int] | None = None
    beast_explicit: bool = False


@dataclass(frozen=True)
class LighthouseMission:
    kind: str
    label: str
    template: str
    center: tuple[int, int]
    confidence: float
    top_left: tuple[int, int] = (0, 0)
    size: tuple[int, int] = (0, 0)


@dataclass(frozen=True)
class LighthouseScanResult:
    mission: LighthouseMission | None
    missions: tuple[LighthouseMission, ...] = ()
    best_confidence: float = 0.0
    best_label: str = ""
    candidate_locations: int = 0

_MAP_BG_ROI: np.ndarray | None = None
_MAP_BG_SCREEN: np.ndarray | None = None
_event_period: bool = False
_scan_interrupt_cb: Callable[[], bool] | None = None


def _check_scan_interrupted() -> None:
    if _scan_interrupt_cb and _scan_interrupt_cb():
        raise InterruptedError("任务已停止")


def _normalize_screen_for_scan(screen: np.ndarray) -> np.ndarray:
    h, w = screen.shape[:2]
    if (w, h) == (PORTRAIT_WIDTH, PORTRAIT_HEIGHT):
        return screen
    return cv2.resize(
        screen,
        (PORTRAIT_WIDTH, PORTRAIT_HEIGHT),
        interpolation=cv2.INTER_LINEAR,
    )

def _is_ignored_background_pin(
    center: tuple[int, int],
    *,
    contour_area: float = 0.0,
    hsv: np.ndarray | None = None,
    roi_offset: tuple[int, int] = (0, 0),
) -> bool:
    """排除中央固定基地、左上角 UI 头像等非任务元素。

    基地附近也可能刷出真实任务钉：若局部鲜艳色足够，则不忽略。
    """
    x, y = center
    bx, by = BASE_EXCLUDE_CENTER
    if abs(x - bx) <= BASE_EXCLUDE_RADIUS and abs(y - by) <= BASE_EXCLUDE_RADIUS:
        if hsv is not None:
            local = (x - roi_offset[0], y - roi_offset[1])
            if (
                0 <= local[0] < hsv.shape[1]
                and 0 <= local[1] < hsv.shape[0]
                and _patch_vivid_ratio(hsv, local) >= BASE_EXCLUDE_VIVID_KEEP_MIN
            ):
                return False
        return True
    if x <= UI_EXCLUDE_X_MAX and y <= UI_EXCLUDE_Y_MAX:
        return True
    return False


def _patch_looks_like_mission_pin(bgr_patch: np.ndarray) -> bool:
    """彩色泪滴图钉（含橙色爪/帐篷），避免被飞机颜色启发误伤。"""
    if bgr_patch.size == 0 or min(bgr_patch.shape[:2]) < 16:
        return False
    hsv = cv2.cvtColor(bgr_patch, cv2.COLOR_BGR2HSV)
    mask = _build_vivid_pin_mask(hsv)
    if cv2.countNonZero(mask) < 80:
        return False
    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    if not contours:
        return False
    biggest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(biggest) < 60:
        return False
    return _is_teardrop_contour(biggest, relaxed=True)


def _is_super_boss_point(
    center: tuple[int, int],
    roi: np.ndarray,
    hsv: np.ndarray,
    roi_offset: tuple[int, int],
) -> bool:
    """橙光兽头大怪：大橙色块 + 内部白色兽脸。

    顶部经典大怪靠位置放宽；中部橙光钉要求更强橙+白脸，避免误伤普通图钉。
    """
    x, y = center
    rx, ry = x - roi_offset[0], y - roi_offset[1]
    radius = 48
    y1 = max(0, ry - radius)
    y2 = min(roi.shape[0], ry + radius)
    x1 = max(0, rx - radius)
    x2 = min(roi.shape[1], rx + radius)
    patch_hsv = hsv[y1:y2, x1:x2]
    if patch_hsv.size == 0:
        return False
    orange = cv2.inRange(
        patch_hsv, np.array((8, 100, 140)), np.array((28, 255, 255))
    )
    orange_ratio = float(orange.mean()) / 255.0
    patch_gray = cv2.cvtColor(roi[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    white_ratio = float((patch_gray > 200).mean())
    # 顶部超级大怪
    if y <= SUPER_BOSS_Y_MAX and orange_ratio >= 0.22:
        return white_ratio >= 0.05 or y <= 290
    # 中部橙光兽头钉（与附近普通任务钉区分）
    return orange_ratio >= 0.35 and white_ratio >= 0.12


def _extract_center_patch(
    bgr: np.ndarray,
    center: tuple[int, int],
    *,
    half: int = PLANE_PATCH_HALF,
) -> np.ndarray:
    x, y = int(center[0]), int(center[1])
    h, w = bgr.shape[:2]
    x1, y1 = max(0, x - half), max(0, y - half)
    x2, y2 = min(w, x + half), min(h, y + half)
    if x2 <= x1 or y2 <= y1:
        return np.array([])
    return bgr[y1:y2, x1:x2]


def _plane_color_structure_scores(bgr_patch: np.ndarray) -> tuple[float, float]:
    """返回 (橙机身占比, 橙块中心浅灰驾驶舱占比)。旋转不变。"""
    if bgr_patch.size == 0 or min(bgr_patch.shape[:2]) < 16:
        return 0.0, 0.0
    hsv = cv2.cvtColor(bgr_patch, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    orange = (hue >= 5) & (hue <= 32) & (sat >= 55) & (val >= 70)
    orange_ratio = float(orange.mean())
    if orange_ratio < 0.08:
        return orange_ratio, 0.0
    ys, xs = np.where(orange)
    if xs.size < 24:
        return orange_ratio, 0.0
    cx, cy = int(xs.mean()), int(ys.mean())
    ph, pw = bgr_patch.shape[:2]
    radius = max(5, min(ph, pw) // 5)
    y1, y2 = max(0, cy - radius), min(ph, cy + radius)
    x1, x2 = max(0, cx - radius), min(pw, cx + radius)
    cockpit = hsv[y1:y2, x1:x2]
    if cockpit.size == 0:
        return orange_ratio, 0.0
    pale = (cockpit[:, :, 1] < 75) & (cockpit[:, :, 2] > 130)
    return orange_ratio, float(pale.mean())


def _match_plane_template_rotated(bgr_patch: np.ndarray) -> float:
    """多角度模板匹配小飞机，返回最高分。"""
    template_path = TEMPLATE_DIR / PLANE_TEMPLATE
    if not template_path.exists() or bgr_patch.size == 0:
        return 0.0
    template = cv2.imread(str(template_path))
    if template is None:
        return 0.0
    th, tw = template.shape[:2]
    if th < 8 or tw < 8:
        return 0.0
    ph, pw = bgr_patch.shape[:2]
    best = 0.0
    for angle in PLANE_ROTATE_ANGLES:
        matrix = cv2.getRotationMatrix2D((tw / 2.0, th / 2.0), float(angle), 1.0)
        cos_a = abs(matrix[0, 0])
        sin_a = abs(matrix[0, 1])
        new_w = max(1, int(th * sin_a + tw * cos_a))
        new_h = max(1, int(th * cos_a + tw * sin_a))
        matrix[0, 2] += new_w / 2.0 - tw / 2.0
        matrix[1, 2] += new_h / 2.0 - th / 2.0
        rotated = cv2.warpAffine(
            template,
            matrix,
            (new_w, new_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        rh, rw = rotated.shape[:2]
        if rh > ph or rw > pw:
            scale = min(ph / rh, pw / rw) * 0.92
            rotated = cv2.resize(
                rotated,
                (max(8, int(rw * scale)), max(8, int(rh * scale))),
                interpolation=cv2.INTER_AREA,
            )
            rh, rw = rotated.shape[:2]
        if rh > ph or rw > pw or rh < 8 or rw < 8:
            continue
        score = float(
            cv2.matchTemplate(bgr_patch, rotated, cv2.TM_CCOEFF_NORMED).max()
        )
        if score > best:
            best = score
    return best


def _is_plane_like_appearance(bgr_patch: np.ndarray) -> bool:
    """任意朝向小飞机：旋转模板为主，橙机身+中心浅驾驶舱为辅。"""
    if bgr_patch.size == 0 or min(bgr_patch.shape[:2]) < 16:
        return False
    tmpl_score = _match_plane_template_rotated(bgr_patch)
    if tmpl_score >= PLANE_ROTATE_MATCH_MIN:
        logger.debug(f"小飞机模板命中 score={tmpl_score:.2f}")
        return True
    # 橙色爪印/帐篷钉也有「橙+白图标」，必须先排除泪滴图钉
    if _patch_looks_like_mission_pin(bgr_patch):
        return False
    orange_ratio, pale_ratio = _plane_color_structure_scores(bgr_patch)
    if (
        orange_ratio >= PLANE_ORANGE_RATIO_MIN
        and pale_ratio >= PLANE_COCKPIT_PALE_MIN
        and tmpl_score >= PLANE_COLOR_TMPL_MIN
    ):
        logger.debug(
            f"小飞机颜色结构命中 orange={orange_ratio:.2f} pale={pale_ratio:.2f} "
            f"tmpl={tmpl_score:.2f}"
        )
        return True
    return False


def _is_plane_point(
    center: tuple[int, int],
    *,
    blob_area: float = 0.0,
    blob_width: int = 0,
    blob_height: int = 0,
    roi: np.ndarray | None = None,
    roi_offset: tuple[int, int] = (0, 0),
) -> bool:
    """排除移动小飞机（任意朝向）。先外观，再回退右下角位置启发。"""
    if roi is not None and roi.size > 0:
        local = (center[0] - roi_offset[0], center[1] - roi_offset[1])
        patch = _extract_center_patch(roi, local)
        if _is_plane_like_appearance(patch):
            return True

    x, y = center
    if y >= 880 and x >= 500:
        return True
    if y >= PLANE_EXCLUDE_Y_MIN and x >= PLANE_EXCLUDE_X_MIN:
        if blob_area >= 3000 or blob_width > blob_height * 1.15:
            return True
    return False


def _is_ui_diff_blob(
    center_y: int, *, blob_height: int, blob_width: int
) -> bool:
    if center_y <= UI_EXCLUDE_Y_MAX:
        return True
    return blob_height <= 20 and blob_width >= 80


def _patch_vivid_ratio(hsv: np.ndarray, center: tuple[int, int], radius: int = 22) -> float:
    y, x = center[1], center[0]
    y1 = max(0, y - radius)
    y2 = min(hsv.shape[0], y + radius)
    x1 = max(0, x - radius)
    x2 = min(hsv.shape[1], x + radius)
    patch = hsv[y1:y2, x1:x2]
    if patch.size == 0:
        return 0.0
    hue, sat, val = patch[:, :, 0], patch[:, :, 1], patch[:, :, 2]
    # 橙/蓝图钉较亮；帐篷紫偏暗，单独放宽 V
    bright = (sat > 100) & (val > 130)
    purple = ((hue >= 125) & (hue <= 175)) & (sat > 90) & (val > 70)
    return float((bright | purple).mean())


def _pin_diff_support(
    center: tuple[int, int],
    diff_mask: np.ndarray,
    roi_offset: tuple[int, int],
    *,
    radius: int = BG_PIN_DIFF_SEARCH_RADIUS,
) -> tuple[float, tuple[int, int]]:
    """在图钉附近搜索差分最强的位置，返回 (差分占比,  refined_center)。"""
    ox, oy = roi_offset
    base_rx, base_ry = center[0] - ox, center[1] - oy
    best_ratio = 0.0
    best_center = center
    patch_r = 20

    for dy in range(-radius, radius + 1, 4):
        for dx in range(-radius, radius + 1, 4):
            rx, ry = base_rx + dx, base_ry + dy
            if (
                rx < patch_r
                or ry < patch_r
                or rx >= diff_mask.shape[1] - patch_r
                or ry >= diff_mask.shape[0] - patch_r
            ):
                continue
            patch = diff_mask[ry - patch_r : ry + patch_r, rx - patch_r : rx + patch_r]
            ratio = float(patch.mean()) / 255.0
            if ratio > best_ratio:
                best_ratio = ratio
                best_center = (ox + rx, oy + ry)
    return best_ratio, best_center


def _merge_nearby_pin_candidates(
    candidates: list[tuple[tuple[int, int], float]],
    *,
    merge_distance: int = MISSION_PIN_MERGE_DISTANCE,
) -> list[tuple[tuple[int, int], float]]:
    """合并邻近图钉候选，保留置信度最高者。"""
    ordered = sorted(candidates, key=lambda item: item[1], reverse=True)
    merged: list[tuple[tuple[int, int], float]] = []
    for center, confidence in ordered:
        if any(
            abs(center[0] - existing[0]) < merge_distance
            and abs(center[1] - existing[1]) < merge_distance
            for existing, _ in merged
        ):
            continue
        merged.append((center, confidence))
    return merged


def _contour_white_ratio(gray: np.ndarray, contour: np.ndarray) -> float:
    x, y, w, h = cv2.boundingRect(contour)
    if w <= 0 or h <= 0:
        return 0.0
    sub_gray = gray[y : y + h, x : x + w]
    return float((sub_gray > 190).mean())

def _pin_contour_passes_filters(
    contour: np.ndarray,
    gray: np.ndarray,
    *,
    area_min: int,
    area_max: int,
    white_max: float,
    aspect_min: float = MISSION_PIN_MIN_ASPECT,
    aspect_max: float = MISSION_PIN_MAX_ASPECT,
    circularity_min: float = MISSION_PIN_MIN_CIRCULARITY,
) -> bool:
    area = cv2.contourArea(contour)
    if area < area_min or area > area_max:
        return False
    if _contour_white_ratio(gray, contour) > white_max:
        return False
    perimeter = cv2.arcLength(contour, True)
    if perimeter <= 0:
        return False
    circularity = 4 * math.pi * area / (perimeter * perimeter)
    if (
        circularity < circularity_min
        or circularity > MISSION_PIN_MAX_CIRCULARITY
    ):
        return False
    x, y, w, h = cv2.boundingRect(contour)
    extent = area / max(w * h, 1)
    aspect = h / max(w, 1)
    if extent < MISSION_PIN_MIN_EXTENT or extent > MISSION_PIN_MAX_EXTENT:
        return False
    if aspect < aspect_min or aspect > aspect_max:
        return False
    return True


def _contour_vivid_peak_center(
    contour: np.ndarray,
    hsv: np.ndarray,
    roi_offset: tuple[int, int],
    *,
    local_offset: tuple[int, int] = (0, 0),
) -> tuple[int, int] | None:
    x, y, w, h = cv2.boundingRect(contour)
    if w <= 0 or h <= 0:
        return None
    lx, ly = local_offset
    mask = np.zeros((h, w), dtype=np.uint8)
    shifted = contour.copy()
    shifted[:, 0, 0] -= x
    shifted[:, 0, 1] -= y
    cv2.drawContours(mask, [shifted], -1, 255, thickness=-1)
    sat = hsv[y : y + h, x : x + w, 1].astype(np.float32)
    val = hsv[y : y + h, x : x + w, 2]
    score_map = sat * (val > 130)
    score_map[mask == 0] = 0.0
    if float(score_map.max()) <= 0:
        return None
    _, _, _, max_loc = cv2.minMaxLoc(score_map)
    ox, oy = roi_offset
    return (ox + lx + x + max_loc[0], oy + ly + y + max_loc[1])


def _contour_screen_center(
    contour: np.ndarray,
    roi_offset: tuple[int, int],
    *,
    local_offset: tuple[int, int] = (0, 0),
    hsv: np.ndarray | None = None,
    use_vivid_peak: bool = False,
) -> tuple[int, int] | None:
    if use_vivid_peak and hsv is not None:
        peak = _contour_vivid_peak_center(
            contour, hsv, roi_offset, local_offset=local_offset
        )
        if peak is not None:
            return peak
    moments = cv2.moments(contour)
    if moments["m00"] <= 0:
        return None
    ox, oy = roi_offset
    lx, ly = local_offset
    cx = int(moments["m10"] / moments["m00"]) + lx
    cy = int(moments["m01"] / moments["m00"]) + ly
    return (ox + cx, oy + cy)


def _screen_center_in_map(center: tuple[int, int]) -> bool:
    x, y = center
    return (
        MAP_PIN_MIN_SCREEN_Y <= y <= MAP_PIN_MAX_SCREEN_Y
        and MAP_PIN_MIN_SCREEN_X <= x <= MAP_PIN_MAX_SCREEN_X
    )

def _append_pin_candidates_from_contours(
    contours: list[np.ndarray],
    gray: np.ndarray,
    roi_offset: tuple[int, int],
    candidates: list[tuple[tuple[int, int], float]],
    *,
    area_min: int = MISSION_PIN_MIN_AREA,
    area_max: int = MISSION_PIN_MAX_AREA,
    white_max: float = PIN_BLOB_MAX_WHITE_RATIO,
    aspect_min: float = MISSION_PIN_MIN_ASPECT,
    aspect_max: float = MISSION_PIN_MAX_ASPECT,
    circularity_min: float = MISSION_PIN_MIN_CIRCULARITY,
    local_offset: tuple[int, int] = (0, 0),
    screen_y_min: int = MAP_PIN_MIN_SCREEN_Y,
    hsv: np.ndarray | None = None,
    vivid_peak_min_area: float = 0.0,
) -> None:
    for contour in contours:
        _check_scan_interrupted()
        if not _pin_contour_passes_filters(
            contour,
            gray,
            area_min=area_min,
            area_max=area_max,
            white_max=white_max,
            aspect_min=aspect_min,
            aspect_max=aspect_max,
            circularity_min=circularity_min,
        ):
            continue
        area = cv2.contourArea(contour)
        center = _contour_screen_center(
            contour,
            roi_offset,
            local_offset=local_offset,
            hsv=hsv,
            use_vivid_peak=(
                hsv is not None
                and vivid_peak_min_area > 0
                and area >= vivid_peak_min_area
            ),
        )
        if center is None or center[1] < screen_y_min:
            continue
        if not _screen_center_in_map(center):
            continue
        if _is_ignored_background_pin(
            center, contour_area=area, hsv=hsv, roi_offset=roi_offset
        ):
            continue
        candidates.append((center, cv2.contourArea(contour)))


def _append_fragment_centers_from_large_blobs(
    pin_mask: np.ndarray,
    gray: np.ndarray,
    roi_offset: tuple[int, int],
    candidates: list[tuple[tuple[int, int], float]],
    *,
    hsv: np.ndarray | None = None,
    min_area: float = PIN_LARGE_BLOB_MIN_AREA,
    erode_iters: int = 2,
) -> None:
    """大地形色块内用腐蚀切分，找回被粘连的任务图钉。"""
    kernel = np.ones((7, 7), np.uint8)
    for contour in cv2.findContours(
        pin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )[0]:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        local = pin_mask[y : y + h, x : x + w]
        if local.size == 0:
            continue
        eroded = cv2.erode(local, kernel, iterations=erode_iters)
        eroded = cv2.morphologyEx(eroded, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        sub_gray = gray[y : y + h, x : x + w]
        sub_hsv = hsv[y : y + h, x : x + w] if hsv is not None else None
        _append_pin_candidates_from_contours(
            cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0],
            sub_gray,
            roi_offset,
            candidates,
            area_min=60,
            area_max=600,
            local_offset=(x, y),
            hsv=sub_hsv,
            vivid_peak_min_area=120.0,
        )


def _find_mission_pin_centers(
    roi: np.ndarray,
    roi_offset: tuple[int, int],
) -> tuple[tuple[int, int], ...]:
    """情报地图上彩色任务图钉中心（橙/紫/蓝紧凑水滴形）。"""
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    pin_mask = np.zeros(roi.shape[:2], dtype=np.uint8)
    for lower, upper in (
        ((10, 130, 150), (28, 255, 255)),
        ((125, 60, 100), (155, 255, 255)),
        ((98, 70, 110), (118, 255, 255)),
    ):
        pin_mask = cv2.bitwise_or(
            pin_mask, cv2.inRange(hsv, np.array(lower), np.array(upper))
        )

    pin_closed = cv2.morphologyEx(pin_mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    pin_opened = cv2.morphologyEx(
        pin_closed,
        cv2.MORPH_OPEN,
        np.ones((PIN_BLOB_OPEN_KERNEL, PIN_BLOB_OPEN_KERNEL), np.uint8),
    )

    candidates: list[tuple[tuple[int, int], float]] = []
    _append_pin_candidates_from_contours(
        cv2.findContours(pin_opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0],
        gray,
        roi_offset,
        candidates,
        area_min=60,
        hsv=hsv,
        vivid_peak_min_area=180.0,
    )
    _append_fragment_centers_from_large_blobs(
        pin_closed, gray, roi_offset, candidates, hsv=hsv, min_area=950.0
    )

    orange_hero_mask = cv2.inRange(
        hsv,
        np.array((8, 90, 130)),
        np.array((28, 255, 255)),
    )
    orange_hero_mask = cv2.morphologyEx(
        orange_hero_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)
    )
    priority_candidates: list[tuple[tuple[int, int], float]] = []
    _append_pin_candidates_from_contours(
        cv2.findContours(orange_hero_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0],
        gray,
        roi_offset,
        priority_candidates,
        area_min=50,
        area_max=450,
    )

    hero_mask = cv2.inRange(
        hsv,
        np.array((155, 85, 150)),
        np.array((175, 255, 255)),
    )
    hero_mask = cv2.morphologyEx(hero_mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    hero_mask = cv2.morphologyEx(hero_mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    _append_pin_candidates_from_contours(
        cv2.findContours(hero_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0],
        gray,
        roi_offset,
        candidates,
        area_min=60,
        area_max=500,
    )

    blue_claw_mask = cv2.inRange(
        hsv,
        np.array((98, 80, 120)),
        np.array((118, 255, 255)),
    )
    blue_claw_mask = cv2.morphologyEx(
        blue_claw_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)
    )
    _append_pin_candidates_from_contours(
        cv2.findContours(blue_claw_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0],
        gray,
        roi_offset,
        priority_candidates,
        area_min=50,
        area_max=400,
    )

    orange_claw_mask = cv2.inRange(
        hsv,
        np.array((10, 100, 140)),
        np.array((26, 255, 255)),
    )
    orange_claw_mask = cv2.morphologyEx(
        orange_claw_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)
    )
    _append_pin_candidates_from_contours(
        cv2.findContours(orange_claw_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0],
        gray,
        roi_offset,
        priority_candidates,
        area_min=50,
        area_max=450,
    )
    candidates.extend(priority_candidates)

    tent_mask = cv2.inRange(
        hsv,
        np.array((150, 18, 170)),
        np.array((175, 55, 255)),
    )
    tent_mask = cv2.morphologyEx(tent_mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    tent_mask = cv2.morphologyEx(tent_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    _append_pin_candidates_from_contours(
        cv2.findContours(tent_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0],
        gray,
        roi_offset,
        candidates,
        area_min=TENT_PIN_BLOB_MIN_AREA,
        area_max=TENT_PIN_BLOB_MAX_AREA,
        white_max=PIN_BLOB_TENT_MAX_WHITE_RATIO,
        aspect_max=3.2,
        circularity_min=0.15,
    )

    blue_tent_mask = cv2.inRange(
        hsv,
        np.array((98, 70, 120)),
        np.array((118, 255, 255)),
    )
    blue_tent_mask = cv2.morphologyEx(
        blue_tent_mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8)
    )
    blue_tent_mask = cv2.morphologyEx(
        blue_tent_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)
    )
    _append_pin_candidates_from_contours(
        cv2.findContours(blue_tent_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0],
        gray,
        roi_offset,
        candidates,
        area_min=60,
        area_max=900,
        white_max=PIN_BLOB_TENT_MAX_WHITE_RATIO,
        aspect_max=3.5,
        circularity_min=0.12,
    )
    _append_fragment_centers_from_large_blobs(
        blue_tent_mask,
        gray,
        roi_offset,
        candidates,
        hsv=hsv,
        min_area=750.0,
        erode_iters=3,
    )

    candidates.sort(key=lambda item: item[1], reverse=True)
    merged: list[tuple[int, int]] = []
    priority_centers = [c for c, _ in priority_candidates]
    for center in priority_centers:
        if any(
            abs(center[0] - existing[0]) < MISSION_PIN_MERGE_DISTANCE
            and abs(center[1] - existing[1]) < MISSION_PIN_MERGE_DISTANCE
            for existing in merged
        ):
            continue
        merged.append(center)
    for center, _area in candidates:
        if any(
            abs(center[0] - existing[0]) < MISSION_PIN_MERGE_DISTANCE
            and abs(center[1] - existing[1]) < MISSION_PIN_MERGE_DISTANCE
            for existing in merged
        ):
            continue
        merged.append(center)

    if len(merged) > LIGHTHOUSE_MAX_PIN_CANDIDATES:
        priority_set = set(priority_centers)
        priority_kept = [c for c in merged if c in priority_set]
        others = [c for c in merged if c not in priority_set]
        merged = priority_kept + others[: max(0, LIGHTHOUSE_MAX_PIN_CANDIDATES - len(priority_kept))]
    return tuple(merged)
def _mission_slot_distance(a: LighthouseMission, b: LighthouseMission) -> float:
    return max(abs(a.center[0] - b.center[0]), abs(a.center[1] - b.center[1]))


def _dedupe_missions_by_slot(
    missions: list[LighthouseMission],
    *,
    merge_distance: float = LIGHTHOUSE_SLOT_MERGE_DISTANCE,
) -> list[LighthouseMission]:
    ordered = sorted(missions, key=lambda item: item.confidence, reverse=True)
    kept: list[LighthouseMission] = []
    for mission in ordered:
        if any(
            _mission_slot_distance(mission, existing) < merge_distance
            for existing in kept
        ):
            continue
        kept.append(mission)
    return kept
def _detail_action_roi_patch(
    screen: np.ndarray,
    roi: tuple[int, int, int, int],
) -> tuple[np.ndarray, tuple[int, int]]:
    screen = _normalize_screen_for_scan(screen)
    x1, y1, x2, y2 = roi
    return screen[y1:y2, x1:x2], (x1, y1)


def _prepare_action_button_template(template_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """裁掉模板四周浅灰背景，只保留按钮本体（含白字）用于匹配。"""
    hsv = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2HSV)
    sat, val = hsv[:, :, 1], hsv[:, :, 2]
    color_mask = ((sat > 45) & (val > 85)).astype(np.uint8) * 255
    white_mask = ((sat < 45) & (val > 165)).astype(np.uint8) * 255
    mask = cv2.bitwise_or(color_mask, white_mask)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    coords = cv2.findNonZero(mask)
    if coords is None:
        h, w = template_bgr.shape[:2]
        return template_bgr, np.full((h, w), 255, dtype=np.uint8)

    x, y, w, h = cv2.boundingRect(coords)
    pad = 2
    y1 = max(0, y - pad)
    x1 = max(0, x - pad)
    y2 = min(template_bgr.shape[0], y + h + pad)
    x2 = min(template_bgr.shape[1], x + w + pad)
    return template_bgr[y1:y2, x1:x2], mask[y1:y2, x1:x2]


def _load_action_button_template(template_name: str, template_dir: Path) -> tuple[np.ndarray, np.ndarray] | None:
    cache_key = f"{template_dir}/{template_name}"
    if cache_key in _ACTION_BTN_TEMPLATE_CACHE:
        return _ACTION_BTN_TEMPLATE_CACHE[cache_key]

    template_path = template_dir / template_name
    if not template_path.is_file():
        return None
    template = cv2.imread(str(template_path))
    if template is None:
        return None

    prepared = _prepare_action_button_template(template)
    _ACTION_BTN_TEMPLATE_CACHE[cache_key] = prepared
    return prepared


def _score_masked_at(
    patch_f: np.ndarray,
    t_z: np.ndarray,
    t_norm: float,
    m: np.ndarray,
    n: float,
    x: int,
    y: int,
    th: int,
    tw: int,
) -> float:
    sub = patch_f[y : y + th, x : x + tw]
    s_mean = (sub * m[..., None]).sum(axis=(0, 1)) / n
    s_z = (sub - s_mean) * m[..., None]
    s_norm = float(np.sqrt((s_z ** 2).sum())) + 1e-6
    return float((s_z * t_z).sum() / (s_norm * t_norm))


def _match_masked_ncc(
    patch: np.ndarray,
    template: np.ndarray,
    mask: np.ndarray,
    *,
    search_radius: int = 24,
) -> tuple[float, int, int, int, int]:
    """仅在 mask 覆盖的按钮像素上计算归一化互相关（忽略模板背景）。"""
    m = (mask > 0).astype(np.float32)
    n = float(m.sum())
    th, tw = template.shape[:2]
    if n < 20:
        return 0.0, 0, 0, tw, th

    tpl_f = template.astype(np.float32)
    t_mean = (tpl_f * m[..., None]).sum(axis=(0, 1)) / n
    t_z = (tpl_f - t_mean) * m[..., None]
    t_norm = float(np.sqrt((t_z ** 2).sum())) + 1e-6

    ph, pw = patch.shape[:2]
    if th > ph or tw > pw:
        return 0.0, 0, 0, tw, th

    patch_f = patch.astype(np.float32)
    cx = (pw - tw) // 2
    cy = (ph - th) // 2
    best_score = -1.0
    best_x = best_y = 0

    for dy in range(-search_radius, search_radius + 1, 2):
        for dx in range(-search_radius, search_radius + 1, 2):
            x, y = cx + dx, cy + dy
            if x < 0 or y < 0 or x + tw > pw or y + th > ph:
                continue
            score = _score_masked_at(patch_f, t_z, t_norm, m, n, x, y, th, tw)
            if score > best_score:
                best_score = score
                best_x, best_y = x, y

    if best_score >= 0:
        return best_score, best_x, best_y, tw, th
    return 0.0, cx, cy, tw, th


def _match_masked_ncc_multiscale(
    patch: np.ndarray,
    template: np.ndarray,
    mask: np.ndarray,
    scales: tuple[float, ...],
) -> tuple[float, tuple[int, int, int, int]]:
    best_score = 0.0
    best_box = (0, 0, template.shape[1], template.shape[0])
    for scale in scales:
        if scale == 1.0:
            tpl, m = template, mask
        else:
            tpl = cv2.resize(
                template,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR,
            )
            m = cv2.resize(mask, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        score, x, y, tw, th = _match_masked_ncc(patch, tpl, m)
        if score > best_score:
            best_score = score
            best_box = (x, y, tw, th)
    return best_score, best_box

def _match_action_button_in_screen_roi(
    screen: np.ndarray,
    template_name: str,
    roi: tuple[int, int, int, int],
    *,
    template_dir: Path = TEMPLATE_DIR,
    threshold: float = MISSION_DETAIL_BTN_THRESHOLD,
    scales: tuple[float, ...] = MISSION_DETAIL_BTN_SCALES,
) -> MatchResult:
    """行动按钮：去背景 + 掩膜匹配（避免模板浅灰底拉低得分）。"""
    patch, (ox, oy) = _detail_action_roi_patch(screen, roi)
    if patch.size == 0:
        return MatchResult(found=False)

    prepared = _load_action_button_template(template_name, template_dir)
    if prepared is None:
        logger.warning(f"灯塔详情模板缺失: {template_name}")
        return MatchResult(found=False)

    template, mask = prepared
    best_conf, (x, y, tw, th) = _match_masked_ncc_multiscale(patch, template, mask, scales)
    logger.debug(f"行动按钮 {template_name}: masked={best_conf:.2f}")

    center = (ox + x + tw // 2, oy + y + th // 2)
    return MatchResult(
        found=best_conf >= threshold,
        confidence=best_conf,
        center=center,
        top_left=(ox + x, oy + y),
        size=(tw, th),
    )


def _classify_detail_action_by_color(
    screen: np.ndarray,
    action_roi: tuple[int, int, int, int],
) -> tuple[str, str, float, tuple[int, int]] | None:
    """按行动按钮主色兜底分类（模板得分偏低时）。"""
    patch, (ox, oy) = _detail_action_roi_patch(screen, action_roi)
    if patch.size == 0:
        return None

    h, w = patch.shape[:2]
    margin_x = max(2, int(w * 0.04))
    margin_y = max(2, int(h * 0.08))
    center_patch = patch[margin_y : h - margin_y, margin_x : w - margin_x]
    if center_patch.size == 0:
        center_patch = patch

    hsv = cv2.cvtColor(center_patch, cv2.COLOR_BGR2HSV)
    scored: list[tuple[str, str, float]] = []

    for kind, label, _color_key, lower, upper in _DETAIL_BTN_HSV:
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        ratio = float(mask.mean()) / 255.0
        scored.append((kind, label, ratio))
        logger.debug(f"详情按钮颜色 {label} ratio={ratio:.3f}")

    scored.sort(key=lambda item: item[2], reverse=True)
    if len(scored) < 2:
        return None

    best_kind, best_label, best_ratio = scored[0]
    second_ratio = scored[1][2]
    if best_ratio < MISSION_DETAIL_COLOR_MIN_RATIO:
        return None
    if best_ratio < second_ratio + MISSION_DETAIL_COLOR_MIN_MARGIN:
        return None

    # 色块质心作为点击位置
    hsv_full = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    for kind, _label, _key, lower, upper in _DETAIL_BTN_HSV:
        if kind != best_kind:
            continue
        mask = cv2.inRange(hsv_full, np.array(lower), np.array(upper))
        moments = cv2.moments(mask)
        if moments["m00"] > 0:
            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])
            tap = (ox + cx, oy + cy)
        else:
            tap = ((action_roi[0] + action_roi[2]) // 2, (action_roi[1] + action_roi[3]) // 2)
        return best_kind, best_label, best_ratio, tap

    return None


def _template_conf_for_kind(
    scored: list[tuple[str, str, str, MatchResult]], kind: str
) -> float:
    for k, _label, _tpl, res in scored:
        if k == kind:
            return res.confidence
    return 0.0


def _refine_action_with_button_color(
    resolved_kind: str,
    resolved_label: str,
    resolved_conf: float,
    action_center: tuple[int, int] | None,
    scored: list[tuple[str, str, str, MatchResult]],
    screen: np.ndarray,
    action_roi: tuple[int, int, int, int],
) -> tuple[str, str, float, tuple[int, int] | None]:
    """模板与按钮主色不一致时，以颜色为准（蓝=探险，绿=营救，橙=出征）。"""
    if resolved_kind not in ("tent", "hero_journey", "small_monster"):
        return resolved_kind, resolved_label, resolved_conf, action_center

    color_hit = _classify_detail_action_by_color(screen, action_roi)
    if color_hit is None:
        return resolved_kind, resolved_label, resolved_conf, action_center

    c_kind, c_label, c_ratio, c_center = color_hit
    if c_kind == resolved_kind:
        return resolved_kind, resolved_label, resolved_conf, action_center or c_center

    tpl_for_color = _template_conf_for_kind(scored, c_kind)
    color_kinds = {resolved_kind, c_kind}
    is_blue_green_swap = color_kinds == {"tent", "hero_journey"}

    strong_color = (
        c_ratio >= MISSION_DETAIL_COLOR_STRONG_RATIO
        and tpl_for_color >= MISSION_DETAIL_COLOR_STRONG_MIN_TEMPLATE
    )
    soft_color = (
        is_blue_green_swap
        and c_ratio >= MISSION_DETAIL_COLOR_MIN_RATIO
        and tpl_for_color >= MISSION_DETAIL_COLOR_BACKUP_MIN_TEMPLATE
    )

    if strong_color or soft_color:
        logger.info(
            f"行动按钮颜色修正：模板 {resolved_label}({resolved_conf:.2f}) "
            f"→ {c_label}（color={c_ratio:.2f} tpl={tpl_for_color:.2f}）"
        )
        return (
            c_kind,
            c_label,
            max(resolved_conf, c_ratio, tpl_for_color),
            c_center,
        )

    return resolved_kind, resolved_label, resolved_conf, action_center


def _detail_action_center_fallback(
    action_roi: tuple[int, int, int, int],
) -> tuple[int, int]:
    return ((action_roi[0] + action_roi[2]) // 2, (action_roi[1] + action_roi[3]) // 2)


def _match_subtitle_template(
    screen: np.ndarray,
    template_name: str,
    subtitle_roi: tuple[int, int, int, int],
    *,
    template_dir: Path = TEMPLATE_DIR,
    threshold: float = MISSION_DETAIL_BOUNTY_THRESHOLD,
    scales: tuple[float, ...] = BOUNTY_SUBTITLE_SCALES,
    search_side: str = "left",
    search_width_ratio: float = BOUNTY_SUBTITLE_SEARCH_WIDTH_RATIO,
) -> MatchResult:
    """副标题文字模板匹配。

    search_side:
      left  — 仅在 ROI 左侧搜索（「等级」、悬赏等靠左文案）
      right — 仅在 ROI 右侧搜索（大怪「号」靠右）
      full  — 全宽搜索
    """
    patch, (ox, oy) = _detail_action_roi_patch(screen, subtitle_roi)
    if patch.size == 0:
        return MatchResult(found=False)

    pw = patch.shape[1]
    search_w = max(1, int(pw * search_width_ratio))
    if search_side == "right":
        search_start = max(0, pw - search_w)
        search_patch = patch[:, search_start:]
        x_offset = search_start
    elif search_side == "full":
        search_patch = patch
        x_offset = 0
    else:
        search_patch = patch[:, :search_w]
        x_offset = 0

    template_path = template_dir / template_name
    if not template_path.is_file():
        logger.warning(f"灯塔详情模板缺失: {template_name}")
        return MatchResult(found=False)

    template = cv2.imread(str(template_path))
    if template is None:
        return MatchResult(found=False)

    patch_gray = cv2.cvtColor(search_patch, cv2.COLOR_BGR2GRAY)
    tpl_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    best_conf = 0.0
    best_loc: tuple[int, int, int, int] | None = None

    for scale in scales:
        scaled_gray = cv2.resize(
            tpl_gray,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR,
        )
        scaled_bgr = cv2.resize(
            template,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR,
        )
        th, tw = scaled_gray.shape[:2]
        if th > patch_gray.shape[0] or tw > patch_gray.shape[1]:
            continue

        gray_result = cv2.matchTemplate(patch_gray, scaled_gray, cv2.TM_CCOEFF_NORMED)
        color_result = cv2.matchTemplate(search_patch, scaled_bgr, cv2.TM_CCOEFF_NORMED)
        gray_score = float(gray_result.max())
        color_score = float(color_result.max())
        if color_score >= gray_score:
            max_val = color_score
            _, _, _, max_loc = cv2.minMaxLoc(color_result)
        else:
            max_val = gray_score
            _, _, _, max_loc = cv2.minMaxLoc(gray_result)
        if max_val <= best_conf:
            continue
        x, y = max_loc
        best_conf = max_val
        best_loc = (x, y, tw, th)

    if best_loc is None:
        return MatchResult(found=False, confidence=0.0)

    x, y, tw, th = best_loc
    center = (ox + x_offset + x + tw // 2, oy + y + th // 2)
    found = best_conf >= threshold
    return MatchResult(
        found=found,
        confidence=best_conf,
        center=center,
        top_left=(ox + x_offset + x, oy + y),
        size=(tw, th),
    )


def _normalize_subtitle_ocr(raw: str) -> str:
    text = (raw or "").replace(" ", "").replace("\n", "").replace("　", "")
    text = text.replace("：", ":").replace(":", "")
    for wrong, right in (
        ("等缓", "等级"),
        ("等经", "等级"),
        ("等缓", "等级"),
        ("大师悬赏", "大师悬赏"),
        ("宗师悬赏", "宗师悬赏"),
        ("悬赏", "悬赏"),
    ):
        text = text.replace(wrong, right)
    return text


def _prepare_subtitle_ocr_patch(patch: np.ndarray) -> np.ndarray:
    if patch.size == 0:
        return patch
    scaled = cv2.resize(patch, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(scaled, cv2.COLOR_BGR2GRAY)
    norm = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.cvtColor(norm, cv2.COLOR_GRAY2BGR)


def _ocr_subtitle_text(
    screen: np.ndarray,
    subtitle_roi: tuple[int, int, int, int],
) -> str:
    patch, _origin = _detail_action_roi_patch(screen, subtitle_roi)
    if patch.size == 0:
        return ""
    best = ""
    variants = (
        _prepare_subtitle_ocr_patch(patch),
        cv2.resize(patch, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC),
    )
    for variant in variants:
        if variant.size == 0:
            continue
        text, _engine = ocr_chip_text(variant)
        cleaned = _normalize_subtitle_ocr(text)
        if len(cleaned) > len(best):
            best = cleaned
        if best:
            break
    return best


def _normalize_action_button_ocr(raw: str) -> str:
    text = (raw or "").replace(" ", "").replace("\n", "").replace("　", "")
    text = re.sub(r"\d+", "", text)
    if re.fullmatch(r"出[征前证径无开成]?", text):
        return "出征"
    for wrong, right in (
        ("出证", "出征"),
        ("出前", "出征"),
        ("出无", "出征"),
        ("出径", "出征"),
        ("出成", "出征"),
        ("出开", "出征"),
        ("探检", "探险"),
        ("探除", "探险"),
        ("探验", "探险"),
        ("营教", "营救"),
        ("救营", "营救"),
        ("营求", "营救"),
    ):
        text = text.replace(wrong, right)
    return text


def _parse_action_button_ocr(text: str) -> tuple[str, str] | None:
    """从 OCR 文本解析行动按钮类型。"""
    normalized = _normalize_action_button_ocr(text)
    if not normalized:
        return None
    for keyword, kind, label in DETAIL_ACTION_TEXT_KINDS:
        if keyword in normalized:
            return kind, label
    if normalized.startswith("出") and len(normalized) <= 3:
        if any(ch in normalized for ch in "征前证径"):
            return "small_monster", "出征"
    return None


def _prepare_action_button_ocr_patch(patch: np.ndarray) -> np.ndarray:
    """裁掉体力图标行，增强对比度后供 OCR。"""
    if patch.size == 0:
        return patch
    h, w = patch.shape[:2]
    text_area = patch[: max(1, int(h * 0.58)), :]
    scaled = cv2.resize(
        text_area, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC
    )
    gray = cv2.cvtColor(scaled, cv2.COLOR_BGR2GRAY)
    norm = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.cvtColor(norm, cv2.COLOR_GRAY2BGR)


def _action_button_ocr_variants(patch: np.ndarray) -> list[np.ndarray]:
    """行动按钮 OCR 多路预处理。"""
    if patch.size == 0:
        return []

    variants: list[np.ndarray] = [_prepare_action_button_ocr_patch(patch)]
    full_scaled = cv2.resize(patch, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
    variants.append(full_scaled)

    top = patch[: max(1, int(patch.shape[0] * 0.58)), :]
    top_scaled = cv2.resize(top, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(top_scaled, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(top_scaled, cv2.COLOR_BGR2HSV)
    white = (gray > 175) & (hsv[:, :, 1] < 90)
    isolated = np.zeros_like(top_scaled)
    isolated[white] = (255, 255, 255)
    variants.append(isolated)
    return variants


def _classify_detail_action_by_text(
    screen: np.ndarray,
    action_roi: tuple[int, int, int, int],
) -> tuple[str, str, float, tuple[int, int]] | None:
    """OCR 识别行动按钮主文字（出征 / 探险 / 营救）。"""
    patch, _origin = _detail_action_roi_patch(screen, action_roi)
    if patch.size == 0:
        return None

    variants = _action_button_ocr_variants(patch)
    for variant in variants:
        if variant.size == 0:
            continue
        text, _engine = ocr_chip_text(variant)
        parsed = _parse_action_button_ocr(text)
        if parsed is None:
            continue
        kind, label = parsed
        action_center = _detail_action_center_fallback(action_roi)
        color_hit = _classify_detail_action_by_color(screen, action_roi)
        if color_hit is not None and color_hit[0] == kind:
            action_center = color_hit[3]
        return kind, label, MISSION_DETAIL_ACTION_OCR_CONF, action_center
    return None


def _classify_march_subtitle_from_text(
    text: str,
) -> MissionDetailClassification | None:
    """出征任务：根据副标题 OCR 文本区分悬赏 / 灯塔小怪 / 特殊大怪。"""
    if not text:
        return None

    if "大师悬赏" in text or "宗师悬赏" in text:
        return MissionDetailClassification(
            kind="bounty_skip",
            label="大师/宗师悬赏",
            confidence=MISSION_DETAIL_ACTION_OCR_CONF,
        )

    if text.startswith("等级") or re.match(r"^等[级经缓]", text):
        return None

    if re.search(r"\d+号", text):
        return MissionDetailClassification(
            kind="beast_skip",
            label="特殊大怪",
            confidence=MISSION_DETAIL_ACTION_OCR_CONF,
            beast_explicit=True,
        )

    return None


def _classify_march_subtitle(
    screen: np.ndarray,
    *,
    subtitle_roi: tuple[int, int, int, int] = MISSION_DETAIL_SUBTITLE_ROI,
    template_dir: Path = TEMPLATE_DIR,
) -> tuple[MissionDetailClassification | None, str]:
    """出征任务副标题：仅 OCR 文本（悬赏 / 大怪 / 等级怪）。"""
    _ = template_dir
    text = _ocr_subtitle_text(screen, subtitle_roi)
    return _classify_march_subtitle_from_text(text), text


def _score_detail_action_templates(
    screen: np.ndarray,
    action_roi: tuple[int, int, int, int],
    *,
    template_dir: Path = TEMPLATE_DIR,
) -> list[tuple[str, str, str, MatchResult]]:
    scored: list[tuple[str, str, str, MatchResult]] = []
    for kind, label, template_name in DETAIL_ACTION_BUTTON_TEMPLATES:
        result = _match_action_button_in_screen_roi(
            screen,
            template_name,
            action_roi,
            template_dir=template_dir,
        )
        scored.append((kind, label, template_name, result))
    scored.sort(key=lambda item: item[3].confidence, reverse=True)
    return scored


def _pick_detail_action_template(
    scored: list[tuple[str, str, str, MatchResult]],
    action_roi: tuple[int, int, int, int],
) -> tuple[str, str, float, tuple[int, int]] | None:
    if not scored:
        return None
    best_kind, best_label, _template_name, best_result = scored[0]
    if best_result.confidence < MISSION_DETAIL_BTN_ACCEPT_SCORE:
        return None
    second_conf = scored[1][3].confidence if len(scored) > 1 else 0.0
    if best_result.confidence < second_conf + MISSION_DETAIL_BTN_MIN_MARGIN:
        return None
    center = best_result.center or _detail_action_center_fallback(action_roi)
    return best_kind, best_label, best_result.confidence, center


def _pick_detail_action_template_relaxed(
    scored: list[tuple[str, str, str, MatchResult]],
    action_roi: tuple[int, int, int, int],
) -> tuple[str, str, float, tuple[int, int]] | None:
    """放宽阈值：弹窗布局略有偏差时仍接受明显领先的模板。"""
    if not scored:
        return None
    best_kind, best_label, _template_name, best_result = scored[0]
    if best_result.confidence < DETAIL_ACTION_RELAXED_ACCEPT_SCORE:
        return None
    second_conf = scored[1][3].confidence if len(scored) > 1 else 0.0
    if best_result.confidence < second_conf + DETAIL_ACTION_RELAXED_MIN_MARGIN:
        return None
    center = best_result.center or _detail_action_center_fallback(action_roi)
    return best_kind, best_label, best_result.confidence, center


def _resolve_detail_action(
    screen: np.ndarray,
    *,
    template_dir: Path = TEMPLATE_DIR,
    action_roi: tuple[int, int, int, int] = MISSION_DETAIL_ACTION_ROI,
) -> tuple[str, str, float, tuple[int, int]] | None:
    """行动按钮：OCR → 模板 → 颜色，逐级兜底。"""
    rois = (action_roi, DETAIL_ACTION_ROI_FALLBACK, DETAIL_ACTION_BOTTOM_BAND)
    unique_rois: list[tuple[int, int, int, int]] = []
    for roi in rois:
        if roi not in unique_rois:
            unique_rois.append(roi)

    scored: list[tuple[str, str, str, MatchResult]] = []
    for roi in unique_rois:
        ocr_hit = _classify_detail_action_by_text(screen, roi)
        scored = _score_detail_action_templates(screen, roi, template_dir=template_dir)
        tpl_hit = _pick_detail_action_template(scored, roi)

        if ocr_hit is not None:
            kind, label, conf, center = ocr_hit
            if scored:
                kind, label, conf, center = _refine_action_with_button_color(
                    kind, label, conf, center, scored, screen, roi
                )
            logger.info(f"详情识别: 行动按钮 OCR → {label}({conf:.2f})")
            return kind, label, conf, center

        if tpl_hit is not None:
            kind, label, conf, center = tpl_hit
            if scored:
                kind, label, conf, center = _refine_action_with_button_color(
                    kind, label, conf, center, scored, screen, roi
                )
            logger.info(f"详情识别: 行动按钮模板 → {label}({conf:.2f})")
            return kind, label, max(conf, MISSION_DETAIL_BTN_ACCEPT_SCORE), center

        color_hit = _classify_detail_action_by_color(screen, roi)
        if color_hit is not None:
            c_kind, c_label, c_ratio, c_center = color_hit
            tpl_conf = _template_conf_for_kind(scored, c_kind)
            strong = c_ratio >= MISSION_DETAIL_COLOR_STRONG_RATIO and tpl_conf >= (
                MISSION_DETAIL_COLOR_STRONG_MIN_TEMPLATE
            )
            soft = (
                c_ratio >= MISSION_DETAIL_COLOR_MIN_RATIO
                and tpl_conf >= MISSION_DETAIL_COLOR_BACKUP_MIN_TEMPLATE
            )
            if strong or soft:
                conf = max(c_ratio, tpl_conf, MISSION_DETAIL_BTN_ACCEPT_SCORE)
                logger.info(
                    f"详情识别: 行动按钮颜色 → {c_label} "
                    f"(color={c_ratio:.2f} tpl={tpl_conf:.2f})"
                )
                return c_kind, c_label, conf, c_center

        relaxed_hit = _pick_detail_action_template_relaxed(scored, roi)
        if relaxed_hit is not None:
            kind, label, conf, center = relaxed_hit
            logger.info(f"详情识别: 行动按钮模板(放宽) → {label}({conf:.2f})")
            return kind, label, max(conf, MISSION_DETAIL_BTN_ACCEPT_SCORE), center

        if roi != unique_rois[-1]:
            logger.debug(
                "详情行动按钮 ROI 未命中，尝试下一档区域 "
                f"({roi[0]},{roi[1]})-({roi[2]},{roi[3]})"
            )

    if scored:
        best_kind, best_label, _tpl, best_result = scored[0]
        color_hit = _classify_detail_action_by_color(
            screen, unique_rois[-1]
        )
        if (
            color_hit is not None
            and color_hit[0] == "small_monster"
            and color_hit[2] >= DETAIL_ORANGE_ONLY_MIN_RATIO
            and best_kind == "small_monster"
            and best_result.confidence >= 0.28
        ):
            conf = max(
                color_hit[2],
                best_result.confidence,
                MISSION_DETAIL_BTN_ACCEPT_SCORE,
            )
            logger.info(
                f"详情识别: 底部橙色出征兜底 "
                f"(color={color_hit[2]:.2f} tpl={best_result.confidence:.2f})"
            )
            return "small_monster", "出征", conf, color_hit[3]

        logger.info(
            "详情识别: 行动按钮模板得分 "
            + ", ".join(
                f"{label}={result.confidence:.2f}"
                for _kind, label, _tpl, result in scored
            )
        )
    return None


def _classify_detail_from_subtitle_first(
    screen: np.ndarray,
    *,
    subtitle_roi: tuple[int, int, int, int] = MISSION_DETAIL_SUBTITLE_ROI,
    action_roi: tuple[int, int, int, int] = MISSION_DETAIL_ACTION_ROI,
    template_dir: Path = TEMPLATE_DIR,
) -> MissionDetailClassification | None:
    """行动按钮未命中时：仅 OCR 副标题文字兜底（悬赏 / 大怪）。"""
    march_sub, subtitle = _classify_march_subtitle(
        screen,
        subtitle_roi=subtitle_roi,
        template_dir=template_dir,
    )
    if march_sub is None:
        return None
    if march_sub.kind in ("bounty_skip", "beast_skip"):
        logger.info(f"详情识别: 副标题 OCR → {march_sub.label}（{subtitle}）")
        return MissionDetailClassification(
            kind=march_sub.kind,
            label=march_sub.label,
            confidence=march_sub.confidence,
            action_center=_detail_action_center_fallback(action_roi),
            beast_explicit=march_sub.beast_explicit,
        )
    return None


def mission_detail_action_ready(
    screen: np.ndarray,
    *,
    action_roi: tuple[int, int, int, int] = MISSION_DETAIL_ACTION_ROI,
    template_dir: Path = TEMPLATE_DIR,
) -> bool:
    """详情页是否可分类（行动按钮或强副标题特征）。"""
    if (
        _resolve_detail_action(
            screen,
            template_dir=template_dir,
            action_roi=action_roi,
        )
        is not None
    ):
        return True
    return (
        _classify_detail_from_subtitle_first(
            screen,
            action_roi=action_roi,
            template_dir=template_dir,
        )
        is not None
    )


def classify_mission_detail_screen(
    screen: np.ndarray,
    *,
    template_dir: Path = TEMPLATE_DIR,
    action_roi: tuple[int, int, int, int] = MISSION_DETAIL_ACTION_ROI,
    subtitle_roi: tuple[int, int, int, int] = MISSION_DETAIL_SUBTITLE_ROI,
) -> MissionDetailClassification:
    """详情页：行动按钮 OCR/模板/颜色；副标题仅 OCR 文字。"""
    action_hit = _resolve_detail_action(
        screen,
        template_dir=template_dir,
        action_roi=action_roi,
    )
    if action_hit is None:
        subtitle_first = _classify_detail_from_subtitle_first(
            screen,
            subtitle_roi=subtitle_roi,
            action_roi=action_roi,
            template_dir=template_dir,
        )
        if subtitle_first is not None:
            return subtitle_first
        logger.info("详情识别: 未识别（行动按钮与副标题均未命中）")
        return MissionDetailClassification(
            kind="unknown",
            label="未识别",
            confidence=0.0,
        )

    resolved_kind, resolved_label, resolved_conf, action_center = action_hit
    if action_center is None:
        action_center = _detail_action_center_fallback(action_roi)

    if resolved_kind == "small_monster":
        march_sub, subtitle = _classify_march_subtitle(
            screen,
            subtitle_roi=subtitle_roi,
            template_dir=template_dir,
        )
        if march_sub is not None:
            logger.info(
                f"详情识别: 出征 + 副标题 OCR「{subtitle}」→ {march_sub.label}"
            )
            return MissionDetailClassification(
                kind=march_sub.kind,
                label=march_sub.label,
                confidence=max(resolved_conf, march_sub.confidence),
                action_center=action_center,
                beast_explicit=march_sub.beast_explicit,
            )
        logger.info(f"详情识别: 出征 + 副标题 OCR「{subtitle}」→ 灯塔小怪")
        return MissionDetailClassification(
            kind="small_monster",
            label="灯塔小怪",
            confidence=resolved_conf,
            action_center=action_center,
        )

    logger.info(f"详情识别: {resolved_label}")
    return MissionDetailClassification(
        kind=resolved_kind,
        label=resolved_label,
        confidence=resolved_conf,
        action_center=action_center,
    )


def configure_lighthouse_scan(*, event_period: bool = False) -> str:
    """按是否活动期间切换差分背景图，切换时清空缓存。"""
    global _active_map_bg_name, _MAP_BG_SCREEN, _MAP_BG_ROI, _event_period
    desired = LIGHTHOUSE_MAP_BG_EVENT if event_period else LIGHTHOUSE_MAP_BG_NORMAL
    _event_period = event_period
    if desired != _active_map_bg_name:
        _active_map_bg_name = desired
        _MAP_BG_SCREEN = None
        _MAP_BG_ROI = None
    logger.info(f"灯塔扫描背景图: {_active_map_bg_name}")
    return _active_map_bg_name


def _is_event_period() -> bool:
    return _event_period


def _extract_pin_patch(
    roi: np.ndarray,
    screen_center: tuple[int, int],
    roi_offset: tuple[int, int],
    *,
    half: int = LIGHTHOUSE_PIN_PATCH_HALF,
) -> np.ndarray:
    cx = screen_center[0] - roi_offset[0]
    cy = screen_center[1] - roi_offset[1]
    x1 = max(0, cx - half)
    y1 = max(0, cy - half)
    x2 = min(roi.shape[1], cx + half)
    y2 = min(roi.shape[0], cy + half)
    return roi[y1:y2, x1:x2]


def is_beast_map_pin(
    screen: np.ndarray,
    center: tuple[int, int],
    *,
    scan_roi: tuple[int, int, int, int] = LIGHTHOUSE_SCAN_ROI,
) -> MatchResult:
    """地图图钉位置是否匹配特殊大怪（熊头/兽形橙钉）模板。"""
    screen = _normalize_screen_for_scan(screen)
    x1, y1, x2, y2 = scan_roi
    roi = screen[y1:y2, x1:x2]
    patch = _extract_pin_patch(roi, center, (x1, y1))
    if patch.size == 0:
        return MatchResult(found=False)
    template_path = TEMPLATE_DIR / BEAST_PIN_TEMPLATE
    if not template_path.is_file():
        return MatchResult(found=False)
    vision = Vision(TEMPLATE_DIR, threshold=BEAST_PIN_MATCH_MIN)
    result = vision.match_template_multiscale(patch, BEAST_PIN_TEMPLATE)
    if result.found:
        logger.debug(
            f"地图图钉 ({center[0]},{center[1]}) 匹配特殊大怪 "
            f"conf={result.confidence:.2f}"
        )
    return result


def tag_scanned_missions(
    screen: np.ndarray,
    missions: tuple[LighthouseMission, ...],
    *,
    scan_roi: tuple[int, int, int, int] = LIGHTHOUSE_SCAN_ROI,
) -> tuple[LighthouseMission, ...]:
    """扫描后补类型：标记特殊大怪图钉，供任务层跳过。"""
    if not missions:
        return missions

    tagged: list[LighthouseMission] = []
    for mission in missions:
        beast = is_beast_map_pin(screen, mission.center, scan_roi=scan_roi)
        if beast.found:
            tagged.append(
                LighthouseMission(
                    kind="small_monster_beast",
                    label="特殊大怪",
                    template=BEAST_PIN_TEMPLATE,
                    center=mission.center,
                    confidence=beast.confidence,
                    top_left=mission.top_left,
                    size=mission.size,
                )
            )
        else:
            tagged.append(mission)
    return tuple(tagged)


def scan_mission_icons(
    screen: np.ndarray,
    *,
    scan_roi: tuple[int, int, int, int] = LIGHTHOUSE_SCAN_ROI,
    interrupted: Callable[[], bool] | None = None,
) -> LighthouseScanResult:
    """检测地图上的任务图钉位置，不区分任务类型。

    相对固定背景图做差分，再结合鲜艳色 + 倒水滴轮廓过滤；
    自动排除背景中的飞机、超级大怪及中央基地等干扰。
    """
    global _scan_interrupt_cb
    prev_interrupt = _scan_interrupt_cb
    _scan_interrupt_cb = interrupted
    try:
        return _scan_icons_impl(screen, scan_roi=scan_roi)
    except InterruptedError:
        return LighthouseScanResult(mission=None, best_label="已停止")
    finally:
        _scan_interrupt_cb = prev_interrupt


def _load_map_background_screen() -> np.ndarray | None:
    """加载完整灯塔情报页背景（720×1280）。"""
    global _MAP_BG_SCREEN
    if _MAP_BG_SCREEN is not None:
        return _MAP_BG_SCREEN

    path = TEMPLATE_DIR / _active_map_bg_name
    image = cv2.imread(str(path))
    if image is None:
        logger.warning(f"灯塔地图背景缺失: {path}")
        return None

    _MAP_BG_SCREEN = _normalize_screen_for_scan(image)
    return _MAP_BG_SCREEN


def _load_map_background_roi() -> np.ndarray | None:
    """加载灯塔情报页地图背景（720×1280 竖屏 ROI 切片）。"""
    global _MAP_BG_ROI
    if _MAP_BG_ROI is not None:
        return _MAP_BG_ROI

    screen = _load_map_background_screen()
    if screen is None:
        return None

    x1, y1, x2, y2 = LIGHTHOUSE_SCAN_ROI
    _MAP_BG_ROI = screen[y1:y2, x1:x2].copy()
    return _MAP_BG_ROI


def is_lighthouse_intel_screen(screen: np.ndarray) -> bool:
    """当前截图是否为灯塔情报页（非野外大地图/出征界面）。"""
    normalized = _normalize_screen_for_scan(screen)
    bg = _load_map_background_screen()
    if bg is None:
        return False

    x1, y1, x2, y2 = LIGHTHOUSE_HEADER_ROI
    patch = normalized[y1:y2, x1:x2]
    reference = bg[y1:y2, x1:x2]
    if patch.shape != reference.shape:
        return False
    mean_diff = float(cv2.absdiff(patch, reference).mean())
    return mean_diff <= LIGHTHOUSE_HEADER_MEAN_DIFF_MAX


def _align_background_roi(
    bg_roi: np.ndarray, roi_shape: tuple[int, ...]
) -> np.ndarray:
    if bg_roi.shape[:2] == roi_shape[:2]:
        return bg_roi
    return cv2.resize(
        bg_roi,
        (roi_shape[1], roi_shape[0]),
        interpolation=cv2.INTER_LINEAR,
    )


def _build_vivid_pin_mask(hsv: np.ndarray) -> np.ndarray:
    """任务图钉常见高饱和色（橙 / 紫 / 蓝）。"""
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lower, upper in (
        ((10, 120, 140), (30, 255, 255)),
        # 帐篷紫偏暗，V 下限到 70，避免漏检
        ((125, 55, 70), (155, 255, 255)),
        ((98, 65, 110), (118, 255, 255)),
        ((155, 80, 70), (175, 255, 255)),
    ):
        mask = cv2.bitwise_or(
            mask, cv2.inRange(hsv, np.array(lower), np.array(upper))
        )
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return mask


def _contour_vivid_ratio(contour: np.ndarray, hsv: np.ndarray) -> float:
    x, y, w, h = cv2.boundingRect(contour)
    if w <= 0 or h <= 0:
        return 0.0
    patch_hsv = hsv[y : y + h, x : x + w]
    mask = np.zeros((h, w), dtype=np.uint8)
    shifted = contour.copy()
    shifted[:, 0, 0] -= x
    shifted[:, 0, 1] -= y
    cv2.drawContours(mask, [shifted], -1, 255, thickness=-1)
    sat = patch_hsv[:, :, 1]
    val = patch_hsv[:, :, 2]
    vivid = ((sat > 100) & (val > 130)).astype(np.float32)
    vivid[mask == 0] = 0.0
    if mask.sum() <= 0:
        return 0.0
    return float(vivid.sum() / (mask.sum() / 255.0))


def _is_teardrop_contour(contour: np.ndarray, *, relaxed: bool = False) -> bool:
    """倒水滴：上宽下窄、竖向略长。"""
    x, y, w, h = cv2.boundingRect(contour)
    area = cv2.contourArea(contour)
    area_max = BG_PIN_AREA_MAX if not relaxed else 3200
    if area < BG_PIN_AREA_MIN or area > area_max:
        return False
    aspect = h / max(w, 1)
    aspect_min = BG_TEARDROP_ASPECT_MIN if not relaxed else 0.75
    if aspect < aspect_min or aspect > BG_TEARDROP_ASPECT_MAX:
        return False

    mask = np.zeros((h, w), dtype=np.uint8)
    shifted = contour.copy()
    shifted[:, 0, 0] -= x
    shifted[:, 0, 1] -= y
    cv2.drawContours(mask, [shifted], -1, 255, thickness=-1)

    top_band = mask[: max(1, h // 4), :]
    bottom_band = mask[3 * h // 4 :, :]
    top_width = int(np.count_nonzero(top_band.any(axis=0)))
    bottom_width = int(np.count_nonzero(bottom_band.any(axis=0)))
    if bottom_width <= 0:
        return False
    ratio_min = BG_TEARDROP_TOP_BOTTOM_MIN if not relaxed else 1.0
    return (top_width / bottom_width) >= ratio_min


def _snap_to_vivid_peak(
    hsv: np.ndarray,
    center: tuple[int, int],
    roi_offset: tuple[int, int],
    *,
    radius: int = 36,
) -> tuple[tuple[int, int], float]:
    """将点击点吸附到附近的鲜艳色图钉头部。"""
    ox, oy = roi_offset
    base_rx, base_ry = center[0] - ox, center[1] - oy
    best_vivid = 0.0
    best_center = center
    for dy in range(-radius, radius + 1, 4):
        for dx in range(-radius, radius + 1, 4):
            rx, ry = base_rx + dx, base_ry + dy
            if rx < 0 or ry < 0 or rx >= hsv.shape[1] or ry >= hsv.shape[0]:
                continue
            vivid = _patch_vivid_ratio(hsv, (rx, ry), radius=14)
            if vivid > best_vivid:
                best_vivid = vivid
                best_center = (ox + rx, oy + ry)
    return best_center, best_vivid


def _accept_pin_candidate(
    center: tuple[int, int],
    confidence: float,
    roi: np.ndarray,
    hsv: np.ndarray,
    roi_offset: tuple[int, int],
    diff_mask: np.ndarray,
    *,
    blob_area: float = 0.0,
    blob_width: int = 0,
    blob_height: int = 0,
) -> tuple[tuple[int, int], float] | None:
    if not _screen_center_in_map(center):
        return None
    if _is_super_boss_point(center, roi, hsv, roi_offset):
        return None
    if _is_plane_point(
        center,
        blob_area=blob_area,
        blob_width=blob_width,
        blob_height=blob_height,
        roi=roi,
        roi_offset=roi_offset,
    ):
        return None

    diff_ratio, refined = _pin_diff_support(center, diff_mask, roi_offset)
    if diff_ratio < BG_DIFF_PIN_SUPPORT_MIN:
        return None

    snapped, vivid = _snap_to_vivid_peak(hsv, refined, roi_offset)
    # 空雪地/飞机等无图钉色区域：差分再大也不接受
    if vivid < BG_DIFF_VIVID_MIN:
        return None
    refined = snapped
    if _is_plane_point(refined, roi=roi, roi_offset=roi_offset):
        return None
    if _is_super_boss_point(refined, roi, hsv, roi_offset):
        return None
    if not _screen_center_in_map(refined):
        return None
    if _is_ignored_background_pin(refined, hsv=hsv, roi_offset=roi_offset):
        return None
    return refined, float(max(confidence, vivid, diff_ratio))


def _max_pins_for_diff_blob(blob_area: float) -> int:
    if blob_area < BG_LARGE_BLOB_AREA:
        return BG_MAX_PINS_PER_DIFF_BLOB
    return min(8, max(BG_MAX_PINS_PER_DIFF_BLOB, int(blob_area / 1800)))


def _extract_vivid_peaks_from_blob(
    blob_contour: np.ndarray,
    search_mask: np.ndarray,
    hsv: np.ndarray,
    roi: np.ndarray,
    roi_offset: tuple[int, int],
    diff_mask: np.ndarray,
) -> list[tuple[tuple[int, int], float]]:
    """大片差分块：距离变换 + 峰值抑制，拆分重叠图钉。"""
    blob_area = cv2.contourArea(blob_contour)
    if blob_area < BG_LARGE_BLOB_AREA:
        return []

    bx, by, bw, bh = cv2.boundingRect(blob_contour)
    pad = 6
    lx = max(0, bx - pad)
    ly = max(0, by - pad)
    rx = min(search_mask.shape[1], bx + bw + pad)
    ry = min(search_mask.shape[0], by + pad + bh)
    local_search = search_mask[ly:ry, lx:rx]
    if local_search.size == 0 or cv2.countNonZero(local_search) < 80:
        return []

    dist = cv2.distanceTransform(local_search, cv2.DIST_L2, 5)
    work = dist.copy()
    limit = _max_pins_for_diff_blob(blob_area)
    found: list[tuple[tuple[int, int], float]] = []
    spacing = BG_LARGE_BLOB_PIN_SPACING

    for _ in range(limit):
        _min_val, max_val, _min_loc, max_loc = cv2.minMaxLoc(work)
        if max_val < BG_PEAK_MIN_DISTANCE:
            break
        px, py = max_loc
        center = (roi_offset[0] + lx + px, roi_offset[1] + ly + py)
        vivid = _patch_vivid_ratio(
            hsv,
            (center[0] - roi_offset[0], center[1] - roi_offset[1]),
        )
        accepted = _accept_pin_candidate(
            center,
            vivid,
            roi,
            hsv,
            roi_offset,
            diff_mask,
            blob_area=blob_area,
            blob_width=bw,
            blob_height=bh,
        )
        if accepted is not None:
            found.append(accepted)
            cv2.circle(work, max_loc, spacing, 0, -1)
            continue
        # 峰值落在无效区域时仍抑制，避免反复选中同一点
        cv2.circle(work, max_loc, max(8, spacing // 2), 0, -1)

    return found


def _extract_pins_from_diff_blob(
    blob_contour: np.ndarray,
    search_mask: np.ndarray,
    hsv: np.ndarray,
    roi: np.ndarray,
    roi_offset: tuple[int, int],
    diff_mask: np.ndarray,
) -> list[tuple[tuple[int, int], float]]:
    bx, by, bw, bh = cv2.boundingRect(blob_contour)
    pad = 6
    lx = max(0, bx - pad)
    ly = max(0, by - pad)
    rx = min(search_mask.shape[1], bx + bw + pad)
    ry = min(search_mask.shape[0], by + pad + bh)
    local_search = search_mask[ly:ry, lx:rx]
    local_hsv = hsv[ly:ry, lx:rx]
    if local_search.size == 0:
        return []

    blob_area = cv2.contourArea(blob_contour)
    found: list[tuple[tuple[int, int], float]] = []
    seen: set[tuple[int, int]] = set()

    for erode_iters in (0, 2, 4, 6):
        mask = local_search
        if erode_iters:
            mask = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=erode_iters)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        sub_contours = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )[0]
        for sub in sub_contours:
            sub_area = cv2.contourArea(sub)
            if sub_area < 50:
                continue
            center: tuple[int, int] | None = None
            confidence = _contour_vivid_ratio(sub, local_hsv)
            if _is_teardrop_contour(sub):
                center = _contour_screen_center(
                    sub,
                    (roi_offset[0] + lx, roi_offset[1] + ly),
                    hsv=local_hsv,
                    use_vivid_peak=True,
                )
            elif sub_area >= 120:
                sx, sy, sw, sh = cv2.boundingRect(sub)
                sub_hsv = local_hsv[sy : sy + sh, sx : sx + sw]
                sat = sub_hsv[:, :, 1].astype(np.float32)
                val = sub_hsv[:, :, 2]
                score = sat * (val > 130)
                if float(score.max()) > 0:
                    _, _, _, peak = cv2.minMaxLoc(score)
                    center = (
                        roi_offset[0] + lx + sx + peak[0],
                        roi_offset[1] + ly + sy + peak[1],
                    )
            if center is None:
                continue
            key = (center[0] // 8, center[1] // 8)
            if key in seen:
                continue
            accepted = _accept_pin_candidate(
                center,
                confidence,
                roi,
                hsv,
                roi_offset,
                diff_mask,
                blob_area=blob_area,
                blob_width=bw,
                blob_height=bh,
            )
            if accepted is None:
                continue
            seen.add(key)
            found.append(accepted)

    peak_found = _extract_vivid_peaks_from_blob(
        blob_contour, search_mask, hsv, roi, roi_offset, diff_mask
    )
    for item in peak_found:
        key = (item[0][0] // 8, item[0][1] // 8)
        if key in seen:
            continue
        if any(
            abs(item[0][0] - existing[0]) < MISSION_PIN_MERGE_DISTANCE
            and abs(item[0][1] - existing[1]) < MISSION_PIN_MERGE_DISTANCE
            for existing, _ in found
        ):
            continue
        seen.add(key)
        found.append(item)

    return _merge_nearby_pin_candidates(found)[
        : _max_pins_for_diff_blob(blob_area)
    ]


def _find_mission_pins_bg_diff(
    roi: np.ndarray,
    bg_roi: np.ndarray,
    roi_offset: tuple[int, int],
) -> list[tuple[tuple[int, int], float]]:
    """背景差分定位新增任务图钉（图钉 + 地面光晕会连成大片，需分块提取）。"""
    bg_roi = _align_background_roi(bg_roi, roi.shape)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    diff = cv2.absdiff(roi, bg_roi)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, diff_mask = cv2.threshold(
        diff_gray, BG_DIFF_THRESHOLD, 255, cv2.THRESH_BINARY
    )
    diff_mask = cv2.morphologyEx(
        diff_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)
    )
    search_mask = cv2.bitwise_and(_build_vivid_pin_mask(hsv), diff_mask)

    candidates: list[tuple[tuple[int, int], float]] = []
    rejected = {"ui": 0, "plane": 0, "boss": 0, "support": 0, "merged": 0}

    for blob in cv2.findContours(
        diff_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )[0]:
        _check_scan_interrupted()
        blob_area = cv2.contourArea(blob)
        if blob_area < BG_DIFF_BLOB_MIN_AREA:
            continue

        bx, by, bw, bh = cv2.boundingRect(blob)
        moments = cv2.moments(blob)
        if moments["m00"] <= 0:
            continue
        blob_cx = int(moments["m10"] / moments["m00"]) + roi_offset[0]
        blob_cy = int(moments["m01"] / moments["m00"]) + roi_offset[1]

        if _is_ui_diff_blob(blob_cy, blob_height=bh, blob_width=bw):
            rejected["ui"] += 1
            continue
        # 小飞机 / 橙光大怪可能与附近图钉粘成一大片。小块可整片剔除；
        # 大块继续拆分，由单点 plane/boss 过滤掉干扰物本身。
        plane_blob = _is_plane_point(
            (blob_cx, blob_cy),
            blob_area=blob_area,
            blob_width=bw,
            blob_height=bh,
            roi=roi,
            roi_offset=roi_offset,
        )
        boss_blob = _is_super_boss_point(
            (blob_cx, blob_cy), roi, hsv, roi_offset
        )
        if plane_blob or boss_blob:
            if blob_area < BG_LARGE_BLOB_AREA:
                rejected["plane" if plane_blob else "boss"] += 1
                continue

        for item in _extract_pins_from_diff_blob(
            blob, search_mask, hsv, roi, roi_offset, diff_mask
        ):
            candidates.append(item)

    color_pins = _find_mission_pin_centers(roi, roi_offset)
    for pin in color_pins:
        _check_scan_interrupted()
        accepted = _accept_pin_candidate(
            pin,
            _patch_vivid_ratio(
                hsv,
                (pin[0] - roi_offset[0], pin[1] - roi_offset[1]),
            ),
            roi,
            hsv,
            roi_offset,
            diff_mask,
        )
        if accepted is None:
            rejected["support"] += 1
            continue
        if any(
            abs(accepted[0][0] - c[0][0]) < MISSION_PIN_MERGE_DISTANCE
            and abs(accepted[0][1] - c[0][1]) < MISSION_PIN_MERGE_DISTANCE
            for c in candidates
        ):
            rejected["merged"] += 1
            continue
        candidates.append(accepted)

    candidates = _merge_nearby_pin_candidates(candidates)

    if candidates or any(rejected.values()):
        logger.debug(
            "灯塔差分扫描：保留 {} 个，剔除 ui={} plane={} boss={} "
            "support={} merged={}",
            len(candidates),
            rejected["ui"],
            rejected["plane"],
            rejected["boss"],
            rejected["support"],
            rejected["merged"],
        )
    return candidates


def _scan_icons_impl(
    screen: np.ndarray,
    *,
    scan_roi: tuple[int, int, int, int],
) -> LighthouseScanResult:
    """背景差分定位任务图钉，不区分类型。"""
    screen = _normalize_screen_for_scan(screen)
    x1, y1, x2, y2 = scan_roi
    roi = screen[y1:y2, x1:x2]
    if roi.size == 0:
        return LighthouseScanResult(mission=None)

    bg_roi = _load_map_background_roi()
    if bg_roi is None:
        logger.error(
            f"缺少灯塔背景图 {_active_map_bg_name}，无法扫描任务图标"
        )
        return LighthouseScanResult(mission=None)

    pin_candidates = _find_mission_pins_bg_diff(roi, bg_roi, (x1, y1))
    total_candidates = len(pin_candidates)
    if not pin_candidates:
        return LighthouseScanResult(mission=None, candidate_locations=0)

    half = LIGHTHOUSE_PIN_PATCH_HALF
    missions: list[LighthouseMission] = []
    for center, confidence in pin_candidates:
        roi_cx = center[0] - x1
        roi_cy = center[1] - y1
        x1p = max(0, roi_cx - half)
        y1p = max(0, roi_cy - half)
        x2p = min(roi.shape[1], roi_cx + half)
        y2p = min(roi.shape[0], roi_cy + half)
        missions.append(
            LighthouseMission(
                kind="",
                label="图标",
                template="",
                center=center,
                confidence=confidence,
                top_left=(x1 + x1p, y1 + y1p),
                size=(x2p - x1p, y2p - y1p),
            )
        )

    missions = _dedupe_missions_by_slot(missions, merge_distance=28)
    missions.sort(key=lambda item: item.center[1])

    best = missions[0] if missions else None
    return LighthouseScanResult(
        mission=best,
        missions=tuple(missions),
        best_confidence=best.confidence if best else 0.0,
        best_label="图标",
        candidate_locations=total_candidates,
    )
