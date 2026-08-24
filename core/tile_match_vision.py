"""消消乐（三消进槽）视觉识别与道具按钮。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from loguru import logger

# 底部道具按钮区域（用户标定）—— 功能区
TOOL_REMOVE_ROI = (38, 1124, 240, 1252)  # 移出
TOOL_UNDO_ROI = (262, 1122, 470, 1238)  # 撤回
TOOL_SHUFFLE_ROI = (472, 1126, 678, 1236)  # 重构造

# 四个模块（720×1280）：
#   常规区：堆叠消牌区（暂移区上方）
#   暂移区：使用「移出」后放置区前 3 张落到这里，仍可点回放置区消除（最多 3）
#   放置区：底部七格槽（最多 7，不可点选「槽内」牌面）
#   功能区：移出 / 撤回 / 重构造
DEFAULT_BOARD_ROI = (8, 70, 708, 842)  # 常规区（上沿须含孤立顶牌，信封顶约 y=81）
DEFAULT_TEMP_ROI = (222, 842, 504, 948)  # 暂移区
DEFAULT_SLOT_ROI = (4, 924, 712, 1122)  # 放置区
DEFAULT_GAME_ROI = (10, 158, 708, 1272)
TEMP_CAPACITY = 3

FREE_BADGE_NAMES = ("free_badge.png", "免费.png", "mianfei.png")


@dataclass
class TileHit:
    kind: str
    center: tuple[int, int]
    confidence: float
    size: tuple[int, int]
    free: bool = True  # True=可点上层；False=被挡，仅供库存推演


@dataclass
class BoardScan:
    """常规区+暂移区扫描结果。"""

    free_tiles: list[TileHit]
    board_counts: dict[str, int]  # 可点 + 被挡 总库存
    buried_counts: dict[str, int]  # 仅被挡
    all_tiles: list[TileHit] | None = None  # 含被挡，供遮挡推演


def tile_bbox(
    t: TileHit, *, expand: int = 0
) -> tuple[int, int, int, int]:
    tw, th = t.size
    cx, cy = t.center
    return (
        cx - tw // 2 - expand,
        cy - th // 2 - expand,
        cx + tw // 2 + expand,
        cy + th // 2 + expand,
    )


def _is_stacked_under(upper: TileHit, lower: TileHit) -> bool:
    """lower 是否在 upper 下方/同位被压住（含几乎同心叠层）。"""
    dy = lower.center[1] - upper.center[1]
    dist_x = abs(upper.center[0] - lower.center[0])
    # 几乎同一中心：异种叠在一起（靴压锅）
    if dist_x <= 18 and abs(dy) <= 18:
        return upper.kind != lower.kind
    # 下一层大约半个牌高；太深是再下一层
    if dy < -8 or dy > 96:
        return False
    cover_r = max(upper.size[0], lower.size[0]) * 0.9 + 28
    if dist_x <= cover_r:
        return True
    # 水平有重叠也算压住
    ux1, _, ux2, _ = tile_bbox(upper, expand=8)
    lx1, _, lx2, _ = tile_bbox(lower, expand=0)
    overlap = max(0, min(ux2, lx2) - max(ux1, lx1))
    return overlap >= min(upper.size[0], lower.size[0]) * 0.12


def predict_exposed_after_remove(
    all_tiles: list[TileHit],
    removed: TileHit,
) -> list[TileHit]:
    """预测点掉 removed 后，下一层哪些牌会露出来（变为可点）。

    规则：
      - 只考虑被 removed 压住、且 y 更靠下的被挡牌
      - 取「最近一层」y 簇，避免一次把再下层也算露出来
      - 若仍被其它当前可点牌压住，则不算露出
    """
    if not all_tiles:
        return []

    def _same(a: TileHit, b: TileHit) -> bool:
        return (
            a.kind == b.kind
            and abs(a.center[0] - b.center[0]) <= 8
            and abs(a.center[1] - b.center[1]) <= 8
        )

    remaining = [t for t in all_tiles if not _same(t, removed)]
    still_free = [t for t in remaining if t.free]
    buried = [t for t in remaining if not t.free]

    unders = [b for b in buried if _is_stacked_under(removed, b)]
    if not unders:
        return []

    unders.sort(key=lambda b: (b.center[1], -b.confidence))
    first_y = unders[0].center[1]
    # 同一层：y 接近
    layer = [b for b in unders if b.center[1] <= first_y + 28]

    exposed: list[TileHit] = []
    for b in layer:
        if any(_is_stacked_under(f, b) for f in still_free):
            continue
        exposed.append(b)
    # 同种同位置去重；异种叠层都要留下（靴下可能同时有锅/米袋误检）
    exposed.sort(key=lambda t: t.confidence, reverse=True)
    kept: list[TileHit] = []
    for hit in exposed:
        if any(
            hit.kind == k.kind
            and abs(hit.center[0] - k.center[0]) < 28
            and abs(hit.center[1] - k.center[1]) < 28
            for k in kept
        ):
            continue
        kept.append(hit)
    return kept


def assign_tile_layers(
    all_tiles: list[TileHit],
    *,
    max_layer: int = 12,
) -> dict[tuple[str, int, int], int]:
    """给每张牌标层号：1=当前可点，2+=需先点掉上层后露出。"""
    layers: dict[tuple[str, int, int], int] = {}

    def _key(t: TileHit) -> tuple[str, int, int]:
        return (t.kind, t.center[0], t.center[1])

    frees = [t for t in all_tiles if t.free]
    for t in frees:
        layers[_key(t)] = 1

    # 异种几乎同心 / 正下方：直接标为 L2（靴压锅）
    for t in all_tiles:
        k = _key(t)
        if k in layers:
            continue
        if any(_is_stacked_under(f, t) for f in frees):
            layers[k] = min(2, max_layer)

    for _ in range(max_layer + 2):
        progress = False
        for t in all_tiles:
            k = _key(t)
            if k in layers:
                continue
            uppers = [
                u
                for u in all_tiles
                if _key(u) != k and _is_stacked_under(u, t)
            ]
            if not uppers:
                continue
            uppers.sort(key=lambda u: t.center[1] - u.center[1])
            closest_dy = t.center[1] - uppers[0].center[1]
            immediate = [
                u
                for u in uppers
                if (t.center[1] - u.center[1]) <= closest_dy + 22
            ]
            known = [layers[_key(u)] for u in immediate if _key(u) in layers]
            if len(known) < len(immediate):
                continue
            cand = 1 + min(known)
            if cand <= max_layer:
                layers[k] = cand
                progress = True
        if not progress:
            break
    return layers


def tiles_up_to_layer(
    all_tiles: list[TileHit],
    layer_map: dict[tuple[str, int, int], int],
    max_layer: int,
) -> list[TileHit]:
    out: list[TileHit] = []
    for t in all_tiles:
        layer = layer_map.get((t.kind, t.center[0], t.center[1]))
        if layer is not None and layer <= max_layer:
            out.append(t)
    return out


def inventory_within_layers(
    slot: list[str],
    all_tiles: list[TileHit],
    *,
    max_layer: int = 3,
) -> dict[str, int]:
    """放置区 + 第 1..max_layer 层场上牌的种类计数。"""
    from collections import Counter

    layer_map = assign_tile_layers(all_tiles, max_layer=max_layer)
    pool = tiles_up_to_layer(all_tiles, layer_map, max_layer)
    counts = Counter(slot)
    for t in pool:
        counts[t.kind] += 1
    return dict(counts)


def has_clearable_within_layers(
    slot: list[str],
    all_tiles: list[TileHit],
    *,
    max_layer: int = 3,
) -> bool:
    """L1..max_layer（含槽）是否存在某种类 ≥3（能三消）。

    挖更深层会把中间层牌塞进槽，成本极高；1+2+3 都凑不齐就该重构。
    """
    if not all_tiles and not slot:
        return False
    inv = inventory_within_layers(slot, all_tiles, max_layer=max_layer)
    return any(n >= 3 for n in inv.values())


def predict_expose_map(
    all_tiles: list[TileHit],
) -> dict[tuple[str, tuple[int, int]], list[TileHit]]:
    """对每个当前可点牌：移除后会露出哪些。"""
    result: dict[tuple[str, tuple[int, int]], list[TileHit]] = {}
    for free in all_tiles:
        if not free.free:
            continue
        result[(free.kind, free.center)] = predict_exposed_after_remove(
            all_tiles, free
        )
    return result


@dataclass
class ToolButton:
    key: str
    label: str
    roi: tuple[int, int, int, int]

    @property
    def center(self) -> tuple[int, int]:
        x1, y1, x2, y2 = self.roi
        return (x1 + x2) // 2, (y1 + y2) // 2


TOOLS: tuple[ToolButton, ...] = (
    ToolButton("remove", "移出", TOOL_REMOVE_ROI),
    ToolButton("undo", "撤回", TOOL_UNDO_ROI),
    ToolButton("shuffle", "重构造", TOOL_SHUFFLE_ROI),
)


def list_tile_templates(template_dir: Path) -> list[Path]:
    """牌面模板；忽略 _ 开头，以及免费角标文件名。"""
    if not template_dir.is_dir():
        return []
    skip = {n.lower() for n in FREE_BADGE_NAMES}
    return sorted(
        p
        for p in template_dir.glob("*.png")
        if p.is_file()
        and not p.name.startswith("_")
        and p.name.lower() not in skip
    )


def _nms(hits: list[TileHit], min_dist: int) -> list[TileHit]:
    ordered = sorted(hits, key=lambda h: h.confidence, reverse=True)
    kept: list[TileHit] = []
    for hit in ordered:
        if any(
            abs(hit.center[0] - k.center[0]) < min_dist
            and abs(hit.center[1] - k.center[1]) < min_dist
            for k in kept
        ):
            continue
        kept.append(hit)
    return kept


def _peak_nms(
    peaks: list[tuple[float, int, int]], min_dist: int
) -> list[tuple[float, int, int]]:
    """peaks: (score, x, y) 相对 crop 坐标。"""
    ordered = sorted(peaks, key=lambda p: p[0], reverse=True)
    kept: list[tuple[float, int, int]] = []
    for score, x, y in ordered:
        if any(abs(x - kx) < min_dist and abs(y - ky) < min_dist for _, kx, ky in kept):
            continue
        kept.append((score, x, y))
    return kept


def _center_brightness(
    gray: np.ndarray, cx: int, cy: int, *, radius: int = 18
) -> float:
    """中心小窗亮度（比整模板块更稳，孤立小牌不会被拉低）。"""
    h, w = gray.shape[:2]
    x1, y1 = max(0, cx - radius), max(0, cy - radius)
    x2, y2 = min(w, cx + radius), min(h, cy + radius)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    return float(np.mean(gray[y1:y2, x1:x2]))


def _patch_brightness(gray: np.ndarray, x: int, y: int, tw: int, th: int) -> float:
    """模板中心附近亮度。"""
    return _center_brightness(gray, x + tw // 2, y + th // 2, radius=max(8, min(tw, th) // 4))


def _match_kind_peaks(
    gray: np.ndarray,
    tmpl: np.ndarray,
    *,
    threshold: float,
    min_dist: int,
) -> list[tuple[float, int, int]]:
    th, tw = tmpl.shape[:2]
    if th >= gray.shape[0] or tw >= gray.shape[1]:
        return []
    result = cv2.matchTemplate(gray, tmpl, cv2.TM_CCOEFF_NORMED)
    ys, xs = np.where(result >= threshold)
    peaks = [(float(result[yy, xx]), int(xx), int(yy)) for yy, xx in zip(ys, xs)]
    return _peak_nms(peaks, min_dist=min_dist)


def _match_multiscale_peaks(
    gray: np.ndarray,
    tmpl0: np.ndarray,
    *,
    threshold: float,
    min_dist: int,
    scales: tuple[float, ...] = (0.65, 0.75, 0.85, 1.0, 1.15),
    color_img: np.ndarray | None = None,
    color_tmpl0: np.ndarray | None = None,
) -> list[tuple[float, int, int, int, int]]:
    """多尺度匹配。优先彩色；同时试「整牌」和「图标核心」。

    孤立小堆（右上角靴）整牌底和主堆不一致，整模板常只有 ~0.69；
    图标核心可达 0.95+。返回 (conf, x, y, tw, th)。
    """
    use_color = (
        color_img is not None
        and color_tmpl0 is not None
        and color_img.ndim == 3
        and color_tmpl0.ndim == 3
    )
    src = color_img if use_color else gray
    variants: list[np.ndarray] = []
    if use_color:
        variants.append(color_tmpl0)
        core = _extract_icon_core(color_tmpl0)
        if core is not None and core.shape[0] >= 12 and core.shape[1] >= 12:
            variants.append(core)
    else:
        variants.append(tmpl0)

    all_peaks: list[tuple[float, int, int, int, int]] = []
    for variant in variants:
        for s in scales:
            tw = max(8, int(variant.shape[1] * s))
            th = max(8, int(variant.shape[0] * s))
            if th >= src.shape[0] or tw >= src.shape[1]:
                continue
            interp = cv2.INTER_AREA if s < 1.0 else cv2.INTER_LINEAR
            tmpl = cv2.resize(variant, (tw, th), interpolation=interp)
            result = cv2.matchTemplate(src, tmpl, cv2.TM_CCOEFF_NORMED)
            ys, xs = np.where(result >= threshold)
            for yy, xx in zip(ys, xs):
                all_peaks.append((float(result[yy, xx]), int(xx), int(yy), tw, th))
    all_peaks.sort(key=lambda p: p[0], reverse=True)
    kept: list[tuple[float, int, int, int, int]] = []
    for conf, x, y, tw, th in all_peaks:
        cx, cy = x + tw // 2, y + th // 2
        if any(
            abs(cx - (kx + ktw // 2)) < min_dist
            and abs(cy - (ky + kth // 2)) < min_dist
            for _, kx, ky, ktw, kth in kept
        ):
            continue
        kept.append((conf, x, y, tw, th))
    return kept


def find_tiles_in_roi(
    screen: np.ndarray,
    template_dir: Path,
    roi: tuple[int, int, int, int],
    *,
    threshold: float = 0.72,
    min_dist: int = 40,
) -> list[TileHit]:
    """在 ROI 内多尺度匹配牌面（高阈值 ≈ 可点上层）。"""
    x1, y1, x2, y2 = roi
    h, w = screen.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return []

    crop = screen[y1:y2, x1:x2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    templates = list_tile_templates(template_dir)
    if not templates:
        logger.warning("消消乐模板目录为空: {}", template_dir)
        return []

    raw: list[TileHit] = []
    for path in templates:
        tmpl = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if tmpl is None:
            continue
        peaks = _match_multiscale_peaks(
            gray, tmpl, threshold=threshold, min_dist=min_dist
        )
        kind = path.stem
        for conf, xx, yy, tw, th in peaks:
            raw.append(
                TileHit(
                    kind=kind,
                    center=(x1 + xx + tw // 2, y1 + yy + th // 2),
                    confidence=conf,
                    size=(tw, th),
                    free=True,
                )
            )
    return _nms(raw, min_dist=min_dist)


def _extract_icon_core(tmpl: np.ndarray) -> np.ndarray | None:
    """去掉浅色牌面底，只留图标高饱和核心（半遮挡时靠这认）。"""
    hsv = cv2.cvtColor(tmpl, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    mask = ((sat > 35) & (val < 245) & (val > 35)).astype(np.uint8) * 255
    # 去掉极靠边框噪点
    mask[:2, :] = 0
    mask[-2:, :] = 0
    mask[:, :2] = 0
    mask[:, -2:] = 0
    ys, xs = np.where(mask > 0)
    if len(xs) < 40:
        return None
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    if x2 - x1 < 10 or y2 - y1 < 10:
        return None
    core = tmpl[y1:y2, x1:x2].copy()
    return core


def _partial_views(tmpl: np.ndarray) -> list[np.ndarray]:
    """半遮挡视图：图标核心的底/顶/角 + 整核。"""
    core = _extract_icon_core(tmpl)
    src = core if core is not None else tmpl
    th, tw = src.shape[:2]
    views = [
        src[int(th * 0.50) :, :],
        src[: int(th * 0.50), :],
        src[:, : int(tw * 0.50)],
        src[:, int(tw * 0.50) :],
        src[int(th * 0.55) :, : int(tw * 0.55)],
        src[int(th * 0.55) :, int(tw * 0.45) :],
        src,
    ]
    # 原模板底边（含牌沿）有时也有用
    th0, tw0 = tmpl.shape[:2]
    views.append(tmpl[int(th0 * 0.65) :, :])
    return [v for v in views if v.size > 0 and min(v.shape[:2]) >= 6]


def _hit_patch(screen: np.ndarray, hit: TileHit, *, pad: int = 8) -> np.ndarray | None:
    h, w = screen.shape[:2]
    tw, th = hit.size
    cx, cy = hit.center
    x1 = max(0, cx - tw // 2 - pad)
    y1 = max(0, cy - th // 2 - pad)
    x2 = min(w, cx + tw // 2 + pad)
    y2 = min(h, cy + th // 2 + pad)
    if x2 <= x1 or y2 <= y1:
        return None
    return screen[y1:y2, x1:x2]


def _reject_false_kind(screen: np.ndarray, hit: TileHit) -> bool:
    """颜色门：挡掉「灯笼下摆 → 齿轮」「半挡棕块 → 靴」等弱误检。"""
    patch = _hit_patch(screen, hit)
    if patch is None or patch.size == 0:
        return False
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    n = max(1, patch.shape[0] * patch.shape[1])
    if hit.kind == "chilun":
        # 真齿轮不应有灯笼黄珠；且要有足够蓝
        yellow = cv2.inRange(hsv, np.array([14, 70, 90]), np.array([40, 255, 255]))
        blue = cv2.inRange(hsv, np.array([95, 70, 50]), np.array([125, 255, 255]))
        y_r = cv2.countNonZero(yellow) / n
        b_r = cv2.countNonZero(blue) / n
        if y_r >= 0.012 or b_r < 0.06 or hit.confidence < 0.78:
            return True
    if hit.kind == "xiezi" and not hit.free:
        # 被挡靴：弱分+过小框多半是棕块误检（曾把 1 只真靴扩成 10 只）
        if hit.confidence < 0.72:
            return True
        if min(hit.size) < 48 and hit.confidence < 0.82:
            return True
        white = cv2.inRange(hsv, np.array([0, 0, 170]), np.array([180, 60, 255]))
        brown = cv2.inRange(hsv, np.array([5, 40, 40]), np.array([25, 255, 220]))
        blue = cv2.inRange(hsv, np.array([90, 40, 40]), np.array([130, 255, 255]))
        if cv2.countNonZero(blue) / n > 0.25 and cv2.countNonZero(brown) / n < 0.08:
            return True
        if (
            cv2.countNonZero(white) / n < 0.015
            and cv2.countNonZero(brown) / n < 0.05
            and hit.confidence < 0.85
        ):
            return True
    return False


def _color_prior_bonus(cell: np.ndarray, kind: str) -> float:
    """露出很少时靠颜色拉一把；红靴/米袋/灯笼要分清。"""
    hsv = cv2.cvtColor(cell, cv2.COLOR_BGR2HSV)
    n = max(1, cell.shape[0] * cell.shape[1])
    if kind == "denglong":
        blue = cv2.inRange(hsv, np.array([90, 60, 40]), np.array([130, 255, 220]))
        yellow = cv2.inRange(hsv, np.array([14, 80, 100]), np.array([40, 255, 255]))
        # 灯笼：蓝体为主；平底锅也偏蓝，黄心能区分
        return 0.05 * (cv2.countNonZero(blue) / n) + 0.12 * (
            cv2.countNonZero(yellow) / n
        )
    if kind == "chilun":
        # 齿轮：实心蓝齿；灯笼下摆也偏蓝但带黄珠——有黄就扣分
        blue = cv2.inRange(hsv, np.array([95, 80, 60]), np.array([125, 255, 255]))
        yellow = cv2.inRange(hsv, np.array([14, 80, 100]), np.array([40, 255, 255]))
        return 0.10 * (cv2.countNonZero(blue) / n) - 0.20 * (
            cv2.countNonZero(yellow) / n
        )
    if kind == "pingdiguo":
        # 锅：银灰蓝 + 棕色柄；无黄心
        blue = cv2.inRange(hsv, np.array([85, 20, 80]), np.array([120, 120, 230]))
        brown = cv2.inRange(hsv, np.array([8, 40, 40]), np.array([28, 255, 200]))
        yellow = cv2.inRange(hsv, np.array([14, 80, 100]), np.array([40, 255, 255]))
        return (
            0.03 * (cv2.countNonZero(blue) / n)
            + 0.05 * (cv2.countNonZero(brown) / n)
            - 0.08 * (cv2.countNonZero(yellow) / n)
        )
    if kind == "midai":
        # 米袋棕黄，不要和纯红靴抢
        brown = cv2.inRange(hsv, np.array([8, 45, 50]), np.array([28, 220, 220]))
        return 0.10 * (cv2.countNonZero(brown) / n)
    if kind == "xiezi":
        # 只要鲜红，避开米袋棕
        r1 = cv2.inRange(hsv, np.array([0, 90, 70]), np.array([8, 255, 255]))
        r2 = cv2.inRange(hsv, np.array([170, 90, 70]), np.array([180, 255, 255]))
        return 0.10 * (cv2.countNonZero(cv2.bitwise_or(r1, r2)) / n)
    if kind == "snowflake":
        blue = cv2.inRange(hsv, np.array([95, 50, 140]), np.array([125, 255, 255]))
        return 0.06 * (cv2.countNonZero(blue) / n)
    return 0.0


def _best_partial_score(
    cell: np.ndarray,
    tmpl: np.ndarray,
    scales: list[float] | np.ndarray,
    kind: str = "",
) -> float:
    best = -1.0
    for view in _partial_views(tmpl):
        for s in scales:
            nw = max(8, int(view.shape[1] * s))
            nh = max(8, int(view.shape[0] * s))
            if nh >= cell.shape[0] or nw >= cell.shape[1]:
                continue
            interp = cv2.INTER_AREA if s < 1.0 else cv2.INTER_LINEAR
            small = cv2.resize(view, (nw, nh), interpolation=interp)
            res = cv2.matchTemplate(cell, small, cv2.TM_CCOEFF_NORMED)
            best = max(best, float(res.max()))
    if kind:
        best += _color_prior_bonus(cell, kind)
    return best


def _probe_partial_grid(
    crop: np.ndarray,
    abs_x1: int,
    abs_y1: int,
    color_tmpls: list[tuple[str, np.ndarray]],
    gray: np.ndarray,
    *,
    cell: int = 84,
    step: int = 56,
    min_score: float = 0.85,
    min_margin: float = 0.035,
) -> list[TileHit]:
    """半遮挡探针：上/中/下三带 × 多列（校验 ROI：雪花/靴/米袋/米袋/灯笼）。"""
    _ = cell, step
    h, w = crop.shape[:2]
    if h < 24 or w < 24 or not color_tmpls:
        return []

    cols = max(2, int(round(w / 78.0)))
    # 三带略重叠：上层高亮、中层半挡、底层约 20% 露出
    bands = (
        (0.00, 0.45),
        (0.22, 0.62),
        (0.50, 1.00),
    )
    scales = [0.45, 0.55, 0.7, 0.85, 1.0, 1.15]
    hits: list[TileHit] = []

    for ya, yb in bands:
        for c in range(cols):
            x0 = c * w // cols
            x1c = (c + 1) * w // cols
            y0 = int(h * ya)
            y1c = int(h * yb)
            if y1c - y0 < 20 or x1c - x0 < 20:
                continue
            patch = crop[y0:y1c, x0:x1c]
            if float(np.std(patch)) < 12:
                continue
            ranked: list[tuple[float, str]] = []
            for kind, tmpl in color_tmpls:
                ranked.append(
                    (_best_partial_score(patch, tmpl, scales, kind=kind), kind)
                )
            ranked.sort(reverse=True)
            top_score, top_kind = ranked[0]
            second = ranked[1][0] if len(ranked) > 1 else 0.0
            if top_score < min_score:
                continue
            if top_score - second < min_margin and top_score < 0.93:
                continue
            cx = abs_x1 + (x0 + x1c) // 2
            cy = abs_y1 + (y0 + y1c) // 2
            bri = float(np.mean(gray[y0:y1c, x0:x1c]))
            hit = TileHit(
                kind=top_kind,
                center=(cx, cy),
                confidence=min(0.999, float(top_score)),
                size=(x1c - x0, y1c - y0),
                free=False,
            )
            setattr(hit, "_bri", bri)
            setattr(hit, "_partial", True)
            hits.append(hit)

    return _nms(hits, min_dist=max(22, min(h, w) // 6))


def find_tiles_layered_in_roi(
    screen: np.ndarray,
    template_dir: Path,
    roi: tuple[int, int, int, int],
    *,
    free_threshold: float = 0.68,
    buried_threshold: float = 0.55,
    min_dist: int = 28,
) -> tuple[list[TileHit], dict[str, int], list[TileHit]]:
    """分层识别：整图多尺度（可点/半露）+ 网格边角探针（约 20% 露出的下层）。

    返回 (free_tiles, board_counts, all_tiles)。
    """
    x1, y1, x2, y2 = roi
    h, w = screen.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return [], {}, []

    crop = screen[y1:y2, x1:x2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    templates = list_tile_templates(template_dir)
    if not templates:
        return [], {}, []

    buried_thr = min(buried_threshold, free_threshold - 0.08)
    kind_dist = max(16, int(min_dist * 0.75))

    color_tmpls: list[tuple[str, np.ndarray]] = []
    raw_all: list[TileHit] = []
    for path in templates:
        tmpl_g = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        tmpl_c = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if tmpl_g is None or tmpl_c is None:
            continue
        color_tmpls.append((path.stem, tmpl_c))
        peaks = _match_multiscale_peaks(
            gray,
            tmpl_g,
            threshold=buried_thr,
            min_dist=kind_dist,
            color_img=crop,
            color_tmpl0=tmpl_c,
        )
        kind = path.stem
        for conf, xx, yy, tw, th in peaks:
            cx, cy = x1 + xx + tw // 2, y1 + yy + th // 2
            # 用中心小窗亮度；整模板块会把孤立右上角靴亮度算得过低
            bri = _center_brightness(gray, xx + tw // 2, yy + th // 2, radius=18)
            hit = TileHit(
                kind=kind,
                center=(cx, cy),
                confidence=conf,
                size=(tw, th),
                free=False,
            )
            setattr(hit, "_bri", bri)
            setattr(hit, "_partial", False)
            raw_all.append(hit)

    # 半遮挡网格探针极慢（整板可到 30s+），默认关闭；需要时再开
    use_partial_probe = False
    if use_partial_probe:
        partial_hits = _probe_partial_grid(
            crop,
            x1,
            y1,
            color_tmpls,
            gray,
            min_score=0.85,
            min_margin=0.035,
        )
        raw_all.extend(partial_hits)

    if not raw_all:
        return [], {}, []

    conflict_dist = max(14, min_dist // 2)
    # 整图匹配优先于探针（同分时）
    raw_all.sort(
        key=lambda t: (
            t.confidence,
            0 if getattr(t, "_partial", False) else 1,
        ),
        reverse=True,
    )
    unique: list[TileHit] = []
    for hit in raw_all:
        clash = False
        for u in unique:
            if (
                abs(hit.center[0] - u.center[0]) < conflict_dist
                and abs(hit.center[1] - u.center[1]) < conflict_dist
            ):
                if hit.kind == u.kind:
                    clash = True
                    break
                # 异种同位置：保留最多 3 个不同种类（上层 + 可能的两层被压）
                same_pos = [
                    x
                    for x in unique
                    if abs(hit.center[0] - x.center[0]) < conflict_dist
                    and abs(hit.center[1] - x.center[1]) < conflict_dist
                ]
                kinds_here = {x.kind for x in same_pos}
                if hit.kind in kinds_here:
                    clash = True
                    break
                if len(same_pos) >= 3:
                    clash = True
                    break
                best = max(same_pos, key=lambda x: x.confidence)
                if hit.confidence < 0.72 or hit.confidence < best.confidence - 0.35:
                    clash = True
                    break
        if not clash:
            unique.append(hit)

    def _bbox(t: TileHit) -> tuple[int, int, int, int]:
        tw, th = t.size
        cx, cy = t.center
        return cx - tw // 2, cy - th // 2, cx + tw // 2, cy + th // 2

    def _center_in(
        t: TileHit, box: tuple[int, int, int, int], shrink: float = 0.22
    ) -> bool:
        x1b, y1b, x2b, y2b = box
        bw, bh = x2b - x1b, y2b - y1b
        x1b += int(bw * shrink)
        x2b -= int(bw * shrink)
        y1b += int(bh * shrink)
        y2b -= int(bh * shrink)
        cx, cy = t.center
        return x1b <= cx <= x2b and y1b <= cy <= y2b

    for hit in unique:
        bri = float(getattr(hit, "_bri", 0.0))
        is_partial = bool(getattr(hit, "_partial", False))
        covered = False
        for other in unique:
            if other is hit:
                continue
            obri = float(getattr(other, "_bri", 0.0))
            near = (
                abs(hit.center[0] - other.center[0]) <= 18
                and abs(hit.center[1] - other.center[1]) <= 18
            )
            # 异种几乎同心：只认置信度更高的为上层（避免误检用亮度压住真可点）
            if near and hit.kind != other.kind:
                if other.confidence >= hit.confidence + 0.03:
                    covered = True
                    break
                continue
            if other.confidence < hit.confidence - 0.02 and obri < bri - 8:
                continue
            if _center_in(hit, _bbox(other)):
                if (
                    abs(hit.center[0] - other.center[0]) > 10
                    or abs(hit.center[1] - other.center[1]) > 10
                ):
                    if obri >= bri - 5 or other.confidence >= hit.confidence:
                        covered = True
                        break
        # 探针命中一律视为被挡；可点要高置信（图标核心匹配后孤立靴应 ≥0.9）
        if is_partial:
            is_free = False
        else:
            is_free = (not covered) and hit.confidence >= max(
                free_threshold, 0.78
            )
        hit.free = is_free

    free_tiles = _nms([t for t in unique if t.free], min_dist=min_dist)
    free_ids = {(t.kind, t.center) for t in free_tiles}
    for t in unique:
        if (t.kind, t.center) not in free_ids:
            t.free = False

    # 被挡库存：抬高门槛 + 颜色门（灯笼底≠齿轮；弱棕块≠靴）
    min_buried = max(0.70, buried_threshold + 0.10)
    filtered: list[TileHit] = []
    for t in unique:
        if t.free:
            filtered.append(t)
            continue
        if t.confidence < min_buried:
            continue
        if _reject_false_kind(screen, t):
            continue
        filtered.append(t)
    unique = filtered
    free_tiles = [t for t in unique if t.free]

    board_counts: dict[str, int] = {}
    for t in unique:
        board_counts[t.kind] = board_counts.get(t.kind, 0) + 1

    return free_tiles, board_counts, unique


def read_slot_contents(
    screen: np.ndarray,
    template_dir: Path,
    slot_roi: tuple[int, int, int, int],
    *,
    capacity: int = 7,
    threshold: float = 0.88,
) -> list[str]:
    """从画面识别放置区从左到右的牌（图标核心+彩色局部匹配）。"""
    x1, y1, x2, y2 = slot_roi
    h, w = screen.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return []

    templates = list_tile_templates(template_dir)
    if not templates:
        return []

    loaded: list[tuple[str, np.ndarray]] = []
    for path in templates:
        tmpl = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if tmpl is not None:
            loaded.append((path.stem, tmpl))
    if not loaded:
        return []

    scales = [0.55, 0.75, 1.0]
    cell_w = (x2 - x1) / float(capacity)
    pad_x = max(2, int(cell_w * 0.06))
    pad_y = max(3, int((y2 - y1) * 0.10))
    slot: list[str] = []
    for i in range(capacity):
        cx1 = int(x1 + i * cell_w) + pad_x
        cx2 = int(x1 + (i + 1) * cell_w) - pad_x
        cy1, cy2 = y1 + pad_y, y2 - pad_y
        if cx2 <= cx1 or cy2 <= cy1:
            break
        cell = screen[cy1:cy2, cx1:cx2]
        if cell.size == 0:
            break
        gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
        if float(np.std(gray)) < 12.0:
            break

        ranked: list[tuple[float, str]] = []
        for kind, tmpl in loaded:
            ranked.append(
                (_best_partial_score(cell, tmpl, scales, kind=kind), kind)
            )
        ranked.sort(reverse=True)
        best_score, best_kind = ranked[0]
        second = ranked[1][0] if len(ranked) > 1 else 0.0
        if best_score < threshold or best_score - second < 0.04:
            break
        slot.append(best_kind)
    return slot


def _load_free_badge(template_dir: Path) -> np.ndarray | None:
    for name in FREE_BADGE_NAMES:
        path = template_dir / name
        if path.is_file():
            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                return img
    return None


def _crop_tool_roi(screen: np.ndarray, tool: ToolButton) -> np.ndarray | None:
    x1, y1, x2, y2 = tool.roi
    h, w = screen.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    crop = screen[y1:y2, x1:x2]
    return crop if crop.size else None


def _tool_green_ratio(crop: np.ndarray) -> float:
    """免费态按钮整体变绿（原蓝色）。返回绿色像素占比。"""
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    # 亮绿色按钮主体（避开极暗边框/图标）
    lower = np.array([35, 60, 60], dtype=np.uint8)
    upper = np.array([95, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    return float(cv2.countNonZero(mask)) / float(mask.size)


def _tool_blue_ratio(crop: np.ndarray) -> float:
    """付费/未免费态偏蓝。"""
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    lower = np.array([95, 50, 50], dtype=np.uint8)
    upper = np.array([135, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    return float(cv2.countNonZero(mask)) / float(mask.size)


def _badge_match_score(
    screen: np.ndarray,
    tool: ToolButton,
    badge: np.ndarray,
) -> float:
    """右上角「免费」角标模板分；无匹配返回 0。"""
    x1, y1, x2, y2 = tool.roi
    mid_x = (x1 + x2) // 2
    mid_y = (y1 + y2) // 2
    # 右上：免费字样区域
    rx1, ry1, rx2, ry2 = mid_x - 10, y1, x2, mid_y + 24
    h, w = screen.shape[:2]
    rx1, ry1 = max(0, rx1), max(0, ry1)
    rx2, ry2 = min(w, rx2), min(h, ry2)
    if rx2 - rx1 < badge.shape[1] or ry2 - ry1 < badge.shape[0]:
        rx1, ry1, rx2, ry2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    crop = screen[ry1:ry2, rx1:rx2]
    if crop.size == 0:
        return 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    if badge.shape[0] >= gray.shape[0] or badge.shape[1] >= gray.shape[1]:
        return 0.0
    result = cv2.matchTemplate(gray, badge, cv2.TM_CCOEFF_NORMED)
    return float(result.max())


def is_tool_free(
    screen: np.ndarray,
    tool: ToolButton,
    template_dir: Path,
    *,
    threshold: float = 0.70,
    green_ratio: float = 0.18,
) -> bool:
    """免费可用：按钮整体变绿（主信号）；可选「免费」角标模板作加强。

    游戏表现：未免费=蓝色；免费=整钮变绿，右上有「免费」中文字样。
    无角标模板时仅靠绿色占比判断。
    """
    crop = _crop_tool_roi(screen, tool)
    if crop is None:
        return False

    g = _tool_green_ratio(crop)
    b = _tool_blue_ratio(crop)
    # 绿明显多于蓝 → 免费态
    color_free = g >= green_ratio and g > b * 1.15

    badge = _load_free_badge(template_dir)
    if badge is not None:
        score = _badge_match_score(screen, tool, badge)
        if score >= threshold:
            return True
        # 有模板但没对上：仍允许纯绿色判定（模板可能裁偏）
        return color_free

    return color_free


def detect_free_tools(
    screen: np.ndarray,
    template_dir: Path,
    *,
    threshold: float = 0.70,
) -> dict[str, bool]:
    return {
        tool.key: is_tool_free(screen, tool, template_dir, threshold=threshold)
        for tool in TOOLS
    }


def apply_slot_click(slot: list[str], kind: str) -> list[str]:
    """模拟点入槽后的消除。"""
    slot = list(slot) + [kind]
    if slot.count(kind) >= 3:
        removed = 0
        new_slot: list[str] = []
        for item in reversed(slot):
            if item == kind and removed < 3:
                removed += 1
                continue
            new_slot.append(item)
        slot = list(reversed(new_slot))
    return slot


def _slot_danger(slot: list[str], capacity: int = 7) -> float:
    """槽位危险度：越高越危险。"""
    from collections import Counter

    n = len(slot)
    if n >= capacity:
        return 1e6
    kinds = Counter(slot)
    # 每种未凑齐的牌至少还要再占 (3-cnt) 空位才消得掉
    debt = sum(max(0, 3 - c) for c in kinds.values())
    unique = len(kinds)
    return n * 12.0 + debt * 8.0 + unique * 5.0 + (0 if n < capacity - 2 else 40.0)


def _evaluate_state(
    slot: list[str],
    free_counts: dict[str, int],
    supply_counts: dict[str, int],
    *,
    capacity: int,
) -> float:
    """局面分：越高越好。supply 含被挡库存。"""
    from collections import Counter

    if len(slot) >= capacity:
        return -1e9
    score = 0.0
    score += (capacity - len(slot)) * 15.0
    kinds = Counter(slot)
    for kind, cnt in kinds.items():
        free_n = free_counts.get(kind, 0)
        supply = supply_counts.get(kind, 0)
        need = 3 - cnt
        if cnt == 2 and free_n >= 1:
            score += 90.0
        elif cnt == 2 and supply >= 1:
            score += 35.0  # 被挡里有，暂时点不到
        elif cnt == 2 and supply == 0:
            score -= 80.0
        elif cnt == 1 and free_n >= 2:
            score += 30.0
        elif cnt == 1 and supply >= 2:
            score += 12.0
        elif cnt == 1 and supply == 1:
            score -= 25.0  # 凑不齐 3
        elif cnt == 1 and supply == 0:
            score -= 45.0
        # 总量能否按 3 消完
        total = cnt + supply
        if total % 3 != 0:
            score -= 20.0
        if need > supply:
            score -= 40.0
    score -= _slot_danger(slot, capacity)
    score += min(40, sum(supply_counts.values())) * 0.25
    return score


def choose_tile(
    free_tiles: list[TileHit],
    slot: list[str],
    *,
    board_counts: dict[str, int] | None = None,
    all_tiles: list[TileHit] | None = None,
    max_layer: int = 3,
    slot_capacity: int = 7,
    search_depth: int = 5,
    prefer_roi: tuple[int, int, int, int] | None = None,
) -> TileHit | None:
    """渐进选牌（每步只点一张，点完重扫）：

    1) 看 放置区 + L1：某种 ≥3 → 点可点的那张
    2) 不够 → 放置区 + L1 + L2
    3) 再不够 → + L3
    槽内差一张在下层：只点覆盖层一张。1+2+3 都凑不齐 → None（重构）。
    """
    _ = search_depth
    if not free_tiles or len(slot) >= slot_capacity:
        return None

    from collections import Counter

    plan_depth = min(3, max_layer)

    free_by_kind: dict[str, list[TileHit]] = {}
    for t in free_tiles:
        if t.confidence < 0.72:
            continue
        free_by_kind.setdefault(t.kind, []).append(t)
    if not free_by_kind:
        for t in free_tiles:
            free_by_kind.setdefault(t.kind, []).append(t)
    for kind in free_by_kind:

        def _rank(t: TileHit, _prefer=prefer_roi) -> tuple:
            in_pref = (
                0
                if _prefer and point_in_roi(t.center[0], t.center[1], _prefer)
                else 1
            )
            return (in_pref, -t.confidence)

        free_by_kind[kind].sort(key=_rank)

    free_counts = {k: len(v) for k, v in free_by_kind.items()}
    slot_counts = Counter(slot)
    incomplete = [k for k, c in slot_counts.items() if 0 < c < 3]

    def best_hit(kind: str) -> TileHit | None:
        items = free_by_kind.get(kind) or []
        return items[0] if items else None

    def dig_cover_for(need_kind: str) -> TileHit | None:
        if all_tiles is None or len(slot) > slot_capacity - 2:
            return None
        buried_targets = [
            t for t in all_tiles if t.kind == need_kind and not t.free
        ]
        if not buried_targets:
            return None
        best: TileHit | None = None
        best_key: tuple | None = None
        for hit in free_tiles:
            if hit.confidence < 0.72:
                continue
            exposed = predict_exposed_after_remove(all_tiles, hit)
            exposes = any(e.kind == need_kind for e in exposed)
            overlaps = False
            if not exposes:
                hx, hy = hit.center
                hw, hh = hit.size
                for b in buried_targets:
                    bx, by = b.center
                    if abs(bx - hx) <= 18 and abs(by - hy) <= 18:
                        overlaps = True
                        break
                    if abs(bx - hx) <= hw * 0.7 and 0 <= (by - hy) <= max(96, hh):
                        overlaps = True
                        break
            if not exposes and not overlaps:
                continue
            junk = sum(1 for e in exposed if e.kind != need_kind)
            cover_helps = 1 if 0 < slot_counts.get(hit.kind, 0) < 3 else 0
            new_cover = 0 if hit.kind in slot_counts else 1
            # 优先：目标几乎在正下方/同心（靴压锅），再少带垃圾
            concentric = any(
                e.kind == need_kind
                and abs(e.center[0] - hit.center[0]) <= 22
                and abs(e.center[1] - hit.center[1]) <= 22
                for e in exposed
            )
            key = (
                1 if concentric else 0,
                1 if exposes else 0,
                cover_helps,
                -new_cover,
                -junk,
                hit.confidence,
            )
            if best_key is None or key > best_key:
                best_key = key
                best = hit
        return best

    # 槽内已有 2：先点同种 / 挖一层（靴下锅），不要被库存误判直接重构
    for kind in incomplete:
        if slot_counts[kind] != 2:
            continue
        if free_counts.get(kind, 0) > 0:
            return best_hit(kind)
        dig = dig_cover_for(kind)
        if dig is not None:
            return dig

    if all_tiles is not None and not has_clearable_within_layers(
        slot, all_tiles, max_layer=plan_depth
    ):
        return None

    def try_at_depth(supply: dict[str, int]) -> TileHit | None:
        """当前层深下：放置区 + 该层库存能凑 ≥3 的，就动手。"""
        inv = dict(supply)
        for k, n in free_counts.items():
            inv[k] = max(inv.get(k, 0), n)

        def can_clear(kind: str) -> bool:
            return slot_counts.get(kind, 0) + inv.get(kind, 0) >= 3

        # A. 槽内已有 2：必须先处理；点不到第三张就不要改开新种
        unresolved_pair = False
        for kind in incomplete:
            if slot_counts[kind] != 2:
                continue
            if free_counts.get(kind, 0) > 0:
                return best_hit(kind)
            dig = dig_cover_for(kind)
            if dig is not None:
                return dig
            unresolved_pair = True

        if unresolved_pair:
            # 两锅在槽却 L1 没有、也挖不出 → 本层深停，交给更深或重构
            return None

        # B. 槽内已有 1：本层深能凑满 → 点可点 / 挖
        for kind in incomplete:
            if slot_counts[kind] != 1:
                continue
            if not can_clear(kind):
                continue
            if free_counts.get(kind, 0) > 0:
                return best_hit(kind)
            dig = dig_cover_for(kind)
            if dig is not None:
                return dig

        # C. 开新种：放置区+本层深 ≥3，且 L1 有可点
        candidates: list[str] = []
        for kind, n_free in free_counts.items():
            if n_free <= 0 or not can_clear(kind):
                continue
            if kind not in slot_counts and incomplete:
                if slot_capacity - len(slot) < 3:
                    continue
            candidates.append(kind)
        if not candidates:
            return None
        candidates.sort(
            key=lambda k: (
                slot_counts.get(k, 0),
                free_counts.get(k, 0),
                inv.get(k, 0),
            ),
            reverse=True,
        )
        return best_hit(candidates[0])

    if all_tiles:
        layer_map = assign_tile_layers(all_tiles, max_layer=plan_depth)
        for depth in range(1, plan_depth + 1):
            pool = tiles_up_to_layer(all_tiles, layer_map, depth)
            supply = dict(Counter(t.kind for t in pool))
            choice = try_at_depth(supply)
            if choice is not None:
                return choice
        return None

    # 无分层信息：只用可点 + 总库存
    supply = dict(board_counts) if board_counts else dict(free_counts)
    return try_at_depth(supply)


def point_in_roi(x: int, y: int, roi: tuple[int, int, int, int]) -> bool:
    x1, y1, x2, y2 = roi
    return x1 <= x <= x2 and y1 <= y <= y2


def filter_tiles_outside_roi(
    tiles: list[TileHit], forbidden: tuple[int, int, int, int]
) -> list[TileHit]:
    """丢掉落在禁止区（如七格放置槽）内的匹配点。"""
    return [t for t in tiles if not point_in_roi(t.center[0], t.center[1], forbidden)]


def filter_placement_tiles(
    tiles: list[TileHit],
    slot_roi: tuple[int, int, int, int],
    temp_roi: tuple[int, int, int, int],
) -> list[TileHit]:
    """禁止点放置区槽内牌；暂移区与放置区 Y 有重叠，暂移区内的牌保留可点。"""
    kept: list[TileHit] = []
    for t in tiles:
        x, y = t.center
        in_slot = point_in_roi(x, y, slot_roi)
        in_temp = point_in_roi(x, y, temp_roi)
        if in_slot and not in_temp:
            continue
        kept.append(t)
    return kept


def scan_play_area(
    screen: np.ndarray,
    template_dir: Path,
    board_roi: tuple[int, int, int, int],
    temp_roi: tuple[int, int, int, int],
    slot_roi: tuple[int, int, int, int],
    *,
    free_threshold: float = 0.68,
    buried_threshold: float = 0.55,
    min_dist: int = 28,
) -> BoardScan:
    """扫描常规区+暂移区：可点牌 + 含被挡的总库存。"""
    from collections import Counter

    free_b, counts_b, all_b = find_tiles_layered_in_roi(
        screen,
        template_dir,
        board_roi,
        free_threshold=free_threshold,
        buried_threshold=buried_threshold,
        min_dist=min_dist,
    )
    free_t, counts_t, all_t = find_tiles_layered_in_roi(
        screen,
        template_dir,
        temp_roi,
        free_threshold=max(0.58, free_threshold - 0.08),
        buried_threshold=max(0.50, buried_threshold - 0.03),
        min_dist=max(22, min_dist - 4),
    )
    for t in free_t:
        t.free = True
    if counts_t and not free_t:
        free_t = find_tiles_in_roi(
            screen,
            template_dir,
            temp_roi,
            threshold=max(0.55, free_threshold - 0.12),
            min_dist=max(22, min_dist - 4),
        )

    free_all = filter_placement_tiles(
        _nms(free_b + free_t, min_dist=min_dist), slot_roi, temp_roi
    )
    # 被挡也计入库存（常规区 all + 暂移）
    all_kept = filter_placement_tiles(all_b + all_t, slot_roi, temp_roi)
    board_counts: dict[str, int] = dict(Counter(t.kind for t in all_kept))
    free_counter = Counter(t.kind for t in free_all)
    for k, n in free_counter.items():
        board_counts[k] = max(board_counts.get(k, 0), n)
    buried_counts = {
        k: max(0, board_counts.get(k, 0) - free_counter.get(k, 0))
        for k in board_counts
    }

    return BoardScan(
        free_tiles=free_all,
        board_counts=board_counts,
        buried_counts=buried_counts,
        all_tiles=all_kept,
    )


def find_playable_tiles(
    screen: np.ndarray,
    template_dir: Path,
    board_roi: tuple[int, int, int, int],
    temp_roi: tuple[int, int, int, int],
    slot_roi: tuple[int, int, int, int],
    *,
    threshold: float = 0.72,
    min_dist: int = 40,
) -> list[TileHit]:
    """兼容旧接口：仅返回可点牌。"""
    scan = scan_play_area(
        screen,
        template_dir,
        board_roi,
        temp_roi,
        slot_roi,
        free_threshold=threshold,
        buried_threshold=max(0.48, threshold - 0.20),
        min_dist=min_dist,
    )
    return scan.free_tiles


def apply_remove_tool(slot: list[str], count: int = TEMP_CAPACITY) -> list[str]:
    """「移出」：放置区前 count 张进暂移区（本地只清槽，暂移牌靠视觉再识别）。"""
    if count <= 0:
        return list(slot)
    return list(slot[count:])
