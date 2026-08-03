"""寻梦记忆：底栏目标识别与坐标解析。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from loguru import logger

from core.dream_memory.chip_match import fuzzy_match_map_key
from core.dream_memory.maps import DreamMemoryMap
from core.dream_memory.ocr_engine import ocr_chip_text


def normalize_item_name(text: str) -> str:
    return re.sub(r"\s+", "", (text or "").strip())


@dataclass(frozen=True)
class TargetChip:
    slot_index: int
    text: str
    active: bool
    roi: tuple[int, int, int, int]


def _crop(screen: np.ndarray, roi: tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = roi
    h, w = screen.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return np.array([])
    return screen[y1:y2, x1:x2]


def chip_is_active(
    chip_bgr: np.ndarray,
    *,
    min_brightness: float = 95.0,
) -> bool:
    """未找到的目标按钮较亮；已划线/变灰的跳过（普通模式）。"""
    if chip_bgr.size == 0:
        return False
    gray = cv2.cvtColor(chip_bgr, cv2.COLOR_BGR2GRAY)
    if float(gray.mean()) < min_brightness:
        return False

    # 检测中间横线（删除线）：中间带比上下带更暗
    h = gray.shape[0]
    if h >= 12:
        band = max(2, h // 5)
        mid_y = h // 2
        mid = gray[mid_y - band : mid_y + band, :]
        top = gray[max(0, mid_y - band * 3) : max(1, mid_y - band), :]
        bottom = gray[min(h - 1, mid_y + band) : min(h, mid_y + band * 3), :]
        if top.size and bottom.size and mid.size:
            mid_mean = float(mid.mean())
            surround = float(np.mean([top.mean(), bottom.mean()]))
            if mid_mean + 12 < surround:
                logger.debug(
                    f"chip 中间横线检测 mid={mid_mean:.1f} surround={surround:.1f}"
                )
                return False
    return True


def pk_slot_has_content(chip_bgr: np.ndarray) -> bool:
    """PK 槽位是否仍存在（不检测变灰/划线；消失后多为空白均匀区）。"""
    if chip_bgr.size == 0:
        return False
    gray = cv2.cvtColor(chip_bgr, cv2.COLOR_BGR2GRAY)
    mean = float(gray.mean())
    std = float(gray.std())
    if std < 10.0:
        return False
    if mean < 35.0 and std < 18.0:
        return False
    return True


def pk_last_present_slot_index(patches: list[np.ndarray]) -> int:
    """PK 自槽位 6 起向左消失，返回仍存在的最高槽位索引（无则 -1）。"""
    for index in range(len(patches) - 1, -1, -1):
        if pk_slot_has_content(patches[index]):
            return index
    return -1


def _label_from_ocr_text(
    ocr_text: str,
    map_keys: tuple[str, ...] | list[str],
    engine_name: str,
    *,
    fuzzy_min_ratio: float = 0.72,
) -> tuple[str, str]:
    keys_set = set(map_keys)
    if ocr_text in keys_set:
        return ocr_text, f"{engine_name}({ocr_text!r})"
    fuzzy = fuzzy_match_map_key(ocr_text, map_keys, min_ratio=fuzzy_min_ratio)
    if fuzzy:
        name, score = fuzzy
        if name != ocr_text:
            return name, f"{engine_name}_fuzzy({ocr_text!r}->{name}, {score:.2f})"
        return name, f"{engine_name}({ocr_text!r}, {score:.2f})"
    if ocr_text:
        return ocr_text, f"{engine_name}({ocr_text!r})"
    return "", ""


def recognize_chip_label(
    chip_bgr: np.ndarray,
    map_keys: tuple[str, ...] | list[str],
    *,
    ocr_engine: str | None = None,
    tesseract_cmd: Path | str | None = None,
    refs_dir: Path | None = None,
    template_min_score: float = 0.88,
    template_min_margin: float = 0.08,
    fuzzy_min_ratio: float = 0.72,
    map_aliases: dict[str, str] | None = None,
    strict: bool = False,
) -> tuple[str, str]:
    """识别槽位文字，strict 时仅精确/别名/OCR 纠错命中地图名。"""
    if chip_bgr.size == 0:
        return "", ""

    ocr_text = ""

    try:
        from core.dream_memory.ocr_engine import resolve_ocr_engine

        if resolve_ocr_engine(ocr_engine) == "rapidocr":
            from core.dream_memory.ocr_rapid import ocr_chip_rapid_robust

            ocr_text = ocr_chip_rapid_robust(chip_bgr, map_keys)
        else:
            ocr_text, _engine = ocr_chip_text(
                chip_bgr,
                engine=ocr_engine,
                tesseract_cmd=tesseract_cmd,
            )
    except (FileNotFoundError, RuntimeError) as exc:
        logger.warning(f"OCR 失败: {exc}")

    from core.dream_memory.label_resolve import resolve_chip_label

    return resolve_chip_label(
        chip_bgr,
        ocr_text,
        map_keys,
        map_aliases=map_aliases,
        refs_dir=refs_dir,
        fuzzy_min_ratio=fuzzy_min_ratio,
        template_min_score=min(template_min_score, 0.75),
        template_min_margin=min(template_min_margin, 0.05),
        strict=strict,
    )


def split_bar_into_slots(
    bar: tuple[int, int, int, int],
    slot_count: int,
) -> tuple[tuple[int, int, int, int], ...]:
    """将底栏 ROI 均分为 slot_count 个槽位。"""
    x1, y1, x2, y2 = bar
    count = max(1, slot_count)
    width = x2 - x1
    if width <= 0:
        return ()
    rois: list[tuple[int, int, int, int]] = []
    for index in range(count):
        sx1 = x1 + int(round(index * width / count))
        sx2 = x1 + int(round((index + 1) * width / count)) if index < count - 1 else x2
        if sx2 > sx1:
            rois.append((sx1, y1, sx2, y2))
    return tuple(rois)


def split_bar_grid_slots(
    width: int,
    height: int,
    *,
    rows: int = 2,
    cols: int = 3,
) -> tuple[tuple[int, int, int, int], ...]:
    """将底栏图像按 rows×cols 均分为槽位 ROI（相对左上角）。

    PK 六字栏为上 3 + 下 3，顺序与 DEFAULT_PK_TARGET_SLOTS 一致。
    """
    rows = max(1, int(rows))
    cols = max(1, int(cols))
    width = max(1, int(width))
    height = max(1, int(height))
    cell_w = width / cols
    cell_h = height / rows
    slots: list[tuple[int, int, int, int]] = []
    for row in range(rows):
        for col in range(cols):
            x1 = int(round(col * cell_w))
            y1 = int(round(row * cell_h))
            x2 = int(round((col + 1) * cell_w))
            y2 = int(round((row + 1) * cell_h))
            slots.append((x1, y1, max(x1 + 1, x2), max(y1 + 1, y2)))
    return tuple(slots)


def estimate_slot_count(
    bar_bgr: np.ndarray,
    *,
    min_slots: int = 3,
    max_slots: int = 4,
) -> int:
    """根据底栏竖向分隔/空白估计槽位数量（3 或 4）。"""
    min_slots = max(1, min_slots)
    max_slots = max(min_slots, max_slots)
    if bar_bgr.size == 0:
        return min_slots

    gray = cv2.cvtColor(bar_bgr, cv2.COLOR_BGR2GRAY)
    _, width = gray.shape
    if width < 40:
        return min_slots

    # 槽位间竖缝：列标准差较低（背景均匀）；文字区域标准差较高
    col_std = gray.std(axis=0)
    kernel = max(5, width // 35)
    smoothed = np.convolve(col_std, np.ones(kernel) / kernel, mode="same")

    margin = int(width * 0.06)
    inner = smoothed[margin : width - margin]
    if inner.size == 0:
        return min_slots

    low = float(np.percentile(inner, 35))
    high = float(np.percentile(inner, 65))
    gap_thresh = (low + high) / 2
    min_gap = max(4, width // 45)

    gap_count = 0
    index = 0
    while index < len(inner):
        if inner[index] < gap_thresh:
            start = index
            while index < len(inner) and inner[index] < gap_thresh:
                index += 1
            if index - start >= min_gap:
                gap_count += 1
        else:
            index += 1

    # n 个槽位之间通常有 n-1 条内部分隔缝
    estimated = gap_count + 1
    if estimated < min_slots or estimated > max_slots:
        # 竖缝不明显时，用边缘峰值再试一次
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        edge = np.abs(sobel_x).mean(axis=0)
        edge = np.convolve(edge, np.ones(kernel) / kernel, mode="same")
        segment = edge[margin : width - margin]
        peak_thresh = float(np.percentile(segment, 78))
        min_peak_dist = width // (max_slots + 1)
        peaks: list[int] = []
        for i in range(1, len(segment) - 1):
            if segment[i] < peak_thresh:
                continue
            if segment[i] < segment[i - 1] or segment[i] < segment[i + 1]:
                continue
            if peaks and i - peaks[-1] < min_peak_dist:
                if segment[i] > segment[peaks[-1]]:
                    peaks[-1] = i
            else:
                peaks.append(i)
        estimated = len(peaks) + 1

    return max(min_slots, min(max_slots, estimated))


def resolve_target_slots(
    screen: np.ndarray,
    *,
    target_bar: tuple[int, int, int, int],
    min_slots: int = 3,
    max_slots: int = 4,
    fixed_slots: tuple[tuple[int, int, int, int], ...] | None = None,
) -> tuple[tuple[int, int, int, int], ...]:
    """解析本轮 OCR 使用的槽位 ROI；fixed_slots 非空时直接返回。"""
    if fixed_slots:
        return fixed_slots
    bar_patch = _crop(screen, target_bar)
    slot_count = estimate_slot_count(
        bar_patch,
        min_slots=min_slots,
        max_slots=max_slots,
    )
    slots = split_bar_into_slots(target_bar, slot_count)
    logger.debug(
        f"底栏 {target_bar} 检测到 {slot_count} 槽: {slots}"
    )
    return slots


def read_target_chips(
    screen: np.ndarray,
    slots: tuple[tuple[int, int, int, int], ...] | None = None,
    *,
    map_keys: tuple[str, ...] | list[str] | None = None,
    map_aliases: dict[str, str] | None = None,
    target_bar: tuple[int, int, int, int] | None = None,
    min_slots: int = 3,
    max_slots: int = 4,
    tesseract_cmd: Path | str | None = None,
    ocr_engine: str | None = None,
    min_brightness: float = 95.0,
    refs_dir: Path | None = None,
    template_min_score: float = 0.88,
    template_min_margin: float = 0.08,
    fuzzy_min_ratio: float = 0.72,
    save_matched_refs: bool = False,
    pk_mode: bool = False,
) -> list[TargetChip]:
    if slots is None:
        if target_bar is None:
            raise ValueError("read_target_chips 需要 slots 或 target_bar")
        slots = resolve_target_slots(
            screen,
            target_bar=target_bar,
            min_slots=min_slots,
            max_slots=max_slots,
        )
    results: list[TargetChip] = []
    patches: list[np.ndarray] = []
    actives: list[bool] = []
    slot_limit = len(slots)
    for roi in slots:
        patch = _crop(screen, roi)
        patches.append(patch)

    if pk_mode:
        last_present = pk_last_present_slot_index(patches)
        if last_present < 0:
            return []
        slot_limit = last_present + 1
        actives = [pk_slot_has_content(patch) for patch in patches[:slot_limit]]
    else:
        for patch in patches:
            actives.append(chip_is_active(patch, min_brightness=min_brightness))

    batch_labels: list[tuple[str, str]] | None = None
    if map_keys:
        from core.dream_memory.ocr_engine import resolve_ocr_engine

        if resolve_ocr_engine(ocr_engine) == "rapidocr" and sum(actives) >= 1:
            from core.dream_memory.label_resolve import resolve_chip_label
            from core.dream_memory.ocr_rapid import ocr_chip_rapid_robust, ocr_slots_batch

            keys_set = set(map_keys)
            batch_labels = [("", "")] * len(patches)
            active_indices = [i for i, ok in enumerate(actives[:slot_limit]) if ok]
            active_patches = [patches[i] for i in active_indices]
            batch_texts = ocr_slots_batch(active_patches)
            for slot_index, raw_text in zip(active_indices, batch_texts):
                ocr_text = raw_text or ocr_chip_rapid_robust(
                    patches[slot_index], map_keys
                )
                name, method = resolve_chip_label(
                    patches[slot_index],
                    ocr_text,
                    map_keys,
                    map_aliases=map_aliases,
                    refs_dir=refs_dir,
                    fuzzy_min_ratio=fuzzy_min_ratio,
                    template_min_score=min(template_min_score, 0.75),
                    template_min_margin=min(template_min_margin, 0.05),
                    strict=pk_mode,
                )
                if pk_mode and name not in keys_set:
                    name, method = "", ""
                batch_labels[slot_index] = (name, method)

    for index, roi in enumerate(slots):
        if pk_mode and index >= slot_limit:
            break
        patch = patches[index]
        active = actives[index]
        text = ""
        if active:
            if batch_labels is not None:
                text, method = batch_labels[index]
                if text and method:
                    logger.debug(f"槽位 {index + 1} {method} -> {text!r}")
            elif map_keys:
                text, method = recognize_chip_label(
                    patch,
                    map_keys,
                    ocr_engine=ocr_engine,
                    tesseract_cmd=tesseract_cmd,
                    refs_dir=refs_dir,
                    template_min_score=template_min_score,
                    template_min_margin=template_min_margin,
                    fuzzy_min_ratio=fuzzy_min_ratio,
                    map_aliases=map_aliases,
                    strict=pk_mode,
                )
                if text and method:
                    logger.debug(f"槽位 {index + 1} {method} -> {text!r}")
            else:
                try:
                    text, _engine = ocr_chip_text(
                        patch,
                        engine=ocr_engine,
                        tesseract_cmd=tesseract_cmd,
                    )
                except (FileNotFoundError, RuntimeError) as exc:
                    logger.warning(f"OCR 失败 slot={index}: {exc}")
        if pk_mode:
            active = active and bool(text)
        results.append(
            TargetChip(
                slot_index=index,
                text=text,
                active=active,
                roi=roi,
            )
        )
    if pk_mode:
        logger.debug(f"PK 有效槽 {slot_limit}/{len(slots)}")
    return results


def resolve_item_coord(
    game_map: DreamMemoryMap,
    label: str,
) -> tuple[int, int] | None:
    return game_map.lookup(label)
