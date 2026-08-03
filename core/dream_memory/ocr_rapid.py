"""RapidOCR 引擎（寻梦记忆底栏，中文游戏字体识别更准）。"""

from __future__ import annotations

import threading
import time

import cv2
import numpy as np
from loguru import logger

try:
    from rapidocr_onnxruntime import RapidOCR
except ImportError:  # pragma: no cover
    RapidOCR = None  # type: ignore[assignment,misc]

_rec_engine: RapidOCR | None = None
_engine_lock = threading.Lock()
_warmed = False

AMBIGUOUS_LABELS = frozenset({"梯子", "灯塔", "瞭望塔", "喇叭", "毛巾", "石像"})
DEFAULT_SCALE = 2.0
RETRY_SCALE = 3.0
# PK 典型槽位尺寸，预热时用真实 scale 走一遍推理路径
_WARMUP_CHIP_WH = (210, 58)
_WARMUP_SLOT_COUNT = 6


def rapidocr_available() -> bool:
    return RapidOCR is not None


def _get_rec_engine() -> RapidOCR:
    """纯识别引擎：底栏 ROI 固定，跳过文字检测可快一个数量级。"""
    global _rec_engine
    if RapidOCR is None:
        raise RuntimeError(
            "未安装 rapidocr-onnxruntime，请运行:\n"
            "  .venv\\Scripts\\pip.exe install rapidocr-onnxruntime onnxruntime"
        )
    if _rec_engine is None:
        _rec_engine = RapidOCR(use_det=False, use_cls=False)
        logger.debug("RapidOCR 纯识别引擎已初始化 (use_det=False)")
    return _rec_engine


def warmup_rapidocr(
    *,
    slot_count: int = _WARMUP_SLOT_COUNT,
    scale: float = DEFAULT_SCALE,
) -> None:
    """预加载模型并用真实槽位尺寸跑一遍 batch，避免首帧 ONNX 再编译。"""
    global _warmed
    if _warmed:
        return
    _get_rec_engine()
    w, h = _WARMUP_CHIP_WH
    patches = [np.full((h, w, 3), 180, dtype=np.uint8) for _ in range(slot_count)]
    ocr_slots_batch(patches, scale=scale)
    _warmed = True
    logger.debug(f"RapidOCR 预热完成 ({slot_count} 槽, scale={scale})")


def _prepare_chip(chip_bgr, *, scale: float = DEFAULT_SCALE):
    if chip_bgr.size == 0:
        return chip_bgr
    return cv2.resize(chip_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)


def _parse_rec_result(result, *, clean: bool = True) -> str:
    if not result:
        return ""
    line = result[0]
    if not line:
        return ""
    text = line[0]
    if not isinstance(text, str):
        return ""
    if not clean:
        return text.replace("\n", "").strip()
    from core.dream_memory.ocr import clean_ocr_text

    return clean_ocr_text(text)


def ocr_chip_rapid(chip_bgr, *, scale: float = DEFAULT_SCALE, clean: bool = True) -> str:
    engine = _get_rec_engine()
    image = _prepare_chip(chip_bgr, scale=scale)
    if image.size == 0:
        return ""
    with _engine_lock:
        result, elapse = engine(image)
    total = sum(elapse) if elapse else 0.0
    text = _parse_rec_result(result, clean=clean)
    if text:
        logger.debug(f"RapidOCR rec scale={scale}: {text!r} ({total:.3f}s)")
    return text


def ocr_slots_batch(
    patches: list[np.ndarray],
    *,
    scale: float = DEFAULT_SCALE,
) -> list[str]:
    """逐槽纯识别（底栏 ROI 固定，比合并检测快且不会串字）。"""
    if not patches:
        return []

    t0 = time.perf_counter()
    engine = _get_rec_engine()
    prepared = [_prepare_chip(patch, scale=scale) for patch in patches]
    texts: list[str] = []
    with _engine_lock:
        for image in prepared:
            if image.size == 0:
                texts.append("")
                continue
            result, elapse = engine(image)
            total = sum(elapse) if elapse else 0.0
            text = _parse_rec_result(result)
            if text:
                logger.debug(f"RapidOCR rec scale={scale}: {text!r} ({total:.3f}s)")
            texts.append(text)
    elapsed = time.perf_counter() - t0
    logger.info(f"RapidOCR {len(patches)} 槽识别: {texts} ({elapsed:.2f}s)")
    return texts


def ocr_chip_rapid_robust(
    chip_bgr,
    map_keys: tuple[str, ...] | list[str] | None = None,
) -> str:
    """单槽复识：仅易混词或未命中地图名时高倍率再识一次。"""
    keys = set(map_keys or [])
    primary = ocr_chip_rapid(chip_bgr, scale=DEFAULT_SCALE)
    if not primary:
        retry = ocr_chip_rapid(chip_bgr, scale=RETRY_SCALE)
        return retry
    if primary in keys and primary not in AMBIGUOUS_LABELS:
        return primary
    if primary in AMBIGUOUS_LABELS or (primary and primary not in keys):
        retry = ocr_chip_rapid(chip_bgr, scale=RETRY_SCALE)
        if retry in keys:
            if primary != retry:
                logger.info(f"RapidOCR 复识: {primary!r} -> {retry!r}")
            return retry
        if primary in AMBIGUOUS_LABELS and retry and retry not in AMBIGUOUS_LABELS:
            return retry
        return primary or retry
    return primary
