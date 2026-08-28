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
# 末字漏识：槽位右缘太贴字 → 用按钮底色做恒定边距（勿用 REPLICATE，易幻影出「東」）
CHIP_PAD_RATIO = 0.22
# RapidOCR 偶发在字前加的噪点前缀
_OCR_NOISE_PREFIXES = ("東", "东", "電", "电")
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


def _chip_pad_color(chip_bgr: np.ndarray) -> tuple[int, int, int]:
    """取四角中位色作为衬底，避免 REPLICATE 把笔画拖出假字。"""
    h, w = chip_bgr.shape[:2]
    if h < 2 or w < 2:
        return (40, 40, 40)
    corners = np.array(
        [
            chip_bgr[0, 0],
            chip_bgr[0, w - 1],
            chip_bgr[h - 1, 0],
            chip_bgr[h - 1, w - 1],
        ],
        dtype=np.float32,
    )
    color = np.median(corners, axis=0)
    return (int(color[0]), int(color[1]), int(color[2]))


def _strip_ocr_noise_prefix(text: str) -> str:
    raw = (text or "").strip()
    for prefix in _OCR_NOISE_PREFIXES:
        if raw.startswith(prefix) and len(raw) > len(prefix):
            return raw[len(prefix) :]
    return raw


def _prepare_chip(chip_bgr, *, scale: float = DEFAULT_SCALE, pad_ratio: float = 0.0):
    """放大槽位图；可选恒定色边距，减轻末字被裁切/漏识。"""
    if chip_bgr.size == 0:
        return chip_bgr
    image = chip_bgr
    if pad_ratio > 0:
        h, w = image.shape[:2]
        pad_x = max(10, int(round(w * pad_ratio)))
        pad_y = max(6, int(round(h * pad_ratio * 0.4)))
        fill = _chip_pad_color(image)
        image = cv2.copyMakeBorder(
            image,
            pad_y,
            pad_y,
            pad_x,
            pad_x,
            cv2.BORDER_CONSTANT,
            value=fill,
        )
    if scale != 1.0:
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return image


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

    return _strip_ocr_noise_prefix(clean_ocr_text(text))


def ocr_chip_rapid(
    chip_bgr,
    *,
    scale: float = DEFAULT_SCALE,
    clean: bool = True,
    pad_ratio: float = 0.0,
) -> str:
    engine = _get_rec_engine()
    image = _prepare_chip(chip_bgr, scale=scale, pad_ratio=pad_ratio)
    if image.size == 0:
        return ""
    with _engine_lock:
        result, elapse = engine(image)
    total = sum(elapse) if elapse else 0.0
    text = _parse_rec_result(result, clean=clean)
    if text:
        logger.debug(
            f"RapidOCR rec scale={scale} pad={pad_ratio:.2f}: {text!r} ({total:.3f}s)"
        )
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
    prepared = [
        _prepare_chip(patch, scale=scale, pad_ratio=CHIP_PAD_RATIO) for patch in patches
    ]
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


def _ocr_right_half(chip_bgr, *, scale: float = RETRY_SCALE) -> str:
    """只认右半槽，专门补读末字（如「研杵」的「杵」）。"""
    if chip_bgr.size == 0:
        return ""
    _h, w = chip_bgr.shape[:2]
    if w < 8:
        return ""
    # 略重叠，避免把末字切没
    x0 = max(0, int(w * 0.38))
    right = chip_bgr[:, x0:]
    return ocr_chip_rapid(right, scale=scale, pad_ratio=CHIP_PAD_RATIO)


def _merge_left_right(left: str, right: str) -> str:
    """把整槽首字与右半槽结果拼成完整词（右半是真实 OCR，不是字典补字）。"""
    left = (left or "").strip()
    right = (right or "").strip()
    if not right:
        return left
    if not left:
        return right
    if right.startswith(left):
        return right
    if left.endswith(right):
        return left
    # 研 + 杵 -> 研杵；研 + 研杵 -> 研杵
    if len(right) == 1:
        return left + right
    if left and right.startswith(left[-1]):
        return left + right[1:]
    return left + right


def ocr_chip_rapid_robust(
    chip_bgr,
    map_keys: tuple[str, ...] | list[str] | None = None,
) -> str:
    """单槽复识：未命中时换预处理；仍短时再认右半槽拼末字。"""
    keys = set(map_keys or [])

    def _better(a: str, b: str) -> str:
        """只在命中地图名时采纳复识；绝不因为更长就收下噪点（如 東研）。"""
        if b in keys and b != a:
            logger.info(f"RapidOCR 复识: {a!r} -> {b!r}")
            return b
        if a in keys:
            return a
        if b in keys:
            return b
        if a in AMBIGUOUS_LABELS and b and b not in AMBIGUOUS_LABELS:
            return b
        return a or b

    primary = ocr_chip_rapid(chip_bgr, scale=DEFAULT_SCALE)
    if primary in keys and primary not in AMBIGUOUS_LABELS:
        return primary

    retry = ocr_chip_rapid(chip_bgr, scale=RETRY_SCALE)
    chosen = _better(primary, retry)
    if chosen in keys and chosen not in AMBIGUOUS_LABELS:
        return chosen

    padded = ocr_chip_rapid(chip_bgr, scale=RETRY_SCALE, pad_ratio=CHIP_PAD_RATIO)
    chosen = _better(chosen, padded)
    if chosen in keys and chosen not in AMBIGUOUS_LABELS:
        return chosen

    # 仍未命中且结果偏短：右半槽单独 OCR，把末字真正读出来再拼接
    if chosen and len(chosen) <= 2:
        right = _ocr_right_half(chip_bgr)
        if right:
            merged = _merge_left_right(chosen, right)
            picked = _better(chosen, merged)
            if picked != chosen:
                logger.info(
                    f"RapidOCR 右半拼字: {chosen!r} + {right!r} -> {picked!r}"
                )
            elif merged == chosen and right not in (chosen, ""):
                logger.debug(
                    f"RapidOCR 右半未拼出地图名: left={chosen!r} right={right!r}"
                )
            return picked
    return chosen
