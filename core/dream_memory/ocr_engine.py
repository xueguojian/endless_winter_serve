"""寻梦记忆 OCR 统一入口。"""

from __future__ import annotations

from pathlib import Path

from loguru import logger

from core.dream_memory.ocr import ocr_chip, tesseract_available
from core.dream_memory.ocr_rapid import ocr_chip_rapid, rapidocr_available

OCR_ENGINE_RAPID = "rapidocr"
OCR_ENGINE_TESSERACT = "tesseract"
OCR_ENGINE_AUTO = "auto"


def resolve_ocr_engine(preferred: str | None) -> str:
    choice = (preferred or OCR_ENGINE_AUTO).strip().lower()
    if choice == OCR_ENGINE_AUTO:
        if rapidocr_available():
            return OCR_ENGINE_RAPID
        if tesseract_available():
            return OCR_ENGINE_TESSERACT
        return OCR_ENGINE_TESSERACT
    if choice == OCR_ENGINE_RAPID and not rapidocr_available():
        logger.warning("RapidOCR 不可用，回退 Tesseract")
        return OCR_ENGINE_TESSERACT
    return choice


def ocr_engine_available(preferred: str | None = None) -> bool:
    engine = resolve_ocr_engine(preferred)
    if engine == OCR_ENGINE_RAPID:
        return rapidocr_available()
    return tesseract_available()


def ocr_chip_text(
    chip_bgr,
    *,
    engine: str | None = None,
    tesseract_cmd: Path | str | None = None,
    clean: bool = True,
) -> tuple[str, str]:
    """识别槽位文字，返回 (文本, 引擎标识)。

    clean=False 时保留括号等符号（用于账号名联盟简称）。
    """
    if chip_bgr.size == 0:
        return "", ""

    resolved = resolve_ocr_engine(engine)
    if resolved == OCR_ENGINE_RAPID:
        try:
            return ocr_chip_rapid(chip_bgr, clean=clean), OCR_ENGINE_RAPID
        except Exception as exc:
            logger.warning(f"RapidOCR 失败，回退 Tesseract: {exc}")
            resolved = OCR_ENGINE_TESSERACT

    text = ocr_chip(
        chip_bgr,
        tesseract_cmd=tesseract_cmd,
        keep_brackets=not clean,
    )
    return text, OCR_ENGINE_TESSERACT


def warmup_ocr(engine: str | None = None) -> None:
    """后台预加载 OCR 模型与首扫依赖，减少首次识别等待。"""
    if resolve_ocr_engine(engine) != OCR_ENGINE_RAPID:
        return
    from core.dream_memory.chip_match import _ensure_manifest
    from core.dream_memory.config import CHIP_REFS_DIR
    from core.dream_memory.label_resolve import resolve_chip_label  # noqa: F401
    from core.dream_memory.ocr_rapid import warmup_rapidocr

    try:
        _ensure_manifest(CHIP_REFS_DIR)
    except Exception as exc:
        logger.debug(f"底栏模板 manifest 预加载跳过: {exc}")
    warmup_rapidocr()
