"""Tesseract OCR 封装（寻梦记忆底栏目标文字）。"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import cv2
import numpy as np
from loguru import logger

try:
    import pytesseract
except ImportError:  # pragma: no cover
    pytesseract = None  # type: ignore[assignment]


def tesseract_available(tesseract_cmd: Path | str | None = None) -> bool:
    if pytesseract is None:
        return False
    cmd = Path(str(tesseract_cmd)) if tesseract_cmd else None
    if cmd and cmd.is_file():
        return True
    return shutil.which("tesseract") is not None


def configure_tesseract(tesseract_cmd: Path | str) -> None:
    if pytesseract is None:
        raise RuntimeError(
            "未安装 pytesseract，请运行: .venv\\Scripts\\pip.exe install pytesseract"
        )
    path = Path(str(tesseract_cmd))
    if path.is_file():
        pytesseract.pytesseract.tesseract_cmd = str(path)
    elif not shutil.which("tesseract"):
        raise FileNotFoundError(
            f"未找到 Tesseract: {path}\n"
            "请安装: https://github.com/UB-Mannheim/tesseract/wiki"
        )


def _preprocess_chip(chip_bgr: np.ndarray) -> np.ndarray:
    """白字蓝底按钮 → 二值图，便于 chi_sim OCR。"""
    if chip_bgr.size == 0:
        return chip_bgr
    from core.dream_memory.chip_image import normalize_chip

    binary = normalize_chip(chip_bgr, width=420, height=108)
    pad = cv2.copyMakeBorder(binary, 24, 24, 24, 24, cv2.BORDER_CONSTANT, value=0)
    return pad


def normalize_item_name_for_match(text: str) -> str:
    return clean_ocr_text(text)


def clean_ocr_text(raw: str, *, keep_brackets: bool = False) -> str:
    text = raw.replace("\n", "").replace(" ", "").strip()
    if keep_brackets:
        text = re.sub(r"[^\u4e00-\u9fffA-Za-z0-9\[\]【】〔〕［］「」『』]", "", text)
    else:
        text = re.sub(r"[^\u4e00-\u9fffA-Za-z0-9]", "", text)
    return text


def ocr_chip(
    chip_bgr: np.ndarray,
    *,
    tesseract_cmd: Path | str | None = None,
    lang: str = "chi_sim",
    keep_brackets: bool = False,
) -> str:
    if pytesseract is None:
        raise RuntimeError("pytesseract 未安装")
    if tesseract_cmd:
        configure_tesseract(tesseract_cmd)

    processed = _preprocess_chip(chip_bgr)
    if processed.size == 0:
        return ""

    try:
        raw = pytesseract.image_to_string(
            processed,
            lang=lang,
            config="--psm 7 -c preserve_interword_spaces=0",
        )
    except pytesseract.TesseractNotFoundError as exc:
        raise FileNotFoundError(
            "未找到 tesseract.exe，请在 config_555x.yaml dream_memory.tesseract_cmd 配置路径"
        ) from exc

    text = clean_ocr_text(raw, keep_brackets=keep_brackets)
    if text:
        logger.debug(f"OCR chip: {raw!r} -> {text!r}")
    return text
