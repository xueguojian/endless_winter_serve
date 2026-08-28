"""底栏 OCR 结果解析：别名、易混组消歧、模板辅助。"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from loguru import logger

from core.dream_memory.chip_match import fuzzy_match_map_key, match_chip_template
from core.dream_memory.chip_image import normalize_chip
from core.dream_memory.config import CHIP_REFS_DIR

# 地图 YAML aliases 之外的全局 OCR 纠错（仅当 canonical 在本图 items 中且 alias 不是独立物品）
DEFAULT_OCR_ALIASES: dict[str, str] = {
    "销": "锁",
    "琐": "锁",
    "所": "锁",
    "引": "弓",
    "東茄子": "茄子",
    "东茄子": "茄子",
    "電卷轴電": "卷轴",
    "电卷轴电": "卷轴",
    "電卷轴": "卷轴",
    "卷轴電": "卷轴",
    # 口哨：哨常被识成号，并夹杂「電」噪点
    "電口号": "口哨",
    "电口号": "口哨",
    "口号": "口哨",
    "電口哨": "口哨",
    "电口哨": "口哨",
    "口哨電": "口哨",

    # 单数字「2」：中文模型常识成乙/二（勿把 Z→2，车间同图有 F）
    "乙": "2",
    "贰": "2",
    "二": "2",
    "貳": "2",}

# 同图并存时需视觉/多路 OCR 消歧（不能单靠 OCR 字面）
CONFUSABLE_GROUPS: tuple[frozenset[str], ...] = (
    frozenset({"X", "弓"}),
)


def _script_kind(text: str) -> str:
    if not text:
        return ""
    if all(c.isascii() for c in text):
        return "ascii"
    if all("\u4e00" <= c <= "\u9fff" for c in text):
        return "cjk"
    return "mixed"


def _try_prefix_suffix_correction(key: str, keys_set: set[str]) -> str | None:
    """OCR 多识 1~2 字前缀或后缀（如 東茄子 → 茄子）。"""
    prefix_hits = [
        map_key
        for map_key in keys_set
        if len(map_key) >= 2 and key.startswith(map_key) and 0 < len(key) - len(map_key) <= 2
    ]
    if len(prefix_hits) == 1:
        return prefix_hits[0]

    suffix_hits = [
        map_key
        for map_key in keys_set
        if len(map_key) >= 2 and key.endswith(map_key) and 0 < len(key) - len(map_key) <= 2
    ]
    if len(suffix_hits) == 1:
        return suffix_hits[0]
    return None


def _try_substring_correction(key: str, keys_set: set[str], *, max_noise: int = 3) -> str | None:
    """OCR 前后夹杂杂字（如 電卷轴電 → 卷轴）。"""
    if len(key) < 2:
        return None
    hits = [
        map_key
        for map_key in keys_set
        if len(map_key) >= 2 and map_key in key and map_key != key
    ]
    if not hits:
        return None
    max_len = max(len(map_key) for map_key in hits)
    longest = [map_key for map_key in hits if len(map_key) == max_len]
    if len(longest) != 1:
        return None
    hit = longest[0]
    if len(key) - len(hit) > max_noise:
        return None
    return hit


def apply_ocr_aliases(
    text: str,
    map_keys: tuple[str, ...] | list[str],
    map_aliases: dict[str, str] | None = None,
) -> str:
    key = (text or "").strip()
    if not key:
        return ""

    keys_set = set(map_keys)
    merged = dict(DEFAULT_OCR_ALIASES)
    if map_aliases:
        merged.update(map_aliases)

    if key in keys_set:
        return key

    canonical = merged.get(key)
    if canonical and canonical in keys_set:
        logger.debug(f"OCR 别名: {key!r} -> {canonical!r}")
        return canonical

    corrected = _try_prefix_suffix_correction(key, keys_set)
    if corrected:
        logger.debug(f"OCR 前后缀纠错: {key!r} -> {corrected!r}")
        return corrected

    corrected = _try_substring_correction(key, keys_set)
    if corrected:
        logger.debug(f"OCR 子串纠错: {key!r} -> {corrected!r}")
        return corrected

    return key


def confusable_peers(label: str, map_keys: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    keys_set = set(map_keys)
    for group in CONFUSABLE_GROUPS:
        if label not in group:
            continue
        peers = tuple(k for k in group if k in keys_set)
        if len(peers) >= 2:
            return peers
    return ()


def _binary_as_bgr(chip_bgr: np.ndarray) -> np.ndarray:
    gray = normalize_chip(chip_bgr)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def _ocr_vote_among(
    chip_bgr: np.ndarray,
    candidates: tuple[str, ...] | list[str],
) -> str | None:
    from core.dream_memory.ocr_rapid import ocr_chip_rapid

    cand_set = set(candidates)
    votes: Counter[str] = Counter()
    variants: list[tuple[np.ndarray, float]] = [
        (chip_bgr, 2.0),
        (chip_bgr, 3.0),
        (_binary_as_bgr(chip_bgr), 2.0),
        (_binary_as_bgr(chip_bgr), 3.0),
    ]
    for image, scale in variants:
        raw = ocr_chip_rapid(image, scale=scale)
        if raw in cand_set:
            votes[raw] += 1
            continue
        alias = apply_ocr_aliases(raw, candidates)
        if alias in cand_set:
            votes[alias] += 1
    if not votes:
        return None
    best, count = votes.most_common(1)[0]
    if count >= 2 or len(votes) == 1:
        return best
    return None


def disambiguate_confusable(
    chip_bgr: np.ndarray,
    ocr_text: str,
    map_keys: tuple[str, ...] | list[str],
    *,
    refs_dir: Path | None = None,
    template_min_score: float = 0.72,
    template_min_margin: float = 0.05,
) -> str | None:
    """在易混组（如 X / 弓）内用模板 + 多路 OCR 投票选出正确项。"""
    peers = confusable_peers(ocr_text, map_keys)
    if len(peers) < 2:
        return None

    root = refs_dir or CHIP_REFS_DIR
    matched = match_chip_template(
        chip_bgr,
        peers,
        refs_dir=root,
        min_score=template_min_score,
        min_margin=template_min_margin,
    )
    if matched:
        name, score = matched
        logger.info(f"易混消歧 template: OCR={ocr_text!r} -> {name!r} ({score:.2f})")
        return name

    voted = _ocr_vote_among(chip_bgr, peers)
    if voted and voted != ocr_text:
        logger.info(f"易混消歧 vote: OCR={ocr_text!r} -> {voted!r}")
        return voted
    return None


def _normalize_ascii_case(text: str, keys_set: set[str]) -> str:
    if len(text) == 1 and text.isalpha() and text.isascii() and text not in keys_set:
        for variant in (text.upper(), text.lower()):
            if variant in keys_set:
                return variant
    return text


_ASCII_CONFUSION = frozenset({"乙", "己", "已", "Ｚ", "ｚ", "Ｎ", "ｎ", "Ａ", "ａ"})


def _looks_like_ascii_chip(raw: str) -> bool:
    """是否值得做字母/数字补识（空串、ASCII、或常见误识）。中文实识结果不要走 eng。"""
    if not raw:
        return True
    if len(raw) == 1 and _script_kind(raw) == "ascii":
        return True
    if raw.isdigit() or raw in _ASCII_CONFUSION:
        return True
    return False


def _match_ascii_glyphs(
    chip_bgr: np.ndarray,
    candidates: tuple[str, ...] | list[str],
    *,
    min_score: float = 0.32,
    min_margin: float = 0.05,
) -> str | None:
    """用矢量字/系统字体做单字母模板匹配（通常 <50ms），能命中则跳过 Tesseract。"""
    ascii_keys = [
        k for k in candidates if len(k) == 1 and _script_kind(k) == "ascii" and k.isalnum()
    ]
    if chip_bgr.size == 0 or not ascii_keys:
        return None

    probe = normalize_chip(chip_bgr)
    th, tw = probe.shape[:2]
    fonts = (cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX)
    scored: list[tuple[str, float]] = []

    pil_fonts: list = []
    try:
        from PIL import ImageFont

        for path in (
            r"C:\Windows\Fonts\arialbd.ttf",
            r"C:\Windows\Fonts\arial.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ):
            try:
                pil_fonts.append(ImageFont.truetype(path, size=max(28, th - 8)))
                if len(pil_fonts) >= 2:
                    break
            except OSError:
                continue
    except Exception:
        pil_fonts = []

    for name in ascii_keys:
        glyphs = {name, name.upper(), name.lower()} if name.isalpha() else {name}
        best = 0.0
        for glyph in glyphs:
            for font in fonts:
                for scale in (1.6, 2.0):
                    canvas = np.zeros((th, tw), dtype=np.uint8)
                    (gw, gh), baseline = cv2.getTextSize(glyph, font, scale, 3)
                    if gw <= 0 or gh <= 0:
                        continue
                    x = max(0, (tw - gw) // 2)
                    y = min(th - 1, max(gh, (th + gh) // 2 - baseline // 2))
                    cv2.putText(canvas, glyph, (x, y), font, scale, 255, 3, cv2.LINE_AA)
                    best = max(
                        best,
                        float(cv2.matchTemplate(probe, canvas, cv2.TM_CCOEFF_NORMED)[0, 0]),
                    )
            for pil_font in pil_fonts:
                try:
                    from PIL import Image, ImageDraw

                    im = Image.new("L", (tw, th), 0)
                    draw = ImageDraw.Draw(im)
                    bbox = draw.textbbox((0, 0), glyph, font=pil_font)
                    gw, gh = bbox[2] - bbox[0], bbox[3] - bbox[1]
                    x = max(0, (tw - gw) // 2 - bbox[0])
                    y = max(0, (th - gh) // 2 - bbox[1])
                    draw.text((x, y), glyph, fill=255, font=pil_font)
                    canvas = np.asarray(im)
                    best = max(
                        best,
                        float(cv2.matchTemplate(probe, canvas, cv2.TM_CCOEFF_NORMED)[0, 0]),
                    )
                except Exception:
                    continue
            if best >= 0.72:
                break
        scored.append((name, best))

    if not scored:
        return None
    scored.sort(key=lambda item: item[1], reverse=True)
    best_name, best_score = scored[0]
    second = scored[1][1] if len(scored) > 1 else 0.0
    if best_score < min_score:
        return None
    if len(scored) > 1 and (best_score - second) < min_margin:
        return None
    logger.debug(f"ascii_glyph {best_name!r} score={best_score:.2f}")
    return best_name


def _tesseract_ascii_fallback(
    chip_bgr: np.ndarray,
    candidates: tuple[str, ...] | list[str],
) -> str | None:
    """RapidOCR 对单字母/数字常空识或误识，用 Tesseract eng + 白名单补识（仅 1 次）。"""
    ascii_keys = [k for k in candidates if len(k) == 1 and _script_kind(k) == "ascii"]
    if not ascii_keys:
        return None
    from core.dream_memory.config import DEFAULT_TESSERACT_CMD
    from core.dream_memory.ocr import ocr_chip, tesseract_available

    if not tesseract_available(DEFAULT_TESSERACT_CMD):
        return None

    whitelist = "".join(dict.fromkeys(ascii_keys))
    # 大小写都放进白名单，便于 N/n、Z/z
    for ch in list(whitelist):
        if ch.isalpha():
            whitelist += ch.upper() + ch.lower()
    whitelist = "".join(dict.fromkeys(whitelist))
    cand_set = set(candidates)
    try:
        text = ocr_chip(
            chip_bgr,
            tesseract_cmd=DEFAULT_TESSERACT_CMD,
            lang="eng",
            whitelist=whitelist,
            psm=10,
        )
    except (FileNotFoundError, RuntimeError):
        return None
    if not text:
        return None
    if text in cand_set:
        return text
    text = _normalize_ascii_case(text, cand_set)
    if text in cand_set:
        return text
    alias = apply_ocr_aliases(text, candidates)
    return alias if alias in cand_set else None


def _tesseract_cjk_fallback(
    chip_bgr: np.ndarray,
    candidates: tuple[str, ...] | list[str],
) -> str | None:
    """RapidOCR 对单汉字（弩/锁等）常空识，用 Tesseract chi_sim 补识。"""
    cjk_keys = [k for k in candidates if len(k) == 1 and _script_kind(k) == "cjk"]
    if not cjk_keys:
        return None
    from core.dream_memory.config import DEFAULT_TESSERACT_CMD
    from core.dream_memory.ocr import ocr_chip, tesseract_available

    if not tesseract_available(DEFAULT_TESSERACT_CMD):
        return None
    whitelist = "".join(dict.fromkeys(cjk_keys))
    try:
        text = ocr_chip(
            chip_bgr,
            tesseract_cmd=DEFAULT_TESSERACT_CMD,
            lang="chi_sim",
            whitelist=whitelist,
        )
    except (FileNotFoundError, RuntimeError):
        return None
    if not text:
        return None
    cand_set = set(candidates)
    if text in cand_set:
        return text
    alias = apply_ocr_aliases(text, candidates)
    return alias if alias in cand_set else None


def resolve_chip_label(
    chip_bgr: np.ndarray,
    ocr_text: str,
    map_keys: tuple[str, ...] | list[str],
    *,
    map_aliases: dict[str, str] | None = None,
    refs_dir: Path | None = None,
    fuzzy_min_ratio: float = 0.72,
    template_min_score: float = 0.72,
    template_min_margin: float = 0.05,
    strict: bool = False,
) -> tuple[str, str]:
    """OCR 原始文本 → 地图物品名。strict=True 时未命中地图名则返回空（PK 用）。"""
    keys_set = set(map_keys)
    raw = (ocr_text or "").strip()
    text = apply_ocr_aliases(raw, map_keys, map_aliases)
    text = _normalize_ascii_case(text, keys_set)

    peers = confusable_peers(text, map_keys) if text in keys_set else ()
    if len(peers) >= 2:
        picked = disambiguate_confusable(
            chip_bgr,
            text,
            map_keys,
            refs_dir=refs_dir,
            template_min_score=template_min_score,
            template_min_margin=template_min_margin,
        )
        if picked:
            return picked, f"disambig({raw!r}->{picked})"

    if text in keys_set:
        return text, f"rapidocr({raw!r})"

    if not strict:
        fuzzy = fuzzy_match_map_key(text, map_keys, min_ratio=fuzzy_min_ratio)
        if fuzzy:
            name, score = fuzzy
            peer_group = confusable_peers(name, map_keys)
            if len(peer_group) >= 2:
                picked = disambiguate_confusable(
                    chip_bgr,
                    name,
                    map_keys,
                    refs_dir=refs_dir,
                    template_min_score=template_min_score,
                    template_min_margin=template_min_margin,
                )
                if picked:
                    return picked, f"disambig_fuzzy({raw!r}->{picked})"
            tag = "rapidocr_fuzzy" if name != text else "rapidocr"
            return name, f"{tag}({raw!r}->{name}, {score:.2f})"

    single_keys = tuple(
        k for k in map_keys if len(k) == 1 and _script_kind(k) in ("ascii", "cjk")
    )
    ascii_singles = tuple(k for k in single_keys if _script_kind(k) == "ascii")
    cjk_singles = tuple(k for k in single_keys if _script_kind(k) == "cjk")

    # 单字母/数字：先毫秒级字形匹配，避免每个中文误识都拖进 Tesseract
    if ascii_singles and _looks_like_ascii_chip(raw):
        glyph = _match_ascii_glyphs(chip_bgr, ascii_singles)
        if glyph and glyph in keys_set:
            return glyph, f"ascii_glyph({raw!r}->{glyph})"
        # 不再做 4 路 RapidOCR 投票（几乎和 Tesseract 一样慢）；直接一次 eng 兜底
        digit_keys = tuple(k for k in ascii_singles if k.isdigit())
        digit_hints = {"乙", "二", "贰", "貳"}
        tess_pool: tuple[str, ...] = ascii_singles
        if digit_keys and (raw.isdigit() or raw in digit_hints):
            tess_pool = digit_keys
        tess = _tesseract_ascii_fallback(chip_bgr, tess_pool)
        if tess and tess in keys_set:
            return tess, f"tesseract_ascii({raw!r}->{tess})"

    # 单汉字（弩等）：仅 RapidOCR 空串时才用 chi_sim，避免「盆」等误识再跑一遍慢 OCR
    if cjk_singles and not raw:
        tess = _tesseract_cjk_fallback(chip_bgr, cjk_singles)
        if tess and tess in keys_set:
            return tess, f"tesseract_cjk({raw!r}->{tess})"

    if chip_bgr.size > 0:
        root = refs_dir or CHIP_REFS_DIR
        matched = match_chip_template(
            chip_bgr,
            map_keys,
            refs_dir=root,
            min_score=template_min_score,
            min_margin=template_min_margin,
        )
        if matched:
            name, score = matched
            if name in keys_set:
                return name, f"template({raw!r}->{name}, {score:.2f})"

    if strict:
        return "", ""

    return "", ""
