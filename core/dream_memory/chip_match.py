"""寻梦记忆底栏文字：模板匹配 + 模糊纠错（游戏字体 Tesseract 识别率低）。"""

from __future__ import annotations

import hashlib
import re
from difflib import SequenceMatcher
from pathlib import Path

import cv2
import numpy as np
import yaml
from loguru import logger

from core.dream_memory.chip_image import normalize_chip
from core.dream_memory.config import CHIP_REFS_DIR

MANIFEST_NAME = "manifest.yaml"

# 禁止 OCR 结果被模糊匹配到这些「另一项」（字形/读音易混）
_FORBIDDEN_FUZZY: frozenset[tuple[str, str]] = frozenset(
    {
        ("梯子", "灯塔"),
        ("梯子", "瞭望塔"),
        ("灯塔", "梯子"),
        ("瞭望塔", "梯子"),
        ("X", "弓"),
        ("弓", "X"),
    }
)


def _script_kind(text: str) -> str:
    if not text:
        return ""
    if all(c.isascii() for c in text):
        return "ascii"
    if all("\u4e00" <= c <= "\u9fff" for c in text):
        return "cjk"
    return "mixed"


def _is_valid_item_name(name: str) -> bool:
    name = name.strip()
    if not name or len(name) > 10:
        return False
    return all("\u4e00" <= ch <= "\u9fff" for ch in name)


def _ref_hash(item_name: str) -> str:
    return hashlib.sha1(item_name.strip().encode("utf-8")).hexdigest()[:12]


def _manifest_path(refs_dir: Path) -> Path:
    return refs_dir / MANIFEST_NAME


def _load_manifest(refs_dir: Path) -> dict[str, str]:
    """hash -> 物品名"""
    path = _manifest_path(refs_dir)
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    by_hash = raw.get("by_hash") or {}
    return {str(k): str(v) for k, v in by_hash.items()}


def _save_manifest(refs_dir: Path, by_hash: dict[str, str]) -> None:
    refs_dir.mkdir(parents=True, exist_ok=True)
    by_name = {name: hid for hid, name in by_hash.items()}
    payload = {"by_hash": dict(sorted(by_hash.items())), "by_name": dict(sorted(by_name.items()))}
    with _manifest_path(refs_dir).open("w", encoding="utf-8") as fh:
        yaml.safe_dump(payload, fh, allow_unicode=True, sort_keys=False)


def _migrate_legacy_pngs(refs_dir: Path, by_hash: dict[str, str]) -> dict[str, str]:
    """把旧版中文文件名的模板迁入 manifest（仅保留能正确解码的文件名）。"""
    known_names = set(by_hash.values())
    changed = False
    for path in refs_dir.glob("*.png"):
        if path.stem in by_hash:
            continue
        name = path.stem
        if not name or name in known_names:
            continue
        if not _is_valid_item_name(name):
            logger.warning(f"跳过无效底栏模板文件名: {path.name}")
            continue
        try:
            img = _imread_gray(path)
        except OSError:
            img = None
        if img is None or img.size == 0:
            continue
        hid = _ref_hash(name)
        target = refs_dir / f"{hid}.png"
        if not target.is_file():
            _imwrite_gray(target, img)
        by_hash[hid] = name
        known_names.add(name)
        changed = True
        logger.info(f"迁移底栏模板: {name} -> {target.name}")
    if changed:
        _save_manifest(refs_dir, by_hash)
    return by_hash


def _ensure_manifest(refs_dir: Path) -> dict[str, str]:
    by_hash = _load_manifest(refs_dir)
    if refs_dir.is_dir():
        by_hash = _migrate_legacy_pngs(refs_dir, by_hash)
    return by_hash


def _imread_gray(path: Path) -> np.ndarray | None:
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)


def _imwrite_gray(path: Path, image: np.ndarray) -> None:
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise OSError(f"无法写入图片: {path}")
    encoded.tofile(str(path))


def _text_fill_width(image: np.ndarray) -> tuple[float, int]:
    if image.size == 0:
        return 0.0, 0
    cols = np.where(image.max(axis=0) > 0)[0]
    if cols.size == 0:
        return 0.0, 0
    width = int(cols[-1] - cols[0] + 1)
    fill = float(np.count_nonzero(image)) / float(image.size)
    return fill, width


def _length_penalty(probe: np.ndarray, ref: np.ndarray, name: str) -> float:
    _, probe_w = _text_fill_width(probe)
    _, ref_w = _text_fill_width(ref)
    if probe_w <= 0 or ref_w <= 0:
        return 0.6
    width_ratio = min(probe_w, ref_w) / max(probe_w, ref_w)
    char_ratio = min(len(name), max(2, probe_w // 18)) / max(len(name), 1)
    char_ratio = min(1.0, max(0.35, char_ratio))
    return 0.35 + 0.65 * width_ratio * char_ratio


def save_chip_ref(
    chip_bgr: np.ndarray,
    item_name: str,
    *,
    refs_dir: Path | None = None,
) -> Path:
    root = refs_dir or CHIP_REFS_DIR
    root.mkdir(parents=True, exist_ok=True)
    safe = item_name.strip()
    if not safe:
        raise ValueError("物品名不能为空")

    hid = _ref_hash(safe)
    out = root / f"{hid}.png"
    _imwrite_gray(out, normalize_chip(chip_bgr))

    by_hash = _ensure_manifest(root)
    by_hash[hid] = safe
    _save_manifest(root, by_hash)
    logger.debug(f"已保存底栏模板: {safe} ({out.name})")
    return out


def match_chip_template(
    chip_bgr: np.ndarray,
    candidate_names: tuple[str, ...] | list[str],
    *,
    refs_dir: Path | None = None,
    min_score: float = 0.72,
    min_margin: float = 0.05,
) -> tuple[str, float] | None:
    """与 chip_refs 中已保存模板比对，返回最佳物品名。"""
    if chip_bgr.size == 0 or not candidate_names:
        return None

    root = refs_dir or CHIP_REFS_DIR
    by_hash = _ensure_manifest(root)
    if not by_hash:
        return None

    probe = normalize_chip(chip_bgr)
    scored: list[tuple[str, float]] = []

    for name in candidate_names:
        safe = name.strip()
        hid = _ref_hash(safe)
        ref_path = root / f"{hid}.png"
        if not ref_path.is_file():
            continue
        ref = _imread_gray(ref_path)
        if ref is None or ref.size == 0:
            continue
        if ref.shape != probe.shape:
            ref = cv2.resize(ref, (probe.shape[1], probe.shape[0]))
        raw_score = float(cv2.matchTemplate(probe, ref, cv2.TM_CCOEFF_NORMED)[0, 0])
        score = raw_score * _length_penalty(probe, ref, safe)
        scored.append((safe, score))

    if not scored:
        return None

    scored.sort(key=lambda item: item[1], reverse=True)
    best_name, best_score = scored[0]
    second_score = scored[1][1] if len(scored) > 1 else 0.0

    if best_score < min_score:
        return None
    if len(scored) > 1 and (best_score - second_score) < min_margin:
        logger.debug(
            f"模板匹配不确定: {best_name}={best_score:.2f}, "
            f"次优={scored[1][0]}={second_score:.2f}"
        )
        return None
    return best_name, best_score


def _match_key_score(normalized: str, key: str) -> float:
    nk = re.sub(r"\s+", "", (key or "").strip())
    if not normalized or not nk:
        return 0.0
    if normalized == nk:
        return 1.0

    norm_script = _script_kind(normalized)
    key_script = _script_kind(nk)
    if norm_script and key_script and norm_script != key_script:
        return 0.0

    ratio = SequenceMatcher(None, normalized, nk).ratio()
    best = ratio

    if len(normalized) == len(nk):
        if len(normalized) == 1:
            # 单字不能靠「差一字给 0.86」硬凑（否则 销/琐 会误匹配 X）
            best = ratio
        else:
            diffs = sum(a != b for a, b in zip(normalized, nk))
            if diffs == 1:
                best = max(best, 0.86)
            elif diffs == 2 and len(nk) <= 4:
                best = max(best, 0.72)
    elif abs(len(normalized) - len(nk)) >= 2:
        # 长度差太多（如 灯塔 vs 单筒望远镜）时不靠 ratio 硬凑
        best = min(best, 0.55)

    if nk in normalized:
        # OCR 读全了，优先更长的地图名（如 单筒望远镜 > 望远镜）
        best = max(best, 0.78 + min(0.12, len(nk) * 0.015))
    elif normalized in nk:
        # OCR 只读到一部分：仅当覆盖率足够高时才认
        cover = len(normalized) / len(nk)
        if cover >= 0.66:
            best = max(best, 0.74 + 0.18 * cover)
        else:
            best = max(best, ratio)

    return best


def fuzzy_match_map_key(
    text: str,
    map_keys: tuple[str, ...] | list[str],
    *,
    min_ratio: float = 0.68,
) -> tuple[str, float] | None:
    """OCR 结果与地图物品名模糊匹配，返回 (名称, 分数)。"""
    normalized = re.sub(r"\s+", "", (text or "").strip())
    if not normalized:
        return None

    ranked: list[tuple[float, int, str]] = []
    for key in map_keys:
        score = _match_key_score(normalized, key)
        if score <= 0:
            continue
        if (normalized, key.strip()) in _FORBIDDEN_FUZZY:
            continue
        ranked.append((score, len(key.strip()), key))

    if not ranked:
        return None

    ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
    best_score, _, best_name = ranked[0]
    if best_score < min_ratio:
        return None
    return best_name, best_score
