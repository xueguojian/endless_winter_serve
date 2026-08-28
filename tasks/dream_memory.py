"""云控寻梦记忆：客户端连续上传截图，服务端 OCR 后下发点击。"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from loguru import logger

from core.adb_client import AdbClient
from core.dream_memory.config import DreamMemoryConfig, _build_config, sample_tap_between_delay
from core.dream_memory.maps import DreamMemoryMap, load_map
from core.dream_memory.ocr_engine import ocr_engine_available, resolve_ocr_engine, warmup_ocr
from core.dream_memory.vision import TargetChip, chip_is_active, read_target_chips, resolve_item_coord

StatusCallback = Callable[[str], None]


@dataclass(frozen=True)
class _TapItem:
    slot_index: int
    text: str
    x: int
    y: int


def _format_chip_slot(chip: TargetChip) -> str:
    """单槽识别摘要，供云控状态栏展示。"""
    n = chip.slot_index + 1
    if not chip.active:
        return f"槽{n}:—"
    if chip.text:
        if chip.ocr_raw and chip.ocr_raw != chip.text:
            return f"槽{n}:{chip.ocr_raw}→{chip.text}"
        return f"槽{n}:{chip.text}"
    if chip.ocr_raw:
        return f"槽{n}:OCR「{chip.ocr_raw}」未匹配(可能未标定)"
    return f"槽{n}:未识别"


def _format_scan_line(chips: list[TargetChip]) -> str:
    if not chips:
        return ""
    return "识别 " + " ".join(_format_chip_slot(chip) for chip in chips)


class DreamMemoryTask:
    """普通寻梦：识别底栏 → 点地图物品，直到 stop。"""

    name = "寻梦记忆"

    def __init__(
        self,
        adb: AdbClient,
        *,
        map_id: str,
        period: int | None = None,
        config: DreamMemoryConfig | None = None,
        on_status: StatusCallback | None = None,
    ):
        self.adb = adb
        self.map_id = str(map_id).strip()
        if not self.map_id:
            raise ValueError("未选择寻梦地图")
        self.game_map: DreamMemoryMap = load_map(self.map_id)
        if period is not None and int(period) >= 1:
            if int(self.game_map.period) != int(period):
                raise ValueError(
                    f"地图「{self.game_map.name}」属于第 {self.game_map.period} 期，"
                    f"与所选第 {period} 期不符"
                )
        self.config = config or _build_config({}, pk=False)
        self.on_status = on_status
        self._stop = threading.Event()
        self._unmatched_logged: set[str] = set()

    def stop(self) -> None:
        self._stop.set()

    def _emit(self, message: str) -> None:
        logger.info("[{}] {}", self.name, message)
        if self.on_status:
            self.on_status(message)

    def _warn_unmatched_map(self, slot_index: int, raw: str) -> None:
        """OCR 有字，但地图无此物品/无相似项 → 提示可能未标定。"""
        key = (raw or "").strip()
        if not key:
            return
        msg = (
            f"槽位 {slot_index + 1} OCR「{key}」未匹配地图（无相似项），"
            f"可能尚未标定"
        )
        if key not in self._unmatched_logged:
            self._unmatched_logged.add(key)
            logger.warning("[{}] {}", self.name, msg)
            self._emit(msg)
        else:
            logger.debug("[{}] {}（已提示过）", self.name, msg)

    def _interrupted(self) -> bool:
        return self._stop.is_set()

    def _map_keys(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys((*self.game_map.items.keys(), *self.game_map.aliases.keys()))
        )

    def _slots(self) -> tuple[tuple[int, int, int, int], ...]:
        return tuple(self.config.target_slots or ())

    def _slot_patch(self, screen: np.ndarray, slot_index: int) -> np.ndarray:
        slots = self._slots()
        if slot_index >= len(slots):
            return np.array([])
        x1, y1, x2, y2 = slots[slot_index]
        h, w = screen.shape[:2]
        return screen[max(0, y1) : min(h, y2), max(0, x1) : min(w, x2)]

    @staticmethod
    def _patch_mean(patch: np.ndarray) -> float:
        if patch.size == 0:
            return 0.0
        gray = patch.mean(axis=2) if patch.ndim == 3 else patch
        return float(gray.mean())

    def _slot_fingerprints(
        self, screen: np.ndarray, batch: list[_TapItem]
    ) -> dict[int, float]:
        return {
            item.slot_index: self._patch_mean(self._slot_patch(screen, item.slot_index))
            for item in batch
        }

    def _slot_hit(self, screen: np.ndarray, slot_index: int, before_mean: float) -> bool:
        """该槽是否已划掉或亮度明显变化（视为点击生效）。"""
        patch = self._slot_patch(screen, slot_index)
        if patch.size == 0:
            return False
        if not chip_is_active(
            patch,
            min_brightness=self.config.chip_active_min_brightness,
        ):
            return True
        delta = abs(self._patch_mean(patch) - before_mean)
        return delta >= float(self.config.bar_change_mean_delta)

    def _enqueue_taps(self, items: list[_TapItem]) -> None:
        """一批 tap 合并一次下发（单次 HTTP 往返）。"""
        queue_sleep = getattr(self.adb, "queue_sleep", None)
        for index, item in enumerate(items):
            if self._interrupted():
                return
            self.adb.tap(item.x, item.y)
            if index < len(items) - 1:
                gap = sample_tap_between_delay(self.config)
                if queue_sleep is not None:
                    queue_sleep(gap)
        self.adb.sleep(max(0.1, float(self.config.tap_delay)))

    def _wait_bar_refresh(
        self,
        batch: list[_TapItem],
        before_means: dict[int, float],
    ) -> None:
        """点完一批后等底栏刷新，避免下一轮 OCR 仍读到同一批。"""
        if not batch:
            return
        floor = max(0.12, float(self.config.bar_refresh_min_wait))
        time.sleep(floor)
        deadline = time.time() + float(self.config.bar_refresh_timeout)
        poll = max(0.08, float(self.config.bar_refresh_poll))
        while time.time() < deadline:
            if self._interrupted():
                return
            try:
                screen = self.adb.screenshot()
            except Exception:
                time.sleep(poll)
                continue
            cleared = 0
            for item in batch:
                if self._slot_hit(screen, item.slot_index, before_means[item.slot_index]):
                    cleared += 1
            if cleared > 0:
                logger.debug("底栏已刷新 ({}/{})", cleared, len(batch))
                return
            time.sleep(poll)
        logger.debug("等待底栏刷新超时，继续下一轮")

    def _click_batch(self, batch: list[_TapItem], before_means: dict[int, float]) -> None:
        """一批连点后等底栏刷新；不做点后截图校验/补点（云控往返太慢）。"""
        self._enqueue_taps(batch)
        self._wait_bar_refresh(batch, before_means)

    def run_once(self, *, force: bool = False) -> bool:
        _ = force
        if not self.game_map.items:
            raise ValueError(
                f"地图「{self.game_map.name}」尚无标定物品，请先在单机版标定后发布"
            )
        if not ocr_engine_available(self.config.ocr_engine):
            engine = resolve_ocr_engine(self.config.ocr_engine)
            raise FileNotFoundError(f"OCR 引擎不可用: {engine}")

        warmup_ocr(self.config.ocr_engine)
        engine = resolve_ocr_engine(self.config.ocr_engine)
        self._emit(
            f"开始 — 地图「{self.game_map.name}」"
            f"（第 {self.game_map.period} 期，{len(self.game_map.items)} 个物品，OCR={engine}）"
        )

        empty_rounds = 0
        while not self._interrupted():
            try:
                screen = self.adb.screenshot()
            except Exception as exc:
                if self._interrupted():
                    break
                self._emit(f"截图失败: {exc}")
                time.sleep(0.5)
                continue

            chips = read_target_chips(
                screen,
                self.config.target_slots or None,
                map_keys=self._map_keys(),
                map_aliases=self.game_map.aliases,
                target_bar=self.config.target_bar,
                min_slots=self.config.min_target_slots,
                max_slots=self.config.max_target_slots,
                ocr_engine=self.config.ocr_engine,
                min_brightness=self.config.chip_active_min_brightness,
                refs_dir=self.config.chip_refs_dir,
                template_min_score=self.config.chip_template_min_score,
                template_min_margin=self.config.chip_template_min_margin,
                fuzzy_min_ratio=self.config.chip_fuzzy_min_ratio,
                pk_mode=False,
            )
            batch: list[_TapItem] = []
            for chip in chips:
                if not chip.active:
                    continue
                raw = (chip.ocr_raw or chip.text or "").strip()
                if not chip.text:
                    if raw:
                        self._warn_unmatched_map(chip.slot_index, raw)
                    continue
                coord = resolve_item_coord(self.game_map, chip.text)
                if coord is None:
                    self._warn_unmatched_map(chip.slot_index, raw or chip.text)
                    continue
                batch.append(
                    _TapItem(
                        chip.slot_index,
                        chip.text,
                        int(coord[0]),
                        int(coord[1]),
                    )
                )

            if not batch:
                empty_rounds += 1
                scan_line = _format_scan_line(chips)
                if empty_rounds == 1 or empty_rounds % 20 == 0:
                    if scan_line and any(chip.active for chip in chips):
                        self._emit(f"{scan_line}，等待中…")
                    else:
                        self._emit("未识别到可点目标，等待中…")
                time.sleep(max(0.15, float(self.config.scan_interval)))
                continue

            empty_rounds = 0
            scan_line = _format_scan_line(chips)
            labels = "、".join(item.text for item in batch)
            if scan_line:
                self._emit(f"{scan_line} → 本批 {len(batch)} 个: {labels}")
            else:
                self._emit(f"本批 {len(batch)} 个: {labels}")

            before_means = self._slot_fingerprints(screen, batch)
            self._click_batch(batch, before_means)

        self._emit("已结束")
        return True


def build_dream_memory_task(
    adb: AdbClient,
    cfg: dict[str, Any],
    on_status: StatusCallback | None = None,
) -> DreamMemoryTask:
    map_id = str(cfg.get("selected_map") or cfg.get("map_id") or "").strip()
    period_raw = cfg.get("selected_period", cfg.get("period"))
    period = int(period_raw) if period_raw not in (None, "") else None
    config = _build_config(dict(cfg or {}), pk=False)
    return DreamMemoryTask(
        adb,
        map_id=map_id,
        period=period,
        config=config,
        on_status=on_status,
    )
