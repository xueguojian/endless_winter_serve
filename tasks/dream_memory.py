"""云控寻梦记忆：客户端连续上传截图，服务端 OCR 后下发点击。"""

from __future__ import annotations

import threading
import time
from typing import Any, Callable

from loguru import logger

from core.adb_client import AdbClient
from core.dream_memory.config import DreamMemoryConfig, _build_config
from core.dream_memory.maps import DreamMemoryMap, load_map
from core.dream_memory.ocr_engine import ocr_engine_available, resolve_ocr_engine, warmup_ocr
from core.dream_memory.vision import read_target_chips, resolve_item_coord

StatusCallback = Callable[[str], None]


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

    def stop(self) -> None:
        self._stop.set()

    def _emit(self, message: str) -> None:
        logger.info("[{}] {}", self.name, message)
        if self.on_status:
            self.on_status(message)

    def _interrupted(self) -> bool:
        return self._stop.is_set()

    def _map_keys(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys((*self.game_map.items.keys(), *self.game_map.aliases.keys()))
        )

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
            taps: list[tuple[str, int, int]] = []
            for chip in chips:
                if not chip.active or not chip.text:
                    continue
                coord = resolve_item_coord(self.game_map, chip.text)
                if coord is None:
                    continue
                taps.append((chip.text, int(coord[0]), int(coord[1])))

            if not taps:
                empty_rounds += 1
                if empty_rounds == 1 or empty_rounds % 20 == 0:
                    self._emit("未识别到可点目标，等待中…")
                time.sleep(max(0.15, float(self.config.scan_interval)))
                continue

            empty_rounds = 0
            labels = "、".join(text for text, _, _ in taps)
            self._emit(f"本批 {len(taps)} 个: {labels}")
            for text, x, y in taps:
                if self._interrupted():
                    break
                self.adb.tap(x, y)
                time.sleep(max(0.05, float(self.config.tap_between_delay)))
            time.sleep(max(0.1, float(self.config.tap_delay)))

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
    # 用 task_defaults + 客户端 override 拼出运行参数
    config = _build_config(dict(cfg or {}), pk=False)
    return DreamMemoryTask(
        adb,
        map_id=map_id,
        period=period,
        config=config,
        on_status=on_status,
    )
