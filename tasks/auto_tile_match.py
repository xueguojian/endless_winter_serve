"""自动玩消消乐（三消进槽小游戏）。

四个模块（720×1280）：
  常规区 board_roi   — 堆叠消牌（识别可点 + 被挡库存）
  暂移区 temp_roi    — 「移出」后放置区前 3 张落到这里，仍要点回消除（最多 3）
  放置区 slot_roi    — 底部七格槽（最多 7；每步从画面识别，不靠本地瞎记）
  功能区             — 移出 / 撤回 / 重构造（绿钮=免费）

模板：assets/templates/tile_match/*.png
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Callable

from loguru import logger

from core.adb_client import AdbClient
from core.tile_match_vision import (
    TOOLS,
    TEMP_CAPACITY,
    _slot_danger,
    apply_remove_tool,
    apply_slot_click,
    choose_tile,
    detect_free_tools,
    has_clearable_within_layers,
    list_tile_templates,
    point_in_roi,
    read_slot_contents,
    scan_play_area,
)

StatusCallback = Callable[[str], None]

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_DIR = ROOT / "assets" / "templates" / "tile_match"

DEFAULT_BOARD_ROI = [8, 70, 708, 842]
DEFAULT_TEMP_ROI = [222, 842, 504, 948]
DEFAULT_SLOT_ROI = [4, 924, 712, 1122]
DEFAULT_SLOT_CAPACITY = 7
DEFAULT_STEP_DELAY = 1.4
DEFAULT_MATCH_THRESHOLD = 0.72
DEFAULT_BURIED_THRESHOLD = 0.52
DEFAULT_SLOT_THRESHOLD = 0.88
DEFAULT_FREE_THRESHOLD = 0.70
DEFAULT_MIN_DIST = 36
DEFAULT_MAX_STEPS = 200
DEFAULT_TOOL_DELAY = 2.0
DEFAULT_SEARCH_DEPTH = 5


def merge_task_config(cfg: dict) -> dict:
    return {
        "step_delay": float(cfg.get("step_delay", DEFAULT_STEP_DELAY)),
        "tool_delay": float(cfg.get("tool_delay", DEFAULT_TOOL_DELAY)),
        "match_threshold": float(cfg.get("match_threshold", DEFAULT_MATCH_THRESHOLD)),
        "buried_threshold": float(
            cfg.get("buried_threshold", DEFAULT_BURIED_THRESHOLD)
        ),
        "slot_threshold": float(cfg.get("slot_threshold", DEFAULT_SLOT_THRESHOLD)),
        "free_threshold": float(cfg.get("free_threshold", DEFAULT_FREE_THRESHOLD)),
        "min_dist": int(cfg.get("min_dist", DEFAULT_MIN_DIST)),
        "slot_capacity": int(cfg.get("slot_capacity", DEFAULT_SLOT_CAPACITY)),
        "max_steps": int(cfg.get("max_steps", DEFAULT_MAX_STEPS)),
        "search_depth": int(cfg.get("search_depth", DEFAULT_SEARCH_DEPTH)),
        "board_roi": list(cfg.get("board_roi") or DEFAULT_BOARD_ROI),
        "temp_roi": list(cfg.get("temp_roi") or DEFAULT_TEMP_ROI),
        "slot_roi": list(cfg.get("slot_roi") or DEFAULT_SLOT_ROI),
    }


class AutoTileMatchTask:
    """截图识别 → 落子；卡住时按优先级用免费道具：移出 > 重构造 > 撤回。"""

    # 小游戏退出逻辑与野外不同：结束后不要 return_to_main_screen
    skip_return_to_main = True

    def __init__(
        self,
        adb: AdbClient,
        *,
        step_delay: float = DEFAULT_STEP_DELAY,
        tool_delay: float = DEFAULT_TOOL_DELAY,
        match_threshold: float = DEFAULT_MATCH_THRESHOLD,
        buried_threshold: float = DEFAULT_BURIED_THRESHOLD,
        slot_threshold: float = DEFAULT_SLOT_THRESHOLD,
        free_threshold: float = DEFAULT_FREE_THRESHOLD,
        min_dist: int = DEFAULT_MIN_DIST,
        slot_capacity: int = DEFAULT_SLOT_CAPACITY,
        max_steps: int = DEFAULT_MAX_STEPS,
        search_depth: int = DEFAULT_SEARCH_DEPTH,
        board_roi: list[int] | None = None,
        temp_roi: list[int] | None = None,
        slot_roi: list[int] | None = None,
        template_dir: str | Path | None = None,
        on_status: StatusCallback | None = None,
    ):
        merged = merge_task_config(
            {
                "step_delay": step_delay,
                "tool_delay": tool_delay,
                "match_threshold": match_threshold,
                "buried_threshold": buried_threshold,
                "slot_threshold": slot_threshold,
                "free_threshold": free_threshold,
                "min_dist": min_dist,
                "slot_capacity": slot_capacity,
                "max_steps": max_steps,
                "search_depth": search_depth,
                "board_roi": board_roi or DEFAULT_BOARD_ROI,
                "temp_roi": temp_roi or DEFAULT_TEMP_ROI,
                "slot_roi": slot_roi or DEFAULT_SLOT_ROI,
            }
        )
        self.adb = adb
        self.step_delay = merged["step_delay"]
        self.tool_delay = merged["tool_delay"]
        self.match_threshold = merged["match_threshold"]
        self.buried_threshold = merged["buried_threshold"]
        self.slot_threshold = merged["slot_threshold"]
        self.free_threshold = merged["free_threshold"]
        self.min_dist = merged["min_dist"]
        self.slot_capacity = merged["slot_capacity"]
        self.max_steps = merged["max_steps"]
        self.search_depth = merged["search_depth"]
        self.board_roi = tuple(int(v) for v in merged["board_roi"])
        self.temp_roi = tuple(int(v) for v in merged["temp_roi"])
        self.slot_roi = tuple(int(v) for v in merged["slot_roi"])
        self.template_dir = Path(template_dir) if template_dir else TEMPLATE_DIR
        self.on_status = on_status
        self._stop_event = threading.Event()
        self._slot: list[str] = []
        self._history: list[list[str]] = []
        self._slot_vision_miss = 0
        self._tool_used: dict[str, bool] = {
            "remove": False,
            "undo": False,
            "shuffle": False,
        }

    @property
    def name(self) -> str:
        return "消消乐"

    def stop(self) -> None:
        self._stop_event.set()

    def reset_stop(self) -> None:
        self._stop_event.clear()

    def _emit(self, message: str) -> None:
        logger.info(f"[{self.name}] {message}")
        if self.on_status:
            self.on_status(message)

    def _check_stop(self) -> None:
        if self._stop_event.is_set():
            raise InterruptedError("任务已停止")

    def _tap_xy(self, x: int, y: int, delay: float | None = None) -> None:
        self._check_stop()
        self.adb.tap(int(x), int(y))
        time.sleep(delay if delay is not None else self.step_delay)

    def _tool_by_key(self, key: str):
        for tool in TOOLS:
            if tool.key == key:
                return tool
        raise KeyError(key)

    def _read_slot(self, screen) -> list[str]:
        return read_slot_contents(
            screen,
            self.template_dir,
            self.slot_roi,
            capacity=self.slot_capacity,
            threshold=self.slot_threshold,
        )

    def _scan(self, screen):
        return scan_play_area(
            screen,
            self.template_dir,
            self.board_roi,
            self.temp_roi,
            self.slot_roi,
            free_threshold=self.match_threshold,
            buried_threshold=self.buried_threshold,
            min_dist=self.min_dist,
        )

    def _sync_slot_from_screen(self, screen) -> list[str]:
        """画面识别 + 本地记账：画面读空时不立刻清空（避免每步当空槽乱开新种）。"""
        seen = self._read_slot(screen)
        if seen:
            if seen != self._slot:
                self._emit(f"放置区识别：{seen}（原记账 {self._slot}）")
            self._slot = list(seen)
            self._slot_vision_miss = 0
            return self._slot

        if self._slot:
            self._slot_vision_miss += 1
            if self._slot_vision_miss <= 8:
                if self._slot_vision_miss <= 2:
                    self._emit(f"放置区画面未识别，沿用记账 {self._slot}")
                return self._slot
            self._emit("放置区连续未识别，清空记账")
            self._slot = []
            self._slot_vision_miss = 0
        return self._slot

    def _try_use_tool(
        self,
        screen,
        *,
        tiles_count: int,
        proactive: bool = False,
        prefer_shuffle: bool = False,
    ) -> bool:
        """用免费道具。prefer_shuffle：1+2+3 凑不出三消时优先重构。"""
        free = detect_free_tools(
            screen, self.template_dir, threshold=self.free_threshold
        )
        danger = _slot_danger(self._slot, self.slot_capacity)
        slot_n = len(self._slot)

        # 0) 1+2+3 无三消 → 优先重构（不要往 L4+ 挖把槽塞爆）
        if (
            prefer_shuffle
            and not self._tool_used["shuffle"]
            and free.get("shuffle")
            and tiles_count > 0
        ):
            tool = self._tool_by_key("shuffle")
            self._emit(
                f"L1+L2+L3 无三消，使用免费「重构造」@ {tool.center}"
            )
            self._tap_xy(*tool.center, delay=self.tool_delay)
            self._tool_used["shuffle"] = True
            self._history.clear()
            return True

        need_remove = slot_n >= TEMP_CAPACITY and (
            not proactive or slot_n >= 4 or danger >= 80
        )
        if (
            need_remove
            and not self._tool_used["remove"]
            and free.get("remove")
        ):
            tool = self._tool_by_key("remove")
            self._emit(f"使用免费「移出」@ {tool.center}（放置区 {self._slot}）")
            self._tap_xy(*tool.center, delay=self.tool_delay)
            self._history.clear()
            self._tool_used["remove"] = True
            time.sleep(0.3)
            self._sync_slot_from_screen(self.adb.screenshot())
            self._emit(f"移出后放置区 {self._slot}")
            return True

        need_shuffle = tiles_count > 0 and (
            not proactive or slot_n >= 4 or danger >= 90
        )
        if (
            need_shuffle
            and not self._tool_used["shuffle"]
            and free.get("shuffle")
        ):
            tool = self._tool_by_key("shuffle")
            self._emit(f"使用免费「重构造」@ {tool.center}")
            self._tap_xy(*tool.center, delay=self.tool_delay)
            self._tool_used["shuffle"] = True
            self._history.clear()
            return True

        if (
            not self._tool_used["undo"]
            and free.get("undo")
            and self._history
            and slot_n >= self.slot_capacity - 2
        ):
            tool = self._tool_by_key("undo")
            self._emit(f"使用免费「撤回」@ {tool.center}（放置区 {self._slot}）")
            self._tap_xy(*tool.center, delay=self.tool_delay)
            self._tool_used["undo"] = True
            if self._history:
                self._history.pop()
            time.sleep(0.3)
            self._sync_slot_from_screen(self.adb.screenshot())
            return True

        return False

    def run_once(self, *, force: bool = False) -> bool:
        _ = force
        templates = list_tile_templates(self.template_dir)
        if not templates:
            self._emit(
                f"未找到模板，请先裁剪到目录：{self.template_dir} "
                f"（如 shu.png / snowflake.png）"
            )
            return False

        self._emit(
            f"开始消消乐（模板 {len(templates)} 种，深度 {self.search_depth}，"
            f"可点阈值 {self.match_threshold} / 遮挡阈值 {self.buried_threshold}）"
        )
        self._slot = []
        self._history = []
        self._slot_vision_miss = 0
        self._tool_used = {"remove": False, "undo": False, "shuffle": False}
        idle_rounds = 0

        for step in range(1, self.max_steps + 1):
            self._check_stop()
            screen = self.adb.screenshot()
            self._sync_slot_from_screen(screen)
            scan = self._scan(screen)
            tiles = scan.free_tiles
            buried = {k: v for k, v in scan.buried_counts.items() if v > 0}
            temp_n = sum(
                1
                for t in tiles
                if point_in_roi(t.center[0], t.center[1], self.temp_roi)
            )

            # 槽内未完成种类数（防乱开）
            from collections import Counter

            incomplete_n = sum(
                1 for c in Counter(self._slot).values() if 0 < c < 3
            )

            # L1+L2+L3（含槽）凑不出三消 → 优先重构（挖一层能凑的交给 choose_tile）
            need_shuffle = not has_clearable_within_layers(
                self._slot, scan.all_tiles, max_layer=3
            )
            if need_shuffle and self._try_use_tool(
                screen,
                tiles_count=len(tiles),
                prefer_shuffle=True,
            ):
                idle_rounds = 0
                continue

            if len(self._slot) >= 4 and self._try_use_tool(
                screen, tiles_count=len(tiles), proactive=True
            ):
                idle_rounds = 0
                continue

            choice = choose_tile(
                tiles,
                self._slot,
                board_counts=scan.board_counts,
                all_tiles=scan.all_tiles,
                max_layer=3,
                slot_capacity=self.slot_capacity,
                search_depth=self.search_depth,
                prefer_roi=self.temp_roi,
            )

            if choice is None:
                idle_rounds += 1
                reason = (
                    "L1+L2+L3无三消"
                    if need_shuffle
                    else "无好棋"
                )
                self._emit(
                    f"第 {step} 步{reason}（可点 {len(tiles)}，暂移 {temp_n}，"
                    f"未完成种类 {incomplete_n}，放置区 {self._slot}，"
                    f"被挡 {buried or '{}'}）"
                )
                if self._try_use_tool(
                    screen,
                    tiles_count=len(tiles),
                    proactive=False,
                    prefer_shuffle=need_shuffle,
                ):
                    idle_rounds = 0
                    continue
                # 仍无棋：仅在「层内仍可能三消」时硬点；无三消绝不硬挖更深
                if tiles and not need_shuffle:
                    tiles_sorted = sorted(
                        tiles, key=lambda t: t.confidence, reverse=True
                    )
                    choice = tiles_sorted[0]
                    self._emit(
                        f"无规划好棋，硬点可点牌 {choice.kind} @ {choice.center}"
                    )
                    idle_rounds = 0
                elif idle_rounds >= 3:
                    if not self._slot and not tiles:
                        self._emit("场上空净，本局结束")
                        return True
                    self._emit(
                        "无法在 L1–L3 推进且无重构可用，判定本局失败（等待重开）"
                    )
                    return False
                else:
                    time.sleep(self.step_delay)
                    continue

            idle_rounds = 0
            if len(self._slot) >= self.slot_capacity:
                if self._try_use_tool(screen, tiles_count=len(tiles)):
                    continue
                self._emit(f"放置区已满 {self._slot}，失败停止")
                return False

            kind = choice.kind
            # 信任 choose_tile 的渐进策略，不再二次拦截（拦截会造成死循环）

            cx, cy = choice.center
            where = (
                "暂移区"
                if point_in_roi(cx, cy, self.temp_roi)
                else "常规区"
            )
            self._history.append(list(self._slot))
            preview = apply_slot_click(self._slot, kind)
            self._emit(
                f"第 {step} 步点{where} {kind} @ ({cx},{cy}) "
                f"| 放置区 {self._slot} → {preview}"
            )
            self._tap_xy(cx, cy)
            # 先本地更新，再与画面合并（画面读空也不丢记账）
            self._slot = preview
            self._slot_vision_miss = 0
            screen2 = self.adb.screenshot()
            seen = self._read_slot(screen2)
            if seen:
                if seen != self._slot:
                    self._emit(f"落子后放置区校正：{seen}（记账 {self._slot}）")
                self._slot = list(seen)
            if len(self._history) > 30:
                self._history = self._history[-20:]

        self._emit(f"达到步数上限 {self.max_steps}，停止")
        return False
