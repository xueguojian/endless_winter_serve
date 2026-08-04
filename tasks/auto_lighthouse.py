"""自动灯塔任务：扫描地图图标并按类型执行。"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Callable

from loguru import logger

from core.adb_client import AdbClient
from core.common_task_opts import resolve_use_formation
from core.deploy_march import DeployMarchHelper, StaminaInsufficientError
from core.stamina_use import DEFAULT_STAMINA_CAN_LIMIT, StaminaCanLimitReached

from core.lighthouse_vision import (
    LIGHTHOUSE_SCAN_ROI,
    MISSION_DETAIL_ABSOLUTE_MIN,
    LighthouseMission,
    LighthouseScanResult,
    MissionDetailClassification,
    SKIP_MISSION_KINDS,
    classify_mission_detail_screen,
    configure_lighthouse_scan,
    is_lighthouse_intel_screen,
    mission_detail_action_ready,
    scan_mission_icons,
    tag_scanned_missions,
)
from core.navigation import (
    WildernessNavigator,
    is_in_town as nav_is_in_town,
    is_in_wilderness as nav_is_in_wilderness,
    is_on_main_map as nav_is_on_main_map,
)
from tasks.hunt_ice_beast import HuntIceBeastTask, MarchHeroCheckError
from core.vision import Vision

StatusCallback = Callable[[str], None]

TEMPLATE_DIR = Path(__file__).parent.parent / "assets" / "templates"

HERO_JOURNEY_WAIT_SEC = 5.0
MISSION_CONFIRM_PRE_WAIT_SEC = 1.0
MISSION_ICON_WAIT_SEC = 2.0
DEFAULT_INTERVAL = 3600.0
DEFAULT_STEP_DELAY = 1.5
DEFAULT_MONSTER_COOLDOWN = 120.0

MISSION_SCAN_SETTLE_SEC = 0.5
SCAN_EMPTY_RETRY_WAIT_SEC = 1.0
SCAN_REOPEN_WAIT_SEC = 1.2
# 详情页：先轮询底部按钮出现，再多帧 OCR 采样
DETAIL_PAGE_READY_MAX_WAIT_SEC = 4.5
DETAIL_PAGE_READY_POLL_SEC = 0.45
DETAIL_CLASSIFY_FIRST_WAIT_SEC = 0.6
DETAIL_CLASSIFY_RETRY_WAIT_SEC = 0.8
DETAIL_CLASSIFY_MAX_ATTEMPTS = 4
MISSION_DETAIL_VIEW_SETTLE_SEC = 2.0
MISSION_SLOT_MAX_ATTEMPTS = 3
MISSION_SLOT_TRACK_RADIUS = 28
MAX_LIGHTHOUSE_CYCLE_ITERATIONS = 100
MIN_PIN_CONFIDENCE = 0.20

DEFAULT_COORDS: dict[str, list[int]] = {
    "lighthouse_open": [666, 862],
    "hide_completed": [364, 1124],
    "view_immediately": [354, 908],
    "hero_journey_finish": [528, 1196],
    "march": [560, 1200],
    "dialog_cancel": [250, 780],
}


def merge_task_config(cfg: dict) -> dict:
    coords = {**DEFAULT_COORDS, **cfg.get("coords", {})}
    if "view_immediately" not in coords and "mission_confirm_1" in coords:
        coords["view_immediately"] = list(coords["mission_confirm_1"])
    return {
        "step_delay": cfg.get("step_delay", DEFAULT_STEP_DELAY),
        "coords": coords,
        "monster_cooldown": float(cfg.get("monster_cooldown", DEFAULT_MONSTER_COOLDOWN)),
        "event_period": bool(cfg.get("event_period", False)),
        "use_formation": resolve_use_formation(cfg),
    }


def _prefer_detail_classification(
    current: MissionDetailClassification,
    previous: MissionDetailClassification,
) -> bool:
    """多帧采样：低置信度可执行类型不能覆盖 unknown/跳过类结果。"""
    actionable = frozenset({"small_monster", "tent", "hero_journey"})

    def _actionable_reliable(detail: MissionDetailClassification) -> bool:
        return (
            detail.kind not in actionable
            or detail.confidence >= MISSION_DETAIL_ABSOLUTE_MIN
        )

    if current.kind in actionable and not _actionable_reliable(current):
        return False
    if previous.kind in actionable and not _actionable_reliable(previous):
        return True
    if current.kind in actionable and previous.kind not in actionable:
        return _actionable_reliable(current)
    if previous.kind in actionable and current.kind not in actionable:
        return False
    if current.kind == "beast_skip" and not current.beast_explicit:
        return False
    if previous.kind == "beast_skip" and not previous.beast_explicit:
        return True
    return current.confidence > previous.confidence


class AutoLighthouseTask:
    """野外 → 灯塔任务页 → 扫描并处理英雄之旅 / 帐篷 / 小怪。"""

    def __init__(
        self,
        adb: AdbClient,
        coords: dict[str, list[int]] | None = None,
        interval: float = DEFAULT_INTERVAL,
        formation_slot: int = 8,
        use_stamina: bool = True,
        stamina_can_limit: int = DEFAULT_STAMINA_CAN_LIMIT,
        use_formation: bool = True,
        step_delay: float = DEFAULT_STEP_DELAY,
        monster_cooldown: float = DEFAULT_MONSTER_COOLDOWN,
        event_period: bool = False,
        on_status: StatusCallback | None = None,
    ):
        merged = merge_task_config(
            {
                "coords": coords or {},
                "step_delay": step_delay,
                "monster_cooldown": monster_cooldown,
                "event_period": event_period,
                "use_formation": use_formation,
            }
        )
        self.adb = adb
        self.coords = merged["coords"]
        self.interval = interval
        self.formation_slot = int(formation_slot)
        self.use_stamina = use_stamina
        self.use_formation = merged["use_formation"]
        if self.formation_slot <= 0:
            self.use_formation = False
            self.formation_slot = 0
        self.step_delay = merged["step_delay"]
        self.monster_cooldown = merged["monster_cooldown"]
        self.event_period = merged["event_period"]
        self.on_status = on_status
        self._last_run = 0.0
        self._last_monster_dispatch_at = 0.0
        self._stop_event = threading.Event()
        self._slot_attempts: list[tuple[int, int, int]] = []
        self._skipped_centers: list[tuple[int, int]] = []
        self._monster_in_progress: list[tuple[int, int, float]] = []
        # 图钉坐标缓存：首次全图识别后沿用，处理约半数后再重扫
        self._pin_cache: list[LighthouseMission] = []
        self._pin_cache_origin_count: int = 0
        self._pin_actions_since_rescan: int = 0
        self.vision = Vision(TEMPLATE_DIR, threshold=0.70)
        self._deploy = DeployMarchHelper(
            adb,
            formation_slot=self.formation_slot,
            use_stamina=use_stamina,
            stamina_can_limit=stamina_can_limit,
            coords=self.coords,
            on_status=on_status,
            interrupted=self._interrupted,
            step_delay=self.step_delay,
        )
        self._wilderness = WildernessNavigator.from_task(
            self,
            is_lighthouse_intel=is_lighthouse_intel_screen,
            is_deploy_screen=self._deploy.is_deploy_screen,
        )
        configure_lighthouse_scan(event_period=self.event_period)

    @property
    def name(self) -> str:
        return "自动灯塔任务"

    def stop(self) -> None:
        self._stop_event.set()
        self._emit("正在停止…")

    def reset_stop(self) -> None:
        self._stop_event.clear()

    def _emit(self, message: str) -> None:
        logger.info(f"[{self.name}] {message}")
        if self.on_status:
            self.on_status(message)

    def _interrupted(self) -> bool:
        return self._stop_event.is_set()

    def should_run(self) -> bool:
        return time.time() - self._last_run >= self.interval

    def _tap_xy(self, x: int, y: int, delay: float | None = None) -> None:
        if self._interrupted():
            raise InterruptedError("任务已停止")
        self.adb.tap(x, y)
        time.sleep(delay if delay is not None else self.step_delay)

    def _tap(self, key: str, delay: float | None = None) -> None:
        x, y = self.coords[key]
        logger.debug(f"[{self.name}] 点击 {key} ({x}, {y})")
        self._tap_xy(x, y, delay)

    def _back(self, times: int = 1) -> None:
        for _ in range(times):
            if self._interrupted():
                raise InterruptedError("任务已停止")
            self.adb.back()
            time.sleep(0.5)

    def _is_in_wilderness(self, screen) -> bool:
        return nav_is_in_wilderness(self.vision, screen)

    def _is_in_town(self, screen) -> bool:
        return nav_is_in_town(self.vision, screen)

    def _ensure_clean_ui(self) -> None:
        self._emit("清理界面…")
        if self._wilderness.return_to_wilderness():
            self._emit("界面已就绪")
        else:
            self._emit("已尝试清理界面，继续执行")

    def _is_on_main_map(self, screen) -> bool:
        return nav_is_on_main_map(self.vision, screen)

    def _is_on_lighthouse_intel_page(self, screen) -> bool:
        """是否在灯塔情报页（区别于野外大地图）。"""
        if self._is_on_main_map(screen):
            return False
        if self._deploy.is_deploy_screen(screen):
            return False
        return is_lighthouse_intel_screen(screen)

    def _ensure_lighthouse_page(self) -> None:
        """确保当前在情报页；若在野外/出征界面则重新打开。"""
        for attempt in range(3):
            if self._interrupted():
                raise InterruptedError("任务已停止")
            screen = self.adb.screenshot()
            if self._is_on_lighthouse_intel_page(screen):
                return
            if self._deploy.is_deploy_screen(screen):
                self._emit("当前在出征界面，返回并重新打开情报页")
                self._wilderness.return_to_wilderness()
                continue
            if self._is_on_main_map(screen):
                self._emit("当前在野外大地图，重新打开情报页")
                self._open_lighthouse_page()
                return
            self._wilderness.return_to_wilderness()
        self._emit("未能确认情报页，尝试从野外重新打开")
        self._return_to_wilderness_main()
        self._open_lighthouse_page()

    def _ensure_wilderness(self) -> None:
        self._wilderness.ensure_wilderness()

    def _return_to_wilderness_main(self) -> None:
        self._emit("返回野外主界面…")
        if not self._wilderness.return_to_wilderness():
            self._emit("未能确认已回到野外，已尽力返回")

    def _open_lighthouse_page(self) -> None:
        self._ensure_wilderness()
        self._emit("打开灯塔任务页面")
        self._tap("lighthouse_open", delay=2.0)
        self._prepare_lighthouse_scan()

    def _prepare_lighthouse_scan(self) -> None:
        self._emit("隐藏已完成任务")
        self._tap("hide_completed", delay=1.0)
        self._tap("hide_completed", delay=1.0)

    def _monster_cooldown_remaining(self) -> float:
        if self._last_monster_dispatch_at <= 0:
            return 0.0
        elapsed = time.time() - self._last_monster_dispatch_at
        return max(0.0, self.monster_cooldown - elapsed)

    def _sleep_interruptible(self, seconds: float) -> None:
        deadline = time.time() + seconds
        while time.time() < deadline:
            if self._interrupted():
                raise InterruptedError("任务已停止")
            time.sleep(min(1.0, deadline - time.time()))

    def _format_missions_summary(self, missions: tuple[LighthouseMission, ...]) -> str:
        if not missions:
            return "无"
        return "，".join(
            f"{m.label}({m.center[0]},{m.center[1]},{m.confidence:.2f})"
            for m in missions
        )

    def _slot_attempt_count(self, center: tuple[int, int]) -> int:
        x, y = center
        for sx, sy, count in self._slot_attempts:
            if abs(x - sx) < MISSION_SLOT_TRACK_RADIUS and abs(y - sy) < MISSION_SLOT_TRACK_RADIUS:
                return count
        return 0

    def _is_slot_exhausted(self, center: tuple[int, int]) -> bool:
        return self._slot_attempt_count(center) >= MISSION_SLOT_MAX_ATTEMPTS

    def _record_slot_attempt(self, center: tuple[int, int]) -> int:
        """记录某坐标槽位的点击次数，返回当前次数。"""
        x, y = center
        for i, (sx, sy, count) in enumerate(self._slot_attempts):
            if abs(x - sx) < MISSION_SLOT_TRACK_RADIUS and abs(y - sy) < MISSION_SLOT_TRACK_RADIUS:
                count += 1
                self._slot_attempts[i] = (sx, sy, count)
                return count
        self._slot_attempts.append((x, y, 1))
        return 1

    def _purge_expired_monster_in_progress(self) -> None:
        now = time.time()
        self._monster_in_progress = [
            item for item in self._monster_in_progress if item[2] > now
        ]

    def _is_monster_in_progress(self, center: tuple[int, int]) -> bool:
        self._purge_expired_monster_in_progress()
        x, y = center
        for sx, sy, _until in self._monster_in_progress:
            if abs(x - sx) < MISSION_SLOT_TRACK_RADIUS and abs(y - sy) < MISSION_SLOT_TRACK_RADIUS:
                return True
        return False

    def _mark_monster_in_progress(self, center: tuple[int, int]) -> None:
        until = time.time() + self.monster_cooldown
        x, y = center
        for i, (sx, sy, _old_until) in enumerate(self._monster_in_progress):
            if abs(x - sx) < MISSION_SLOT_TRACK_RADIUS and abs(y - sy) < MISSION_SLOT_TRACK_RADIUS:
                self._monster_in_progress[i] = (sx, sy, until)
                return
        self._monster_in_progress.append((x, y, until))

    def _is_skipped_center(self, center: tuple[int, int]) -> bool:
        if self._is_monster_in_progress(center):
            return True
        x, y = center
        for sx, sy in self._skipped_centers:
            if abs(x - sx) < MISSION_SLOT_TRACK_RADIUS and abs(y - sy) < MISSION_SLOT_TRACK_RADIUS:
                return True
        return False

    def _mark_skipped_center(self, center: tuple[int, int]) -> None:
        if self._is_skipped_center(center):
            return
        self._skipped_centers.append(center)

    def _pick_executable_mission(
        self, missions: tuple[LighthouseMission, ...]
    ) -> LighthouseMission | None:
        """从扫描结果中取出下一个可点击的图标（跳过已标记不处理与已达尝试上限的坐标）。"""
        if not missions:
            return None
        for mission in missions:
            if mission.kind in SKIP_MISSION_KINDS:
                logger.debug(
                    f"[{self.name}] 跳过特殊大怪 "
                    f"({mission.center[0]},{mission.center[1]}) "
                    f"conf={mission.confidence:.2f}"
                )
                self._mark_skipped_center(mission.center)
                continue
            if self._is_skipped_center(mission.center):
                logger.debug(
                    f"[{self.name}] 跳过已标记不处理坐标 "
                    f"({mission.center[0]},{mission.center[1]})"
                )
                continue
            if self._is_slot_exhausted(mission.center):
                logger.debug(
                    f"[{self.name}] 跳过已达上限坐标 "
                    f"({mission.center[0]},{mission.center[1]}) "
                    f"{self._slot_attempt_count(mission.center)}/{MISSION_SLOT_MAX_ATTEMPTS}"
                )
                continue
            return mission
        return None

    def _only_skippable_missions_remain(
        self, missions: tuple[LighthouseMission, ...]
    ) -> bool:
        if not missions:
            return False
        return all(
            self._is_skipped_center(m.center) or self._is_slot_exhausted(m.center)
            for m in missions
        )

    def _scan_missions(self, screen):
        if self._interrupted():
            raise InterruptedError("任务已停止")
        result = scan_mission_icons(
            screen,
            interrupted=self._interrupted,
        )
        if self._interrupted():
            raise InterruptedError("任务已停止")
        return result

    def _capture_and_scan(self) -> tuple[object, object]:
        if self._interrupted():
            raise InterruptedError("任务已停止")
        screen = self.adb.screenshot()
        self._emit("扫描任务图标…")
        result = self._scan_missions(screen)
        if result.missions:
            missions = tag_scanned_missions(screen, result.missions)
            self._emit(
                f"扫描到 {len(missions)} 个图钉候选"
                + (
                    f"（最高置信度 {result.best_confidence:.2f}）"
                    if result.best_confidence > 0
                    else ""
                )
            )
            if any(m.kind in SKIP_MISSION_KINDS for m in missions):
                beast_count = sum(
                    1 for m in missions if m.kind in SKIP_MISSION_KINDS
                )
                self._emit(f"扫描识别到 {beast_count} 个特殊大怪，将自动跳过")
            result = type(result)(
                mission=next(
                    (m for m in missions if m.kind not in SKIP_MISSION_KINDS),
                    missions[0] if missions else None,
                ),
                missions=missions,
                best_confidence=result.best_confidence,
                best_label=result.best_label,
                candidate_locations=result.candidate_locations,
            )
            logger.debug(
                f"[{self.name}] 扫描结果: {self._format_missions_summary(missions)}"
            )
        return screen, result

    def _pin_rescan_threshold(self) -> int:
        """首次识别 N 个任务后，约处理 N/2 次再全图重扫。"""
        return max(1, int(self._pin_cache_origin_count) // 2)

    def _clear_pin_cache(self) -> None:
        self._pin_cache = []
        self._pin_cache_origin_count = 0
        self._pin_actions_since_rescan = 0

    def _should_full_rescan_pins(self) -> bool:
        if self._pin_cache_origin_count <= 0 or not self._pin_cache:
            return True
        return self._pin_actions_since_rescan >= self._pin_rescan_threshold()

    def _store_pin_cache(self, missions: tuple[LighthouseMission, ...] | list) -> None:
        self._pin_cache = list(missions)
        self._pin_cache_origin_count = len(self._pin_cache)
        self._pin_actions_since_rescan = 0
        if self._pin_cache_origin_count:
            self._emit(
                f"已缓存 {self._pin_cache_origin_count} 个图钉坐标，"
                f"约处理 {self._pin_rescan_threshold()} 次后重扫"
            )

    def _result_from_pin_cache(self) -> LighthouseScanResult:
        missions = tuple(self._pin_cache)
        best = max((m.confidence for m in missions), default=0.0)
        primary = self._pick_executable_mission(list(missions))
        if primary is None and missions:
            primary = missions[0]
        return LighthouseScanResult(
            mission=primary,
            missions=missions,
            best_confidence=float(best),
            best_label=primary.label if primary else "",
            candidate_locations=len(missions),
        )

    def _find_executable_mission(self) -> tuple[object | None, object, LighthouseMission | None]:
        """优先沿用图钉缓存；首次或达到半数动作后再全图识别。"""
        self._sleep_interruptible(MISSION_SCAN_SETTLE_SEC)

        if not self._should_full_rescan_pins():
            remain = self._pin_rescan_threshold() - self._pin_actions_since_rescan
            self._emit(
                f"沿用图钉缓存（{len(self._pin_cache)} 个，"
                f"{remain} 次动作后重扫）"
            )
            result = self._result_from_pin_cache()
            mission = self._pick_executable_mission(list(result.missions))
            return None, result, mission

        screen, result = self._capture_and_scan()
        if result.missions:
            self._store_pin_cache(result.missions)
        else:
            self._clear_pin_cache()

        mission = self._pick_executable_mission(result.missions)
        if mission is not None:
            return screen, result, mission

        if result.candidate_locations > 0:
            return screen, result, None

        self._emit(
            f"未发现任务图标，等待 {SCAN_EMPTY_RETRY_WAIT_SEC:.1f} 秒后重试…"
        )
        self._sleep_interruptible(SCAN_EMPTY_RETRY_WAIT_SEC)
        self._emit("刷新情报页筛选后重试…")
        self._prepare_lighthouse_scan()
        self._sleep_interruptible(0.8)
        screen, result = self._capture_and_scan()
        if result.missions:
            self._store_pin_cache(result.missions)
        mission = self._pick_executable_mission(result.missions)
        if mission is not None or result.candidate_locations > 0:
            return screen, result, mission

        self._emit("再次刷新情报页筛选后重试…")
        self._prepare_lighthouse_scan()
        self._sleep_interruptible(SCAN_REOPEN_WAIT_SEC)
        screen, result = self._capture_and_scan()
        if result.missions:
            self._store_pin_cache(result.missions)
        else:
            self._clear_pin_cache()
        mission = self._pick_executable_mission(result.missions)
        return screen, result, mission

    def _log_scan_miss(self, result, screen) -> None:
        if result.candidate_locations > 0:
            self._emit(
                f"发现 {result.candidate_locations} 个颜色图钉候选，"
                f"但轮廓过滤后未能确认可点击图标"
            )
        else:
            self._emit("未发现任何任务图标")

    def _handle_stamina_after_action(self) -> bool:
        """处理体力弹窗。返回 True 表示已自动使用体力。"""
        try:
            return self._deploy.handle_stamina_popup_if_any()
        except StaminaInsufficientError:
            raise

    def _before_next_mission(self) -> None:
        """执行下一个小任务前：野外 → 打开灯塔情报页。"""
        screen = self.adb.screenshot()
        if self._is_on_lighthouse_intel_page(screen):
            self._emit("已在情报页，准备扫描")
            self._prepare_lighthouse_scan()
            return
        self._emit("准备下一个小任务，打开情报页")
        self._open_lighthouse_page()

    def _after_mission_completed(self) -> None:
        """每个小任务结束后：回到野外主界面。"""
        self._emit("小任务完成，返回野外")
        self._return_to_wilderness_main()

    def _back_from_detail_page(self) -> None:
        """从任务详情 / 出征 / 误点界面安全退回野外主界面。"""
        self._return_to_wilderness_main()

    def _click_icon_and_view_detail(self, mission: LighthouseMission) -> bool:
        """点击地图图标 → 立即查看，进入任务详情页。返回是否成功进入详情。"""
        self._emit(f"点击图标 @ ({mission.center[0]},{mission.center[1]})")
        self._tap_xy(*mission.center, delay=MISSION_ICON_WAIT_SEC)
        time.sleep(MISSION_CONFIRM_PRE_WAIT_SEC)
        x, y = self.coords["view_immediately"]
        self._emit(f"点击立即查看 @ ({x},{y})")
        self._tap_xy(x, y, delay=MISSION_DETAIL_VIEW_SETTLE_SEC)
        screen = self.adb.screenshot()
        if self._is_on_lighthouse_intel_page(screen):
            self._emit("点击后仍在情报页，疑为误点空地")
            return False
        return True

    def _wait_mission_detail_ready(self) -> None:
        """轮询直到详情页底部按钮可识别，避免弹窗动画未结束就截图。"""
        deadline = time.time() + DETAIL_PAGE_READY_MAX_WAIT_SEC
        polls = 0
        while time.time() < deadline:
            if self._interrupted():
                raise InterruptedError("任务已停止")
            screen = self.adb.screenshot()
            if mission_detail_action_ready(screen):
                if polls > 0:
                    self._emit(
                        f"详情页已就绪（等待 {polls * DETAIL_PAGE_READY_POLL_SEC:.1f}s）"
                    )
                return
            polls += 1
            time.sleep(DETAIL_PAGE_READY_POLL_SEC)
        self._emit(
            f"详情页 {DETAIL_PAGE_READY_MAX_WAIT_SEC:.0f}s 内未检测到行动按钮，"
            f"继续尝试识别"
        )

    def _classify_mission_detail(self) -> MissionDetailClassification:
        """详情页底部行动按钮 + 副标题分类（多帧采样，避免 Toast 遮挡误判）。"""
        self._wait_mission_detail_ready()
        detail: MissionDetailClassification | None = None
        actionable = frozenset({"small_monster", "tent", "hero_journey"})
        for attempt in range(DETAIL_CLASSIFY_MAX_ATTEMPTS):
            wait = (
                DETAIL_CLASSIFY_FIRST_WAIT_SEC
                if attempt == 0
                else DETAIL_CLASSIFY_RETRY_WAIT_SEC
            )
            time.sleep(wait)
            screen = self.adb.screenshot()
            current = classify_mission_detail_screen(screen)
            if current.kind == "beast_skip" and current.beast_explicit:
                detail = current
                break
            if current.kind == "bounty_skip":
                detail = current
                break
            if detail is None or _prefer_detail_classification(current, detail):
                detail = current
            if (
                current.kind in actionable
                and current.confidence >= MISSION_DETAIL_ABSOLUTE_MIN
            ):
                break
        assert detail is not None
        if (
            detail.kind in actionable
            and detail.confidence < MISSION_DETAIL_ABSOLUTE_MIN
        ):
            detail = MissionDetailClassification(
                kind="unknown",
                label="未识别",
                confidence=detail.confidence,
            )
        self._emit(
            f"详情页识别为：{detail.label}（{detail.kind}，"
            f"置信度 {detail.confidence:.2f}）"
        )
        return detail

    def _check_march_heroes(self) -> None:
        """小怪出征前检查英雄栏（仅启用编队时）。"""
        if not self.use_formation:
            return
        # select_formation 已确认出征页，此处不再二次长等待
        time.sleep(0.5)
        self._emit("检查出征英雄…")
        screen = self.adb.screenshot()
        empty_slots = HuntIceBeastTask._find_empty_march_hero_slots(screen)
        if empty_slots:
            slots_text = "、".join(str(i) for i in empty_slots)
            raise MarchHeroCheckError(f"第 {slots_text} 个英雄位为空，跳过出征")
        self._emit("出征英雄已配满（3/3）")

    def _execute_detail_action(self, detail: MissionDetailClassification) -> None:
        """按详情页分类结果执行后续操作。"""
        if detail.kind == "tent":
            if detail.action_center:
                self._emit(f"点击营救 @ {detail.action_center}")
                self._tap_xy(*detail.action_center, delay=1.5)
            self._emit("帐篷任务完成")
            return

        if detail.kind == "hero_journey":
            if detail.action_center:
                self._emit(f"点击探险 @ {detail.action_center}")
                self._tap_xy(*detail.action_center, delay=1.5)
            self._tap("hero_journey_finish", delay=1.0)
            self._emit(f"等待 {int(HERO_JOURNEY_WAIT_SEC)} 秒…")
            time.sleep(HERO_JOURNEY_WAIT_SEC)
            self._handle_stamina_after_action()
            return

        if detail.kind == "small_monster":
            if detail.action_center:
                self._emit(f"点击出征 @ {detail.action_center}")
                self._tap_xy(*detail.action_center, delay=1.5)
            self._emit("执行小怪出征…")
            if self.use_formation:
                self._deploy.select_formation()
                self._check_march_heroes()
            else:
                self._emit("未启用编队，跳过编队选择与英雄校验")
                self._deploy.wait_for_deploy_screen()
            self._deploy.tap_march()
            return

        raise RuntimeError(f"无法执行的任务类型：{detail.kind}")

    def run_lighthouse_cycle(self) -> int:
        """扫描图标 → 点击 → 立即查看 → 详情页分类 → 处理 → 回野外。"""
        handled = 0
        self._slot_attempts.clear()
        self._skipped_centers.clear()
        self._clear_pin_cache()
        iteration = 0

        while not self._interrupted():
            iteration += 1
            if iteration > MAX_LIGHTHOUSE_CYCLE_ITERATIONS:
                self._emit(
                    f"已达到本轮上限 {MAX_LIGHTHOUSE_CYCLE_ITERATIONS} 次，"
                    f"强制结束以防死循环"
                )
                self._after_mission_completed()
                break

            self._before_next_mission()

            screen, result, mission = self._find_executable_mission()
            if self._interrupted():
                raise InterruptedError("任务已停止")

            if mission is None:
                if result.missions and self._only_skippable_missions_remain(
                    result.missions
                ):
                    skipped = sum(
                        1 for m in result.missions if self._is_skipped_center(m.center)
                    )
                    if skipped:
                        self._emit(
                            f"剩余 {skipped} 个已标记不处理的任务，结束本轮"
                        )
                    else:
                        self._emit(
                            f"剩余 {len(result.missions)} 个候选坐标均已尝试 "
                            f"{MISSION_SLOT_MAX_ATTEMPTS} 次，结束本轮"
                        )
                elif result.missions:
                    self._emit(
                        f"有 {len(result.missions)} 个图钉候选但均不可执行，"
                        f"结束本轮"
                    )
                else:
                    self._log_scan_miss(result, screen)
                    self._emit(
                        "页面上没有可处理的灯塔任务，扫描结束。"
                        "若肉眼可见图钉，请重裁 assets/templates/lighthouse/"
                        "lighthouse_map_bg.png（无任务时的情报页全屏截图）"
                    )
                self._after_mission_completed()
                break

            self._emit(
                f"发现任务图标 ({mission.center[0]},{mission.center[1]})，"
                f"共 {len(result.missions)} 个候选"
            )

            tap_target = mission
            # 无论成功失败，都计入「动作次数」，用于半数重扫
            self._pin_actions_since_rescan += 1

            if tap_target.confidence < MIN_PIN_CONFIDENCE:
                self._emit(
                    f"图钉置信度 {tap_target.confidence:.2f} 过低，"
                    f"跳过 ({tap_target.center[0]},{tap_target.center[1]})"
                )
                self._mark_skipped_center(tap_target.center)
                continue

            attempt_no = self._record_slot_attempt(tap_target.center)
            self._emit(
                f"坐标 ({tap_target.center[0]},{tap_target.center[1]}) "
                f"第 {attempt_no}/{MISSION_SLOT_MAX_ATTEMPTS} 次尝试"
            )

            if not self._click_icon_and_view_detail(tap_target):
                self._mark_skipped_center(tap_target.center)
                continue
            detail = self._classify_mission_detail()

            if detail.kind == "bounty_skip":
                self._emit("大师/宗师悬赏，跳过不打")
                self._mark_skipped_center(tap_target.center)
                self._back_from_detail_page()
                continue

            if (
                detail.kind == "beast_skip"
                and detail.beast_explicit
            ) or tap_target.kind in SKIP_MISSION_KINDS:
                self._emit("特殊大怪，跳过不打")
                self._mark_skipped_center(tap_target.center)
                self._back_from_detail_page()
                continue

            if detail.kind == "unknown":
                if self._is_slot_exhausted(tap_target.center):
                    self._emit(
                        f"详情页 {MISSION_SLOT_MAX_ATTEMPTS} 次仍未能识别，"
                        f"跳过此坐标"
                    )
                    self._mark_skipped_center(tap_target.center)
                else:
                    self._emit(
                        "详情页未能识别任务类型，"
                        f"稍后重试（{self._slot_attempt_count(tap_target.center)}"
                        f"/{MISSION_SLOT_MAX_ATTEMPTS}）"
                    )
                self._back_from_detail_page()
                continue

            if detail.kind == "small_monster":
                cooldown = self._monster_cooldown_remaining()
                if cooldown > 0:
                    mins, secs = divmod(int(cooldown), 60)
                    self._emit(
                        f"识别为灯塔小怪但冷却中（剩余 {mins} 分 {secs} 秒），"
                        f"跳过此图标"
                    )
                    self._back_from_detail_page()
                    continue

            try:
                self._execute_detail_action(detail)
            except MarchHeroCheckError as exc:
                self._emit(str(exc))
                self._back_from_detail_page()
                continue
            except (StaminaInsufficientError, StaminaCanLimitReached):
                raise
            except Exception as exc:
                self._emit(f"执行失败：{exc}")
                self._back_from_detail_page()
                continue

            if detail.kind == "small_monster":
                self._last_monster_dispatch_at = time.time()
                self._mark_monster_in_progress(tap_target.center)

            handled += 1
            self._after_mission_completed()

        return handled

    def run_once(self, *, force: bool = False) -> bool:
        _ = force
        configure_lighthouse_scan(event_period=self.event_period)
        self._last_run = time.time()
        try:
            count = self.run_lighthouse_cycle()
            self._return_to_wilderness_main()
            self._emit(f"本轮处理 {count} 个灯塔任务，扫描结束")
            return True
        except StaminaCanLimitReached as exc:
            self._emit(str(exc))
            self._stop_event.set()
            try:
                self._return_to_wilderness_main()
            except Exception as nav_exc:
                logger.warning(f"[{self.name}] 返回野外失败: {nav_exc}")
            raise InterruptedError(str(exc)) from exc
        except InterruptedError:
            self._emit("任务已停止")
            raise
        except StaminaInsufficientError as exc:
            self._emit(str(exc))
            self._return_to_wilderness_main()
            return False
        except Exception as exc:
            logger.exception(f"[{self.name}] 执行失败")
            self._emit(f"执行失败：{exc}")
            try:
                self._return_to_wilderness_main()
            except Exception as nav_exc:
                logger.warning(f"[{self.name}] 返回野外失败: {nav_exc}")
            return False
