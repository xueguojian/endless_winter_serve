"""FastAPI 入口。"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, Header, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from loguru import logger
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from server.auth import AuthManager
from server.config import load_server_config, load_task_defaults
from server.db import LogStore
from server.session import SUPPORTED_TASKS, SessionHub

_cfg = load_server_config()
auth = AuthManager(
    users=list(_cfg.get("users") or []),
    token_ttl_hours=float(_cfg.get("token_ttl_hours") or 24),
)
hub = SessionHub()

_db_path = Path(str(_cfg.get("db_path") or "data/app.db"))
if not _db_path.is_absolute():
    _db_path = ROOT / _db_path
logs = LogStore(_db_path)

_admin_users = {
    str(name).strip()
    for name in (_cfg.get("admin_users") or ["admin"])
    if str(name).strip()
}

app = FastAPI(title="Endless Winter Serve", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_ADMIN_HTML = Path(__file__).resolve().parent / "static" / "admin.html"


def _require_session(authorization: str | None):
    try:
        return auth.require(_token_from_header(authorization))
    except PermissionError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc


def _require_admin(authorization: str | None):
    info = _require_session(authorization)
    if info.username not in _admin_users:
        raise HTTPException(status_code=403, detail="需要管理员权限")
    return info


class LoginBody(BaseModel):
    username: str
    password: str
    device_id: str = Field(default="unknown-device")


class StartTaskBody(BaseModel):
    task_id: str
    config: dict[str, Any] = Field(default_factory=dict)


def _token_from_header(authorization: str | None) -> str | None:
    if not authorization:
        return None
    text = authorization.strip()
    if text.lower().startswith("bearer "):
        return text[7:].strip()
    return text


def _client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    if request.client:
        return request.client.host
    return "0.0.0.0"


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {"ok": True, "tasks": list(SUPPORTED_TASKS)}


@app.get("/admin")
def admin_page() -> FileResponse:
    if not _ADMIN_HTML.is_file():
        raise HTTPException(status_code=404, detail="管理页不存在")
    return FileResponse(_ADMIN_HTML, media_type="text/html; charset=utf-8")


@app.get("/api/task_defaults")
def task_defaults(authorization: str | None = Header(default=None)) -> dict[str, Any]:
    _require_session(authorization)
    return load_task_defaults()


@app.post("/api/login")
def login(body: LoginBody, request: Request) -> dict[str, Any]:
    ip = _client_ip(request)
    try:
        info = auth.login(body.username, body.password, body.device_id, ip)
    except PermissionError as exc:
        logs.add_login(
            username=body.username,
            device_id=body.device_id,
            client_ip=ip,
            success=False,
            message=str(exc),
        )
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=401)
    hub.drop_conflicts(username=info.username, device_id=info.device_id)
    hub.bind(info.token, info.username, info.device_id)
    logs.add_login(
        username=info.username,
        device_id=info.device_id,
        client_ip=ip,
        success=True,
        message="login ok",
    )
    logs.add_op(
        username=info.username,
        device_id=info.device_id,
        client_ip=ip,
        action="login",
        detail={"device_id": info.device_id},
    )
    logger.info(
        "登录成功 user={} device={} ip={}",
        info.username,
        info.device_id,
        ip,
    )
    return {
        "ok": True,
        "token": info.token,
        "username": info.username,
        "device_id": info.device_id,
        "tasks": list(SUPPORTED_TASKS),
        "is_admin": info.username in _admin_users,
    }


@app.post("/api/logout")
def logout(
    request: Request, authorization: str | None = Header(default=None)
) -> dict[str, Any]:
    token = _token_from_header(authorization)
    ip = _client_ip(request)
    if token:
        try:
            info = auth.require(token)
            logs.add_op(
                username=info.username,
                device_id=info.device_id,
                client_ip=ip,
                action="logout",
                detail="",
            )
        except PermissionError:
            info = None
        hub.unbind(token)
        auth.logout(token)
    return {"ok": True}


@app.post("/api/task/start")
def task_start(
    body: StartTaskBody,
    request: Request,
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    sess_info = _require_session(authorization)
    session = hub.get(sess_info.token)
    if session is None:
        session = hub.bind(sess_info.token, sess_info.username, sess_info.device_id)
    try:
        session.start_task(body.task_id, body.config)
    except Exception as exc:
        logs.add_op(
            username=sess_info.username,
            device_id=sess_info.device_id,
            client_ip=_client_ip(request),
            action="task_start_fail",
            detail={"task_id": body.task_id, "error": str(exc)},
        )
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)
    logs.add_op(
        username=sess_info.username,
        device_id=sess_info.device_id,
        client_ip=_client_ip(request),
        action="task_start",
        detail={"task_id": body.task_id},
    )
    return {"ok": True, "task_id": body.task_id}


@app.post("/api/task/stop")
def task_stop(
    request: Request, authorization: str | None = Header(default=None)
) -> dict[str, Any]:
    sess_info = _require_session(authorization)
    session = hub.get(sess_info.token)
    if session is not None:
        session.stop_task()
    logs.add_op(
        username=sess_info.username,
        device_id=sess_info.device_id,
        client_ip=_client_ip(request),
        action="task_stop",
        detail="",
    )
    return {"ok": True}


@app.post("/api/task/tick")
async def task_tick(
    authorization: str | None = Header(default=None),
    screenshot: UploadFile | None = File(default=None),
    has_frame: str = Form(default="0"),
) -> dict[str, Any]:
    sess_info = _require_session(authorization)
    session = hub.get(sess_info.token)
    if session is None:
        return JSONResponse({"ok": False, "error": "会话不存在"}, status_code=401)
    jpeg: bytes | None = None
    if has_frame in {"1", "true", "True"} and screenshot is not None:
        jpeg = await screenshot.read()
        if not jpeg:
            jpeg = None
    try:
        result = session.tick(jpeg)
    except Exception as exc:
        logger.exception("tick 失败")
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)
    return {"ok": True, **result}


@app.get("/api/admin/login_logs")
def admin_login_logs(
    authorization: str | None = Header(default=None),
    username: str | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> dict[str, Any]:
    _require_admin(authorization)
    items = logs.query_login_logs(username=username or None, limit=limit, offset=offset)
    return {"ok": True, "items": items}


@app.get("/api/admin/op_logs")
def admin_op_logs(
    authorization: str | None = Header(default=None),
    username: str | None = Query(default=None),
    action: str | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> dict[str, Any]:
    _require_admin(authorization)
    items = logs.query_op_logs(
        username=username or None,
        action=action or None,
        limit=limit,
        offset=offset,
    )
    return {"ok": True, "items": items}


@app.get("/api/admin/sessions")
def admin_sessions(authorization: str | None = Header(default=None)) -> dict[str, Any]:
    _require_admin(authorization)
    return {"ok": True, "items": auth.dump_sessions()}


def _dream_map_payload(dream_map: Any, *, base: str) -> dict[str, Any]:
    from core.dream_memory.maps import find_map_preview

    preview = find_map_preview(dream_map.map_id)
    preview_url = (
        f"{base}/api/dream_memory/preview/{dream_map.map_id}"
        if preview is not None
        else None
    )
    return {
        "map_id": dream_map.map_id,
        "name": dream_map.name,
        "period": int(dream_map.period),
        "item_count": len(dream_map.items),
        "preview_url": preview_url,
    }


@app.get("/api/dream_memory/catalog")
def dream_memory_catalog(
    authorization: str | None = Header(default=None),
    period: int | None = Query(default=None),
    request: Request = None,  # type: ignore[assignment]
) -> dict[str, Any]:
    """寻梦地图目录（期数 + 地图列表），供云控客户端下拉。"""
    _require_session(authorization)
    from core.dream_memory.config import CURRENT_MAP_PERIOD
    from core.dream_memory.maps import format_period_choice, list_maps

    base = str(request.base_url).rstrip("/") if request is not None else ""
    # 云控可视/选图只开放当前活动期（第 7 期）
    periods = [CURRENT_MAP_PERIOD]
    # 若磁盘上尚无第 7 期 yaml，仍返回当期空列表，避免误选旧期
    query_period = int(period) if period is not None else CURRENT_MAP_PERIOD
    if query_period != CURRENT_MAP_PERIOD:
        query_period = CURRENT_MAP_PERIOD
    maps = list_maps(period=query_period)
    return {
        "ok": True,
        "current_period": CURRENT_MAP_PERIOD,
        "default_period": CURRENT_MAP_PERIOD,
        "periods": [
            {"period": p, "label": format_period_choice(p)} for p in periods
        ],
        "maps": [_dream_map_payload(m, base=base) for m in maps],
    }


@app.get("/api/dream_memory/preview/{map_id}")
def dream_memory_preview(map_id: str) -> FileResponse:
    from core.dream_memory.maps import find_map_preview

    path = find_map_preview(map_id)
    if path is None or not path.is_file():
        raise HTTPException(status_code=404, detail="预览图不存在")
    return FileResponse(path, media_type="image/png")


@app.get("/dream_memory/maps", response_class=HTMLResponse)
def dream_memory_maps_page(
    period: int | None = Query(default=None),
) -> HTMLResponse:
    """给用户看的地图一览页（名称+预览图），云控里用链接打开。只展示第 7 期。"""
    from core.dream_memory.config import CURRENT_MAP_PERIOD
    from core.dream_memory.maps import format_period_choice, list_maps

    # 可视链接固定只查当前活动期，忽略其它 period 参数
    show_period = CURRENT_MAP_PERIOD
    _ = period
    maps = list_maps(period=show_period)
    cards: list[str] = []
    for m in maps:
        from core.dream_memory.maps import find_map_preview

        preview = find_map_preview(m.map_id)
        img = (
            f'<img src="/api/dream_memory/preview/{m.map_id}" '
            f'alt="{m.name}" style="width:100%;max-width:280px;border-radius:8px;" />'
            if preview is not None
            else '<div style="height:160px;background:#eee;border-radius:8px;'
            'display:flex;align-items:center;justify-content:center;color:#888;">无预览</div>'
        )
        cards.append(
            f'<div style="border:1px solid #ddd;border-radius:10px;padding:12px;'
            f'background:#fff;width:300px;">'
            f"{img}"
            f'<div style="margin-top:8px;font-size:16px;font-weight:600;">{m.name}</div>'
            f'<div style="color:#666;font-size:13px;">id: {m.map_id} · '
            f"物品 {len(m.items)} · {format_period_choice(m.period)}</div>"
            f"</div>"
        )
    body = "\n".join(cards) if cards else "<p>第 7 期暂无地图（请先在单机版标定后发布到云控）</p>"
    html = f"""<!doctype html>
<html lang="zh-CN"><head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>寻梦地图一览 · 第 7 期</title>
<style>
body{{font-family:sans-serif;background:#f6f7fb;margin:0;padding:20px;color:#222;}}
h1{{margin:0 0 8px;}}
.grid{{display:flex;flex-wrap:wrap;gap:16px;}}
.tip{{color:#666;margin-bottom:16px;}}
.badge{{display:inline-block;background:#1a5fb4;color:#fff;padding:2px 10px;
border-radius:999px;font-size:13px;margin-bottom:12px;}}
</style></head><body>
<h1>寻梦记忆 · 地图一览</h1>
<div class="badge">{format_period_choice(show_period)}</div>
<p class="tip">本页仅展示第 7 期地图。请在云控客户端选择对应地图 id，进入关卡后再点开始。</p>
<div class="grid">{body}</div>
</body></html>"""
    return HTMLResponse(html)
