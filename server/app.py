"""FastAPI 入口。"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from server.auth import AuthManager
from server.config import load_server_config, load_task_defaults
from server.session import SUPPORTED_TASKS, SessionHub

_cfg = load_server_config()
auth = AuthManager(
    users=list(_cfg.get("users") or []),
    token_ttl_hours=float(_cfg.get("token_ttl_hours") or 24),
)
hub = SessionHub()

app = FastAPI(title="Endless Winter Serve", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _require_session(authorization: str | None):
    try:
        return auth.require(_token_from_header(authorization))
    except PermissionError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc


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
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=401)
    hub.drop_conflicts(username=info.username, device_id=info.device_id)
    hub.bind(info.token, info.username, info.device_id)
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
    }


@app.post("/api/logout")
def logout(authorization: str | None = Header(default=None)) -> dict[str, Any]:
    token = _token_from_header(authorization)
    if token:
        hub.unbind(token)
        auth.logout(token)
    return {"ok": True}


@app.post("/api/task/start")
def task_start(
    body: StartTaskBody, authorization: str | None = Header(default=None)
) -> dict[str, Any]:
    sess_info = _require_session(authorization)
    session = hub.get(sess_info.token)
    if session is None:
        session = hub.bind(sess_info.token, sess_info.username, sess_info.device_id)
    try:
        session.start_task(body.task_id, body.config)
    except Exception as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)
    return {"ok": True, "task_id": body.task_id}


@app.post("/api/task/stop")
def task_stop(authorization: str | None = Header(default=None)) -> dict[str, Any]:
    sess_info = _require_session(authorization)
    session = hub.get(sess_info.token)
    if session is not None:
        session.stop_task()
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
