"""登录 / Token / 单设备单会话。"""

from __future__ import annotations

import secrets
import threading
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SessionInfo:
    token: str
    username: str
    device_id: str
    client_ip: str
    created_at: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)


class AuthManager:
    """同一 username 同时只允许一个有效会话（重新登录会挤掉自己旧会话）。"""

    def __init__(self, users: list[dict[str, str]], token_ttl_hours: float = 24.0):
        self._users = {
            str(item["username"]): str(item["password"])
            for item in users
            if item.get("username")
        }
        self._ttl = max(1.0, float(token_ttl_hours)) * 3600.0
        self._lock = threading.RLock()
        self._sessions: dict[str, SessionInfo] = {}

    def verify_password(self, username: str, password: str) -> bool:
        return self._users.get(username) == password

    def login(self, username: str, password: str, device_id: str, client_ip: str) -> SessionInfo:
        if not self.verify_password(username, password):
            raise PermissionError("用户名或密码错误")
        device_id = (device_id or "").strip() or "unknown-device"
        client_ip = (client_ip or "").strip() or "0.0.0.0"
        with self._lock:
            self._purge_expired_locked()
            # 只挤掉「同一账号」的旧会话，不同用户可并存
            victims = [
                token
                for token, sess in self._sessions.items()
                if sess.username == username
            ]
            for token in victims:
                self._sessions.pop(token, None)
            token = secrets.token_urlsafe(32)
            info = SessionInfo(
                token=token,
                username=username,
                device_id=device_id,
                client_ip=client_ip,
            )
            self._sessions[token] = info
            return info

    def logout(self, token: str) -> None:
        with self._lock:
            self._sessions.pop(token, None)

    def require(self, token: str | None) -> SessionInfo:
        if not token:
            raise PermissionError("未登录")
        with self._lock:
            self._purge_expired_locked()
            info = self._sessions.get(token)
            if info is None:
                raise PermissionError("登录已失效，请重新登录")
            info.last_seen = time.time()
            return info

    def kick_token(self, token: str) -> None:
        self.logout(token)

    def _purge_expired_locked(self) -> None:
        now = time.time()
        expired = [
            token
            for token, sess in self._sessions.items()
            if now - sess.last_seen > self._ttl
        ]
        for token in expired:
            self._sessions.pop(token, None)

    def list_users(self) -> list[str]:
        return sorted(self._users.keys())

    def dump_sessions(self) -> list[dict[str, Any]]:
        with self._lock:
            self._purge_expired_locked()
            return [
                {
                    "username": s.username,
                    "device_id": s.device_id,
                    "client_ip": s.client_ip,
                    "last_seen": s.last_seen,
                }
                for s in self._sessions.values()
            ]
