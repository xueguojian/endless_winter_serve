"""SQLite 日志库：登录日志 / 操作日志。Python 自带 sqlite3，无需单独装库。"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any


_SCHEMA = """
CREATE TABLE IF NOT EXISTS login_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL DEFAULT '',
    device_id TEXT NOT NULL DEFAULT '',
    client_ip TEXT NOT NULL DEFAULT '',
    success INTEGER NOT NULL DEFAULT 0,
    message TEXT NOT NULL DEFAULT '',
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS op_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL DEFAULT '',
    device_id TEXT NOT NULL DEFAULT '',
    client_ip TEXT NOT NULL DEFAULT '',
    action TEXT NOT NULL DEFAULT '',
    detail TEXT NOT NULL DEFAULT '',
    created_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_login_logs_created ON login_logs(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_login_logs_user ON login_logs(username, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_op_logs_created ON op_logs(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_op_logs_user ON op_logs(username, created_at DESC);
"""


class LogStore:
    def __init__(self, db_path: str | Path):
        self.path = Path(db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.path), timeout=5.0, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _init_schema(self) -> None:
        with self._lock:
            conn = self._connect()
            try:
                conn.executescript(_SCHEMA)
                conn.commit()
            finally:
                conn.close()

    def add_login(
        self,
        *,
        username: str,
        device_id: str = "",
        client_ip: str = "",
        success: bool,
        message: str = "",
    ) -> None:
        now = time.time()
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    "INSERT INTO login_logs"
                    " (username, device_id, client_ip, success, message, created_at)"
                    " VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        username or "",
                        device_id or "",
                        client_ip or "",
                        1 if success else 0,
                        message or "",
                        now,
                    ),
                )
                conn.commit()
            finally:
                conn.close()

    def add_op(
        self,
        *,
        username: str,
        action: str,
        detail: Any = "",
        device_id: str = "",
        client_ip: str = "",
    ) -> None:
        if isinstance(detail, (dict, list)):
            detail_text = json.dumps(detail, ensure_ascii=False)
        else:
            detail_text = str(detail or "")
        now = time.time()
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    "INSERT INTO op_logs"
                    " (username, device_id, client_ip, action, detail, created_at)"
                    " VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        username or "",
                        device_id or "",
                        client_ip or "",
                        action or "",
                        detail_text,
                        now,
                    ),
                )
                conn.commit()
            finally:
                conn.close()

    def query_login_logs(
        self,
        *,
        username: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        limit = max(1, min(500, int(limit)))
        offset = max(0, int(offset))
        sql = (
            "SELECT id, username, device_id, client_ip, success, message, created_at"
            " FROM login_logs"
        )
        args: list[Any] = []
        if username:
            sql += " WHERE username = ?"
            args.append(username)
        sql += " ORDER BY id DESC LIMIT ? OFFSET ?"
        args.extend([limit, offset])
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(sql, args).fetchall()
                return [dict(row) for row in rows]
            finally:
                conn.close()

    def query_op_logs(
        self,
        *,
        username: str | None = None,
        action: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        limit = max(1, min(500, int(limit)))
        offset = max(0, int(offset))
        sql = (
            "SELECT id, username, device_id, client_ip, action, detail, created_at"
            " FROM op_logs"
        )
        wheres: list[str] = []
        args: list[Any] = []
        if username:
            wheres.append("username = ?")
            args.append(username)
        if action:
            wheres.append("action = ?")
            args.append(action)
        if wheres:
            sql += " WHERE " + " AND ".join(wheres)
        sql += " ORDER BY id DESC LIMIT ? OFFSET ?"
        args.extend([limit, offset])
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(sql, args).fetchall()
                return [dict(row) for row in rows]
            finally:
                conn.close()
