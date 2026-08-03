"""启动云控服务端。

用法:
  .venv\\Scripts\\python.exe run_server.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import uvicorn

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from server.config import load_server_config


def main() -> None:
    cfg = load_server_config()
    host = str(cfg.get("host") or "0.0.0.0")
    port = int(cfg.get("port") or 8787)
    print(f"Endless Winter Serve  http://{host}:{port}")
    print(f"本机调试可把客户端地址写成  http://127.0.0.1:{port}")
    uvicorn.run("server.app:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
