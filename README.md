# 无尽冬日云控 · 服务端

负责任务识别与决策（巨兽 / 灯塔 / 采集）。客户端只负责截图和点击。

## 快速开始

```bat
setup.bat
run_server.bat
```

默认监听：`http://0.0.0.0:8787`

## 配置

- `config.yaml`：端口、账号密码、token 有效期
- `task_defaults.yaml`：任务默认坐标与参数（从原项目同步）

写死账号示例：

```yaml
users:
  - username: admin
    password: admin123
```

同一用户名 / 同一 `device_id` / 同一 IP 再次登录会挤掉旧会话。

## API 概要

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/login` | 登录拿 token |
| POST | `/api/logout` | 注销 |
| POST | `/api/task/start` | 启动任务 |
| POST | `/api/task/stop` | 停止任务 |
| POST | `/api/task/tick` | 上传截图/ACK，取回动作 |

## 上线

把服务器公网 IP/域名填进客户端 `server_url`，防火墙放行 `8787`（或改端口）。
