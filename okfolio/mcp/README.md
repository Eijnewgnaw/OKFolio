# MCP 模块

职责：对数据处理和 AgentWiki 提供统一控制面。

- `server.py`：官方 FastMCP 协议适配层；
- `service.py`：路径约束、写门禁和业务服务；
- `job_runner.py`：长任务状态持久化。

MCP 不内置原始文档或知识结果。PDF、MinerU 输出、Article、Concept 和 Release
均位于运行时挂载卷。
