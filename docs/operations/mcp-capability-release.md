# OKFolio MCP 处理能力包

## 发布边界

本交付物只包含文档处理能力，不包含任何用户语料或知识结果：

- 不包含 `data/`、`sources/`、`wiki/`、`outputs/` 或 `artifacts/`；
- 不包含 Article 原文、Concept、ConceptRef、图谱、图片或实验日志；
- 不包含模型 API Key、SSH 凭据或服务器配置；
- Docker 镜像中的 `/app/runtime/data` 与 `/app/runtime/releases` 初始为空；
- 原始文档只能由使用方在运行时通过挂载目录或 `ingest_markdown` 工具提供。

## 内含能力

处理能力包括文档导入、经典 A—D 增量编译、Agent 全库编译、跨文档联合编译、
质量审计与自动重编译、关系判断、正式 Bundle 发布、图谱和站点生成、结果审计、
知识检索及 Concept → ConceptRef → Article 溯源。

耗时操作以后台 Job 执行；查询默认开放，写操作受
`MCP_ENABLE_WRITES=true` 门禁。

## 离线启动

```bash
sha256sum -c images/okfolio-mcp-capability-amd64.tar.sha256
docker load -i images/okfolio-mcp-capability-amd64.tar
docker compose up -d
```

MCP endpoint：

```text
http://host.example:3099/mcp
```

初次启动时 `runtime/data/` 为空。需要执行处理任务时，在 `.env` 中配置模型，
并显式开启写能力：

```text
LLM_API_BASE=...
LLM_API_KEY=...
LLM_MODEL=...
MCP_ENABLE_WRITES=true
```

随后可以通过 MCP：

1. `ingest_markdown` / `ingest_asset` 导入运行时语料；
2. `start_agent_compile` 启动编译；
3. `get_job` 轮询任务；
4. `audit_agent_run` 与 `report_agent_run` 验收；
5. `start_relation_judgement` 判断关系；
6. `publish_release` 生成使用方自己的 Bundle；
7. `audit_release` 完成最终门禁。

所有运行结果只写入挂载的 `runtime/`，不会回写镜像。
