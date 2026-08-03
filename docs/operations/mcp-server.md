# OKFolio MCP Server

## 交付结论

MCP Server 不是只读查询壳，而是 OKFolio 的统一控制面。它覆盖：

```text
源文档/图片导入
  → 经典 A—D 增量编译或 Agent 全库编译
  → ConceptRef 跨文档分组
  → Concept 编译、质量评分、自动重编译
  → 关系判断
  → 运行审计和实验报告
  → 正式 OKF Bundle / 图谱 / Wiki 发布
  → 发布审计
  → linux/amd64 镜像和离线 Release 打包
  → Concept 检索及 ConceptRef / Article 溯源
```

语义决策仍由模型负责；路径安全、JSON 合同、Ref 全覆盖、资产字节、发布门禁、
后台任务和溯源查询由确定性 Python 代码负责。

## 公开发布边界

对外交付使用独立的纯能力镜像和白名单压缩包。交付物只包含处理代码、Prompt、
离线 Python 依赖和空运行目录，不包含：

- 任何原始 Article 文档或图片；
- 已生成的 ConceptRef、Concept、关系、图谱、站点或 Bundle；
- 私有实验输出、日志、模型密钥或部署凭据。

原始文档由使用方在运行时导入，所有结果仅写入挂载的 `runtime/`。打包程序还会
展开 Docker 镜像逐层扫描，并使用内部语料文件名作为泄漏指纹进行复核。独立部署
方式见 `docs/operations/mcp-capability-release.md`。

## 架构

```text
MCP Client
   │ stdio / Streamable HTTP
   ▼
kmpro_wiki/mcp/server.py                 FastMCP 协议与工具声明
   ▼
kmpro_wiki/mcp/
  service.py                  路径门禁、查询、流水线编排
   ├─ 直接读取正式 Bundle     快速查询
   ├─ 调用审计函数             确定性短任务
   └─ kmpro_wiki/mcp/job_runner.py        模型调用、发布、构建等长任务
          ▼
     原有 CLI 与核心模块
```

MCP 不复制编译算法。原有 CLI、Prompt、Schema 和编译模块仍是唯一实现，MCP
只提供结构化调用入口，因此命令行、Docker 与 MCP 不会形成三套行为。

## 能力清单

### 语料与运行

| 工具 | 作用 |
|---|---|
| `get_capabilities` | 返回流水线、版本和安全门禁 |
| `get_system_status` | 返回语料、Run、Release、Job 状态 |
| `list_sources` | 列出源文档、字节数与 SHA-256 |
| `ingest_markdown` | 导入 Markdown，可原子激活到 sources |
| `ingest_asset` | 以 base64 导入图片，保持相对路径 |
| `sync_inbox` | 同步 Markdown 和图片；PDF 仍标记 deferred |

### 编译、审计与发布

| 工具 | 作用 |
|---|---|
| `start_incremental_compile` | 启动经典 A—D 单文档增量编译 |
| `start_agent_compile` | 启动完整 Agent 全库编译 |
| `start_relation_judgement` | 判断跨 Concept 语义关系 |
| `audit_agent_run` | 验收 Ref 覆盖、来源、质量和资产 |
| `report_agent_run` | 生成 Token、耗时、路由和质量报告 |
| `publish_release` | 发布 Bundle、图谱、站点和溯源数据 |
| `audit_release` | 验收正式 Release 及校验清单 |
| `build_release_image` | 构建并可导出 linux/amd64 镜像 |
| `package_release` | 打包源码、结果、镜像及 SHA-256 清单 |

### 后台任务

| 工具 | 作用 |
|---|---|
| `list_jobs` | 列出最近任务 |
| `get_job` | 查询状态、退出码和日志尾部 |
| `get_job_log` | 读取更长的任务日志尾部 |

长任务状态保存在 `runtime/data/.mcp/jobs/`。MCP Server 重启不会丢失任务记录，模型密钥
不会写入 Job JSON 或日志元数据。

### 查询与溯源

| 工具 | 作用 |
|---|---|
| `search_concepts` | 检索标题、摘要、来源和正文 |
| `get_concept` | 返回 Concept frontmatter、正文和 Markdown |
| `trace_concept` | 展开 ConceptRef、Article 与关系证据 |
| `get_article` | 返回 Article 原文页面 |
| `get_graph` | 返回三维图谱文件信息与 SHA-256 |

静态资源与资源模板 URI：

```text
kmpro://status
kmpro://concept/{concept_id}
kmpro://article/{article_id}
kmpro://graph
```

工作流 Prompt：

```text
compile_corpus(run_id, release_name)
research_topic(topic)
```

## 启动方式

### 本机 stdio

适合 Codex、Claude Desktop 或其他能启动本地进程的 MCP Client：

```json
{
  "mcpServers": {
    "okfolio": {
      "command": "python3",
      "args": [
        "-m",
        "kmpro_wiki.mcp.server",
        "--transport",
        "stdio"
      ],
      "env": {
        "OKFOLIO_PROJECT_ROOT": "/absolute/path/okfolio",
        "PYTHONPATH": "/absolute/path/okfolio",
        "DATA_DIR": "/absolute/path/okfolio/runtime/data",
        "PROMPTS_DIR": "/absolute/path/okfolio/prompts",
        "RELEASES_DIR": "/absolute/path/okfolio/runtime/releases",
        "MCP_ENABLE_WRITES": "false"
      }
    }
  }
}
```

需要调用模型时，客户端进程还需注入：

```text
OPENAI_BASE_URL
OPENAI_API_KEY
OPENAI_MODEL
```

### Docker Streamable HTTP

```bash
docker compose up -d mcp
```

MCP endpoint：

```text
<scheme>://<deployment-host>:<published-port>/mcp
```

默认 `MCP_ENABLE_WRITES=false`。确认数据目录可写并需要执行编译时：

```bash
MCP_ENABLE_WRITES=true docker compose up -d mcp
```

HTTP 服务本身不内置共享密钥。只应监听本机或可信离线网段；跨主机开放时，应由
反向代理或网络 ACL 增加身份认证。不要直接暴露到互联网。

## 标准正式流程

1. `ingest_markdown` / `ingest_asset`，或把文件放入 inbox 后调用 `sync_inbox`。
2. `start_agent_compile`，保存返回的 `job_id`。
3. 用 `get_job` 轮询，只有 `status=complete` 才进入下一步。
4. 调用 `audit_agent_run` 和 `report_agent_run`。
5. `start_relation_judgement` 并轮询完成。
6. 再次调用 `audit_agent_run`。
7. `publish_release` 并轮询完成。
8. `audit_release`，只有 `status=pass` 才是正式成品。
9. 可选：`build_release_image`、`package_release`。

运行中、失败、未完成关系判断或未通过审计的目录都不能称为正式 Bundle。

## 写操作和 Docker 门禁

- 查询工具始终可用。
- 写工具要求 `MCP_ENABLE_WRITES=true`。
- Docker 工具还要求 `MCP_ENABLE_DOCKER=true`。
- HTTP Compose 默认不挂载 Docker socket，因此容器内不能控制宿主机 Docker。
- 镜像构建建议使用本机 stdio MCP，并显式开启 Docker 门禁。
- 覆盖既有 Release 时，除 `replace=true` 外，还必须让 `confirm_replace` 精确等于
  `release_name`。

这些门禁不会替代宿主机权限、目录只读挂载、网络 ACL 和容器隔离。

## 依赖边界

- 官方 MCP Python SDK 固定为 `mcp==1.28.1`。
- Python 与全部传递依赖已锁定在 `requirements.lock`。
- 公开 Dockerfile 按 `requirements.lock` 从 Python 包索引安装依赖。
- 离线环境可以自行建立 wheelhouse，但不应把内部依赖缓存提交到公共仓库。
- MCP 查询正式 Bundle 时不需要 LLM；只有编译和关系判断需要模型服务。
