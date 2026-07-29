# OKFolio

OKFolio 是一套面向研究报告和智库文档的可追溯知识编译系统。它把 PDF 或
Markdown 转换为 ConceptRef、跨文档 Concept、OKF Bundle、关系图谱和静态
LLM Wiki，并通过 MCP 为 RAG 与 Agent Memory 提供统一调用接口。

> **OKF-native LLM Wiki compiler for traceable RAG and Agent Memory.**

当前开源版本：`0.1.0`

> 本仓库只包含处理能力、提示词、测试和公开演示，不包含私有服务器地址、
> 凭据、部署配置、模型权重或未授权语料。

## 能力概览

```text
PDF / Markdown
      │
      ▼
Data Processing
Document IR → Structure → Segment → Article
      │
      ▼
AgentWiki
ConceptRef → Candidate Graph → Concept → Relation → Bundle
      │
      ├── Static Wiki / 3D Graph
      └── MCP
```

- 一篇 Article 可以产生多个可独立引用的 ConceptRef。
- 多篇 Article 的 Ref 可以联合编译成跨文档 Concept。
- Concept 可追溯至 Ref、Article、章节路径、页码和证据块。
- Agent 可决定结构化切分、语义发现、Ref refine、联合编译和质量重编译。
- Python 合同负责 Schema、路径、证据覆盖、资产守恒、原子发布与恢复。
- 发布物包括 OKF 风格 Markdown、三维关系图谱和静态站点。

## 项目结构

```text
okfolio/
├── kmpro_wiki/
│   ├── data_processing/       # PDF → Document IR → Article
│   ├── agentwiki/             # Article → Ref → Concept → Bundle / Graph
│   └── mcp/                   # MCP 协议、任务编排和查询
├── prompts/                   # Agent 与编译阶段的职责合同
├── scripts/                   # 命令行入口
├── tests/                     # 单元、集成和发布验收
├── docs/                      # 架构、运维和公开调研
├── demo/                      # 已脱除基础设施信息的静态演示
├── Dockerfile
└── docker-compose.yml
```

原始文档、解析缓存、模型密钥和运行结果只写入 Git 忽略的 `runtime/`。

## 快速开始

### Python

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.lock
cp .env.example .env
```

在 `.env` 中配置自己的 OpenAI-compatible 模型服务。不要提交 `.env`。

```bash
PYTHONPATH=. python3 -m kmpro_wiki.mcp.server \
  --transport streamable-http \
  --host 127.0.0.1 \
  --port 3099
```

### Docker

```bash
docker buildx build \
  --platform linux/amd64 \
  --target service \
  --load \
  -t okfolio:0.1.0 .

docker compose up -d mcp
```

默认 MCP 地址为 `http://127.0.0.1:3099/mcp`。公开 Dockerfile 从锁定依赖清单
安装包；不包含内部离线 wheel 或任何模型配置。

## PDF 处理

把 PDF 放入 `runtime/data/inbox/`，配置 MinerU-compatible 服务后运行：

```bash
docker compose --profile pdf run --rm pdf-corpus \
  --input-dir /app/runtime/data/inbox \
  --data-dir /app/runtime/data
```

如果已有 MinerU 结果，可以只执行确定性的结构归一化：

```bash
PYTHONPATH=. python3 scripts/normalize_pdf_corpus.py \
  --data-dir runtime/data
```

只有结构门禁为 `complete` 的 Article 才会进入 AgentWiki。

## AgentWiki

```bash
docker compose --profile agent run --rm agentwiki \
  --run-id example-run
```

正式链路包括：

1. Article 结构与溯源审计；
2. ConceptRef discovery/refine；
3. 跨文档候选召回与联合分组；
4. Concept 编译、质量审计和必要时重编译；
5. 跨 Concept 关系判断；
6. Bundle、Graph、Site 发布；
7. 确定性发布验收和校验和检查。

## 公开演示

[打开静态演示](demo/site/index.html)，或直接打开
[三维知识图谱](demo/site/graph.html)。

演示数据用于验证交互和溯源能力，不代表全量基准实验。公开演示由以下命令从
一个已验收 Release 生成：

```bash
PYTHONPATH=. python3 scripts/build_public_demo.py \
  /path/to/private-release \
  demo \
  --replace
```

构建器只复制静态站点，把所有私网资源 URL 替换为本地占位资产，隐藏模型服务
标识，并再次扫描私网 IP 和凭据形式。它不会修改源 Release。

## 测试

```bash
PYTHONPATH=. python3 -m pytest -q
PYTHONPATH=. python3 scripts/audit_open_source.py
docker compose -f docker-compose.yml -f docker-compose.test.yml config
```

完整验收区分：

- 数据完整：页数、Block、Segment 和资产守恒；
- 结构通过：目录、标题、页面角色和证据映射通过；
- 知识完成：Ref、Concept、关系、Bundle 和展示均通过。

## 安全与公开边界

- `.env`、密钥、SSH 配置、私网地址和部署路径不得进入仓库或 Demo。
- `runtime/`、`data/`、`artifacts/` 和模型权重均被排除。
- 静态页面可能嵌入正文和证据；发布者必须确认数据具有公开授权。
- 公共 Demo 与内部完整 Release 必须分开构建，隐藏 UI 不等于删除数据。
- 安全问题请参阅 [SECURITY.md](SECURITY.md)。

## 文档

- [系统架构](docs/design/modular-architecture.md)
- [MCP 使用说明](docs/operations/mcp-server.md)
- [MCP 能力发布](docs/operations/mcp-capability-release.md)
- [超大 PDF 处理调研](docs/research/2026-07-28-超大PDF处理前沿成熟方案调研.md)

## License

Apache License 2.0。详见 [LICENSE](LICENSE)、[NOTICE](NOTICE) 和
[第三方组件声明](THIRD_PARTY_NOTICES.md)。
