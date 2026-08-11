# 模块化架构

## 目标

项目只保留一条生产链和一个真相源。CLI、MCP 与 Docker 都调用同一组业务
模块，不各自复制解析、编译或发布逻辑。

## 分层

| 层 | 包 | 输入 | 输出 |
|---|---|---|---|
| 数据处理 | `okfolio.data_processing` | PDF、MinerU 输出 | Document IR、Structure、Segment、Article |
| 知识编译 | `okfolio.agentwiki` | Article、Prompt、模型 | ConceptRef、Concept、Bundle、Graph、Site |
| 控制面 | `okfolio.mcp` | MCP 调用 | 校验后的后台任务和只读查询 |
| 交付 | `Dockerfile`、`docker-compose.yml`、`scripts/package_source.py` | 能力源码 | 无数据的 x86 镜像与源码包 |

依赖方向只能向下：

```text
MCP ───────────────┐
CLI ───────────────┼──> AgentWiki ──> contracts / storage
CLI ───────────────└──> Data Processing

Data Processing 不导入 AgentWiki
AgentWiki 不导入 MCP
业务模块不导入 scripts
```

## 运行目录

```text
runtime/
├── data/
│   ├── inbox/
│   ├── parser-jobs/
│   ├── processed/
│   ├── normalized-sources/
│   └── agent-runs/
└── releases/
```

运行目录不进入 Git 和镜像。历史版本的发布物也不放在源码项目内部。

## PDF 结构恢复

结构恢复对同类中文智库/政策报告使用文档内自适应策略：

1. 将 MinerU 的 title、text 与 table row 统一为候选结构行；
2. 根据目录标记、印刷页码、编号语法、叙述密度和相邻页连续性识别目录区间；
3. 标题层级按 parser 显式层级、目录对齐、编号语法、文档内相对层级依次决策；
4. 每个标题保存 `level_source` 与置信度；
5. 原始 Block 不被覆盖，规范化视图只决定下游是否可作为证据；
6. 低置信度视觉页或结构异常进入审计，不静默丢弃。

所有阈值集中在 `StructurePolicy`，不得写入书名、固定页码或某本书的标题。

## 发布门禁

```text
MinerU 页完整
  → Block / Segment / Asset 守恒
  → 页面角色与目录区间合理
  → 标题层级质量通过
  → Article 原子激活
  → AgentWiki
```

任一门禁失败只保留原始结果和审计报告，不把半成品激活为 AgentWiki 输入。

## Docker

单一 `Dockerfile` 提供三个 target：

- `service`：默认 MCP 服务；
- `compiler`：经典批处理兼容入口；
- `release`：只读 Graph/Site 成品镜像。

单一 `docker-compose.yml` 通过 profile 暴露 PDF 与 AgentWiki 任务，避免多个
Compose 文件产生配置漂移。

## 保密边界

- 源码包与能力镜像不含 PDF、Markdown 语料、图片和知识结果；
- 所有凭据由运行环境注入；
- 发布审计检查路径、文件类型、源文件名指纹和镜像层；
- 对外展示从正式 Bundle 生成单独的 allowlist 版本。
