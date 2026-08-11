# LightRAG 独立外部基线实验方案

日期：2026-08-09  
状态：仅完成官方资料核对和实验设计；未安装 LightRAG、未下载依赖、未启动实验。

## 1. 结论

LightRAG 应作为 **独立外部系统基线**，而不是 OKFolio 的一个内部检索器：

- T0（固定长度 chunk）、T1（结构化 Parent-Child）、C1（Concept）与 LightRAG 必须读取同一份 10 本公开语料的 MinerU 归一化 Markdown；不能让 LightRAG 重跑 PDF 解析，否则比较混入了解析器差异。
- LightRAG 只从原始归一化文档建自己的 chunk、实体、关系和图，不读取 OKFolio 的 Concept/ConceptRef。
- 主实验不直接采用 LightRAG 自带回答作为最终答案，而是调用 `/query/data` 获取 `entities`、`relationships`、`chunks`、`references`，转成统一 `retrieved_contexts`，再交给与 T0/T1/C1 相同的 Qwen 生成器和回答 Prompt。
- `local/global/hybrid/mix` 先在开发集上比较，正式测试集只冻结一个主模式。官方在启用 reranker 时推荐 `mix`，因此预注册主候选为 `mix`，其余模式作为机制分析。
- LightRAG 的原生回答与引用可作为次级结果，通过 `/query` 加 `include_references=true`、`include_chunk_content=true` 留档，但不能替代统一生成实验。

这样可以分别回答两个问题：

1. C1 相对 T0/T1 的提升是否来自 Concept 表示；
2. OKFolio 完整检索链路相对当前 LightRAG 外部系统的效果、成本与更新能力如何。

## 2. 版本冻结与集成边界

截至 2026-08-09，官方最新稳定 Release 是 `v1.5.6`，标签指向提交：

```text
b33c6b0812cddf39206e48a9810112e51f025274
```

实验必须固定这个标签和提交，不跟随 `main`。LightRAG 最近仍有接口和响应结构变更，例如引用中的 `content` 已由单个字符串改成字符串数组，因此不能只写“安装最新版”。版本依据见 [v1.5.6 Release](https://github.com/HKUDS/LightRAG/releases/tag/v1.5.6) 和 [v1.5.6 API Server 文档](https://github.com/HKUDS/LightRAG/blob/v1.5.6/docs/LightRAG-API-Server.md)。

建议后续使用官方 REST API Server，而不是把 LightRAG Core 直接嵌入 OKFolio。官方也明确建议一般集成优先采用 REST，Core 更适合嵌入式应用或研究用途，见 [Programming With LightRAG Core](https://github.com/HKUDS/LightRAG/blob/v1.5.6/docs/ProgramingWithCore.md#programming-with-lightrag-core)。

建议的隔离目录：

```text
external-eval/
└── lightrag-v1.5.6/
    ├── source/                 # 固定官方代码，不修改
    ├── r0-storage/             # 10 本全量基线索引
    ├── r1-storage/             # 从 R0 快照派生的单文档更新实验
    ├── inputs-manifest.json    # 10 本输入指纹和稳定 source key
    ├── retrieval/              # /query/data 原始响应
    ├── graph-audit/            # 更新前、删除后、重插后的图导出
    └── metrics/                # 统一评测结果
```

R0 永远只读保留；更新和删除只在 R1 上执行。

## 3. 后续安装命令（本次不执行）

为避免 PyPI/GHCR 发布时差，首轮研究实验直接固定官方 Git 标签：

```bash
git clone --branch v1.5.6 --depth 1 \
  https://github.com/HKUDS/LightRAG.git \
  external-eval/lightrag-v1.5.6/source

git -C external-eval/lightrag-v1.5.6/source rev-parse HEAD
# 期望：b33c6b0812cddf39206e48a9810112e51f025274

cd external-eval/lightrag-v1.5.6/source
uv sync --extra api
cp env.example .env
```

官方安装方式与 REST Server 启动方式见 [README v1.5.6](https://github.com/HKUDS/LightRAG/blob/v1.5.6/README.md)。若以后改用官方容器，应固定版本标签或 digest，并按官方 [Docker Deployment](https://github.com/HKUDS/LightRAG/blob/v1.5.6/docs/DockerDeployment.md) 验证签名，不能使用浮动的 `latest`。

## 4. 模型与索引配置

### 4.1 LLM

LightRAG 官方建议索引模型至少约 32B、上下文至少 32K，推荐 64K，并建议索引阶段关闭 reasoning。当前本地 Qwen 的实际加载上下文为 71,936 token，满足其 64K 建议；这里应使用实际加载值而不是模型元数据中的最大值。

建议配置骨架：

```dotenv
HOST=127.0.0.1
PORT=9621

LLM_BINDING=openai
LLM_BINDING_HOST=<OPENAI_COMPATIBLE_CHAT_BASE_URL>
LLM_BINDING_API_KEY=<LOCAL_OR_API_KEY>
LLM_MODEL=qwen3.6-35b-a3b-mlx
MAX_ASYNC_LLM=2
LLM_TIMEOUT=900
OPENAI_LLM_MAX_TOKENS=16384
OPENAI_LLM_EXTRA_BODY={"chat_template_kwargs":{"enable_thinking":false}}

KEYWORD_LLM_BINDING=openai
KEYWORD_LLM_BINDING_HOST=<SAME_CHAT_BASE_URL>
KEYWORD_LLM_MODEL=qwen3.6-35b-a3b-mlx
KEYWORD_MAX_ASYNC_LLM=2

QUERY_LLM_BINDING=openai
QUERY_LLM_BINDING_HOST=<SAME_CHAT_BASE_URL>
QUERY_LLM_MODEL=qwen3.6-35b-a3b-mlx
QUERY_MAX_ASYNC_LLM=2

ENTITY_EXTRACTION_USE_JSON=true
ENABLE_LLM_CACHE=false
```

`OPENAI_LLM_EXTRA_BODY` 是官方 `env.example` 给出的 Qwen/vLLM 关闭 thinking 方式。正式启动前仍须用 LM Studio 做一次索引结构化输出探针，因为“服务接受参数”不等于“模型完全不产生 reasoning”。

### 4.2 Embedding 与 reranker

为了避免模型差异污染比较，LightRAG、T0、T1、C1 必须使用同一版本和同一精度的：

- Embedding：`BAAI/bge-m3`；
- Reranker：`BAAI/bge-reranker-v2-m3`（或者项目最终冻结的同一个 BGE reranker）；
- 查询/文档 prefix、向量归一化、相似度阈值必须相同。

LightRAG 官方推荐 BGE-M3，并建议启用 reranker 时使用 `mix`，见 [README 的模型选择与查询模式说明](https://github.com/HKUDS/LightRAG/blob/v1.5.6/README.md#quick-start)。OpenAI-compatible embedding 与 reranker 的地址应指向独立本地服务，不假设聊天模型端口同时提供 embedding。

配置骨架：

```dotenv
EMBEDDING_BINDING=openai
EMBEDDING_BINDING_HOST=<OPENAI_COMPATIBLE_EMBED_BASE_URL>
EMBEDDING_BINDING_API_KEY=<LOCAL_OR_API_KEY>
EMBEDDING_MODEL=BAAI/bge-m3
EMBEDDING_DIM=1024
EMBEDDING_TOKEN_LIMIT=8192
EMBEDDING_SEND_DIM=false

RERANK_BINDING=cohere
RERANK_BINDING_HOST=<COHERE_COMPATIBLE_RERANK_URL>
RERANK_MODEL=BAAI/bge-reranker-v2-m3
RERANK_BY_DEFAULT=true
MIN_RERANK_SCORE=0.0
```

若 reranker 服务不是 Cohere-compatible，应根据官方支持的 `jina/cohere/aliyun` binding 选择真实协议，不能只改 URL。更换 embedding 模型、维度或 query/document prefix 后必须重建整个 LightRAG workspace；官方明确警告旧向量索引不能复用。

### 4.3 存储

首轮 10 本研究实验使用默认 `JsonKVStorage + JsonDocStatusStorage + NetworkXStorage + NanoVectorDBStorage`，减少数据库变量。官方说明它们适合测试和调试、并不代表生产部署能力。生产性结论不得从这一轮存储性能外推。

## 5. 10 本公开语料的输入方式

### 5.1 输入契约

每本书只输入一份完整的 MinerU 归一化 Markdown，并为它保留：

```json
{
  "document_key": "稳定公开文档键",
  "file_source": "public10/<document_key>.md",
  "sha256": "归一化 Markdown 指纹",
  "structure_sidecar": "同一文档的标题/页码/segment 映射",
  "parser": "MinerU 归一化管线版本"
}
```

`file_source` 必须稳定，因为 LightRAG 用它做引用和同源冲突检查。输入内容中不加入 Concept/ConceptRef，不把标题 sidecar 直接当第二篇文档插入。

### 5.2 使用内置固定 token chunker

禁止调用 `insert_custom_chunks` / `ainsert_custom_chunks`。在 v1.5.6 源码中二者仍存在但已明确标记 `deprecated, use insert/ainsert instead`，见 [lightrag.py v1.5.6](https://github.com/HKUDS/LightRAG/blob/v1.5.6/lightrag/lightrag.py#L1831-L1847)。

采用官方 `/documents/text` 或 `/documents/texts`，显式指定内置 `fixed_token`：

```bash
jq -n \
  --rawfile text "$NORMALIZED_MD" \
  --arg source "public10/$DOCUMENT_KEY.md" \
  '{
    text: $text,
    file_source: $source,
    chunking: {
      strategy: "fixed_token",
      params: {
        chunk_token_size: 1200,
        chunk_overlap_token_size: 100
      }
    }
  }' |
curl -sS -X POST http://127.0.0.1:9621/documents/text \
  -H 'Content-Type: application/json' \
  --data-binary @-
```

`1200/100` 是官方默认附近的起点，不是最终公认最优值。为公平起见，实际值应与 T0 的 tokenizer、chunk 上限和 overlap 对齐，并在开发集冻结。LightRAG v1.5.6 的 `/documents/text(s)` 已正式支持 `fixed_token/recursive_character/semantic_vector/paragraph_semantic` 配置，见 [document_routes.py](https://github.com/HKUDS/LightRAG/blob/v1.5.6/lightrag/api/routers/document_routes.py#L530-L714)。这条正式路径替代 deprecated custom chunks。

每次插入保存返回的 `track_id`，用以下正式接口等待完成：

```bash
curl -sS http://127.0.0.1:9621/documents/track_status/$TRACK_ID
curl -sS http://127.0.0.1:9621/documents/pipeline_status
curl -sS -X POST http://127.0.0.1:9621/documents/paginated \
  -H 'Content-Type: application/json' \
  -d '{"page":1,"page_size":50,"sort_field":"file_path","sort_direction":"asc"}'
```

不要使用已 deprecated 的 `GET /documents`；v1.5.6 源码要求改用 `/documents/paginated`。

R0 索引验收条件：

- 10/10 文档均为 `PROCESSED`；
- 每个 `file_source` 恰好一个 active 文档；
- 输入指纹与冻结 manifest 一致；
- 无 `FAILED/PENDING/PROCESSING` 残留；
- 记录 chunk、entity、relation 数、索引墙钟时间、LLM 调用/输入输出 token、embedding/reranker 请求、磁盘占用。

## 6. 查询模式与统一 retrieved contexts

### 6.1 模式含义

根据官方定义：

- `local`：实体和局部关系为中心，适合对象、具体概念和细节事实；
- `global`：关系和宏观主题为中心，适合跨文档主题、趋势和广域关系；
- `hybrid`：合并 local 与 global；
- `mix`：合并 local、global 与 naive 文本向量检索；
- `naive`：只检索原始文本 chunk，可作为 LightRAG 内部 sanity baseline，但不代替 T0。

正式结构化检索使用 `/query/data`。官方接口说明该端点不做最终 LLM 生成，返回实体、关系、chunk、reference 和处理元信息，并支持所有查询模式，见 [query_routes.py v1.5.6](https://github.com/HKUDS/LightRAG/blob/v1.5.6/lightrag/api/routers/query_routes.py#L930-L1260)。这比 `only_need_context=true` 返回的一段格式化字符串更适合作为统一评测输入。

示例：

```bash
curl -sS -X POST http://127.0.0.1:9621/query/data \
  -H 'Content-Type: application/json' \
  -d '{
    "query":"<FROZEN_QUESTION>",
    "mode":"mix",
    "top_k":40,
    "chunk_top_k":20,
    "max_entity_tokens":4000,
    "max_relation_tokens":6000,
    "max_total_tokens":16000,
    "enable_rerank":true
  }'
```

### 6.2 统一适配器输出

每次 `/query/data` 原始响应先原样落盘，再映射为：

```json
{
  "system": "lightrag",
  "version": "1.5.6+b33c6b0",
  "mode": "mix",
  "question_id": "...",
  "retrieved_contexts": [
    {
      "rank": 1,
      "kind": "entity|relationship|chunk",
      "text": "...",
      "file_path": "...",
      "reference_id": "...",
      "chunk_id": "...",
      "source_id": "...",
      "is_model_synthesized": true
    }
  ],
  "raw_evidence_contexts": [],
  "retrieval_metadata": {}
}
```

其中：

- `entity/relationship` 的 description 是 LightRAG 索引期模型综合出的图知识，`is_model_synthesized=true`；
- `chunk` 是原文证据，进入 `raw_evidence_contexts`；
- 回答生成允许使用三类 context，体现 LightRAG 图检索的完整能力；
- Evidence Recall、页码命中和 citation precision 只能由可映射回原始 segment/page 的 raw chunk 得分，不能让模型综合出的关系描述直接冒充原文证据。

LightRAG 返回 chunk 后，用 `file_path + chunk text` 在冻结归一化 Markdown 中做精确子串定位，再与 structure sidecar 的字符区间相交，得到 canonical `article/page/segment`。重复文本无法唯一定位时取候选并集并标记 `ambiguous_alignment=true`，不得猜一个页码。

### 6.3 固定 token budget 的公平比较

主实验固定生成器可见检索上下文预算 `B=8192` Qwen tokenizer token（最终值可在开发集前冻结）。所有系统遵守：

1. 先返回至少约 `2B` 的候选；
2. 使用同一个 tokenizer、同一个去重器和同一个 stable packer；
3. 严格按各系统返回顺序打包到 `B`，最后一项不得越界；
4. 使用同一个回答 Prompt、同一个 Qwen checkpoint、相同 temperature、相同最大输出 token；
5. 同一题的 `question + system prompt + retrieved contexts` 计数方式一致。

LightRAG 示例中的 `max_total_tokens=16000` 是候选阶段约 `2B` 的上限；最终仍由统一 packer 限到 `8192`。`max_entity_tokens=4000`、`max_relation_tokens=6000` 只在开发集固定一次，测试集不得调参。

为了避免“同名实验、问题不同”，结果拆成：

- **表示消融**：T0 vs T1 vs C1，共用项目的 BM25/BGE-M3/HyDE/RRF/BGE-Reranker；
- **外部系统比较**：最佳 T 系统、C1、LightRAG-mix，共用语料、最终上下文预算和生成器，但保留各自原生检索算法；
- **LightRAG 模式分析**：local/global/hybrid/mix 在开发集全跑，测试集只报告预注册主模式，并把其余模式明确标成 secondary/exploratory。

LightRAG 的 keyword extraction 会调用 LLM，这是其检索算法的一部分。主实验不得给它人工预计算 `hl_keywords/ll_keywords` 来隐去成本；应同时报告检索延迟和 keyword LLM token。若后续做“纯检索结构”消融，可另开实验注入固定关键词，但不能和主结果混写。

## 7. 文档更新与删除共享知识审计

LightRAG 官方文档说明 `adelete_by_doc_id` 会删除目标文档 chunk，删除只属于该文档的实体/关系，并对仍由其他文档贡献的实体/关系进行重建、更新向量索引，见 [Delete by Document ID](https://github.com/HKUDS/LightRAG/blob/v1.5.6/docs/ProgramingWithCore.md#delete-by-document-id)。v1.5.6 源码进一步说明，受部分影响的知识会利用剩余文档的 LLM extraction cache 重建，见 [lightrag.py](https://github.com/HKUDS/LightRAG/blob/v1.5.6/lightrag/lightrag.py#L5387-L5428)。

### 7.1 R0 快照

完成 10 本索引后：

1. 停止写入，确认 pipeline idle；
2. 保存 storage 快照；
3. 用官方 `rag.export_data(...)` 导出实体、关系及其 `source_id`，见 [Data Export Functions](https://github.com/HKUDS/LightRAG/blob/main/docs/AdvancedFeatures.md#data-export-functions)；
4. 冻结一组更新审计 query：
   - 只属于待更新文档的事实；
   - 至少被两本文档支持的共享实体/关系；
   - 只属于其余九本文档的对照事实；
   - 新版本新增事实与旧版本撤回事实。

从 R0 快照复制出 R1；R0 不再变更。

### 7.2 删除旧版本

`/documents/text` 对相同 `file_source` 会返回 409，因此正式更新顺序必须是“按 doc_id 删除旧版本 → 等后台删除完成 → 用同一 file_source 插入新内容”，而不是覆盖写。

```bash
curl -sS -X DELETE http://127.0.0.1:9621/documents/delete_document \
  -H 'Content-Type: application/json' \
  -d '{
    "doc_ids":["<OLD_DOC_ID>"],
    "delete_file":false,
    "delete_llm_cache":false
  }'
```

这个接口只返回 `deletion_started`，删除在后台执行。必须轮询 `/documents/pipeline_status` 和 `/documents/paginated`，直到旧 doc_id 消失且 pipeline idle 后才允许重插。请求模型与后台语义见 [document_routes.py v1.5.6](https://github.com/HKUDS/LightRAG/blob/v1.5.6/lightrag/api/routers/document_routes.py#L6117-L6245)。

`delete_llm_cache=false` 保留默认行为，使共享知识能利用剩余文档的 extraction cache 重建；本实验正要测量这条官方更新路径。不要在删除期间并发 scan、insert 或第二个破坏性操作。

### 7.3 删除后门禁

删除旧文档后、尚未插入新版本时，必须满足：

- `exclusive_removal_precision`：只由旧文档贡献的实体/关系应消失；
- `shared_survival_rate`：删除前由旧文档和其他文档共同贡献的共享实体/关系仍存在；
- `stale_source_leakage=0`：导出图和 `/query/data` 中不能再出现旧 doc/chunk source_id；
- 其余九本文档的冻结 query 在 evidence recall 和答案上无非预期回退；
- 共享实体/关系的 description 可以因“仅由剩余九本重建”而变化，不能要求字符串完全相等，应比较身份、支持来源和关键原子事实。

### 7.4 插入新版本并验收

用相同 `file_source`、新 Markdown 指纹和相同内置 chunk 配置重新插入。完成后检查：

- 新事实进入检索；撤回事实不再检索；
- unchanged 事实仍可检索；
- 不产生重复文档、重复实体或重复关系；
- `old chunk id` 无残留；
- 共享知识仍至少有一个活跃来源；
- 记录更新墙钟时间、实际 LLM/embedding 调用、token、缓存复用、storage delta；
- R0 快照始终可启动和查询。

建议更新指标：

```text
Shared survival rate
Exclusive removal precision / recall
Stale-source leakage count
New-fact incorporation recall
Withdrawn-fact leakage rate
Remaining-9 retrieval regression
Entity/relation duplication rate
Update wall time and token cost
R0 snapshot availability
```

重要边界：LightRAG 的“更新”是删除旧版本后重插的 replacement。若希望新旧时点并存，必须使用不同 `file_source` 作为两篇文档；LightRAG 本身不会自动表达 OKFolio 的 temporal/scenario variant。因此更新能力比较必须同时报告“是否保留历史语义”，不能只比较速度。

## 8. 正式执行清单

### 阶段 L0：冻结

- [ ] 固定 `v1.5.6+b33c6b0`；
- [ ] 固定 10 本 normalized Markdown 和 structure sidecar 的 SHA-256；
- [ ] 固定 Qwen、BGE-M3、BGE reranker 的模型/精度/服务协议指纹；
- [ ] 固定 tokenizer、chunk size/overlap、token budget、gold QA dev/test；
- [ ] 建立完全独立的 LightRAG workspace。

### 阶段 L1：索引

- [ ] 逐文档走 `/documents/text` 内置 `fixed_token`；
- [ ] 保存每个 track_id、doc_id、file_source；
- [ ] 10/10 `PROCESSED` 后导出 R0 图和运行统计；
- [ ] 失败必须续跑/重试并记录，不能悄悄减少文档。

### 阶段 L2：模式开发集

- [ ] 对 frozen dev questions 调用 local/global/hybrid/mix；
- [ ] 每个响应原样保存；
- [ ] 统一转换 retrieved contexts、对齐 raw evidence；
- [ ] 使用统一 packer 和统一生成器；
- [ ] 只在 dev 上选择主模式与 `top_k/chunk_top_k`，预期主模式为 mix。

### 阶段 L3：冻结测试

- [ ] 一次性运行 T0/T1/C1/LightRAG 主配置；
- [ ] 报告 retrieval、answer、citation/refusal、latency、token、index cost；
- [ ] 题序随机但各系统用同一顺序；
- [ ] 缓存条件分开：质量实验允许确定性缓存，延迟实验分别报告 cold/warm。

### 阶段 L4：更新/删除

- [ ] R0 快照复制为 R1；
- [ ] 删除一篇、执行共享知识门禁；
- [ ] 重插合成更新版本、执行新旧事实与重复性门禁；
- [ ] 对比 OKFolio 更新实验的受影响范围、复用率、历史保留和 baseline 可用性。

## 9. 主要风险

1. **版本风险**：v1.5.6 发布很新；固定 commit，并先做 API contract smoke test。不要跟随 main。
2. **文档与代码默认值可能不一致**：部分文档示例仍写旧默认；每个 query 显式传 `mode`、top-k 和 token budget。
3. **reasoning 污染结构化抽取**：即使请求关闭 thinking，本地模型仍可能先推理；先跑一篇文档的 JSON extraction probe，再放大到 10 本。
4. **索引成本不对称**：LightRAG 需要实体关系抽取 LLM，T0/T1 主要是切分/embedding；必须单独报告 index token、时间与能耗，不能只比问答分数。
5. **页码引用不原生**：LightRAG reference 原生主要到 file/chunk；页码必须通过冻结 normalized source + sidecar 的字符区间回映，不能由模型生成。
6. **图描述不是原文**：它们可用于回答，但不得直接记作 gold evidence；证据指标只看 raw chunk。
7. **embedding 漂移**：模型、维度、prefix 或 provider task 行为变化均要求重建 workspace。
8. **删除是后台异步操作**：收到 `deletion_started` 不等于删除完成；过早重插会导致 409 或不一致。
9. **默认存储只适合实验**：10 本结果能比较算法，但不能证明生产高并发和数据库可靠性。
10. **多模式选择偏差**：local/global/hybrid/mix 不能都在测试集上挑最高值；只允许开发集选主模式。

## 10. 验收判定

只有同时满足以下条件，LightRAG 才能作为有效外部基线进入总表：

- 10/10 文档完整进入且输入指纹一致；
- 没有调用 deprecated custom chunks；
- 四种模式响应均可转换为统一 retrieved-context schema；
- 所有系统最终生成上下文不超过同一 token budget；
- raw evidence 能回映到公开文档、页码和 segment；
- 主测试配置在看测试结果前冻结；
- 更新实验共享知识无误删、旧 source 无泄漏、R0 保持可用；
- 报告完整索引成本和查询成本，不只报告答案分数。

## 官方一手资料

- [LightRAG v1.5.6 Release](https://github.com/HKUDS/LightRAG/releases/tag/v1.5.6)
- [LightRAG README v1.5.6](https://github.com/HKUDS/LightRAG/blob/v1.5.6/README.md)
- [LightRAG API Server v1.5.6](https://github.com/HKUDS/LightRAG/blob/v1.5.6/docs/LightRAG-API-Server.md)
- [Programming With LightRAG Core v1.5.6](https://github.com/HKUDS/LightRAG/blob/v1.5.6/docs/ProgramingWithCore.md)
- [QueryParam source v1.5.6](https://github.com/HKUDS/LightRAG/blob/v1.5.6/lightrag/base.py#L89-L166)
- [Query REST routes v1.5.6](https://github.com/HKUDS/LightRAG/blob/v1.5.6/lightrag/api/routers/query_routes.py)
- [Document REST routes v1.5.6](https://github.com/HKUDS/LightRAG/blob/v1.5.6/lightrag/api/routers/document_routes.py)
- [LightRAG Core deletion/custom chunks source v1.5.6](https://github.com/HKUDS/LightRAG/blob/v1.5.6/lightrag/lightrag.py)
- [Advanced Features: graph export](https://github.com/HKUDS/LightRAG/blob/main/docs/AdvancedFeatures.md#data-export-functions)
