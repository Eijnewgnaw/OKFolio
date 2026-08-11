# Concept-as-Chunk 本地 RAG 对照实验：框架选型与评测方案

> 调研日期：2026-08-09  
> 范围：为 OKFolio 的 Concept-as-Chunk 实验选择主实验载体，并把 LightRAG 作为独立外部基线；只依据官方文档、官方仓库、模型卡和论文。

## 结论先行

主实验建议使用 **Haystack 作为编排与评测外壳**，但不让它负责 PDF 解析或再次切块。OKFolio 直接把已生成的传统 chunk 或 Concept 转成带稳定 ID、来源页码和证据元数据的 `Document`，分别写入两个独立索引。两组使用完全相同的 BM25、BGE-M3、HyDE、RRF、BGE-Reranker 和生成 Prompt，唯一核心自变量是“知识单元”。Haystack 支持直接写入 `Document`、BM25/向量检索、RRF、模型重排、OpenAI-compatible 自定义端点以及带真值标签的 Recall/MRR/NDCG/答案评测，组件边界也便于逐段计时。[Document Store](https://docs.haystack.deepset.ai/docs/document-store) · [DocumentJoiner](https://docs.haystack.deepset.ai/docs/documentjoiner) · [Evaluation](https://docs.haystack.deepset.ai/docs/evaluation)

**LlamaIndex 作为备选，不作为第一版主载体。** 它同样可以直接接收手工创建的 `TextNode`，并提供 BM25、融合检索、HyDE、OpenAI-like LLM 和检索评测；但其集成包较分散，而且旧的 Query Pipeline 已进入冻结/弃用阶段并建议迁移到 Workflows。对这次强调“变量隔离、指标审计、代码少而明确”的实验，Haystack 更容易固定协议。[直接管理 TextNode](https://docs.llamaindex.ai/en/v0.10.18.post1/module_guides/indexing/vector_store_index.html) · [Query Pipeline 状态](https://docs.llamaindex.ai/en/stable/module_guides/querying/pipeline/)

**LightRAG 应作为独立 Graph-RAG 基线，而不是承载 Concept-vs-Chunk 的主实验。** LightRAG 在索引时还会做实体/关系抽取、图构建和双层检索，若把它混入主实验，就无法判断增益来自 Concept，还是来自图谱与查询策略。官方论文把其核心定义为“图结构 + 低层/高层双层检索 + 增量更新”。[ACL 论文](https://aclanthology.org/2025.findings-emnlp.568/) · [官方仓库](https://github.com/HKUDS/LightRAG)

## 候选框架比较

| 能力 | Haystack | LlamaIndex | LightRAG |
|---|---|---|---|
| 直接导入预切分单元 | 直接构造 `Document(content, id, meta)` 后写入；不经过 Splitter | 直接构造 `TextNode(text, id_)` 并传给 `VectorStoreIndex(nodes)` | SDK 仍有 `ainsert_custom_chunks`，但当前源码已经标记 deprecated；Server 主路径仍使用自身 pipeline/chunker |
| BM25 | `InMemoryBM25Retriever` 或外部 Store Retriever | `BM25Retriever` | 不是面向“BM25 + dense RRF”的可替换流水线，主检索是图与向量的 low/high/global/local/hybrid/mix 模式 |
| BGE-M3 dense | `SentenceTransformersDocumentEmbedder`，也可写自定义组件直接调用 FlagEmbedding | Hugging Face embedding 集成或自定义 Embedding | 官方推荐 BGE-M3，可注入 embedding function |
| HyDE | 官方 Cookbook 给出完整组件管线 | 内置 `HyDEQueryTransform` | 不是标准主查询算子；不宜为了对齐而改其内部检索 |
| RRF | `DocumentJoiner(join_mode="reciprocal_rank_fusion")` | `QueryFusionRetriever(mode="reciprocal_rerank")` | 无需/不应强行改造成同一 RRF 管线 |
| BGE Reranker | `SentenceTransformersSimilarityRanker` 或 FlagEmbedding 自定义 Ranker | SentenceTransformer/FlagEmbedding 后处理器 | 可注入 rerank function；官方推荐 `BAAI/bge-reranker-v2-m3`，查询模式建议 `mix` |
| 本地 OpenAI-compatible LLM | `OpenAIChatGenerator(api_base_url=...)` | `OpenAILike(api_base=..., api_key=...)` | 官方 OpenAI-like function 支持 `base_url` |
| 离线检索评测 | Recall、MRR、MAP、NDCG 等有标签统计组件 | RetrieverEvaluator 支持 MRR、Hit Rate 等 | 官方提供 RAGAS 评测和 retrieved contexts，但其示例以模型评审为主；应接入本项目统一真值评测器 |
| 增量更新 | 稳定 ID + `DuplicatePolicy` + delete/write；版本影响范围由 OKFolio 管理 | doc_id/hash 检查、缓存、变更后 upsert | 有文档状态、插入/删除/恢复与图贡献更新；适合独立测试在线索引更新 |
| 许可证 | Apache-2.0 | MIT | MIT |
| 本项目定位 | **主实验载体** | 备选/快速原型 | **独立外部基线** |

关键一手依据：

- Haystack 可直接把用户构造的 `Document` 写入 Store，并以稳定 ID 执行覆盖、跳过或删除：[Document Store](https://docs.haystack.deepset.ai/docs/document-store)、[DocumentWriter / DuplicatePolicy](https://docs.haystack.deepset.ai/v2.9/reference/experimental-writers-api)。
- Haystack 有原生 [BM25 Retriever](https://docs.haystack.deepset.ai/docs/inmemorybm25retriever)、[RRF Joiner](https://docs.haystack.deepset.ai/docs/documentjoiner)、[HyDE 示例](https://docs.haystack.deepset.ai/docs/hypothetical-document-embeddings-hyde)、[SentenceTransformers Ranker](https://docs.haystack.deepset.ai/docs/rankers) 和 [OpenAI 自定义 base URL](https://docs.haystack.deepset.ai/docs/openaichatgenerator)。其许可证为 [Apache-2.0](https://github.com/deepset-ai/haystack/blob/main/license-header.txt)。
- LlamaIndex 明确支持直接构造节点而绕过 `from_documents()` 的自动切分：[VectorStoreIndex 节点直入](https://docs.llamaindex.ai/en/v0.10.18.post1/module_guides/indexing/vector_store_index.html)。其 Ingestion Pipeline 可基于 `doc_id -> document_hash` 识别未变/已变文档并 upsert：[Document Management](https://docs.llamaindex.ai/en/v0.10.17/module_guides/loading/ingestion_pipeline/root.html)。许可证为 [MIT](https://github.com/run-llama/llama_index/blob/main/LICENSE)。
- LightRAG 官方源码仍提供 caller-chunked 的 `ainsert_custom_chunks`，并做稳定 chunk ID、幂等和失败恢复，但函数已明确标记 deprecated，因此不应把它作为长期的 Concept 直入接口：[当前源码](https://github.com/HKUDS/LightRAG/blob/main/lightrag/lightrag.py#L1677-L1715)。官方 Core 文档支持 OpenAI-like API、BGE-M3 和 reranker 注入：[Programming With Core](https://github.com/HKUDS/LightRAG/blob/main/docs/ProgramingWithCore.md)。许可证为 [MIT](https://github.com/HKUDS/LightRAG/blob/main/LICENSE)。

## 主实验的严格变量设计

### 两个核心实验臂

1. **Traditional RAG**：由同一批规范化 Article 生成普通语义 chunk；保留 `article_id / page / section_path / segment_id`。
2. **Concept RAG**：每个已接受的 OKFolio Concept 作为检索知识单元；保留其全部 `concept_ref_ids` 和每个 Ref 对应的 `article_id / page / segment_id` 证据集合。

两臂必须共用：

- 同一问题集、同一 Qwen 生成模型、同一生成 Prompt；
- 同一 BGE-M3 版本、向量归一化方式和最大输入长度；
- 同一 BM25 中文分词器、候选 Top-K、RRF 常数、reranker 版本与最终上下文 token 预算；
- 同一温度、最大输出 token、拒答阈值和引用格式。

只有语料单元及其元数据不同。不要让 Traditional 组使用弱检索而 Concept 组使用完整混合检索，否则无法把提升归因于 Concept。

### 中文 BM25 的必要修正

Haystack `InMemoryDocumentStore` 当前默认 BM25 tokenizer 是正则 `(?u)\b\w+\b`。对连续中文，它可能把一长段汉字视为一个 token，不能直接当作可信中文 BM25。第一版应把 BM25 做成显式的、可测试的中文组件：对 query 和文档都使用同一固定词典版本的分词器，将分词结果仅用于稀疏检索；dense、reranker 和生成仍读取原文。官方源码允许设置 tokenization regex，但正则本身不能完成中文词切分，因此预分词或自定义 Retriever 更稳妥：[Haystack InMemoryDocumentStore 源码](https://github.com/deepset-ai/haystack/blob/main/haystack/document_stores/in_memory/document_store.py)。

### BGE-M3 与重排

BAAI 模型卡说明 BGE-M3 支持 dense、sparse、multi-vector，支持 100+ 语言和最长 8192 token，并建议“混合检索 + reranker”。为了不把“换稀疏模型”混入本次实验，首轮只使用 BGE-M3 dense，稀疏路固定为同一个中文 BM25，再通过 RRF 合并。[BGE-M3 模型卡](https://huggingface.co/BAAI/bge-m3)

重排使用 `BAAI/bge-reranker-v2-m3`，模型卡给出 `FlagReranker.compute_score([query, passage])` 用法。两臂候选数量和最终 Top-K 必须一致。[BGE Reranker 模型卡](https://huggingface.co/BAAI/bge-reranker-v2-m3/blob/main/README.md)；FlagEmbedding 项目与模型均为 [MIT](https://github.com/FlagOpen/FlagEmbedding)。

### Concept 过长时怎么处理

“Concept 作为 chunk”需要先统计每个 Concept 的模型 token 数，不能静默截断。建议同时记录但不混为一个指标：

- **Strict Concept**：一个 Concept 一个检索单元；仅在不超过 BGE-M3 8192-token 上限时成立。
- **Concept Parent-Child（工程候选）**：标题、摘要和 Concept 内小节作为检索 child，命中后返回完整 Concept parent；所有 child 共用一个 `concept_id`。它更适合超长 Concept，但属于另一个实验臂，不能拿它的结果冒充 Strict Concept。

首轮先跑 Strict Concept，并报告“超过上限的 Concept 数量和占比”；若非零，再单独跑 Parent-Child，而不是截断后不披露。

## “答案正确率”如何判断

答案正确率不能只让生成答案的同一个模型再打一次分。应先建立一份带证据锚点的金标准 QA：

```json
{
  "question_id": "q-001",
  "question": "...",
  "answerable": true,
  "reference_answer": "...",
  "required_facts": ["事实 A", "事实 B"],
  "forbidden_claims": ["与材料冲突的说法"],
  "evidence": [
    {"article_id": "...", "page": 12, "segment_id": "..."}
  ],
  "question_type": "single_fact|multi_evidence|cross_document|unanswerable"
}
```

### 检索层

- **Evidence Recall@K**：Top-K 结果的 provenance 集合是否覆盖 gold evidence。Concept 命中时按其 Ref 证据集合判断，不能用标题相似度代替。
- **MRR**：第一条含 gold evidence 的结果排在第几位。
- **nDCG@K**：多证据问题中，越多关键证据排在越前面越好。
- **Context Precision**：送入生成器的上下文中，真正相关证据的比例。

Haystack 自带有标签的 [DocumentRecallEvaluator](https://docs.haystack.deepset.ai/docs/documentrecallevaluator)、[DocumentMRREvaluator](https://docs.haystack.deepset.ai/docs/documentmrrevaluator) 和 NDCG/MAP 等组件；实际实现需要用 OKFolio provenance ID 做匹配适配。

### 生成层

每个问题先用确定性规则形成一个 pass/fail：

- 可回答题：所有 `required_facts` 正确，无关键矛盾，且至少一条引用能回到正确页码/证据段；
- 不可回答题：明确拒答，且没有编造材料之外的实质事实；
- 任一关键事实错误、引用指向错误来源或应拒答却作答，均记 fail。

于是：

`答案正确率 = pass 的问题数 / 全部问题数`

同时保留更细的诊断指标：原子事实 Precision/Recall/F1、引用 Precision/Recall、Faithfulness、拒答 Precision/Recall/F1。短答案可增加规范化 Exact Match；开放式政策综合题不能只用字符串匹配。Haystack 官方也明确指出 Exact Match 不适合 LLM 生成答案，建议用 faithfulness 或 semantic answer similarity 等指标：[AnswerExactMatchEvaluator](https://docs.haystack.deepset.ai/docs/answerexactmatchevaluator)。

LLM Judge 只作为辅助：用与回答模型不同的评审模型，固定 rubric，盲掉实验臂名称；再随机抽取至少 20% 样本人工复核。若当前只有一台本地 Qwen，则首轮以金标准规则和人工复核为主，不把“同模型自评”称为答案正确率。

### 问题集分层

建议首轮 60–100 题，至少覆盖：单文档事实、多段证据、跨文档综合、时间/场景冲突、不可回答。两组用完全相同的问题和证据锚点。样本规模不足时只报告置信区间和原始计数，不提前宣称“提升 18%”。

## LightRAG 基线怎样跑才可比

LightRAG 使用相同的 10 本规范化公开 Article 作为输入，采用其官方索引和 `mix` 检索，不把 OKFolio Concept 预先灌进去。这样比较的是：

- Traditional Hybrid RAG；
- OKFolio Concept RAG；
- LightRAG Graph-RAG。

三者共用生成模型、问题集、最终上下文 token 预算和统一评测器；分别记录建库时间、LLM 调用数、索引体积、查询延迟。LightRAG 官方评测已能返回 retrieved contexts，并集成 RAGAS，但项目自己的示例分数不能当成本数据集的结果，仍需进入统一 gold QA 评测。[LightRAG RAGAS 说明](https://github.com/HKUDS/LightRAG/blob/main/lightrag/evaluation/README_EVALUASTION_RAGAS.md)

更新实验另外做：给 10 本中的 1 本增加一个显式新版本，检查每套系统的索引写入量、受影响知识单元、旧版本可用性和更新后 QA 变化。不要把更新成本混进首轮静态效果对比。

### 从 LightRAG 吸收什么

LightRAG 值得吸收的是工程机制和查询思想，不是用它替换 OKFolio 的知识编译：

1. **可恢复的文档状态机**：借鉴其插入、删除、失败恢复和文档状态跟踪，把 OKFolio 已有的 Article/Ref/Concept 版本影响范围继续做成可审计的增量索引事务。
2. **低层/高层查询路由**：把 OKFolio 的 Ref 证据检索视为低层，把 Concept 与类型化关系视为高层；后续增加 `local / global / mix` 查询模式，但仍保留原始 provenance。
3. **统一返回 retrieved contexts**：不只返回最终回答，同时固化送入生成器的证据单元、排序分数和页码，使离线评测与页面溯源共用同一份结果。
4. **图贡献删除测试**：单文档更新或删除时，验证只撤销该文档对共享 Concept/关系的贡献，不误删仍有其他来源支撑的知识。

第一版不吸收其 deprecated custom-chunk 入口，也不把 LightRAG 自带的实体关系抽取混进 Concept-vs-Chunk 主实验；这两项都会破坏自变量隔离。

## 本机 LM Studio 的 TTFT 与吞吐怎么测

不需要在 Mac 上再套 vLLM。**LM Studio/MLX 本身就是模型服务层**；RAG 框架只是客户端和检索编排层。

分两层测量：

1. **轻量功能与单请求基准**：使用 OpenAI Python SDK 对 `/v1/chat/completions` 发 `stream=true` 请求，以 `time.perf_counter_ns()` 记录请求开始、第一段非空 token、最后一段 token。得到 client-side TTFT、端到端时延和输出吞吐。LM Studio 官方说明只需把 OpenAI client 的 `base_url` 指向本地服务：[OpenAI compatibility](https://lmstudio.ai/docs/developer/openai-compat)、[Chat Completions](https://lmstudio.ai/docs/developer/openai-compat/chat-completions)。
2. **交叉核验**：LM Studio native `/api/v0/chat/completions` 响应直接给出 `stats.tokens_per_second`、`stats.time_to_first_token` 和 `generation_time`，可与客户端计时对照；但业务代码仍走通用 OpenAI-compatible 接口，避免锁死 LM Studio。[LM Studio REST API v0](https://lmstudio.ai/docs/developer/rest/endpoints)
3. **并发/压力基准（后续）**：使用 NVIDIA 当前维护的 **AIPerf**，而不是已提示迁移的 GenAI-Perf。AIPerf 支持 OpenAI-compatible chat、streaming、TTFT、ITL、TPS、并发和 JSON/CSV 导出，也提供 `aiperf chat` 做单请求 sanity check。[AIPerf 官方文档](https://docs.nvidia.com/aiperf/dev/reference/command-line-options) · [官方仓库](https://github.com/ai-dynamo/aiperf)

RAG 端到端计时必须拆开：

- `retrieval_ms`：BM25 + dense + HyDE + RRF；
- `rerank_ms`：BGE-Reranker；
- `prompt_build_ms`：证据排序、引用映射与 Prompt 拼装；
- `llm_ttft_ms`：向 LM Studio 发请求到第一段非空 token；
- `e2e_ms`：问题进入到最终答案结束；
- `output_tokens_per_second`。

基准时固定输入长度分桶、最大输出 token、temperature 和 seed；先预热，再分别报告 p50/p90/p99。Traditional 与 Concept 的 prompt 长度往往不同，因此既要报告“真实端到端”结果，也要在相同 prompt token 预算下做一组控制实验。

## 推荐落地顺序

1. 只读盘点本机 10 本公开数据的 Article、ConceptRef、Concept 和 provenance 是否完整；缺任一层先停止 RAG 对比。
2. 建立统一 `rag-unit.jsonl` 契约，至少包含 `unit_id / unit_type / text / article_ids / ref_ids / evidence / content_hash`。
3. 用 Haystack 做两个独立索引，先跑 **无 HyDE** 的 BM25 + BGE-M3 + RRF + reranker 冻结探针，验证证据 ID 和指标实现。
4. 只增加一个变量：打开 HyDE；检查它是否改善 Recall，尤其注意官方 LlamaIndex 示例展示的歧义查询误导风险，不能默认 HyDE 一定提升：[HyDE failure case](https://docs.llamaindex.ai/en/v0.9.48/examples/query_transformations/HyDEQueryTransformDemo.html)。
5. 主实验通过后再跑 LightRAG 独立基线；最后才做单文档更新实验和 AIPerf 并发测试。

模型尚未下载完成时，可以先完成 1–2、写离线索引契约测试和金标准 QA schema；接口探针、HyDE、生成评测和 TTFT 均应等待模型下载并加载后执行。
