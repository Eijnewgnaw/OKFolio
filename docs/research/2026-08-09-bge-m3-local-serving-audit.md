# BGE-M3 本地规格、容量与 LM Studio 接入审计

日期：2026-08-09  
范围：`BAAI/bge-m3`、`BAAI/bge-reranker-v2-m3`，以及 OKFolio 当前的 FlagEmbedding 与 LM Studio 两种接入方式。  
边界：本次只读检查文件、配置与官方资料，没有加载模型，也没有给出真实性能基准。

## 结论

1. 当前项目已经保存了 BAAI 官方 Hugging Face 权重，并非 LM Studio 下载的 GGUF：
   - `${REPO_ROOT}/runtime/models/bge-m3`
   - `${REPO_ROOT}/runtime/models/bge-reranker-v2-m3`
2. BGE-M3 是约 568M 参数的 XLM-RoBERTa 模型，官方模型体积 2.27 GB，输出维度 1024，最长输入 8192 token；同一个模型可以产生 dense、sparse 和 ColBERT multi-vector 三类表示，并支持 100+ 语言。[BGE 官方规格](https://bge-model.com/tutorial/1_Embedding/1.2.1.html#bge-m3) · [BAAI 官方模型卡](https://huggingface.co/BAAI/bge-m3) · [论文](https://arxiv.org/abs/2402.03216)
3. `bge-reranker-v2-m3` 也是约 0.6B 参数、F32 权重的 XLM-RoBERTa 系模型，但职责不同：它把 `query + passage` 一起输入，直接给出一个相关性分数，不生成用于向量库的 embedding。[BAAI 官方模型卡](https://huggingface.co/BAAI/bge-reranker-v2-m3#usage)
4. M3 Max 36 GB 从容量上可以运行其中任一个模型；但这是根据模型体积得出的容量判断，不是速度与峰值内存实测。正式运行应继续采用“Embedding 建索引后卸载，再加载 Reranker”的串行方案，不与约 20.4 GB 的本地 Qwen 长时间共存。
5. LM Studio 里的同名可下载项通常是另外的 GGUF/量化制品。LM Studio 官方 `/v1/embeddings` 可以作为 **dense embedding 服务接口**，但它不等价于 FlagEmbedding 的完整 BGE-M3：官方接口没有返回 sparse/ColBERT 表示，也没有通用 reranker endpoint。[LM Studio OpenAI compatibility](https://lmstudio.ai/docs/developer/openai-compat) · [REST embeddings](https://lmstudio.ai/docs/developer/rest/endpoints#post-api-v0-embeddings)
6. 当前三臂主实验应继续使用已经冻结的 FlagEmbedding 原始权重；如果以后试 LM Studio GGUF，应把它作为单独的“服务化/量化消融”，三臂同时切换，并重新校验向量、排序与指标，不能只替换其中一臂。

## 当前本地资产

本地检查得到：

| 资产 | 仓库相对路径 | 冻结 revision | 主要权重 | 本地磁盘占用 |
|---|---|---|---|---:|
| BGE-M3 | `runtime/models/bge-m3` | `5617a9f61b028005a4858fdac845db406aefb181` | `pytorch_model.bin`，另有 `sparse_linear.pt`、`colbert_linear.pt` | 约 2.1 GiB |
| BGE Reranker v2 M3 | `runtime/models/bge-reranker-v2-m3` | `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e` | `model.safetensors` | 约 2.1 GiB |

revision 来自 Hugging Face 下载元数据；两个目录均在 `runtime/`，不会被当作开源源码提交。当前 `lms ls --json` 中只有 Nomic embedding 模型，没有 BGE-M3，因此用户在 LM Studio Discover 中看到的是可下载候选，而不是本机已经存在的第二份 BGE 权重。

## 两个模型分别做什么

### BGE-M3：先找候选

官方规格如下：[BGE 官方文档](https://bge-model.com/tutorial/1_Embedding/1.2.1.html#bge-m3)

| 项目 | 规格 |
|---|---|
| 参数量 | 568M |
| 官方模型体积 | 2.27 GB |
| 主干 | XLM-RoBERTa |
| Dense 向量维度 | 1024 |
| 最大输入 | 8192 token |
| 语言 | 100+ |
| 输出能力 | dense、sparse、multi-vector/ColBERT |

三种输出的含义：

- **dense**：每个文本一个 1024 维向量，适合向量相似度召回；
- **sparse**：产生词项权重，适合词法匹配；
- **multi-vector/ColBERT**：每个 token 保留向量，用 late interaction 计算细粒度相关性。

BAAI 论文说明三种检索能力可以统一训练和组合，并在长文档实验中分别报告 Dense、Sparse、Multi-vec 及混合结果；同时也明确指出，接近 8192 token 的极长输入会带来计算资源与效率挑战。因此“支持 8192”是模型能力上限，不代表所有文档都应该填满 8192。[BGE-M3 论文](https://arxiv.org/abs/2402.03216)

当前 OKFolio 主实验有意只启用 dense：

```python
return_dense=True
return_sparse=False
return_colbert_vecs=False
```

词法路由固定使用中文 BM25，再通过 RRF 与 BGE-M3 dense 合并。这个选择让三臂只比较 T0/T1/C1 的知识单元，不额外引入 BGE sparse 或 ColBERT 的实验变量。实现位于 [`local_bge_backend.py`](../../kmpro_wiki/evaluation/local_bge_backend.py)。

### BGE Reranker v2 M3：对候选重排

Reranker 不是第二个 embedding 模型。官方定义是把查询与候选段落组成一对，直接输出相关性 logit；需要 0–1 分数时再做 sigmoid。它是 cross-encoder，因此通常只处理召回后的较小候选集，而不是给全库建立向量。[BAAI Reranker 模型卡](https://huggingface.co/BAAI/bge-reranker-v2-m3#usage)

| 项目 | 规格/当前设置 |
|---|---|
| 参数量与权重 | 官方页面标注 0.6B、F32 |
| 主干 | BGE-M3 / XLM-RoBERTa sequence classification |
| 输入 | `query + passage` |
| 输出 | 单个相关性分数，可选 sigmoid 归一化 |
| 模型位置上限 | 官方配置 `max_position_embeddings=8194` |
| OKFolio 当前运行上限 | `reranker_max_length=1024` |

BAAI 官方建议 `use_fp16=True` 可以加速，但可能带来轻微性能下降；因此精度设置必须写入实验指纹，不能运行中途悄悄改变。[BGE Reranker 官方文档](https://bge-model.com/bge/bge_reranker_v2.html#usage)

## M3 Max 36 GB 能否带动

### 容量判断

568M 参数的纯权重理论下限约为：

- FP32：`568M × 4 bytes ≈ 2.27 GB`；
- FP16：`568M × 2 bytes ≈ 1.14 GB`。

这与官方 2.27 GB 和本地约 2.1 GiB 文件相符。实际运行内存还包括 tokenizer、PyTorch/Metal 运行时、激活、中间缓冲区和输入 batch，必然高于纯权重；越长的序列与越大的 batch，峰值越高。因此可作如下判断：

- **容量上能运行**：单个 568M 模型相对 36 GB 统一内存不大；
- **不宜与 35B Qwen 长时间并存**：当前 Qwen 量化文件约 20.4 GB，叠加运行时和长上下文缓存后，继续共驻会增加内存压力；
- **8192 token 不应作为默认长度**：模型支持到 8192，但长输入成本明显更高；
- **本次尚不能声称速度**：需要后续单独做 batch、长度、吞吐和峰值内存探针。

### 当前代码的安全边界

当前本地后端已经按阶段释放模型，并保存 dense 索引为 float32 NumPy 文件：

1. BGE-M3 建立三臂 dense 索引；
2. 保存索引并释放 BGE-M3；
3. 召回后加载 Reranker；
4. 保存候选结果并释放 Reranker；
5. 最后再由 Qwen 生成答案。

示例配置目前是：MPS、dense batch 4、reranker batch 2、dense passage 最大 8192、reranker 最大 1024。第一次真实性能探针应从 batch 1/1 开始，再逐级增大；是否启用 FP16必须作为固定实验变量验证，而不能仅凭“更快”直接替换。

## FlagEmbedding 原始权重与 LM Studio GGUF 的差异

| 维度 | 当前 FlagEmbedding 方案 | LM Studio GGUF / embeddings endpoint |
|---|---|---|
| 权重 | BAAI 官方 PyTorch/Safetensors，当前是 F32 文件 | 通常为 Hugging Face 社区转换的 GGUF，并选择 Q4/Q8 等量化 |
| 调用方式 | Python 进程内调用 `BGEM3FlagModel` / `FlagReranker` | HTTP 或 SDK，兼容 `/v1/embeddings` |
| Dense | 支持，返回 1024 维向量 | 支持时返回一个 dense 浮点数组 |
| Sparse | BGE-M3 原生支持 | 官方 embedding API 未定义 sparse 权重返回值 |
| ColBERT multi-vector | BGE-M3 原生支持 | 官方 embedding API 未定义 token-level multi-vector 返回值 |
| Reranker | `FlagReranker` 对 query-passage 直接打分 | LM Studio 官方 API 当前没有通用 rerank endpoint；不能把 reranker 当普通 embedding 使用 |
| 精度 | F32 或显式 FP16，设置可冻结 | 取决于具体 GGUF 量化；LM Studio 官方说明量化以更小体积换取一定精度损失 |
| 可复现性 | 可固定官方 revision、参数和本地文件哈希 | 必须另外固定 publisher、GGUF 文件、量化等级、runtime 版本和文件哈希 |

LM Studio 的价值主要是统一服务接口和更轻量的量化部署。官方支持通过 `/v1/embeddings` 使用 OpenAI 客户端，也支持列出 embedding 模型的格式、量化和最大上下文。[OpenAI-compatible embeddings](https://lmstudio.ai/docs/developer/openai-compat) · [REST model metadata](https://lmstudio.ai/docs/developer/rest/endpoints#get-api-v0-models)

但对当前实验，直接改成 LM Studio 会同时改变：

- 权重精度；
- 推理运行时；
- 可能的截断与 pooling 实现；
- 能否取到 sparse/ColBERT 输出；
- reranker 服务方式。

这会让实验不再只比较知识单元。因此不能把 FlagEmbedding T0/T1 与 LM Studio C1 混在一起；若测试 LM Studio，三臂必须同时切换，并把它标记为独立消融。

## 推荐决策

### 当前正式三臂实验

继续使用已经下载并冻结的 BAAI 原始权重：

- BGE-M3：`5617a9f61b028005a4858fdac845db406aefb181`；
- BGE Reranker v2 M3：`953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e`；
- 三臂共享完全相同的模型、精度、tokenizer、最大长度和 batch 设置；
- 继续串行加载，避免与 Qwen 抢占统一内存；
- 首轮只使用 BGE-M3 dense，BM25 作为固定稀疏路，HyDE 关闭。

### 后续 LM Studio 服务化消融

如需验证 LM Studio，单独建立 `provider=lmstudio-gguf` 实验：

1. 记录完整模型标识、GGUF 文件哈希、量化、LM Studio/runtime 版本；
2. 核验实际最大输入与返回向量维度；
3. 在冻结文本集上比较原始权重与 GGUF 的 cosine、Top-K overlap、MRR/nDCG；
4. 三臂同时使用同一个 LM Studio embedding 服务；
5. Reranker 仍使用 FlagEmbedding，或另建有明确 cross-encoder rerank 协议的服务；不得把 `/v1/embeddings` 当成 rerank API；
6. 结果只说明“量化服务化带来的速度/质量变化”，不与主实验的 Concept 效果混为一谈。

## 可复核命令

以下命令只查看文件或模型清单，不加载模型：

```bash
du -sh runtime/models/bge-m3 runtime/models/bge-reranker-v2-m3
sed -n '1,160p' runtime/models/bge-m3/config.json
sed -n '1,160p' runtime/models/bge-reranker-v2-m3/config.json
lms ls --json
```

## 一手来源

- [BAAI BGE-M3 官方 Hugging Face 模型卡](https://huggingface.co/BAAI/bge-m3)
- [BGE 官方文档：BGE-M3 参数、体积、维度和能力](https://bge-model.com/tutorial/1_Embedding/1.2.1.html#bge-m3)
- [BGE-M3 论文](https://arxiv.org/abs/2402.03216)
- [BAAI BGE Reranker v2 M3 官方模型卡](https://huggingface.co/BAAI/bge-reranker-v2-m3)
- [BGE 官方 Reranker 文档](https://bge-model.com/bge/bge_reranker_v2.html)
- [FlagEmbedding 官方仓库](https://github.com/FlagOpen/FlagEmbedding)
- [LM Studio OpenAI-compatible endpoints](https://lmstudio.ai/docs/developer/openai-compat)
- [LM Studio REST Embeddings 与模型元数据](https://lmstudio.ai/docs/developer/rest/endpoints)
- [LM Studio 模型下载与量化说明](https://lmstudio.ai/docs/app/basics/download-model)
