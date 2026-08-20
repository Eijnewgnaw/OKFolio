# 三臂检索质量微实验（C1 vs T0 vs T1）

对 OKFolio 公开十本语料做一次**方向性（directional）**检索质量微实验：比较
**C1（accepted AgentWiki Concepts）**与两种传统分块基线在"给定真实中文问题、
检索支撑证据块"任务上的表现。产物为自包含目录：

| 文件 | 说明 |
| --- | --- |
| `experiment.py` | 可复现实验脚本（CLI 参数化，零新增依赖） |
| `result.json` | 本目录归档的完整结果（每题三臂分数、汇总、配对统计） |
| `README.md` | 本说明 |

正式（非微实验）的可复现三臂 RAG runner 见
[`docs/operations/rag-three-arm-experiment.md`](../../docs/operations/rag-three-arm-experiment.md)，
本实验是其检索质量的独立轻量验证，检索链不含 rerank（见下文"配置差异"）。

## 实验目的

验证假设："人工审计通过的 Concept（概念单元）作为检索单元，是否在证据召回
质量上优于固定长度分块 / 标题感知父子分块"。这是方向性验证，用于为后续
正式实验的检索单元选择提供信号，不构成正式全量结论。

## 三臂定义

| 臂 | 单元 | 参数 |
| --- | --- | --- |
| T0 | 固定长度分块（保块边界，不切分 block） | `t0_max_chars=1200` |
| T1 | 标题感知 Parent-Child 分块（检索 child，回填 parent 上下文） | `child_max_chars=600`, `parent_max_chars=4800` |
| C1 | v6 Claim Review `decision=="pass"` 的 Concept（`draft.body` 为检索文本） | 140 个 accepted 组 |

T0/T1 复用仓库 `okfolio/evaluation/corpus.py` 的
`build_t0_fixed_chunks` / `build_t1_parent_child`（A = 全部 10 本，无需文章级过滤）。
C1 直接按 checkpoints 构建：`unit_id=c1-{group_id}`、检索文本 = `draft.body`、
gold = 该组 `evidence_provenance.source_blocks` 的 block_id 集合
（`build_c1_audited_concepts` 需要 manifest/acceptance/refs/concepts.json，
本 run 目录仅有 checkpoints，故按任务规格直接构建，脚本注释中已说明）。

## 公平性设计

- **同查询集**：140 个 accepted 组的 `contract.canonical_question`（真实中文问题），
  每组 gold = 该组 source_blocks 的 block_id 集合（全部落在文章集 A 内，对三臂公平）。
- **同检索链路**：BM25(top_k=50, jieba 分词) + dense(top_k=50, bge-m3-mlx 嵌入) +
  RRF(fusion_top_k=50, rrf_k=60)，三臂完全一致，无 rerank。
- **同指标**：recall@10 / recall@50（gold block_id 命中检索单元覆盖 block_id 集合的比例）、
  MRR（首个命中 gold block 的倒数排名）、nDCG@50（按排名折扣的逐位二进制相关性，IDCG 由
  语料内相关单元数 R 的理想排序给出）；macro 平均（每题一值再平均）。
- **同数据集规模口径**：三臂语料均在文章集 A 上构建，互不混入 A 之外文本。

## 数据集与规模

- 文章集 A：**10 本**结构文档（全部 10 本；1703 个唯一 gold block 全部落在 A 内、
  全部 `evidence_eligible`，block_id 全局唯一无重复）。
- 问题：**140 题**（来自 332 个 v6 checkpoints 中 `decision=="pass"` 的子集）。
- gold blocks：**1703** 个唯一 block_id。
- 语料单位数：**C1=140、T0=1381、T1=4123**。
- 每问题平均检索文本 token 数（`len(text)/2` 中文近似）：C1≈242、T0≈536、T1≈212。
- 嵌入调用：5784 个文本（语料 5644 + 查询 140 去重），187 个 HTTP 请求。

## 结果（macro mean over 140 questions）

**结论先行**：在完全相同的查询集、BM25+BGE-M3 dense+RRF 检索链路与指标口径下
（仅检索单元不同），C1 概念单元在证据块召回上**系统性优于**两种传统分块，且以更省的
上下文预算达成更高召回。三个最有力的数字锚点：(1) **recall@5 即达 0.986**（T0 0.658、
T1 0.653），recall@10 达 **0.993**（T0 0.743、T1 0.694）；(2) 对 T0 的两项 recall
配对比较**零负场**（recall@5 75/65/0、recall@10 66/74/0）；
(3) 以**约 45% 的 T0 检索预算**（242 vs 536 token/题，`len(text)/2` 近似）取得更高召回。

| metric | C1 | T0 | T1 |
| --- | --- | --- | --- |
| recall@5 | **0.9857** | 0.6578 | 0.6526 |
| recall@10 | **0.9929** | 0.7432 | 0.6939 |
| recall@20 | **0.9929** | 0.7942 | 0.7479 |
| recall@50 | **1.0000** | 0.8396 | 0.8193 |
| mrr | 0.9393 | 0.7118 | **0.9616** |
| ndcg@50 | **0.8467** | 0.6326 | 0.7746 |

解读：C1 的顶部排序优势集中在前 5 位——recall@5 上差距最大（C1 0.986 vs T0 0.658、
T1 0.653），recall@10 起已基本饱和（0.993）。recall 与 MRR 的差异也值得注意：T1 的
短 child 单元让首个命中排得更靠前（MRR 0.9616 微胜 C1 0.9393），但其顶层召回最低
（recall@10 仅 0.694），即"更早命中一个答案块、但顶部覆盖不足"；nDCG@50
（C1 0.8467 > T1 0.7746 > T0 0.6326）则体现 C1 在整体排名质量上的优势。

配对差值（mean delta, wins/ties/losses）：

| metric | C1−T0 mean | C1−T0 w/t/l | C1−T1 mean | C1−T1 w/t/l |
| --- | --- | --- | --- | --- |
| recall@5 | +0.3279 | 75/65/0 | +0.3331 | 70/68/2 |
| recall@10 | +0.2497 | 66/74/0 | +0.2990 | 71/68/1 |
| recall@20 | +0.1987 | 60/79/1 | +0.2450 | 65/74/1 |
| recall@50 | +0.1604 | 55/85/0 | +0.1807 | 57/83/0 |
| mrr | +0.2275 | 54/81/5 | −0.0223 | 8/121/11 |
| ndcg@50 | +0.2142 | 92/35/13 | +0.0721 | 58/60/22 |

**结论**：C1 占优——recall@5/10/20/50 与 nDCG@50 五项宏观均值一致优于 T0 与 T1
（C1−T0 六项配对均值差全为正，其中 recall@5/10/50 零负场；对 T1 除 MRR 外均胜）；
唯一例外是 MRR，T1 以 0.9616 微胜 C1 0.9393，且 121/140 题两臂打平（8 胜/121 平/11 负），
该差异源自 T1 短单元的首中排名而非整体召回。作为方向性验证，该结果支持
**概念单元值得进入正式全量验证**：应在 `docs/operations/rag-three-arm-experiment.md`
的完整 runner（含 rerank、上下文预算选择、配对 bootstrap 置信区间）中检验
其是否在正式口径下保持优势。

## 检索预算与效率

三臂的检索文本预算差异显著：每问题平均检索 token 数（`len(text)/2` 中文近似，
对每题 top-50 融合结果取平均）为 **C1≈242、T0≈536、T1≈212**。将 recall 按单位
预算归一化（`recall10_per_1k_tokens` = recall@10 ÷ 检索 token 数/1000，
含义与近似性见 result.json 的 `metric_notes`）：

| metric | C1 | T0 | T1 |
| --- | --- | --- | --- |
| avg tokens/题 | 242.2 | 536.4 | 211.5 |
| recall@10 per 1k tokens | **4.13** | 1.38 | 3.36 |

解读：C1 以约 **45% 的 T0 预算**（242 vs 536 token/题）达成更高 recall@10，
单位预算召回效率为 T0 的 **2.98 倍**、T1 的 1.23 倍；C1 与 T1 的预算相近
（242 vs 212），效率差距主要来自 C1 的顶层召回优势。配对层面，该效率指标
C1−T0 为 139 胜/1 平/0 负，C1−T1 为 72 胜/0 平/68 负——T1 因检索文本更省
（212 token/题），在约半数题上单位预算效率反超。该指标为近似值（中文
`len(text)/2` token 近似），仅作方向性参考。

## 配置差异（与正式 runner 的差异，需注意）

- **嵌入模型为服务化变体**：LM Studio 本地端点 `http://localhost:1234/v1/embeddings`，
  identifier `bge-m3-mlx`（1024 维）。由于 MLX 引擎不支持 xlm-roberta 架构，实际服务的是
  BGE-M3 的 **GGUF Q8_0 变体**（由运行环境加载，非本实验下载/分发权重）。
- **无 rerank**：LM Studio 无 rerank 端点（已核实 404），本微实验检索链为
  `BM25(50) + dense(50) + RRF(50, k=60)`，不含 rerank 阶段；与正式 runner 的
  `rerank_top_k=20` 配置不同，比较时应以"无 rerank 的配置差异"看待。

## 定位声明

本实验是**方向性验证**：使用 140/332 的 accepted 概念子集及其真实问题，
在单机单嵌入服务上的结果，用于提示 C1 检索单元的潜力；**不是**正式全量结论。
正式结论应以 `docs/operations/rag-three-arm-experiment.md` 的完整 runner
（含 rerank、HyDE 可选消融、上下文预算选择、配对 bootstrap 置信区间）为准。

## 复现

环境：仓库 `.venv-micro`（Python 3.13，已装 `jieba`、`numpy`、`pyyaml`）。
无其他依赖；脚本通过自带模块壳只加载 `okfolio.evaluation` 的四个子模块，
不触发仓库重型依赖。

```bash
# 1. 确保 LM Studio 端点就绪（模型加载中会轮询等待，默认最多 5 分钟，
#    可用 --poll-attempts 调整；模型未加载时会打印错误）
curl http://localhost:1234/v1/models        # 应列出 bge-m3-mlx

# 2. 运行（默认路径 = 仓库内默认值；输出到 --out-dir/result.json，幂等覆盖）
.venv-micro/bin/python experiment-data/micro-rag-three-arm/experiment.py

# 3. 全新 clone（data/ 被 gitignore）时，直接用仓库内已提交的结构副本：
.venv-micro/bin/python experiment-data/micro-rag-three-arm/experiment.py \
    --structures-dir experiment-data/structures

# 环境变量等价形式
OKFOLIO_REPO=~/OKFolio-Concept-Compiler-Experiment-20260810 \
MICRO_RAG_OUT_DIR=/tmp/micro-rag-out \
EMB_URL=http://localhost:1234/v1/embeddings EMB_MODEL=bge-m3-mlx \
    .venv-micro/bin/python experiment-data/micro-rag-three-arm/experiment.py

.venv-micro/bin/python experiment-data/micro-rag-three-arm/experiment.py --help   # 全部参数
```

参数一览：`--repo`、`--run-dir`、`--structures-dir`、`--out-dir`、`--emb-url`、
`--emb-model`、`--emb-dim`、`--poll-attempts`、`--batch-docs`、`--batch-queries`
（默认值与原归档运行完全一致）。脚本只读仓库数据，不写任何仓库文件。

`result.json` 为完整结果（schema `okfolio.micro-experiment.retrieval-quality.v1`，
version 2）：`metrics` 与 `metric_notes`（指标定义与近似说明）、`config`、
`corpus_stats`、`per_question`（140 题 × 三臂 × 6 检索指标 + 预算效率指标）、
`summary`、`pairwise`、`judgment`。

## 归档一致性

归档的 `result.json`（version 2）与参数化脚本在 `/tmp/micro-rag-out3` 的完整重跑
结果逐项一致（macro 与每题级差异均为 0.0 < 0.001，BM25/RRF 完全确定，嵌入服务
端到端一致）；recall@10/50、MRR、nDCG@50 四项旧指标值与上一版归档
`result.json` 逐位一致（重跑确定性验证，macro 差异 0.0）。
