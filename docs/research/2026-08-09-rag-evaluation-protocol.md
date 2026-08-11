# OKFolio Concept-as-Chunk RAG 评测协议

> 调研日期：2026-08-09  
> 适用范围：公开 10 本区域经济皮书；Traditional Chunk、OKFolio Concept、后续 LightRAG 对照  
> 证据范围：论文、官方仓库和官方评测文档，不采用博客转述作为方法依据

## 结论先行

先建立带来源证据的人工金标准（gold set），再分别评测检索、答案正确性、忠实性、引用和拒答，是当前 RAG 工程与研究中最稳妥、也最容易审计的做法；但“必须先人工写 60–100 篇完整标准答案”不是通用定律。

更准确的结论是：

1. **开发期可以用无参考 LLM 指标快速回归，正式比较不能只靠它。** RAGAS、TruLens、DeepEval 都提供无参考的 faithfulness / groundedness / relevance 指标；Haystack 同时明确区分需要真值标签的统计评测和不需要标签的模型评测。它们适合快速发现问题，但不能单独证明 Concept 比传统 chunk 的答案更正确。[RAGAS 论文](https://aclanthology.org/2024.eacl-demo.16/) · [Haystack Evaluation](https://docs.haystack.deepset.ai/docs/evaluation) · [TruLens Evaluation](https://www.trulens.org/component_guides/evaluation/) · [DeepEval Faithfulness](https://deepeval.com/docs/metrics-faithfulness)
2. **正式 A/B 必须有独立于两个系统的 query、answerability 和 evidence labels。** 检索 Recall、MRR、nDCG 本身就依赖 qrels；BEIR 官方实现用 qrels 计算 nDCG、MAP、Recall、Precision 和 MRR。HotpotQA 同时标注答案和 supporting facts，KILT 进一步要求答案只有在完整 provenance 被检索到时才得分。[BEIR 官方仓库](https://github.com/beir-cellar/beir) · [HotpotQA 论文](https://aclanthology.org/D18-1259/) · [KILT 论文](https://aclanthology.org/2021.naacl-main.200/)
3. **不必为每题只写一段“唯一标准长答案”。** 对本项目更稳妥的是标注 `required_facts`、`forbidden_facts`、`answerable` 和一个或多个最小 `evidence_sets`；参考答案正文只是便于阅读的派生产物。这样不会因措辞不同误判开放式答案。
4. **“答案正确率”不能取代忠实性、引用和拒答指标。** 正确但没有证据、忠实于错误检索、引用存在但不支持陈述、对不可回答问题强答，是四种不同失败；必须分层报告。
5. **60–100 题适合作为工程 pilot；能否支撑正式结论必须看置信区间。** 针对本项目计划对外报告“90%+、提升 18%”的目标，建议冻结 40 题开发集和至少 200 题测试集，报告逐题配对的 95% 置信区间。不存在脱离效应大小与题目方差的万能样本量。

## 1. 这是否属于通用工程做法

### 1.1 行业工具的共同结构

不同框架名称不一，但实际都把 RAG 拆成多个可诊断环节：

| 来源 | 检索层 | 生成层 | 是否支持真值 |
|---|---|---|---|
| BEIR | nDCG、MAP、Recall、Precision、MRR | 不覆盖 | qrels 必需 |
| Haystack | Document Recall/MRR/MAP/nDCG | Exact Match、SAS、Faithfulness、LLM rubric | 统计指标需要 labels；模型指标可无 labels |
| LlamaIndex | Hit Rate、MRR，对比 expected node IDs | Faithfulness 等 evaluator | 检索评测需要 expected IDs |
| RAGAS | Context Precision/Recall | Faithfulness、Response Relevancy 等 | 同时有 reference-based 和 reference-free 路径 |
| TruLens | Context Relevance | Groundedness、Answer Relevance | 无 gold 时用 feedback functions |
| DeepEval | Contextual Precision/Recall/Relevancy | Answer Relevancy、Faithfulness | 同时支持 reference-based 和 referenceless |

一手依据：

- [Haystack 官方评测文档](https://docs.haystack.deepset.ai/docs/evaluation)明确建议既评估单个组件，也评估端到端系统，并列出哪些统计 evaluator 需要标签。
- [LlamaIndex 官方 RetrievalEvaluator 示例](https://developers.llamaindex.ai/python/framework/module_guides/evaluating/usage_pattern_retrieval/)把检索结果和 `expected_ids` 比较，计算 MRR、Hit Rate。
- [RAGAS Faithfulness](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/faithfulness/)把答案拆成 claims，计算其中可由 retrieved context 支持的比例；[Context Recall](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/context_recall/)也明确指出 recall 需要 reference，并提供按 context ID 计算的非 LLM 路径。
- [TruLens](https://www.trulens.org/component_guides/evaluation/)说明 feedback functions 是在缺少 ground truth 时的程序化替代，而不是证明 ground truth 不再需要。
- [DeepEval](https://deepeval.com/docs/getting-started-rag)明确建议分开测 retriever 和 generator，因为单个端到端分数无法定位错误来源。

因此，本项目提出的“金标准 QA + 分层指标”不是自创的特殊流程，而是把上述框架的共同结构落实到可追溯的政策报告语料上。

### 1.2 人工 gold 是否必要

按目标分三档：

| 目标 | 人工 gold 要求 | 可接受结论 |
|---|---|---|
| 接口/Prompt 日常回归 | 可只用少量固定样例 + 无参考指标 | “本次改动未出现明显回归” |
| 选择候选配置 | 需要一批已审阅的 evidence labels 和答案事实 | “在该开发集上 A 优于 B” |
| 对外声称“90%+、提升 18%” | 必须使用冻结测试集、独立人工核验和置信区间 | “在预先定义的测试协议下，点估计与不确定区间为……” |

RAGAS 的初衷正是 reference-free 快速评测；但 ARES 的研究发现，自动 judge 仍需要少量人工验证集来校准和计算统计置信度。ARES 使用合成数据训练 judge，同时要求约 150 条或更多正负人工标注，并用 prediction-powered inference 生成置信区间。这说明“自动化评测”和“人工金标准”不是二选一，而是不同成本层的组合。[RAGAS](https://aclanthology.org/2024.eacl-demo.16/) · [ARES](https://aclanthology.org/2024.naacl-long.20/)

## 2. Gold QA 应该怎样构造

### 2.1 标注单位必须独立于 chunk 方案

不能把 Traditional Chunk 的 chunk ID 或 Concept ID 直接当成 gold evidence，因为两种方案的边界不同，标签会天然偏向某一方。统一标注到更底层且稳定的证据原子：

```text
evidence_atom_id = article_id + page + normalized_segment_id/block_id
```

每个传统 chunk 和每个 Concept 都必须通过 provenance 映射回这些共同原子。评测时比较“检索结果覆盖了多少 gold atoms / 是否覆盖一个完整 evidence set”，而不是比较它有没有命中自己那套 chunk ID。

HotpotQA 提供了可复用的先例：每道多跳问题标注 supporting fact sentences，并把答案 EM/F1、supporting-fact EM/F1 和 joint EM/F1 分开报告。KILT 则只有在完整 provenance 集被正确召回时才给下游答案分数，强调“答对且有据”。[HotpotQA](https://aclanthology.org/D18-1259/) · [KILT](https://aclanthology.org/2021.naacl-main.200/)

### 2.2 推荐数据契约

```json
{
  "question_id": "policy-0042",
  "question": "……",
  "question_type": "cross_document_synthesis",
  "answerable": true,
  "scope": {
    "time": "……",
    "region": "……",
    "subject": "……"
  },
  "required_facts": [
    {"fact_id": "f1", "claim": "……", "weight": 2},
    {"fact_id": "f2", "claim": "……", "weight": 1}
  ],
  "forbidden_facts": [
    {"claim": "……", "reason": "时期或适用场景冲突"}
  ],
  "evidence_sets": [
    ["article-a:p003:s007", "article-b:p041:s002"],
    ["article-c:p016:s004"]
  ],
  "reference_answer": "供人工阅读的规范答案，不作为唯一字符串真值",
  "annotation": {
    "author": "human-a",
    "reviewer": "human-b",
    "status": "adjudicated"
  }
}
```

`evidence_sets` 是“可替代的最小完整证据集合”：只要覆盖其中任意一组即可，不要求把所有可能支持材料全部检出。跨文档题的完整集合可以包含两个以上 Article 的 evidence atom。

### 2.3 题目构造流程

1. **先分层抽样，再出题。** 以 10 本书、章节层级、题型和时间/区域范围为抽样轴，避免只从容易检索的标题段落出题。
2. **从原始证据出发产生候选题。** LLM 可以根据规范化 Article、页面和结构化目录批量生成候选问题、事实和 evidence IDs，但只能作为草稿；禁止从已经编译好的 Concept 反向出题，否则题目会天然偏向 C1。
3. **人工校验可回答性和范围。** 核对数字、单位、年份、政策主体、地区、表格行列，以及是否存在同名但不同场景的事实。
4. **补充困难负例。** 不可回答题应“主题合理但语料无完整答案”，而不是靠明显荒谬问题取巧。SQuAD 2.0 的核心就是加入看起来与可回答题相似、但上下文实际不支持的对抗式不可回答问题，并要求系统学会 abstain。[SQuAD 2.0](https://aclanthology.org/P18-2124/)
5. **盲审和裁决。** 标注人员不看 Traditional / Concept 的输出；至少 20% 题由第二人独立复核，分歧经裁决后冻结。
6. **开发集与测试集隔离。** BM25 分词、RRF 权重、reranker Top-K、拒答阈值只能在开发集上调整；测试集一次冻结后不再调参。

### 2.4 建议题型配额

正式 240 题可按下表组织，其中 40 题为开发集、200 题为冻结测试集：

| 题型 | 总数 | 主要测试点 |
|---|---:|---|
| 单文档事实 | 60 | 实体、定义、政策措施 |
| 数字/表格/时间范围 | 48 | 数值、单位、年份和场景约束 |
| 单文档多证据综合 | 48 | 同章或跨章证据组合 |
| 跨文档综合 | 48 | 多 Article 的联合知识优势 |
| 不可回答 | 36 | 拒答、过度拒答、外部知识污染 |

每本书至少覆盖 15 道可回答题；跨文档题单独成层，不强行归属一本书。首轮工程试跑可先抽其中 80 题，但不得把 80 题结果包装成最终稳定结论。

## 3. 指标必须分层，而不是只报一个“答案正确率”

### 3.1 L0：输入与索引门禁

- 两组覆盖同样的 10 本 Article 和同样的 active 版本。
- 100% 检索单元能映射回 `article_id / page / evidence_atom_id`。
- 报告索引单元数、总字符/token、超长截断数、重复 evidence atom 数。
- 任何一组缺书、缺页或存在静默截断，正式 A/B 不启动。

### 3.2 L1：检索质量

主指标：

- `Evidence Recall@K`：gold evidence atoms 被检出的比例。
- `Complete Evidence Set Recall@K`：是否完整覆盖任意一个 gold evidence set；多证据题尤其重要。
- `MRR@K`：首个必要证据出现的倒数排名。
- `nDCG@K`：采用预先固定的 graded qrels（必要证据=2、补充证据=1）衡量排序质量。
- `Context Precision@K`：进入生成上下文的 evidence atoms 中，有多少和题目相关。

BEIR 官方实现同时提供 nDCG、MAP、Recall、Precision、MRR；本项目沿用这些 IR 指标，但 qrels 落在公共 evidence atoms，而非不同边界的 chunk。[BEIR 官方仓库](https://github.com/beir-cellar/beir)

除固定 `K` 外，必须再报告**固定上下文 token 预算**下的 Evidence Recall。Concept 通常比普通 chunk 长，只比 Top-5 会让一组获得更多上下文，因此端到端主结论使用固定 token 预算，固定 K 仅作诊断。

### 3.3 L2：答案正确性

按题型计分：

- 短事实：规范化 Exact Match、token/字符 F1。
- 数值题：数值、单位、正负方向、时间范围全部正确才计关键事实；必要时预先定义容差。
- 综合题：`required_facts` 加权 Precision / Recall / F1；命中 `forbidden_facts` 单独记为 contradiction。
- 主观表述：只允许基于逐事实 rubric 的人工或经校准 judge 评分，不能用一个无定义的 1–10 总分。

推荐的主答案指标：

```text
Answer Correct =
  all critical required facts are correct
  AND no critical forbidden fact appears
  AND answerability decision is correct
```

同时报告 `Atomic Fact F1`，避免一处小遗漏让所有部分得分消失。

### 3.4 L3：Faithfulness / Groundedness

Faithfulness 问的是“答案中的每个事实是否由**实际检索到的上下文**支持”，不是“世界上是否正确”。RAGAS 的定义就是 supported response claims / all response claims；DeepEval 也明确把它定位为 generator 对 retrieval context 的一致性评价。[RAGAS Faithfulness](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/faithfulness/) · [DeepEval Faithfulness](https://deepeval.com/docs/metrics-faithfulness)

因此必须与正确率分开：

- 检索到了错误时期的材料，答案忠实复述它：可能 faithfulness 高、correctness 低。
- 模型凭参数记忆答对，但检索材料没有证据：可能 correctness 高、faithfulness 低。

计算方式：把输出拆成 atomic claims，每个 claim 标注 `supported / contradicted / not_in_context`，主报 supported 比例，并单报 contradiction rate。

### 3.5 L4：引用质量与溯源

至少报告：

- `Citation Validity`：引用的 Article、页码、segment 是否真实存在并可打开。
- `Citation Precision`：每条引用是否实际支持与它相邻的 claim。
- `Citation Recall/Completeness`：需要证据的 claims 有多少得到完整支持。
- `Source/Page Accuracy`：Article 和页码是否与 gold evidence 一致。

ALCE 把正确性和引用质量分开，并将 citation recall 定义为输出陈述是否被引用段落完整支持，将 citation precision 用于发现无关引用；其自动指标还和人工标注作了单独一致性检验。[ALCE 论文](https://aclanthology.org/2023.emnlp-main.398/) · [官方代码](https://github.com/princeton-nlp/ALCE)

### 3.6 L5：不可回答与拒答

把拒答看成 `answerable` 二分类，不与普通准确率混在一起：

- `Refusal Recall`：所有不可回答题中，正确拒答的比例。
- `Refusal Precision`：所有拒答中，确实不可回答的比例。
- `Over-refusal Rate`：可回答题被错误拒答的比例。
- `Unsupported Answer Rate`：不可回答题仍生成实质答案的比例。

同时分别报告 answerable 和 unanswerable 子集的答案质量。若一个系统靠全部拒答获得很高“安全分”，Over-refusal 会直接暴露该问题。

### 3.7 L6：端到端联合成功

参考 HotpotQA 的 joint 指标和 KILT 的 provenance-gated 指标，定义严格主指标：

```text
Joint Success =
  Answer Correct
  AND Complete Evidence Set Retrieved
  AND Citation Validity = 1
  AND critical claims are citation-supported
```

Joint Success 是正式端到端主指标；其余分层指标负责解释为什么失败。不要把五六个异质分数加权成一个无法解释的“综合分”。

## 4. Concept-as-Chunk 与传统 chunk 的公平 A/B

### 4.1 三个实验臂

为了避免只击败一个过弱基线，建议：

1. `T0 Fixed Chunk`：固定 token 窗口 + 固定 overlap，代表普通基础 RAG。
2. `T1 Structure-aware Parent-Child`：利用同样目录、页码和元信息的强传统基线。
3. `C1 OKFolio Concept`：完整 10 本重新生成的 Concept；不复用早先 2 本探针。

主要论文式结论比较 `C1 vs T1`；`C1 vs T0` 只说明相对基础方案的提升。

### 4.2 必须锁死的变量

- 同一 10 本 corpus snapshot、同一问题集和 gold labels；
- 同一标题/章节/时间/地区元信息 envelope；
- 同一中文 BM25 分词器与版本；
- 同一 BGE-M3 checkpoint、向量归一化和最大长度；
- 同一候选深度、RRF 公式和权重、BGE-Reranker checkpoint；
- 同一生成模型、system prompt、temperature、max output tokens；
- 同一拒答规则、引用格式和运行硬件；
- final test 前冻结全部参数和代码 commit。

Concept 的自动标题、摘要和跨文档组织属于 C1 的处理效果，可以保留，但要在消融实验中报告：

- `C1-full`：Concept 正文 + 标题/description；
- `C1-body`：仅 Concept 正文；

这样才能判断增益来自真正的联合知识组织，还是仅来自额外摘要字段。

### 4.3 两套预算必须同时报

| 评测视角 | 约束 | 用途 |
|---|---|---|
| 排名诊断 | 固定 Top-K | 比较前几名检索单元的排序 |
| 生成公平性（主） | 固定最大 context tokens | 避免长 Concept 获得更多信息预算 |

装填上下文时采用确定性策略：按 reranker 排名依次加入，超过预算的单元不静默截断；另报未装入数量。若需要截断，必须作为独立实验臂并记录被截断的 provenance。

### 4.4 对照 LightRAG 的边界

LightRAG 会改变实体/关系抽取、图结构和查询模式，不能和“仅替换 chunk 单元”的主 A/B 混在一起。完成 T0/T1/C1 后，再增加：

- `G1 LightRAG-local`
- `G2 LightRAG-global`
- `G3 LightRAG-mix`

所有 G 组继续使用同一 gold QA、固定生成 token 预算和同一分层指标。这样可以回答两个不同问题：

1. Concept 作为知识单元是否优于强传统 chunk？
2. OKFolio 的联合编译是否优于或可补充 LightRAG 的图检索？

## 5. 如何避免 LLM-as-a-Judge 偏差

### 5.1 不让 judge 承担可以确定计算的部分

以下指标全部程序化：ID 命中、页码有效性、Recall/MRR/nDCG、短答案 EM/F1、数值与单位、拒答混淆矩阵、延迟、token 和吞吐。Judge 只处理开放式综合答案中的 claim 对齐与蕴含判断。

### 5.2 Judge 使用规范

1. 不向 judge 暴露实验组名、模型名或生成顺序。
2. 优先 pointwise 对照 `required_facts + evidence` 评分；若使用 pairwise，A/B 顺序互换各评一次，不一致则记 `judge_uncertain`。
3. 固定结构化 rubric，逐事实输出 `supported / contradicted / missing`，禁止只给一个总分。
4. 生成模型与 judge 尽量使用不同模型家族；如果本地只有同一个 Qwen，不能将其自评作为唯一正式结论。
5. 在测试集之外建立人工校准子集，报告 judge 对人工标签的 precision、recall、F1 和一致率。
6. 至少人工复核所有系统结论发生翻转的题、所有 judge 不一致题，以及随机 20% 普通题。

这样做有直接研究依据：MT-Bench/Chatbot Arena 的原始论文发现 LLM judge 存在 position、verbosity 和 self-enhancement bias，并展示交换答案顺序会改变判断；该论文也强调 LLM judge 与传统基准是互补关系，而不是替代关系。[Judging LLM-as-a-Judge](https://proceedings.neurips.cc/paper/2023/hash/91f18a1287b398d378ef22505bf41832-Abstract-Datasets_and_Benchmarks.html)

ARES 进一步说明自动 judge 的预测本身并不完全准确，因此用人工 validation set 和统计校准产生置信区间，而不是把 judge 分数直接当真值。[ARES](https://aclanthology.org/2024.naacl-long.20/)

## 6. 样本量、置信区间和显著性

### 6.1 没有通用的固定样本量

需要多少题取决于：基线水平、题间方差、两系统逐题结果的相关性、预期最小提升和题目聚类。ARES 的约 150 条人工验证数据是其 PPI 设置，不是所有 RAG 的万能阈值；NIST 对常见 IR 指标的研究覆盖了 50–150 个 topics 的多种 TREC/NTCIR 集合，并指出 topic 数增加会缩窄置信区间。[ARES](https://aclanthology.org/2024.naacl-long.20/) · [NIST IR confidence intervals](https://tsapps.nist.gov/publication/get_pdf.cfm?pub_id=917303)

直观上，如果 100 题中 90 题正确，Wilson 95% 区间约为 `[82.6%, 94.5%]`；200 题中 180 题正确约为 `[85.1%, 93.4%]`。因此“点估计 90%”不能等价于“真实水平确定在 90% 以上”。

### 6.2 本项目的可执行建议

- `Pilot-80`：调通 gold schema、评测器和失败分类；只作工程决策。
- `Dev-40`：调 BM25、RRF、reranker、token budget 和拒答阈值。
- `Test-200`：冻结后运行 T0/T1/C1，形成正式结论。
- 若 Test-200 的主指标区间过宽或结果接近，再扩展到 300–400，而不是事后只挑有利题型。

所有系统在**同一题目**上比较，所以统计单位是逐题配对差值：

1. 每题保存两系统的 metric score；
2. 对题目进行 10,000 次 paired bootstrap 重采样；
3. 每次计算 `delta = metric(C1) - metric(T1)`；
4. 报告 delta 点估计和 percentile 95% CI；
5. 只有主指标 delta 的 CI 下界大于 0，才写“有统计支持的提升”；否则写“观察到提升，但区间包含 0”。

NIST 的 IR 研究建议报告置信区间，并验证了标准区间和 bootstrap percentile interval 对多种 IR 指标有良好经验覆盖；其论文也建议比较系统时直接对均值差构造区间。[NIST](https://tsapps.nist.gov/publication/get_pdf.cfm?pub_id=917303)

同时对两个预声明主指标运行 paired randomization/permutation test 作为敏感性分析；CI 是主要呈现，p 值只回答“与零差异是否相容”，不能代替效应大小。Smucker、Allan 与 Carterette 在 TREC runs 上比较了 paired t、bootstrap、randomization、Wilcoxon 与 sign test，为 IR 系统使用逐 topic 配对检验提供了直接依据。[CIIR 原作者版本](https://maroo.cs.umass.edu/pdf/IR-591.pdf)

由于同一本书中的问题相关，最终再做一次分层/簇 bootstrap：先按 Article（跨文档题按 source-set）重采样，再在簇内抽题；同时报告逐书 macro average 和全题 micro average。只按 200 道题独立 bootstrap 可能低估不确定性。

### 6.3 预先声明主指标

避免在大量指标里挑最有利结果，正式实验预先声明两个 primary endpoints：

1. `Complete Evidence Set Recall`（固定 context token budget）；
2. `Joint Success`。

其余指标是诊断性 secondary endpoints。若对 T0/T1/C1/G1/G2/G3 做大量两两显著性检验，使用 Holm 校正；主 A/B `C1 vs T1` 保持预注册的单一核心比较。

## 7. 最终验收协议

### Gate A：Gold 数据质量

- 10 本均覆盖；240 题配额完成；40 dev / 200 frozen test。
- 100% 可回答题有 `required_facts` 和至少一个完整 `evidence_set`。
- 100% 不可回答题经过语料检索和人工复核，确认没有完整答案。
- 至少 20% 双人复核；冲突已 adjudicate。
- annotator 不看任何实验组输出。

### Gate B：A/B 变量隔离

- T0、T1、C1 使用同一 10 本版本，且全部映射到共同 evidence atoms。
- 检索、rerank、生成、Prompt、硬件和 token 预算配置哈希一致。
- C1 是 10 本统一重编结果，不混用早先 2 本探针。
- 测试集运行前锁定 commit、模型 checkpoint 和配置 manifest。

### Gate C：结果完整

- 检索、correctness、faithfulness、citation、refusal、Joint Success、延迟/token 全部输出逐题明细。
- 每个主指标有 paired 95% CI；同时有逐书 macro 和总体 micro。
- Judge 分数带人工校准结果；同模型自评不作为唯一证据。
- 所有失败题可回到 Article、页码、evidence atom、retrieved unit 和最终答案。

### 对外报告模板

```text
在冻结的 200 题公开区域经济报告测试集上，C1 相对强结构化基线 T1：
Complete Evidence Set Recall 提升 X.X 个百分点（paired 95% CI [L, U]）；
Joint Success 提升 Y.Y 个百分点（paired 95% CI [L, U]）。
两组使用相同检索器、reranker、生成模型、Prompt 和 context token 预算。
答案正确性、引用完整性、拒答与性能指标分别报告，不以 LLM 自评分替代人工金标准。
```

如果区间包含 0，则把“提升”改为“样本内点估计差异”，不得写成已证实增益。

## 一手来源清单

- Thakur et al., [BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models](https://arxiv.org/abs/2104.08663), NeurIPS 2021；[官方仓库](https://github.com/beir-cellar/beir)。
- Yang et al., [HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering](https://aclanthology.org/D18-1259/), EMNLP 2018。
- Petroni et al., [KILT: a Benchmark for Knowledge Intensive Language Tasks](https://aclanthology.org/2021.naacl-main.200/), NAACL 2021。
- Es et al., [RAGAS: Automated Evaluation of Retrieval Augmented Generation](https://aclanthology.org/2024.eacl-demo.16/), EACL 2024；[官方指标文档](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/)。
- Saad-Falcon et al., [ARES: An Automated Evaluation Framework for Retrieval-Augmented Generation Systems](https://aclanthology.org/2024.naacl-long.20/), NAACL 2024；[官方仓库](https://github.com/stanford-futuredata/ARES)。
- Gao et al., [Enabling Large Language Models to Generate Text with Citations](https://aclanthology.org/2023.emnlp-main.398/), EMNLP 2023；[官方仓库](https://github.com/princeton-nlp/ALCE)。
- Zheng et al., [Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://proceedings.neurips.cc/paper/2023/hash/91f18a1287b398d378ef22505bf41832-Abstract-Datasets_and_Benchmarks.html), NeurIPS 2023。
- Rajpurkar et al., [Know What You Don't Know: Unanswerable Questions for SQuAD](https://aclanthology.org/P18-2124/), ACL 2018。
- Soboroff, [Computing Confidence Intervals for Common IR Measures](https://tsapps.nist.gov/publication/get_pdf.cfm?pub_id=917303), EVIA/NIST 2014。
- [Haystack Evaluation](https://docs.haystack.deepset.ai/docs/evaluation)、[LlamaIndex Evaluation](https://developers.llamaindex.ai/python/framework/module_guides/evaluating/usage_pattern_retrieval/)、[TruLens Evaluation](https://www.trulens.org/component_guides/evaluation/)、[DeepEval RAG Evaluation](https://deepeval.com/docs/getting-started-rag)。
