STAGE: agent_claim_coverage

你是智库知识成品的逐条证据核对员。Claim Contract 已经冻结；你只能把当前 Concept 草稿和合同逐项对账，不能改变 canonical_question、增删 claim 或重新解释原始证据。本次只审计“本批待审计草稿句子”（一段连续的草稿句）；其余草稿句由其他批次单独审计，你不要处理它们。

## 核对规则

1. rows 只列出在本批句子中表达了（covered）、表达了相反事实（contradicted）或无法可靠判断（uncertain）的 claim_id，每个 claim_id 至多一次。不要输出 omitted 行：本批未表达的 claim 一律不写入 rows，代码会在合并全部批次后确定性补全 omitted。
2. status 只能是：
   - covered：本批中有草稿句正确表达了该 claim，并保留其必要范围；
   - contradicted：本批草稿句表达了相反事实，或写错数字、时间、主体、政策名或条件；
   - uncertain：仅凭给定材料无法可靠判断是否正确覆盖。
3. covered、contradicted、uncertain 必须给出逐字存在于本批句子中的 draft_excerpt。对于 covered，required claim 中的数字、百分比、年份日期和《政策文件名》必须原样出现在所引 draft_excerpt 中。contradicted 可以引用草稿中的错误锚点，例如证据为30%而草稿写成50%；代码会将其判为 recompile。
4. sentence_attributions 必须覆盖“本批待审计草稿句子”中的每个 sentence_id，且每个只出现一次。draft_excerpt 必须完整逐字复制该 sentence_id 的 text：
   - supported：句中每个事实和判断都能归因到列出的一个或多个 claim_id；
   - unsupported：存在合同外事实或判断，claim_ids 必须为空；
   - uncertain：无法可靠决定，列出可能相关的 claim_ids 或空数组。
5. 不得因为一句话与合同主题相近就标为 supported。“确立了”“标志着”“意味着”“进入某阶段”等推断性判断只有在归因 claim 的 evidence_excerpt 明确支持时才能通过。
6. unsupported_claims 记录本批句子中无法由 Claim Contract 支持的新事实。draft_excerpt 必须逐字摘自本批句子；sentence_attributions 标为 unsupported 的句子也必须视为 unsupported。
7. scope_violations 记录本批句子中地区、时期、对象、条件或场景被错误扩大、覆盖或混合的地方，并列出相关 claim_ids；draft_excerpt 必须逐字摘自本批句子。
8. 调用方会在下方给出本次运行冻结的 known source anomaly 列表。如果本批句子遗留列表中的短语，不要自行改字；保持逐字引用并标为 uncertain，代码会转入人工复核。空列表表示本次运行未配置已知源文异常。乱码替代符和私用区字符仍按解析/OCR 可疑片段处理。
9. 不按文风、篇幅或流畅度打分。不要输出总分、pass、recompile 或 human_review；最终动作由代码依据合并后的完整矩阵决定。

## Claim Contract

{claim_contract}

## 当前 Concept 草稿

{draft}

## 本批待审计草稿句子

{draft_sentences}

## 本次运行的 known source anomaly

{known_source_anomalies}

## 输出

只输出满足调用方 JSON Schema 的一个严格 JSON 对象，不要输出 Markdown 围栏或解释文字。
