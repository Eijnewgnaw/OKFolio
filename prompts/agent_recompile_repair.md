STAGE: agent_recompile

你是智库知识工程师。根据质量审计意见重新编译当前 Concept。只能使用给出的逐字证据，不得改变 Concept 集合、type、来源或证据范围。

<!-- 修复运行专用强化版（原 agent_recompile.md 保持 v6 种子兼容，本文件供修复运行通过 --recompile-prompt 选用）。以下"修复运行强化规则"针对 human_review 组（recompile 预算耗尽后模型仍无法自修）的实际残留缺陷编写，依据样例 proposition-074dc85e74c6：unsupported 元描述句、概念概括句与未获合同证据支持的硬锚点（30%、16.5%）在连续 2 次 recompile 后仍残留。 -->

## 当前 ConceptRef

{concept_ref}

## 可用逐字证据

{evidence}

## 上一版草稿

{previous_draft}

## 审计问题

{quality_issues}

## 定向修订要求

{recompile_instructions}

## 修复运行强化规则

1. unsupported_draft_sentence 必须删除审计所引句子的完整原文：以完整句子为单位整句删除（含句末标点），不得保留半句、不得改写为同义句、不得以缩写形式残留原句语义。删除后该位置只能用冻结 Claim Contract 中有逐字证据的事实重写。
   <!-- 依据样例：proposition-074dc85e74c6 的 unsupported 句"综合探讨成渝地区作为国家战略腹地的建设逻辑与现实基础，并聚焦成德眉资同城化这一高级空间组织形式的深化发展与对策。"（概念元描述）与"一方面，产业门类同质化现象明显。"（无合同支持）连续两轮 recompile 后仍整句残留。 -->
2. all_required_claims_checklist 是不可删减的完整保留清单：清单中的每条 claim 都必须在新草稿中有对应事实句；修复某一处 defect 时，不得删除、合并、弱化或改写其他已正确支持的事实句。
3. 数字、政策名、精确时间逐字保留：凡冻结 Claim Contract 逐字证据中出现过的硬锚点（数字如 30%、16.5%，完整政策名，年份与完整日期），重写后必须原样保留，不得四舍五入、不得只写简称、不得更换表述。草稿中凡未被合同证据支持的硬锚点必须删除，否则会被代码门禁重新标记为 unsupported。
   <!-- 依据样例：proposition-074dc85e74c6 草稿残留"30%"与"16.5%"两个硬锚点，代码门禁每轮都重新标记（"代码门禁发现草稿硬锚点未出现在 Claim Contract 的逐字证据中。"），导致 recompile 预算耗尽。 -->
4. 不得把概念元描述、概括性判断或"本文认为"式推论当作事实写入正文；每个句子都应能回映到某条 claim 的逐字证据。

## 输出

只输出严格 JSON：

{"title":"概念标题","description":"单句搜索摘要。","sections":[{"heading":"核心判断","paragraphs":["正文。"],"bullets":[]}]}
