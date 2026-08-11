# LightRAG 与 OKFolio：文档更新策略对比

## 结论

LightRAG 面向在线检索，使用持久化的文档状态、文本块、向量索引和知识图谱；文档可异步进入队列，按文档 ID 删除时会重建仍被其他文档引用的实体、关系和向量。OKFolio 面向可审计的离线知识编译：PDF 用内容 SHA-256 作为解析作业身份，AgentWiki 用源文档哈希和 Concept 分组哈希恢复，最后以完整 Run 为边界发布不可变 Bundle。

两者不是同一个更新模型：LightRAG 是“持续修改一个在线索引”，OKFolio 是“生成下一版经过审计的知识快照”。OKFolio 当前已具备局部重用能力，但尚未把新增、修改、删除统一为显式的文档生命周期事件。

## LightRAG 的官方更新机制

- `insert` 支持单文档、批量文档和显式 ID；后台 pipeline 可以排队并增量处理文档。
- 每个文档在入队时保存 chunk 配置快照；后续修改分块配置只影响新入队文档，旧文档需要重新处理。
- 按文档 ID 删除是异步的：删除文档 chunks，移除仅由该文档贡献的实体/关系，重建仍被其他文档使用的实体/关系，更新向量索引并清理状态。
- 普通上传可以在处理循环运行时继续入队；清除/删除等破坏性操作和扫描分类阶段会拒绝并发入队，以保持存储一致性。
- 文档状态有 pending/processing/processed/failed 等阶段，并通过 Track ID 查询进度；失败文件可重新处理。

官方参考：

- <https://github.com/HKUDS/LightRAG/blob/main/docs/ProgramingWithCore.md>
- <https://github.com/HKUDS/LightRAG/blob/main/docs/LightRAG-API-Server.md>

## OKFolio 当前机制

1. PDF 阶段：`scripts/process_pdf_corpus.py` 以 PDF 内容 SHA-256 建立 `parser-jobs/<sha前20位>` 和 `processed/<sha前20位>`。相同字节且页级结果完整时复用；字节改变会产生新 SHA 和新解析作业。
2. 规范化阶段：`scripts/normalize_pdf_corpus.py` 将完成的 Document IR 规范化为 Article、segments 和 structure manifest，再通过 activation gate 替换当前可见 Article。
3. AgentWiki 阶段：`okfolio/agentwiki/agentic.py` 为每个当前 Article 计算 `source_hash`；相同哈希复用 plan/discovery 和 ConceptRef。编译阶段为每个 Concept 分组计算 group hash，只重编译哈希变化的分组。
4. 发布阶段：`_publish` 会重建当前 Run 的 `concepts/` 和 `drafts/`，随后关系判断、审计和发布脚本生成正式 Bundle、图谱和站点。
5. 当前缺口：PDF 目录扫描没有显式的删除/tombstone 事件；`sync_inbox` 只复制新增或修改文件，不清理已从 inbox 移除的 Article/图片；同一 Agent Run 的旧 source_progress/图片目录没有单独的文档生命周期历史。因此不能把“文件消失”自动等价为“知识删除”。

## 建议的统一更新协议

增加一个文档生命周期清单，稳定的 `document_id` 不随内容修改改变，另存 `version`、`content_hash`、`parser_options_hash`、`supersedes`、`status`（active/changed/deleted/failed）和 `run_id`。每次更新先构造下一版快照，不直接覆盖上一次 accepted Bundle。

- unchanged：复用 Article、ConceptRef、受影响范围外的 Concept 和资产。
- added：只对新 Article 做解析/发现，再与现有 Ref 做候选召回；受影响 Concept 和关系重新计算。
- changed：同一 `document_id` 生成新版本；仅该 Article 的 Ref 失效，按 group hash 重编译受影响 Concept，并重判触及这些 Concept 的关系。
- deleted：写入 tombstone，移除该文档的 Ref；没有剩余 Ref 的 Concept 下线；共享 Concept 重新编译；关系图重新发布。
- failed：保留上一版 accepted Bundle 可查询，失败版本只进入 pending/review，不污染线上结果。

最后以 `run complete + relations complete + audit pass` 作为原子提升条件，将 `current` 指向新 Bundle。这样保留 OKFolio 的证据链和可复现性，同时获得类似 LightRAG 的新增、修改、删除增量语义。
