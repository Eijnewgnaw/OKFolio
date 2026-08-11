# AgentWiki 模块

职责：把已经完成文档解析的 Article 编译为可溯源知识资产。

核心链路：

```text
Article / ArticleSegment
  → Discovery / Refine
  → ConceptRef
  → 跨文档候选与分组
  → Concept compile
  → 质量审计与自动重编译
  → Bundle / Graph / Site
```

实现就在本目录；`scripts/` 只保留命令行适配。公开入口是 `AgentCompiler`
和 `Compiler`。
