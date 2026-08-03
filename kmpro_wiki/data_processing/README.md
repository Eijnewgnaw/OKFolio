# 数据处理模块

职责：把一个原生 PDF 转换为 AgentWiki 可消费、可恢复、可溯源的 Article。

```text
PDF
  → MinerU 整本解析
  → Block Document IR
  → 标题优先 ArticleSegment
  → 图片经 S3Writer / 本地 Writer 归档
  → Article Markdown + provenance manifests
```

一份 PDF 始终是一篇 Article；执行分片和 Segment 都不会改变 Article 身份。
`S3WriterAssetWriter` 适配任何提供 `write(object_key, bytes)` 的内部 S3Writer，
工厂由 `S3_WRITER_FACTORY=package.module:function` 注入，凭据只从运行环境读取。
原始 Document IR 始终保留；只有结构归一化状态为 `complete` 的 Article 才会
以 Markdown + `structure.json` 成对激活到 `normalized-sources/`。

公开入口：

```bash
python3 scripts/process_pdf.py \
  --pdf runtime/data/inbox/report.pdf \
  --mineru-output runtime/data/mineru-output/report \
  --destination runtime/data/processed/report \
  --activate-dir runtime/data/normalized-sources
```

如 MinerU 已在外部完成，增加 `--skip-mineru`。默认图片写入本地运行卷；远程
对象存储可设置 `DATA_ASSET_MODE=s3writer`。

## OpenAI-compatible vision parsing

不需要在 Worker 容器中部署模型。设置：

```bash
export MINERU_PROVIDER=openai-compatible
export MINERU_BASE_URL=<mineru-compatible-endpoint>
export MINERU_API_KEY=...
export MINERU_MODEL=your-mineru-model-id
```

The default provider sends one page at a time to an OpenAI-compatible
`/chat/completions` endpoint. If a deployment exposes the official MinerU 2.5
two-step protocol, select `mineru-http-client` and provide its dedicated base
URL; the rest of the pipeline remains unchanged.

单页闭环：

```bash
python3 scripts/process_pdf.py \
  --pdf runtime/data/inbox/report.pdf \
  --mineru-output runtime/data/mineru-output/report \
  --destination runtime/data/processed/report \
  --mineru-provider mineru-http-client \
  --page-start 1 \
  --page-end 1
```

Worker 使用 Poppler 逐页渲染，页结果写入 `page-results/`，重启后自动复用已经
完成的页面。完整文档仍生成标准 `content_list.json`、Document IR、Segment、
Article 和资产谱系。

## MinIO

设置 `DATA_ASSET_MODE=minio` 并提供 `S3_ENDPOINT`、`S3_ACCESS_KEY`、
`S3_SECRET_KEY`、`S3_BUCKET`。实现使用 S3 Signature V4，不依赖内部私有包，
因此同一代码可连接 MinIO 或其他 S3 兼容存储。
