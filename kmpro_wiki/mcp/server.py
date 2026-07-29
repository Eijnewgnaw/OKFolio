#!/usr/bin/env python3
"""Official FastMCP adapter for the complete OKFolio capability set."""
from __future__ import annotations

import argparse
import json
import os
from typing import Any

from mcp import types
from mcp.server.fastmcp import FastMCP

from .service import MCPConfig, WikiMCPService


READ_ONLY = types.ToolAnnotations(
    readOnlyHint=True,
    destructiveHint=False,
    idempotentHint=True,
    openWorldHint=False,
)
LOCAL_WRITE = types.ToolAnnotations(
    readOnlyHint=False,
    destructiveHint=False,
    idempotentHint=False,
    openWorldHint=False,
)
REPLACE_WRITE = types.ToolAnnotations(
    readOnlyHint=False,
    destructiveHint=True,
    idempotentHint=False,
    openWorldHint=False,
)
MODEL_JOB = types.ToolAnnotations(
    readOnlyHint=False,
    destructiveHint=False,
    idempotentHint=False,
    openWorldHint=True,
)


def create_server(
    service: WikiMCPService,
    *,
    host: str,
    port: int,
) -> FastMCP:
    server = FastMCP(
        name="OKFolio",
        instructions=(
            "将智库文档编译为可追溯 OKF Bundle。长任务先启动后台 job，"
            "随后用 get_job 轮询；正式发布顺序为 Agent 编译、运行审计、"
            "关系判断、再次审计、发布、发布审计。"
        ),
        host=host,
        port=port,
        stateless_http=True,
        json_response=True,
    )

    @server.tool(annotations=READ_ONLY, structured_output=True)
    def get_capabilities() -> dict[str, Any]:
        """返回完整能力、流水线阶段和当前安全门禁。"""
        return service.capabilities()

    @server.tool(annotations=READ_ONLY, structured_output=True)
    def get_system_status() -> dict[str, Any]:
        """返回语料、Agent 运行、正式 Release 和后台任务状态。"""
        return service.system_status()

    @server.tool(annotations=READ_ONLY, structured_output=True)
    def list_sources() -> dict[str, Any]:
        """列出已激活的源 Markdown 及其校验和。"""
        return service.list_sources()

    @server.tool(annotations=READ_ONLY, structured_output=True)
    def list_pdfs() -> dict[str, Any]:
        """列出运行时 inbox 中的 PDF、MinerU 结果和处理状态。"""
        return service.list_pdfs()

    @server.tool(annotations=LOCAL_WRITE, structured_output=True)
    def ingest_markdown(
        filename: str,
        content: str,
        replace: bool = False,
        activate: bool = True,
    ) -> dict[str, Any]:
        """导入一篇 Markdown；可立即同步到 Agent 语料目录。"""
        return service.ingest_markdown(
            filename,
            content,
            replace=replace,
            activate=activate,
        )

    @server.tool(annotations=LOCAL_WRITE, structured_output=True)
    def ingest_asset(
        filename: str,
        content_base64: str,
        replace: bool = False,
        activate: bool = True,
    ) -> dict[str, Any]:
        """以 base64 导入源图片，并保持其相对路径。"""
        return service.ingest_asset(
            filename,
            content_base64,
            replace=replace,
            activate=activate,
        )

    @server.tool(annotations=LOCAL_WRITE, structured_output=True)
    def sync_inbox() -> dict[str, Any]:
        """把 inbox 中的新 Markdown/图片原子同步到 sources。"""
        return service.sync_inbox()

    @server.tool(annotations=MODEL_JOB, structured_output=True)
    def start_pdf_processing(
        filename: str,
        backend: str = "pipeline",
        target_chars: int = 12_000,
        hard_max_chars: int = 24_000,
        reuse_mineru_output: bool = True,
    ) -> dict[str, Any]:
        """后台执行 MinerU、Document IR、语义分段和 Article 激活。"""
        return service.start_pdf_processing(
            filename,
            backend=backend,
            target_chars=target_chars,
            hard_max_chars=hard_max_chars,
            reuse_mineru_output=reuse_mineru_output,
        )

    @server.tool(annotations=READ_ONLY, structured_output=True)
    def get_processed_document(document_name: str) -> dict[str, Any]:
        """读取一个 PDF 处理结果的清单、Segment 和资产统计。"""
        return service.get_processed_document(document_name)

    @server.tool(annotations=MODEL_JOB, structured_output=True)
    def start_incremental_compile(
        sync_before_run: bool = True,
    ) -> dict[str, Any]:
        """后台启动经典 A-D 单文档增量编译。"""
        return service.start_incremental_compile(
            sync_inbox=sync_before_run
        )

    @server.tool(annotations=MODEL_JOB, structured_output=True)
    def start_agent_compile(
        run_id: str,
        resume: bool = False,
        sync_before_run: bool = True,
        quality_threshold: float = 0.82,
        max_recompile_attempts: int = 2,
        max_component_refs: int = 24,
    ) -> dict[str, Any]:
        """后台启动全库 Agent 编译、聚类、质量审计和自动重编译。"""
        return service.start_agent_compile(
            run_id,
            resume=resume,
            sync_inbox=sync_before_run,
            quality_threshold=quality_threshold,
            max_recompile_attempts=max_recompile_attempts,
            max_component_refs=max_component_refs,
        )

    @server.tool(annotations=MODEL_JOB, structured_output=True)
    def start_relation_judgement(
        run_id: str,
        resume: bool = False,
        batch_size: int = 16,
    ) -> dict[str, Any]:
        """后台判断跨 Concept 语义关系，输出可解释关系证据。"""
        return service.start_relation_judgement(
            run_id,
            resume=resume,
            batch_size=batch_size,
        )

    @server.tool(annotations=LOCAL_WRITE, structured_output=True)
    def audit_agent_run(run_id: str) -> dict[str, Any]:
        """确定性审计 Ref 覆盖、质量、来源及资产守恒。"""
        return service.audit_agent_run(run_id)

    @server.tool(annotations=LOCAL_WRITE, structured_output=True)
    def report_agent_run(run_id: str) -> dict[str, Any]:
        """汇总模型调用、Token、耗时、路由及质量指标。"""
        return service.report_agent_run(run_id)

    @server.tool(annotations=REPLACE_WRITE, structured_output=True)
    def publish_release(
        run_id: str,
        release_name: str,
        version: str,
        replace: bool = False,
        confirm_replace: str = "",
    ) -> dict[str, Any]:
        """后台发布正式 Bundle、溯源数据、图谱和静态站点。"""
        return service.publish_release(
            run_id,
            release_name,
            version=version,
            replace=replace,
            confirm_replace=confirm_replace,
        )

    @server.tool(annotations=LOCAL_WRITE, structured_output=True)
    def audit_release(release_name: str = "") -> dict[str, Any]:
        """对正式 Bundle、图谱、站点和校验清单执行发布验收。"""
        return service.audit_release(release_name)

    @server.tool(annotations=MODEL_JOB, structured_output=True)
    def build_release_image(
        release_name: str,
        image_tag: str,
        export_tar: bool = True,
    ) -> dict[str, Any]:
        """后台构建 linux/amd64 离线镜像；需显式启用 Docker 门禁。"""
        return service.build_release_image(
            release_name,
            image_tag=image_tag,
            export_tar=export_tar,
        )

    @server.tool(annotations=LOCAL_WRITE, structured_output=True)
    def package_release(
        release_name: str,
        archive_name: str = "",
    ) -> dict[str, Any]:
        """后台封装源码、Bundle、图谱、镜像和 SHA-256 清单。"""
        return service.package_release(
            release_name,
            archive_name=archive_name,
        )

    @server.tool(annotations=READ_ONLY, structured_output=True)
    def list_jobs(limit: int = 30) -> dict[str, Any]:
        """列出最近的 MCP 后台任务。"""
        return service.list_jobs(limit=limit)

    @server.tool(annotations=READ_ONLY, structured_output=True)
    def get_job(job_id: str) -> dict[str, Any]:
        """获取后台任务状态和末尾日志。"""
        return service.get_job(job_id)

    @server.tool(annotations=READ_ONLY, structured_output=True)
    def get_job_log(
        job_id: str,
        max_chars: int = 20_000,
    ) -> dict[str, Any]:
        """读取后台任务日志末尾，不返回模型密钥。"""
        return service.get_job_log(job_id, max_chars=max_chars)

    @server.tool(annotations=READ_ONLY, structured_output=True)
    def search_concepts(
        query: str = "",
        concept_type: str = "",
        source: str = "",
        limit: int = 20,
        release_name: str = "",
    ) -> dict[str, Any]:
        """按标题、摘要、来源和正文检索 Concept。"""
        return service.search_concepts(
            query,
            concept_type=concept_type,
            source=source,
            limit=limit,
            release_name=release_name,
        )

    @server.tool(annotations=READ_ONLY, structured_output=True)
    def get_concept(
        concept_id: str,
        release_name: str = "",
    ) -> dict[str, Any]:
        """读取一个 Concept 的 frontmatter、正文和完整 Markdown。"""
        return service.get_concept(concept_id, release_name=release_name)

    @server.tool(annotations=READ_ONLY, structured_output=True)
    def trace_concept(
        concept_id: str,
        include_evidence: bool = True,
        release_name: str = "",
    ) -> dict[str, Any]:
        """展开 Concept→ConceptRef→Article 和关联 Concept 证据链。"""
        return service.trace_concept(
            concept_id,
            include_evidence=include_evidence,
            release_name=release_name,
        )

    @server.tool(annotations=READ_ONLY, structured_output=True)
    def get_article(
        article_id: str,
        release_name: str = "",
    ) -> dict[str, Any]:
        """读取原始 Article 页面及其 ConceptRef 元数据。"""
        return service.get_article(article_id, release_name=release_name)

    @server.tool(annotations=READ_ONLY, structured_output=True)
    def get_graph(release_name: str = "") -> dict[str, Any]:
        """返回三维图谱文件信息和展示模式。"""
        return service.graph_info(release_name)

    @server.resource(
        "kmpro://status",
        name="OKFolio status",
        description="当前知识库、运行和发布状态",
        mime_type="application/json",
    )
    def status_resource() -> str:
        return json.dumps(
            service.system_status(),
            ensure_ascii=False,
            indent=2,
        )

    @server.resource(
        "kmpro://concept/{concept_id}",
        name="KMPro Concept",
        description="当前正式 Release 中的 Concept Markdown",
        mime_type="text/markdown",
    )
    def concept_resource(concept_id: str) -> str:
        return service.concept_markdown_resource(concept_id)

    @server.resource(
        "kmpro://article/{article_id}",
        name="KMPro Article",
        description="当前正式 Release 中的原始 Article Markdown",
        mime_type="text/markdown",
    )
    def article_resource(article_id: str) -> str:
        return service.article_markdown_resource(article_id)

    @server.resource(
        "kmpro://graph",
        name="KMPro knowledge graph",
        description="当前正式 Release 的离线三维图谱",
        mime_type="text/html",
    )
    def graph_resource() -> str:
        return service.graph_html()

    @server.prompt(
        name="compile_corpus",
        description="指导 Agent 完成一轮可发布的全库知识编译",
    )
    def compile_corpus_prompt(run_id: str, release_name: str) -> str:
        return f"""请完成 OKFolio 全库编译：
1. 调用 get_system_status 和 list_sources 检查语料。
2. 调用 start_agent_compile(run_id="{run_id}")，用 get_job 轮询至完成。
3. 调用 audit_agent_run 和 report_agent_run；审计不通过时停止发布。
4. 调用 start_relation_judgement(run_id="{run_id}") 并轮询完成。
5. 再次调用 audit_agent_run，确认关系结果未破坏 Ref 全覆盖。
6. 调用 publish_release(run_id="{run_id}", release_name="{release_name}",
   version="{release_name}")，轮询完成后调用 audit_release。
7. 汇报 Article、ConceptRef、Concept、联合 Concept、关系、Token、耗时和限制。
不要把运行中、失败或未审计的产物描述为正式结果。"""

    @server.prompt(
        name="research_topic",
        description="从正式知识库检索主题并逐条溯源",
    )
    def research_topic_prompt(topic: str) -> str:
        return f"""围绕“{topic}”检索 OKFolio：
1. 先调用 search_concepts，必要时按 type/source 缩小范围；
2. 对关键结果调用 get_concept；
3. 对需要引用的判断调用 trace_concept，核对 ConceptRef 与 Article；
4. 明确区分 Concept 综合判断、Ref 证据和原始 Article；
5. 输出结论时附 Concept ID 与 Article ID，不编造未出现的事实。"""

    return server


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the OKFolio MCP server")
    parser.add_argument(
        "--transport",
        choices=("stdio", "streamable-http"),
        default=os.environ.get("MCP_TRANSPORT", "stdio"),
    )
    parser.add_argument("--host", default=os.environ.get("MCP_HOST", "127.0.0.1"))
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("MCP_PORT", "3099")),
    )
    args = parser.parse_args()
    service = WikiMCPService(MCPConfig.from_env())
    server = create_server(service, host=args.host, port=args.port)
    server.run(transport=args.transport)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
