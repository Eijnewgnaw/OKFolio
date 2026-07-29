from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest

from kmpro_wiki.mcp.service import MCPConfig, WikiMCPService


def make_config(
    tmp_path: Path,
    *,
    enable_writes: bool,
) -> MCPConfig:
    return MCPConfig(
        project_root=Path(__file__).parents[1],
        data_dir=tmp_path / "data",
        prompts_dir=Path(__file__).parents[1] / "prompts",
        releases_dir=tmp_path / "releases",
        active_release_dir=None,
        enable_writes=enable_writes,
        enable_docker=False,
    )


def make_release(config: MCPConfig) -> Path:
    release = config.releases_dir / "v-test"
    concepts = release / "data/wiki/concepts"
    articles = release / "data/wiki/articles"
    provenance = release / "data/provenance"
    outputs = release / "data/outputs"
    for directory in (concepts, articles, provenance, outputs):
        directory.mkdir(parents=True, exist_ok=True)
    (concepts / "topic-001.md").write_text(
        """---
type: 分析框架
title: 低空经济发展框架
description: 从基础、问题和政策路径分析低空经济。
source: 报告.md
concept_refs:
- ref-001
articles:
- article-001
relation_count: 1
---
## 核心判断

成都应完善低空经济基础设施和协同机制。
""",
        encoding="utf-8",
    )
    (concepts / "proposition-002.md").write_text(
        """---
type: 政策建议
title: 完善低空基础设施
description: 建设协同运行平台。
source: 报告二.md
concept_refs:
- ref-002
articles:
- article-002
relation_count: 1
---
## 建议

建设跨部门协同运行平台。
""",
        encoding="utf-8",
    )
    (articles / "article-001.md").write_text(
        """---
title: 低空经济报告
source: 报告.md
article_id: article-001
concept_refs:
- ref-001
---
# 低空经济报告

原文证据。
""",
        encoding="utf-8",
    )
    (articles / "article-002.md").write_text(
        """---
title: 基础设施报告
source: 报告二.md
article_id: article-002
concept_refs:
- ref-002
---
# 基础设施报告

原文证据二。
""",
        encoding="utf-8",
    )
    (provenance / "refs.json").write_text(
        json.dumps(
            {
                "refs": [
                    {
                        "ref_id": "ref-001",
                        "article_id": "article-001",
                        "source": "报告.md",
                        "title": "低空经济基础",
                        "type": "分析框架",
                        "evidence": ["原文证据。"],
                    },
                    {
                        "ref_id": "ref-002",
                        "article_id": "article-002",
                        "source": "报告二.md",
                        "title": "低空基础设施建议",
                        "type": "政策建议",
                        "evidence": ["原文证据二。"],
                    },
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (provenance / "groups.json").write_text(
        json.dumps(
            {
                "groups": [
                    {
                        "group_id": "topic-001",
                        "title": "低空经济发展框架",
                        "ref_ids": ["ref-001"],
                    },
                    {
                        "group_id": "proposition-002",
                        "title": "完善低空基础设施",
                        "ref_ids": ["ref-002"],
                    },
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (provenance / "relations.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "judgements": [
                    {
                        "left_ref_id": "ref-001",
                        "right_ref_id": "ref-002",
                        "decision": "related",
                        "reason": "发展框架约束基础设施建设。",
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (outputs / "graph.html").write_text(
        "<!doctype html><title>graph</title>",
        encoding="utf-8",
    )
    (release / "release-manifest.json").write_text(
        json.dumps({"status": "complete", "concepts": 2}),
        encoding="utf-8",
    )
    (release / "acceptance.json").write_text(
        json.dumps({"status": "pass"}),
        encoding="utf-8",
    )
    return release


def test_read_only_query_and_full_provenance(tmp_path: Path):
    config = make_config(tmp_path, enable_writes=False)
    make_release(config)
    service = WikiMCPService(config)

    results = service.search_concepts("低空 经济")
    assert results["release"] == "v-test"
    assert results["total_matches"] == 1
    assert results["results"][0]["concept_id"] == "topic-001"

    trace = service.trace_concept("topic-001")
    assert trace["concept_refs"][0]["ref_id"] == "ref-001"
    assert trace["articles"][0]["article_id"] == "article-001"
    assert trace["related_concepts"][0]["concept_id"] == "proposition-002"
    assert trace["provenance"] == "Concept -> ConceptRef -> Article"


def test_write_gate_and_markdown_ingestion(tmp_path: Path):
    readonly = WikiMCPService(make_config(tmp_path, enable_writes=False))
    with pytest.raises(PermissionError, match="MCP_ENABLE_WRITES"):
        readonly.ingest_markdown("报告.md", "# 报告\n\n正文")

    writable = WikiMCPService(make_config(tmp_path, enable_writes=True))
    result = writable.ingest_markdown("报告.md", "# 报告\n\n正文")
    assert result["activated"] is True
    assert (
        writable.config.data_dir / "sources" / "报告.md"
    ).read_text(encoding="utf-8") == "# 报告\n\n正文"
    with pytest.raises(ValueError, match="safe .md"):
        writable.ingest_markdown("../越界.md", "# 越界")


def test_background_job_persists_status_and_log(tmp_path: Path):
    service = WikiMCPService(make_config(tmp_path, enable_writes=True))
    job = service.jobs.start(
        "test",
        [[sys.executable, "-c", "print('job-ok')"]],
    )
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        current = service.get_job(job["job_id"])
        if current["status"] in {"complete", "failed"}:
            break
        time.sleep(0.05)
    assert current["status"] == "complete"
    assert current["exit_code"] == 0
    assert "job-ok" in current["log_tail"]


def test_pdf_processing_surface_reuses_existing_mineru_output(
    tmp_path: Path,
) -> None:
    service = WikiMCPService(make_config(tmp_path, enable_writes=True))
    inbox = service.config.data_dir / "inbox"
    parsed = service.config.data_dir / "mineru-output" / "报告"
    inbox.mkdir(parents=True)
    parsed.mkdir(parents=True)
    (inbox / "报告.pdf").write_bytes(b"%PDF fixture")
    (parsed / "报告_content_list.json").write_text(
        json.dumps(
            [{"type": "text", "text": "正文", "page_idx": 0}],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    listed = service.list_pdfs()
    assert listed["pdfs"][0]["mineru_output_ready"] is True

    job = service.start_pdf_processing("报告.pdf")
    record = service.get_job(job["job_id"])
    command = record["commands"][0]
    assert "--skip-mineru" in command
    assert "process_pdf.py" in " ".join(command)


def test_pdf_processing_rejects_path_traversal(tmp_path: Path) -> None:
    service = WikiMCPService(make_config(tmp_path, enable_writes=True))
    with pytest.raises(ValueError, match="safe .pdf"):
        service.start_pdf_processing("../报告.pdf")
