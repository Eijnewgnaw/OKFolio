from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import pytest

from kmpro_wiki.agentwiki.agent_contracts import AgentPolicy
from kmpro_wiki.agentwiki.agentic import (
    AgentCompiler,
    AgentRefRecord,
    AgentRunError,
    discover_agent_concepts,
    discover_from_headings,
    plan_compile_groups,
    plan_source,
    profile_document,
    refine_discovery,
    _attach_agent_ref_provenance,
    _load_source_structure,
)
from kmpro_wiki.agentwiki.config import Settings
from kmpro_wiki.agentwiki.global_cluster import CandidateEdge


class FakeLLM:
    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.prompts: list[str] = []
        self.schemas: list[str | None] = []

    def complete(
        self,
        prompt: str,
        *,
        json_schema_name: str | None = None,
        json_schema: dict | None = None,
    ) -> str:
        self.prompts.append(prompt)
        self.schemas.append(json_schema_name)
        if not self.responses:
            raise AssertionError(f"unexpected LLM call for {json_schema_name}")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def source_plan() -> str:
    return json.dumps(
        {
            "discovery_mode": "heading",
            "refine_discovery": False,
            "asset_policy": "auto",
            "reason": "两个同级标题均有完整正文，可确定性分割。",
        },
        ensure_ascii=False,
    )


def draft(title: str) -> str:
    return json.dumps(
        {
            "title": title,
            "description": f"{title}摘要。",
            "sections": [
                {
                    "heading": "核心判断",
                    "paragraphs": [f"{title}正文包含完整证据。"],
                    "bullets": [],
                }
            ],
        },
        ensure_ascii=False,
    )


def quality(
    score: float,
    decision: str,
    *,
    issues: list[str] | None = None,
    instructions: str = "",
) -> str:
    return json.dumps(
        {
            "score": score,
            "decision": decision,
            "issues": issues or [],
            "recompile_instructions": instructions,
        },
        ensure_ascii=False,
    )


def make_project(tmp_path: Path) -> Settings:
    data = tmp_path / "data"
    sources = data / "sources"
    (sources / "images").mkdir(parents=True)
    prompts = tmp_path / "prompts"
    prompts.mkdir()
    for name in (
        "discover",
        "compile",
        "preserve",
        "agent_plan",
        "agent_refine",
        "agent_group",
        "agent_quality",
        "agent_recompile",
    ):
        shutil.copy(Path("prompts") / f"{name}.md", prompts / f"{name}.md")
    return Settings(
        data_dir=data,
        prompts_dir=prompts,
        openai_base_url="https://api.example/v1",
        openai_api_key="unused",
        openai_model="test-model",
    )


def test_heading_discovery_is_deterministic_and_keeps_source_evidence():
    content = """# 报告

## 一、数字产业结构优化

数字产业增加值持续增长，结构优化形成了独立判断。

## 二、行动建议

建立跨部门协同机制，明确责任主体和实施路径。
"""

    refs = discover_from_headings("报告.md", content, ())

    assert [item.type for item in refs] == ["分析框架", "政策建议"]
    assert refs[0].source == "报告.md"
    assert "数字产业增加值" in "\n".join(refs[0].evidence)


def test_heading_discovery_drops_container_heading_without_body():
    content = """# 报告

## 四、成都因地制宜发展新质生产力的重点举措总体框架

围绕产业、创新、人才等领域提出以下一系列重点举措。

## （一）建设创新平台

建设跨部门创新平台，明确牵头单位和实施路径。

## （二）强化人才支持

完善人才认定和项目支持机制，形成可执行政策。
"""

    refs = discover_from_headings("报告.md", content, ())

    assert [item.title for item in refs] == [
        "建设创新平台",
        "强化人才支持",
    ]
    assert all("总体框架" not in item.title for item in refs)


def test_structure_sidecar_attaches_section_and_page_provenance(tmp_path: Path):
    source = tmp_path / "报告.md"
    source.write_text("# 报告\n\n## 一、发展基础\n\n区域协同水平持续提高。")
    structure = {
        "schema_version": "kmpro.document-structure.v1",
        "status": "complete",
        "pages": [],
        "blocks": [
            {
                "block_id": "blk-body",
                "block_type": "text",
                "content": "区域协同水平持续提高。",
                "page_idx": 8,
                "page_number": 9,
                "heading_path": ["综合篇", "一、发展基础"],
                "evidence_eligible": True,
            }
        ],
    }
    source.with_suffix(".structure.json").write_text(
        json.dumps(structure, ensure_ascii=False),
        encoding="utf-8",
    )
    ref = AgentRefRecord(
        ref_id="ref-1",
        article_id="article-1",
        local_id="发展基础",
        type="分析框架",
        title="区域协同发展基础",
        description="区域协同水平持续提高。",
        evidence=("## 一、发展基础\n\n区域协同水平持续提高。",),
        asset_hints=(),
        source=source.name,
    )

    loaded = _load_source_structure(source)
    enriched = _attach_agent_ref_provenance(ref, loaded)

    assert enriched.section_path == ("综合篇", "一、发展基础")
    assert enriched.page_start == 9
    assert enriched.page_end == 9
    assert enriched.evidence_block_ids == ("blk-body",)


def test_structure_sidecar_blocks_agent_when_page_roles_are_unresolved(
    tmp_path: Path,
):
    source = tmp_path / "报告.md"
    source.write_text("# 报告")
    source.with_suffix(".structure.json").write_text(
        json.dumps(
            {
                "schema_version": "kmpro.document-structure.v1",
                "status": "needs_review",
                "pages": [{"page_number": 7, "role": "content_retry"}],
                "blocks": [],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(AgentRunError, match="pages=\\[7\\]"):
        _load_source_structure(source)


def test_agent_compiler_bounds_compile_parallelism(tmp_path: Path):
    settings = make_project(tmp_path)

    with pytest.raises(ValueError, match="between 1 and 4"):
        AgentCompiler(settings, FakeLLM([]), compile_workers=5)


def test_source_planner_repairs_an_invalid_route_once():
    content = "# 报告\n\n正文不足以按标题切分。"
    profile = profile_document("报告.md", content, ())
    invalid = json.dumps(
        {
            "discovery_mode": "heading",
            "refine_discovery": False,
            "asset_policy": "auto",
            "reason": "错误选择。",
        },
        ensure_ascii=False,
    )
    valid = json.dumps(
        {
            "discovery_mode": "llm",
            "refine_discovery": True,
            "asset_policy": "auto",
            "reason": "标题不足，需要语义发现和复核。",
        },
        ensure_ascii=False,
    )
    llm = FakeLLM([invalid, valid])

    plan = plan_source(
        llm,
        Path("prompts/agent_plan.md").read_text(encoding="utf-8"),
        profile,
    )

    assert plan.discovery_mode == "llm"
    assert len(llm.prompts) == 2
    assert "代码合同" in llm.prompts[1]


def test_discovery_refine_can_replace_heading_candidates_with_audited_refs():
    content = """# 报告

## 运行特征

产业规模增长，但资源配置效率仍然不足。

## 行动建议

建立跨部门资源统筹机制并明确实施责任。
"""
    candidates = discover_from_headings("报告.md", content, ())
    refined_response = json.dumps(
        {
            "concepts": [
                {
                    "id": "产业规模与效率约束",
                    "type": "分析框架",
                    "title": "产业规模增长与资源效率约束",
                    "description": "规模增长同时受到资源配置效率约束。",
                    "evidence": ["evidence-0003"],
                    "asset_hints": [],
                },
                {
                    "id": "跨部门资源统筹",
                    "type": "政策建议",
                    "title": "跨部门资源统筹机制",
                    "description": "通过明确责任建立跨部门资源统筹机制。",
                    "evidence": ["evidence-0005"],
                    "asset_hints": [],
                },
            ]
        },
        ensure_ascii=False,
    )
    llm = FakeLLM([refined_response])

    refs = refine_discovery(
        llm,
        Path("prompts/agent_refine.md").read_text(encoding="utf-8"),
        source_name="报告.md",
        source_content=content,
        assets=(),
        refs=candidates,
    )

    assert [item.concept_id for item in refs] == [
        "产业规模与效率约束",
        "跨部门资源统筹",
    ]
    assert llm.schemas == ["agent_refined_discovery"]


def test_agent_discovery_normalizes_model_ids_without_changing_evidence():
    content = "# 报告\n\n互联网+价格服务形成独立政策建议。"
    response = json.dumps(
        {
            "concepts": [
                {
                    "id": "互联网+价格服务",
                    "type": "政策建议",
                    "title": "互联网+价格服务",
                    "description": "推动互联网价格服务。",
                    "evidence": ["evidence-0002"],
                    "asset_hints": [],
                }
            ]
        },
        ensure_ascii=False,
    )
    decisions: list[dict] = []

    refs = discover_agent_concepts(
        FakeLLM([response]),
        Path("prompts/discover.md").read_text(encoding="utf-8"),
        title="报告",
        source_name="报告.md",
        source_content=content,
        assets=(),
        on_decision=decisions.append,
    )

    assert refs[0].concept_id == "互联网-价格服务"
    assert refs[0].evidence == ("互联网+价格服务形成独立政策建议。",)
    assert decisions[0]["id_corrections"] == [
        {"from": "互联网+价格服务", "to": "互联网-价格服务"}
    ]


def test_agent_run_groups_cross_article_refs_and_recompiles_low_quality(
    tmp_path: Path,
):
    settings = make_project(tmp_path)
    (settings.data_dir / "sources" / "a.md").write_text(
        """# 报告 A

## 数字产业结构优化

数字产业结构优化表现为增加值增长和产业链协同增强。

## 财政赤字负担研判

财政赤字负担需要结合偿债期限进行独立研判。
""",
        encoding="utf-8",
    )
    (settings.data_dir / "sources" / "b.md").write_text(
        """# 报告 B

## 数字产业结构优化

数字产业结构优化还受到技术供给和数据资源配置约束。

## 农业劳动力短缺测算

农业劳动力短缺需要依据年龄结构和就业数据测算。
""",
        encoding="utf-8",
    )
    group_plan = json.dumps(
        {
            "groups": [
                {
                    "ref_ids": [
                        "REF_A",
                        "REF_B",
                    ],
                    "title": "数字产业结构优化的表现与约束",
                    "description": "综合产业表现与资源约束形成联合判断。",
                    "reason": "两篇报告分别提供表现和约束证据。",
                }
            ]
        },
        ensure_ascii=False,
    )

    class DynamicFakeLLM(FakeLLM):
        def complete(self, prompt: str, **kwargs) -> str:
            schema_name = kwargs.get("json_schema_name")
            if schema_name == "agent_compile_groups":
                schema = kwargs["json_schema"]
                allowed = schema["properties"]["groups"]["items"]["properties"][
                    "ref_ids"
                ]["items"]["enum"]
                payload = json.loads(group_plan.replace("REF_A", allowed[0]).replace("REF_B", allowed[1]))
                response = json.dumps(payload, ensure_ascii=False)
                self.prompts.append(prompt)
                self.schemas.append(schema_name)
                return response
            return super().complete(prompt, **kwargs)

    llm = DynamicFakeLLM(
        [
            source_plan(),
            source_plan(),
            draft("第一版"),
            quality(
                0.62,
                "recompile",
                issues=["跨来源证据没有融合"],
                instructions="明确区分产业表现和资源约束后形成综合判断。",
            ),
            draft("重编译版"),
            quality(0.91, "pass"),
            draft("单文档概念一"),
            quality(0.88, "pass"),
            draft("单文档概念二"),
            quality(0.87, "pass"),
        ]
    )
    output = settings.data_dir / "agent-runs" / "integration"

    summary = AgentCompiler(
        settings,
        llm,
        policy=AgentPolicy(
            quality_threshold=0.8,
            max_recompile_attempts=2,
        ),
    ).run(output)

    assert summary.status == "complete"
    assert summary.articles == 2
    assert summary.refs == 4
    assert summary.groups == 3
    assert summary.concepts == 3
    assert summary.recompiles == 1
    assert len(list((output / "concepts").glob("*.md"))) == 3
    assert not list((output / "drafts").glob("*.md"))
    trace = json.loads((output / "agent_trace.json").read_text(encoding="utf-8"))
    assert any(item["stage"] == "compile_group_plan" for item in trace["events"])
    assert any(item["stage"] == "recompile" for item in trace["events"])


def test_group_planner_demotes_repeated_same_article_joint_group():
    refs = (
        AgentRefRecord(
            ref_id="r1",
            article_id="a",
            local_id="one",
            type="分析框架",
            title="同文档概念一",
            description="摘要一。",
            evidence=("证据一。",),
            asset_hints=(),
            source="a.md",
        ),
        AgentRefRecord(
            ref_id="r2",
            article_id="a",
            local_id="two",
            type="分析框架",
            title="同文档概念二",
            description="摘要二。",
            evidence=("证据二。",),
            asset_hints=(),
            source="a.md",
        ),
        AgentRefRecord(
            ref_id="r3",
            article_id="b",
            local_id="three",
            type="分析框架",
            title="跨文档概念",
            description="摘要三。",
            evidence=("证据三。",),
            asset_hints=(),
            source="b.md",
        ),
    )
    candidates = (
        CandidateEdge("e1", "r1", "r3", {"score": 0.5}),
        CandidateEdge("e2", "r2", "r3", {"score": 0.4}),
    )
    invalid = json.dumps(
        {
            "groups": [
                {
                    "ref_ids": ["r1", "r2"],
                    "title": "错误的同源联合概念",
                    "description": "不应跨同一篇文章联合。",
                    "reason": "模型误判。",
                },
            ]
        },
        ensure_ascii=False,
    )
    llm = FakeLLM([invalid, invalid])
    decisions: list[dict] = []

    groups = plan_compile_groups(
        llm,
        "Refs: {ref_cards}\nEdges: {candidate_edges}",
        refs,
        candidates,
        on_decision=decisions.append,
    )

    assert [group.ref_ids for group in groups] == [("r1",), ("r2",), ("r3",)]
    assert llm.schemas == ["agent_compile_groups", "agent_compile_groups"]
    assert decisions[0]["contract_recovery"] == {
        "mode": "demote_invalid_joint_groups",
        "ref_ids": ["r1", "r2", "r3"],
    }


def test_asset_review_policy_withholds_drafts_without_losing_the_asset(
    tmp_path: Path,
):
    settings = make_project(tmp_path)
    (settings.data_dir / "sources" / "images" / "chart.png").write_bytes(
        b"chart-bytes"
    )
    (settings.data_dir / "sources" / "asset.md").write_text(
        """# 图表报告

## 产业运行特征

产业运行指标形成了完整判断，图表展示了年度变化情况。

![年度变化](images/chart.png)

## 政策行动安排

建立政策行动台账并明确责任部门和实施时间。
""",
        encoding="utf-8",
    )
    review_plan = json.dumps(
        {
            "discovery_mode": "heading",
            "refine_discovery": False,
            "asset_policy": "human_review",
            "reason": "图表可能同时支持运行判断和政策安排，需要人工确认归属。",
        },
        ensure_ascii=False,
    )
    llm = FakeLLM(
        [
            review_plan,
            draft("资产概念一"),
            quality(0.90, "pass"),
            draft("资产概念二"),
            quality(0.89, "pass"),
        ]
    )
    output = settings.data_dir / "agent-runs" / "asset-review"

    summary = AgentCompiler(settings, llm).run(output)

    assert summary.status == "needs_review"
    assert summary.reviews == 1
    assert summary.concepts == 0
    assert len(list((output / "drafts").glob("*.md"))) == 2
    assert (output / "images" / "chart.png").read_bytes() == b"chart-bytes"
    review = json.loads(
        (output / "review_queue.json").read_text(encoding="utf-8")
    )["reviews"][0]
    assert review["asset_id"] == "image-001"
    assert len(review["candidate_group_ids"]) == 2


def test_auto_asset_policy_reuses_v13_placement_and_byte_preservation(
    tmp_path: Path,
):
    settings = make_project(tmp_path)
    (settings.data_dir / "sources" / "images" / "chart.png").write_bytes(
        b"chart-bytes"
    )
    (settings.data_dir / "sources" / "asset.md").write_text(
        """# 自动图表报告

## 产业运行特征

产业运行指标形成了完整判断，图表展示了年度变化情况。

![年度变化](images/chart.png)

## 政策行动安排

建立政策行动台账并明确责任部门和实施时间。
""",
        encoding="utf-8",
    )

    class AssetFakeLLM(FakeLLM):
        def complete(self, prompt: str, **kwargs) -> str:
            schema_name = kwargs.get("json_schema_name")
            if schema_name == "asset_placements":
                concept_id = re.search(
                    r'"concept_id": "([^"]+)"', prompt
                ).group(1)
                response = json.dumps(
                    {
                        "placements": [
                            {
                                "asset_id": "image-001",
                                "concept_id": concept_id,
                                "anchor_id": "anchor-001",
                                "position": "after",
                                "reason": "图表直接支持该概念。",
                            }
                        ]
                    },
                    ensure_ascii=False,
                )
                self.prompts.append(prompt)
                self.schemas.append(schema_name)
                return response
            return super().complete(prompt, **kwargs)

    llm = AssetFakeLLM(
        [
            source_plan(),
            draft("自动资产概念一"),
            quality(0.90, "pass"),
            draft("自动资产概念二"),
            quality(0.89, "pass"),
        ]
    )
    output = settings.data_dir / "agent-runs" / "asset-auto"

    summary = AgentCompiler(settings, llm).run(output)

    assert summary.status == "complete"
    contents = [
        path.read_text(encoding="utf-8")
        for path in (output / "concepts").glob("*.md")
    ]
    assert sum("](../images/chart.png)" in item for item in contents) == 1
    assert (output / "images" / "chart.png").read_bytes() == b"chart-bytes"


def test_failed_agent_run_resumes_without_repeating_completed_stages(
    tmp_path: Path,
):
    settings = make_project(tmp_path)
    (settings.data_dir / "sources" / "resume.md").write_text(
        """# 恢复测试

## 产业运行判断

产业运行形成了一个内容充分并且可以独立引用的分析判断。

## 政策实施建议

建立实施台账并明确责任主体形成了另一项独立政策建议。
""",
        encoding="utf-8",
    )
    output = settings.data_dir / "agent-runs" / "resume"
    first = FakeLLM(
        [
            source_plan(),
            draft("已完成概念"),
            quality(0.90, "pass"),
            RuntimeError("temporary model failure"),
        ]
    )

    with pytest.raises(RuntimeError, match="temporary"):
        AgentCompiler(settings, first).run(output)

    manifest = json.loads(
        (output / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["status"] == "failed"
    assert manifest["model"] == settings.openai_model

    resumed = FakeLLM([draft("恢复后概念"), quality(0.91, "pass")])
    summary = AgentCompiler(settings, resumed).run(output, resume=True)

    assert summary.status == "complete"
    assert len(resumed.prompts) == 2
    assert "agent_source_plan" not in resumed.schemas
    assert len(list((output / "concepts").glob("*.md"))) == 2
    trace = json.loads((output / "agent_trace.json").read_text(encoding="utf-8"))
    assert any(
        item.get("reused") == "plan_and_discovery"
        for item in trace["events"]
    )
    assert any(
        item.get("reused") == "compile_and_quality"
        for item in trace["events"]
    )


def test_agent_refuses_to_write_into_v13_output_directories(tmp_path: Path):
    settings = make_project(tmp_path)
    compiler = AgentCompiler(settings, FakeLLM([]))

    with pytest.raises(AgentRunError, match="must stay under"):
        compiler.run(settings.data_dir / "wiki" / "agent-output")
