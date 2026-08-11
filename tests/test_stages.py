import json

import pytest

from kmpro_wiki.agentwiki.assets import SourceAsset
from kmpro_wiki.agentwiki.contracts import ConceptRef, ContractError, DraftConcept
from kmpro_wiki.agentwiki.llm import LLMOutputTruncated
from kmpro_wiki.agentwiki.okf import ConceptDocument
from kmpro_wiki.agentwiki.stages import (
    PromptRenderError,
    _complete_structured,
    audit_relations,
    build_evidence_catalog,
    compile_concepts,
    discover_concepts,
    infer_discovery_constraints,
    plan_asset_placements,
    render_prompt,
)


class FakeLLM:
    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.prompts: list[str] = []
        self.schemas: list[tuple[str | None, dict | None]] = []

    def complete(
        self,
        prompt: str,
        *,
        json_schema_name: str | None = None,
        json_schema: dict | None = None,
    ) -> str:
        self.prompts.append(prompt)
        self.schemas.append((json_schema_name, json_schema))
        return self.responses.pop(0)


def ref(concept_id: str) -> ConceptRef:
    return ConceptRef(
        concept_id=concept_id,
        type="分析框架",
        title=f"概念 {concept_id}",
        description=f"{concept_id} 摘要。",
        source="报告.md",
        evidence=(f"{concept_id} 证据。",),
        asset_hints=(),
    )


def draft(concept_id: str) -> DraftConcept:
    return DraftConcept(
        ref=ref(concept_id),
        title=f"概念 {concept_id}",
        description=f"{concept_id} 摘要。",
        body=f"{concept_id} 正文。",
    )


def document(concept_id: str) -> ConceptDocument:
    item = draft(concept_id)
    return ConceptDocument(
        f"{concept_id}.md",
        {
            "type": item.ref.type,
            "title": item.title,
            "description": item.description,
            "source": item.ref.source,
        },
        item.body,
    )


def test_render_prompt_replaces_declared_values_and_rejects_unknown_placeholders():
    assert render_prompt("A={value}", value="甲") == "A=甲"

    with pytest.raises(PromptRenderError, match="unknown placeholders"):
        render_prompt("A={missing}", value="甲")


def test_evidence_catalog_merges_page_break_inside_sentence():
    catalog = build_evidence_catalog(
        "企业异常经营\n\n异常数为21007户。\n\n## 下一节\n\n完整段落。"
    )

    assert list(catalog.values()) == [
        "企业异常经营\n\n异常数为21007户。",
        "## 下一节",
        "完整段落。",
    ]


def test_discovery_constraints_follow_explicit_report_structure():
    source = """# 报告
## 一、运行特征
## （一）信用水平
## （二）融资需求
## （三）融资供给
## （四）融资成本
## （五）融资效率
## 二、下一步工作建议
## （一）信用合规行动
## （二）资源统筹行动
## （三）需求对接行动
# 附件
## 指标设置
## 数据来源
## 计算方式
"""

    constraints = infer_discovery_constraints(source)

    assert constraints.min_concepts == 8
    assert constraints.required_types == (
        "数据口径",
        "分析框架",
        "政策建议",
    )
    assert "二、下一步工作建议" in constraints.outline


def test_discover_serializes_source_and_asset_inventory():
    response = json.dumps(
        {
            "concepts": [
                {
                    "id": "a",
                    "type": "分析框架",
                    "title": "概念 a",
                    "description": "a 摘要。",
                    "evidence": ["evidence-0001"],
                    "asset_hints": ["image-001"],
                }
            ]
        },
        ensure_ascii=False,
    )
    llm = FakeLLM([response])
    source_asset = SourceAsset(
        "image-001",
        "image",
        "![图](images/x.jpg)",
        "images/x.jpg",
        "前文",
        "后文",
        1,
        "abc",
    )

    refs = discover_concepts(
        llm,
        "STAGE=discover\nTITLE={title}\nSOURCE={source_file}\n"
        "ASSETS={asset_inventory}\nEVIDENCE={evidence_catalog}",
        title="报告",
        source_name="报告.md",
        source_content="a 证据。",
        assets=(source_asset,),
    )

    assert refs[0].asset_hints == ("image-001",)
    assert refs[0].evidence == ("a 证据。",)
    assert '"asset_id": "image-001"' in llm.prompts[0]
    assert '"evidence_id": "evidence-0001"' in llm.prompts[0]
    assert "SOURCE=报告.md" in llm.prompts[0]
    assert llm.schemas[0][0] == "concept_discovery"
    assert llm.schemas[0][1]["properties"]["concepts"]["minItems"] == 1


def test_discover_retries_one_concept_for_complex_report():
    one = json.dumps(
        {
            "concepts": [
                {
                    "id": "a",
                    "type": "分析框架",
                    "title": "A",
                    "description": "A 摘要。",
                    "evidence": ["evidence-0001"],
                    "asset_hints": [],
                }
            ]
        },
        ensure_ascii=False,
    )
    two_payload = json.loads(one)
    two_payload["concepts"].append(
        {
            "id": "b",
            "type": "政策建议",
            "title": "B",
            "description": "B 摘要。",
            "evidence": ["evidence-0002"],
            "asset_hints": [],
        }
    )
    llm = FakeLLM([one, json.dumps(two_payload, ensure_ascii=False)])
    content = "\n\n".join(f"第 {index} 段。" for index in range(1, 9))

    refs = discover_concepts(
        llm,
        "{title}{source_file}{asset_inventory}{evidence_catalog}{minimum_concepts}",
        title="复杂报告",
        source_name="复杂报告.md",
        source_content=content,
        assets=(),
    )

    assert len(refs) == 2
    assert len(llm.prompts) == 2
    assert "at least 2" in llm.prompts[1]
    assert llm.schemas[0][1]["properties"]["concepts"]["minItems"] == 2


def test_compile_calls_model_once_per_ref():
    llm = FakeLLM(
        [
            '{"title":"A","description":"摘要 A。","sections":['
            '{"heading":"正文","paragraphs":["a 正文。"],"bullets":[]}]}',
            '{"title":"B","description":"摘要 B。","sections":['
            '{"heading":"正文","paragraphs":["b 正文。"],"bullets":[]}]}',
        ]
    )

    drafts = compile_concepts(
        llm,
        "STAGE=compile\nREF={concept_ref}\nEVIDENCE={evidence}",
        (ref("a"), ref("b")),
    )

    assert [item.ref.concept_id for item in drafts] == ["a", "b"]
    assert len(llm.prompts) == 2
    assert '"concept_id": "a"' in llm.prompts[0]
    assert '"concept_id": "b"' in llm.prompts[1]


def test_preservation_skips_model_when_source_has_no_assets():
    llm = FakeLLM([])

    assert plan_asset_placements(
        llm,
        "{asset_inventory}{concepts}",
        assets=(),
        drafts=(draft("a"),),
    ) == ()
    assert llm.prompts == []


def test_preservation_parses_one_decision_per_asset():
    response = json.dumps(
        {
            "placements": [
                {
                    "asset_id": "image-001",
                    "concept_id": "a",
                    "anchor_id": "anchor-001",
                    "position": "after",
                    "reason": "相关",
                }
            ]
        },
        ensure_ascii=False,
    )
    llm = FakeLLM([response])
    source_asset = SourceAsset(
        "image-001", "image", "![图](images/x.jpg)", "images/x.jpg", "", "", 1, "abc"
    )

    placements = plan_asset_placements(
        llm,
        "ASSETS={asset_inventory}\nCONCEPTS={concepts}\nANCHORS={anchor_catalog}",
        assets=(source_asset,),
        drafts=(draft("a"),),
    )

    assert placements[0].concept_id == "a"
    assert placements[0].anchor == "a 正文。"
    assert '"anchor_id": "anchor-001"' in llm.prompts[0]
    assert '"text": "a 正文。"' in llm.prompts[0]
    assert "anchor_id" in llm.schemas[0][1]["properties"]["placements"]["items"]["properties"]
    assert "anchor" not in llm.schemas[0][1]["properties"]["placements"]["items"]["properties"]


def test_relation_prompt_excludes_current_concept_from_candidates():
    llm = FakeLLM(
        [
            '{"status":"linked","links":[{"anchor_id":"b--anchor-001",'
            '"reason":"关联"}]}',
            '{"status":"no_links","links":[]}',
        ]
    )
    concepts = {
        "a": ConceptDocument(
            "a.md",
            {
                "type": "分析框架",
                "title": "当前概念",
                "description": "共享指标约束。",
                "source": "报告.md",
            },
            "共享指标影响当前结论。",
        ),
        "b": ConceptDocument(
            "b.md",
            {
                "type": "数据口径",
                "title": "共享指标",
                "description": "定义共享指标。",
                "source": "报告.md",
            },
            "定义。",
        ),
    }

    audits = audit_relations(
        llm,
        "CURRENT={current_concept}\nCANDIDATES={candidate_index}"
        "\nANCHORS={anchor_catalog}",
        concepts,
    )

    assert audits["a"].links[0].target_id == "b"
    assert audits["a"].links[0].anchor == "共享指标"
    first_candidates = llm.prompts[0].split("CANDIDATES=", 1)[1]
    assert '"concept_id": "a"' not in first_candidates
    assert '"concept_id": "b"' in first_candidates
    assert '"text": "共享指标"' in llm.prompts[0]
    assert '"anchor_id": "b--anchor-001"' in llm.prompts[0]
    assert "target_id" not in llm.schemas[0][1]["properties"]["links"]["items"]["properties"]
    assert "anchor_id" in llm.schemas[0][1]["properties"]["links"]["items"]["properties"]


def test_relation_can_audit_only_new_concepts_against_whole_bundle():
    llm = FakeLLM(['{"status":"no_links","links":[]}'])
    concepts = {"a": document("a"), "b": document("b")}

    audits = audit_relations(
        llm,
        "CURRENT={current_concept}\nCANDIDATES={candidate_index}",
        concepts,
        current_ids=("b",),
    )

    assert set(audits) == {"b"}
    assert llm.prompts == []
    assert audits["b"].status == "no_links"


def test_relation_prunes_overlapping_stable_anchor_selections():
    current = ConceptDocument(
        "a.md",
        {
            "type": "分析框架",
            "title": "当前概念",
            "description": "当前摘要。",
            "source": "报告.md",
        },
        "企业融资成本需要监测。",
    )
    short = ConceptDocument(
        "b.md",
        {
            "type": "分析框架",
            "title": "融资成本",
            "description": "融资成本分析。",
            "source": "报告.md",
        },
        "定义。",
    )
    long = ConceptDocument(
        "c.md",
        {
            "type": "分析框架",
            "title": "企业融资成",
            "description": "企业融资成分析。",
            "source": "报告.md",
        },
        "定义。",
    )
    overlapping = (
        '{"status":"linked","links":['
        '{"anchor_id":"b--anchor-001","reason":"关联"},'
        '{"anchor_id":"c--anchor-001","reason":"关联"}]}'
    )
    llm = FakeLLM([overlapping])
    events: list[str] = []

    audits = audit_relations(
        llm,
        "CURRENT={current_concept}\nCANDIDATES={candidate_index}"
        "\nANCHORS={anchor_catalog}",
        {"a": current, "b": short, "c": long},
        current_ids=("a",),
        on_event=events.append,
    )

    assert len(llm.prompts) == 1
    assert len(audits["a"].links) == 1
    assert audits["a"].links[0].anchor == "企业融资成"
    assert audits["a"].links[0].target_id == "c"
    assert "relation.pruned concept=a dropped=1" in events


class TruncatingLLM:
    def __init__(self, failures: int, response: str):
        self.failures = failures
        self.response = response
        self.prompts: list[str] = []

    def complete(
        self,
        prompt: str,
        *,
        json_schema_name: str | None = None,
        json_schema: dict | None = None,
    ) -> str:
        self.prompts.append(prompt)
        if len(self.prompts) <= self.failures:
            raise LLMOutputTruncated("length", completion_tokens=8191)
        return self.response


def test_complete_structured_retries_truncation_with_verbatim_prompt():
    llm = TruncatingLLM(failures=1, response='{"ok": true}')

    parsed = _complete_structured(
        llm,
        "原始提示",
        schema_name="probe_schema",
        schema={"type": "object"},
        parser=lambda response: json.loads(response),
        max_attempts=3,
    )

    assert parsed == {"ok": True}
    assert llm.prompts == ["原始提示", "原始提示"]
    # The truncated draw has no usable payload: the retry prompt is byte-identical
    # to the original and never appends the partial output as guidance.
    assert llm.prompts[0] == llm.prompts[1]
    assert "上次输出" not in llm.prompts[1]


def test_complete_structured_raises_when_truncation_never_clears():
    llm = TruncatingLLM(failures=3, response='{"ok": true}')

    with pytest.raises(LLMOutputTruncated, match="finish_reason=length"):
        _complete_structured(
            llm,
            "原始提示",
            schema_name="probe_schema",
            schema={"type": "object"},
            parser=lambda response: json.loads(response),
            max_attempts=3,
        )

    assert llm.prompts == ["原始提示", "原始提示", "原始提示"]


def test_complete_structured_contract_error_still_appends_guidance():
    class FaultyLLM:
        def __init__(self) -> None:
            self.prompts: list[str] = []

        def complete(
            self,
            prompt: str,
            *,
            json_schema_name: str | None = None,
            json_schema: dict | None = None,
        ) -> str:
            self.prompts.append(prompt)
            return '{"bad": true}'

    def parser(response: str) -> object:
        raise ContractError("验证失败详情")

    llm = FaultyLLM()
    with pytest.raises(ContractError, match="验证失败详情"):
        _complete_structured(
            llm,
            "原始提示",
            schema_name="probe_schema",
            schema={"type": "object"},
            parser=parser,
            max_attempts=2,
        )

    assert len(llm.prompts) == 2
    assert llm.prompts[1].startswith("原始提示")
    assert "验证失败详情" in llm.prompts[1]
    assert '{"bad": true}' in llm.prompts[1]
