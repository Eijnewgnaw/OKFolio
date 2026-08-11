import json

import pytest

from okfolio.agentwiki.contracts import (
    AssetPlacement,
    ConceptRef,
    ContractError,
    LinkSuggestion,
    parse_discovery,
    parse_draft,
    parse_placements,
    parse_relation_audit,
)


def discovery_response(**overrides: object) -> str:
    concept = {
        "id": "financing-demand",
        "type": "数据口径",
        "title": "融资需求指数",
        "description": "定义融资需求变化。",
        "evidence": ["evidence-0001"],
        "asset_hints": ["image-001"],
    }
    concept.update(overrides)
    return json.dumps({"concepts": [concept]}, ensure_ascii=False)


def concept_ref() -> ConceptRef:
    return ConceptRef(
        concept_id="financing-demand",
        type="数据口径",
        title="融资需求指数",
        description="定义融资需求变化。",
        source="报告.md",
        evidence=("融资需求指数下降。",),
        asset_hints=("image-001",),
    )


def test_discovery_requires_valid_ids_types_exact_evidence_and_known_assets():
    refs = parse_discovery(
        discovery_response(),
        source_name="报告.md",
        evidence_catalog={"evidence-0001": "融资需求指数下降。"},
        asset_ids={"image-001"},
    )

    assert refs == (concept_ref(),)


def test_discovery_preserves_optional_semantic_and_scope_metadata():
    refs = parse_discovery(
        discovery_response(
            semantic_signature={"key": "financing-demand-index"},
            scope={"time": "2025年", "geography": "成都市"},
            ref_family_hint="financing-demand-index",
        ),
        source_name="报告.md",
        evidence_catalog={"evidence-0001": "融资需求指数下降。"},
        asset_ids={"image-001"},
    )

    assert refs[0].semantic_signature == {"key": "financing-demand-index"}
    assert refs[0].scope == {"time": "2025年", "geography": "成都市"}
    assert refs[0].ref_family_hint == "financing-demand-index"


def test_discovery_rejects_trailing_prose():
    with pytest.raises(ContractError, match="valid JSON"):
        parse_discovery(
            discovery_response() + " done",
            source_name="报告.md",
            evidence_catalog={"evidence-0001": "融资需求指数下降。"},
            asset_ids={"image-001"},
        )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"id": "../unsafe"}, "concept id"),
        ({"type": "新闻"}, "type"),
        ({"evidence": ["evidence-9999"]}, "evidence"),
        ({"asset_hints": ["image-999"]}, "asset hint"),
    ],
)
def test_discovery_rejects_invalid_semantics(overrides: dict, message: str):
    with pytest.raises(ContractError, match=message):
        parse_discovery(
            discovery_response(**overrides),
            source_name="报告.md",
            evidence_catalog={"evidence-0001": "融资需求指数下降。"},
            asset_ids={"image-001"},
        )


def test_discovery_rejects_duplicate_ids():
    item = json.loads(discovery_response())["concepts"][0]
    response = json.dumps({"concepts": [item, item]}, ensure_ascii=False)

    with pytest.raises(ContractError, match="duplicate"):
        parse_discovery(
            response,
            source_name="报告.md",
            evidence_catalog={"evidence-0001": "融资需求指数下降。"},
            asset_ids={"image-001"},
        )


def test_discovery_enforces_report_specific_minimum_concept_count():
    with pytest.raises(ContractError, match="at least 2"):
        parse_discovery(
            discovery_response(),
            source_name="报告.md",
            evidence_catalog={"evidence-0001": "融资需求指数下降。"},
            asset_ids={"image-001"},
            min_concepts=2,
        )


def test_discovery_enforces_source_detected_type_coverage():
    with pytest.raises(ContractError, match="missing required concept types"):
        parse_discovery(
            discovery_response(type="分析框架"),
            source_name="报告.md",
            evidence_catalog={"evidence-0001": "融资需求指数下降。"},
            asset_ids={"image-001"},
            required_types={"分析框架", "政策建议", "数据口径"},
        )


def test_draft_uses_ref_identity_and_accepts_content_only():
    draft = parse_draft(
        json.dumps(
            {
                "title": "融资需求指数",
                "description": "解释需求变化。",
                "sections": [
                    {
                        "heading": "核心判断",
                        "paragraphs": ["第一段。", "第二段。"],
                        "bullets": ["证据一", "证据二"],
                    }
                ],
            },
            ensure_ascii=False,
        ),
        concept_ref(),
    )

    assert draft.ref == concept_ref()
    assert draft.body == (
        "## 核心判断\n\n第一段。\n\n第二段。\n\n- 证据一\n- 证据二"
    )


def test_discovery_resolves_evidence_id_to_exact_latex_source_text():
    original = (
        "公共信用评价良好企业同比增长 $15.10\\%$，"
        "占比为 $91.90\\%$。"
    )

    refs = parse_discovery(
        discovery_response(evidence=["evidence-0042"]),
        source_name="报告.md",
        evidence_catalog={"evidence-0042": original},
        asset_ids={"image-001"},
    )

    assert refs[0].evidence == (original,)


@pytest.mark.parametrize(
    "body",
    [
        "![图](images/x.jpg)",
        "| A |\n|---|\n| 1 |",
        "<table><tr><td>1</td></tr></table>",
        "参见 [概念](../concepts/a.md)",
        "参见 [[概念]]",
    ],
)
def test_draft_rejects_assets_and_concept_links(body: str):
    response = json.dumps(
        {
            "title": "A",
            "description": "摘要。",
            "sections": [
                {"heading": "正文", "paragraphs": [body], "bullets": []}
            ],
        },
        ensure_ascii=False,
    )

    with pytest.raises(ContractError, match="asset or concept link"):
        parse_draft(response, concept_ref())


@pytest.mark.parametrize("value", ["### 内联标题", "包含\n换行"])
def test_draft_rejects_markdown_structure_inside_section_fields(value: str):
    response = json.dumps(
        {
            "title": "A",
            "description": "摘要。",
            "sections": [
                {"heading": "正文", "paragraphs": [value], "bullets": []}
            ],
        },
        ensure_ascii=False,
    )

    with pytest.raises(ContractError, match="Markdown"):
        parse_draft(response, concept_ref())


def test_placements_require_exactly_one_decision_per_asset():
    response = json.dumps(
        {
            "placements": [
                {
                    "asset_id": "image-001",
                    "concept_id": "financing-demand",
                    "anchor_id": "anchor-001",
                    "position": "after",
                    "reason": "图表解释该指标",
                }
            ]
        },
        ensure_ascii=False,
    )

    placements = parse_placements(
        response,
        asset_ids={"image-001"},
        concept_ids={"financing-demand"},
        anchor_catalog={
            ("financing-demand", "anchor-001"): "正文锚点"
        },
    )

    assert placements == (
        AssetPlacement(
            "image-001",
            "financing-demand",
            "正文锚点",
            "after",
            "图表解释该指标",
        ),
    )


def test_placements_reject_missing_asset_decision():
    with pytest.raises(ContractError, match="exactly once"):
        parse_placements(
            '{"placements":[]}',
            asset_ids={"image-001"},
            concept_ids={"financing-demand"},
            anchor_catalog={
                ("financing-demand", "anchor-001"): "正文锚点"
            },
        )


def test_placements_reject_anchor_id_from_another_concept():
    response = json.dumps(
        {
            "placements": [
                {
                    "asset_id": "image-001",
                    "concept_id": "financing-demand",
                    "anchor_id": "anchor-001",
                    "position": "after",
                    "reason": "图表解释该指标",
                }
            ]
        },
        ensure_ascii=False,
    )

    with pytest.raises(ContractError, match="unknown anchor id"):
        parse_placements(
            response,
            asset_ids={"image-001"},
            concept_ids={"financing-demand", "financing-supply"},
            anchor_catalog={
                ("financing-supply", "anchor-001"): "供给正文锚点"
            },
        )


def test_relation_contract_distinguishes_no_links_from_failure():
    audit = parse_relation_audit(
        '{"status":"no_links","links":[]}',
        anchor_catalog={},
        current_body="正文。",
    )

    assert audit.status == "no_links"
    assert audit.links == ()


def test_relation_contract_rejects_linked_without_suggestions():
    with pytest.raises(ContractError, match="at least one"):
        parse_relation_audit(
            '{"status":"linked","links":[]}',
            anchor_catalog={},
            current_body="正文。",
        )


def test_relation_contract_resolves_bound_anchor_id_to_target_and_span():
    audit = parse_relation_audit(
        '{"status":"linked","links":[{"anchor_id":"other--anchor-002",'
        '"reason":"指标复用"}]}',
        anchor_catalog={
            "other--anchor-001": ("other", "融资", 0),
            "other--anchor-002": ("other", "融资成本", 0),
        },
        current_body="融资下降，融资成本仍需监测。",
    )

    assert audit.links == (
        LinkSuggestion("other", "融资成本", "指标复用", occurrence=0),
    )


def test_relation_contract_rejects_unknown_anchor_id():
    with pytest.raises(ContractError, match="unknown relation anchor"):
        parse_relation_audit(
            '{"status":"linked","links":[{"anchor_id":"other--missing",'
            '"reason":"指标复用"}]}',
            anchor_catalog={
                "third--anchor-001": ("third", "融资成本", 0),
            },
            current_body="融资成本下降。",
        )
