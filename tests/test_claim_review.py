from __future__ import annotations

import json
from dataclasses import replace

import pytest

from okfolio.agentwiki.claim_review import (
    ClaimCoverageBatch,
    ClaimCoverageRow,
    DraftSentenceAttribution,
    ScopeViolation,
    UnsupportedClaim,
    build_draft_sentences,
    build_evidence_units,
    claim_contract_json_schema,
    claim_coverage_batch_json_schema,
    claim_coverage_json_schema,
    chunk_draft_sentences,
    merge_claim_coverage_batches,
    parse_claim_contract,
    parse_claim_coverage,
    parse_claim_coverage_batch,
)
from okfolio.agentwiki.contracts import ContractError


KNOWN_ANOMALY = "异常占位短语甲"


def refs() -> tuple[dict[str, object], ...]:
    return (
        {
            "ref_id": "ref-a",
            "type": "政策建议",
            "title": "甲地区专项补贴",
            "scope": {"region": "甲地区"},
            "evidence": [
                "甲地区专项补贴自2024年1月1日起实施，标准为符合条件投入的30%。",
                "该方案由甲地区发展改革部门组织实施。",
            ],
        },
        {
            "ref_id": "ref-b",
            "type": "政策建议",
            "title": "乙地区专项补贴",
            "scope": {"region": "乙地区"},
            "evidence": [
                "乙地区同期执行独立方案，补贴标准为符合条件投入的20%。"
            ],
        },
    )


def contract_response(*, second_relation: str = "contrasts") -> str:
    units = build_evidence_units(refs())
    evidence_ids = [item.evidence_id for item in units]
    return json.dumps(
        {
            "canonical_question": "甲乙地区的专项补贴标准有何差异？",
            "members": [
                {
                    "ref_id": "ref-a",
                    "relation": "supports",
                    "contribution": "给出甲地区标准和实施时间。",
                },
                {
                    "ref_id": "ref-b",
                    "relation": second_relation,
                    "contribution": "给出乙地区可比较标准。",
                },
            ],
            "evidence_units": [
                {
                    "evidence_id": evidence_ids[0],
                    "disposition": "required",
                    "reason": "包含甲地区的时间、对象和比例。",
                    "claims": [
                        {
                            "claim": "甲地区专项补贴自2024年1月1日起实施。",
                            "slot": "time",
                            "kind": "time",
                            "evidence_excerpt": "自2024年1月1日起实施",
                            "scope": {"region": "甲地区"},
                        },
                        {
                            "claim": "甲地区补贴标准为符合条件投入的30%。",
                            "slot": "measure",
                            "kind": "metric",
                            "evidence_excerpt": "标准为符合条件投入的30%",
                            "scope": {"region": "甲地区"},
                        },
                    ],
                },
                {
                    "evidence_id": evidence_ids[1],
                    "disposition": "required",
                    "reason": "包含乙地区的可比较比例。",
                    "claims": [
                        {
                            "claim": "乙地区补贴标准为符合条件投入的20%。",
                            "slot": "measure",
                            "kind": "metric",
                            "evidence_excerpt": "补贴标准为符合条件投入的20%",
                            "scope": {"region": "乙地区"},
                        }
                    ],
                },
            ],
        },
        ensure_ascii=False,
    )


def parsed_contract(*, second_relation: str = "contrasts"):
    return parse_claim_contract(
        contract_response(second_relation=second_relation),
        group_id="topic-test",
        refs=refs(),
    )


def draft() -> dict[str, str]:
    return {
        "title": "甲乙地区专项补贴标准差异",
        "description": "比较两地专项补贴的实施时间和比例。",
        "body": (
            "甲地区专项补贴自2024年1月1日起实施，标准为符合条件投入的30%。"
            "乙地区补贴标准为符合条件投入的20%。"
        ),
    }


def coverage_response(
    contract,
    statuses: dict[str, str] | None = None,
    *,
    draft_value: dict[str, str] | None = None,
) -> str:
    status_by_id = statuses or {}
    sentences = build_draft_sentences(draft_value or draft())
    body_sentences = [item.text for item in sentences if item.field == "body"]
    rows = []
    for claim in contract.claims:
        status = status_by_id.get(claim.claim_id, "covered")
        excerpt = ""
        if status != "omitted":
            marker = "20%" if "20%" in claim.claim else "30%"
            excerpt = next(
                (item for item in body_sentences if marker in item),
                body_sentences[0],
            )
        rows.append(
            {
                "claim_id": claim.claim_id,
                "status": status,
                "draft_excerpt": excerpt,
                "finding": "逐条对照当前草稿。",
            }
        )
    sentence_attributions = [
        {
            "sentence_id": sentence.sentence_id,
            "status": "supported",
            "claim_ids": [item.claim_id for item in contract.claims],
            "draft_excerpt": sentence.text,
            "finding": "该句逐项归因到合同 claim。",
        }
        for sentence in sentences
    ]
    return json.dumps(
        {
            "rows": rows,
            "sentence_attributions": sentence_attributions,
            "unsupported_claims": [],
            "scope_violations": [],
        },
        ensure_ascii=False,
    )


def test_evidence_units_and_claim_ids_are_stable():
    units = build_evidence_units(refs())
    assert len(units) == 2
    assert all(":bundle-" in item.evidence_id for item in units)
    assert build_evidence_units(refs()) == units

    first = parsed_contract()
    second = parsed_contract()

    assert first.canonical_question == "甲乙地区的专项补贴标准有何差异？"
    assert [item.claim_id for item in first.claims] == [
        item.claim_id for item in second.claims
    ]
    assert len(set(item.claim_id for item in first.claims)) == 3
    assert first.to_payload()["schema_version"] == "okfolio.claim-contract.v1"


def test_claim_contract_schema_has_no_score_or_decision():
    schema = claim_contract_json_schema(refs())
    rendered = json.dumps(schema)
    assert '"score"' not in rendered
    assert '"decision"' not in rendered
    assert schema["properties"]["members"]["minItems"] == 2
    assert schema["properties"]["evidence_units"]["minItems"] == 2

    assert schema["properties"]["canonical_question"]["maxLength"] == 160
    member = schema["properties"]["members"]["items"]["properties"]
    assert member["contribution"]["maxLength"] == 120
    unit = schema["properties"]["evidence_units"]["items"]["properties"]
    assert unit["reason"]["maxLength"] == 120
    assert unit["claims"]["maxItems"] == 8
    claim = unit["claims"]["items"]["properties"]
    assert claim["claim"]["maxLength"] == 160
    assert claim["evidence_excerpt"]["maxLength"] == 240

    scope = claim["scope"]
    assert scope["additionalProperties"] is False
    assert set(scope["properties"]) == {
        "region",
        "time",
        "object",
        "condition",
        "scenario",
        "actor",
    }
    for value_schema in scope["properties"].values():
        text_schema, array_schema = value_schema["anyOf"]
        assert text_schema["maxLength"] == 64
        assert array_schema["maxItems"] == 4
        assert array_schema["items"]["maxLength"] == 64


def test_claim_contract_parser_enforces_claim_and_scope_bounds():
    too_many_claims = json.loads(contract_response())
    base_claim = too_many_claims["evidence_units"][0]["claims"][0]
    too_many_claims["evidence_units"][0]["claims"] = [base_claim] * 9
    with pytest.raises(ContractError, match="at most 8 claims"):
        parse_claim_contract(
            json.dumps(too_many_claims, ensure_ascii=False),
            group_id="topic-test",
            refs=refs(),
        )

    unsupported_scope = json.loads(contract_response())
    unsupported_scope["evidence_units"][0]["claims"][0]["scope"] = {
        "industry": "制造业"
    }
    with pytest.raises(ContractError, match="unsupported keys"):
        parse_claim_contract(
            json.dumps(unsupported_scope, ensure_ascii=False),
            group_id="topic-test",
            refs=refs(),
        )


def test_evidence_blocks_take_priority_and_preserve_real_block_ids():
    block_refs = (
        {
            "ref_id": "ref-block",
            "type": "数据口径",
            "title": "指标口径",
            "evidence": ["不应作为首选的旧聚合证据。"],
            "evidence_blocks": [
                {
                    "block_id": "page-12-block-03",
                    "content": "指标统计范围为规模以上企业。",
                    "page_number": 12,
                },
                {
                    "block_id": "page-13-block-01",
                    "content": "指标数据由统计部门按季度发布。",
                    "page_number": 13,
                },
            ],
        },
    )

    units = build_evidence_units(block_refs)

    assert len(units) == 1
    assert units[0].evidence_id.startswith("ref-block:bundle-")
    assert units[0].source_blocks[0].block_id == "page-12-block-03"
    assert units[0].source_blocks[0].page_number == 12
    assert len(units[0].source_blocks) == 2
    assert "按季度发布" in units[0].text

    response = json.dumps(
        {
            "canonical_question": "该指标的统计范围是什么？",
            "members": [
                {
                    "ref_id": "ref-block",
                    "relation": "supports",
                    "contribution": "给出统计范围。",
                }
            ],
            "evidence_units": [
                {
                    "evidence_id": units[0].evidence_id,
                    "disposition": "required",
                    "reason": "直接定义统计边界。",
                    "claims": [
                        {
                            "claim": "指标统计范围为规模以上企业。",
                            "slot": "boundary",
                            "kind": "scope",
                            "evidence_excerpt": "统计范围为规模以上企业",
                            "scope": {},
                        }
                    ],
                }
            ],
        },
        ensure_ascii=False,
    )
    contract = parse_claim_contract(
        response,
        group_id="metric-block-test",
        refs=block_refs,
    )
    assert contract.claims[0].evidence_block_ids == ("page-12-block-03",)
    assert contract.claims[0].page_numbers == (12,)

    cross_block = json.loads(response)
    cross_block["evidence_units"][0]["claims"][0].update(
        {
            "claim": "规模以上企业的指标数据由统计部门发布。",
            "evidence_excerpt": "规模以上企业。指标数据由统计部门",
        }
    )
    with pytest.raises(ContractError, match="cannot be located"):
        parse_claim_contract(
            json.dumps(cross_block, ensure_ascii=False),
            group_id="metric-block-test",
            refs=block_refs,
        )


def test_claim_slot_must_belong_to_the_concept_type():
    payload = json.loads(contract_response())
    payload["evidence_units"][0]["claims"][0]["slot"] = "definition"

    with pytest.raises(ContractError, match="slot is invalid"):
        parse_claim_contract(
            json.dumps(payload, ensure_ascii=False),
            group_id="topic-test",
            refs=refs(),
        )


def test_claim_contract_requires_exact_evidence_coverage():
    payload = json.loads(contract_response())
    payload["evidence_units"].pop()

    with pytest.raises(ContractError, match="classify every evidence unit"):
        parse_claim_contract(
            json.dumps(payload, ensure_ascii=False),
            group_id="topic-test",
            refs=refs(),
        )


def test_claim_contract_rejects_non_verbatim_evidence_excerpt():
    payload = json.loads(contract_response())
    payload["evidence_units"][0]["claims"][0]["evidence_excerpt"] = (
        "2023年开始实施"
    )

    with pytest.raises(ContractError, match="not verbatim evidence"):
        parse_claim_contract(
            json.dumps(payload, ensure_ascii=False),
            group_id="topic-test",
            refs=refs(),
        )


def test_claim_contract_rejects_ellipsis_joined_evidence_with_split_instruction():
    payload = json.loads(contract_response())
    payload["evidence_units"][0]["claims"][1]["evidence_excerpt"] = (
        "2023年为20%...2024年为30%"
    )

    with pytest.raises(ContractError, match="split the compound claim"):
        parse_claim_contract(
            json.dumps(payload, ensure_ascii=False),
            group_id="topic-test",
            refs=refs(),
        )


def test_claim_contract_recovers_punctuation_only_excerpt_variant():
    payload = json.loads(contract_response())
    payload["evidence_units"][0]["claims"][1]["evidence_excerpt"] = (
        "甲地区专项补贴自2024年1月1日起实施,标准为符合条件投入的30%"
    )

    contract = parse_claim_contract(
        json.dumps(payload, ensure_ascii=False),
        group_id="topic-test",
        refs=refs(),
    )

    recovered = next(item for item in contract.claims if item.kind == "metric")
    assert recovered.evidence_excerpt == (
        "甲地区专项补贴自2024年1月1日起实施，标准为符合条件投入的30%"
    )


def test_claim_contract_recovers_mineru_latex_percentage_variant():
    latex_refs = (
        {
            "ref_id": "ref-latex",
            "type": "数据口径",
            "title": "地区生产总值占比",
            "evidence": [
                "2022年实现地区生产总值77587.99亿元，占全国的 "
                r"\(6.4\%\) ，西部的 \(30.2\%\) ，增长极能级稳步提升。"
            ],
        },
    )
    unit = build_evidence_units(latex_refs)[0]
    payload = {
        "canonical_question": "2022年地区生产总值规模及占比是多少？",
        "members": [
            {
                "ref_id": "ref-latex",
                "relation": "supports",
                "contribution": "给出规模与占比。",
            }
        ],
        "evidence_units": [
            {
                "evidence_id": unit.evidence_id,
                "disposition": "required",
                "reason": "包含完整指标。",
                "claims": [
                    {
                        "claim": (
                            "2022年地区生产总值为77587.99亿元，占全国6.4%，"
                            "占西部30.2%。"
                        ),
                        "slot": "indicator",
                        "kind": "metric",
                        "evidence_excerpt": (
                            "2022年实现地区生产总值77587.99亿元，占全国的6.4%，"
                            "西部的30.2%，增长极能级稳步提升。"
                        ),
                        "scope": {"time": "2022年"},
                    }
                ],
            }
        ],
    }

    contract = parse_claim_contract(
        json.dumps(payload, ensure_ascii=False),
        group_id="topic-latex",
        refs=latex_refs,
    )

    assert r"\(6.4\%\)" in contract.claims[0].evidence_excerpt
    assert r"\(30.2\%\)" in contract.claims[0].evidence_excerpt


def test_claim_contract_recovers_mineru_latex_superscript_plus_variant():
    plus_refs = (
        {
            "ref_id": "ref-plus",
            "type": "政策建议",
            "title": "成果转化服务",
            "evidence": [
                r"进一步打造科技成果示范区，构建“线上 \(^+\) 线下”双线服务体系。"
            ],
        },
    )
    unit = build_evidence_units(plus_refs)[0]
    payload = {
        "canonical_question": "如何构建成果转化服务体系？",
        "members": [
            {
                "ref_id": "ref-plus",
                "relation": "supports",
                "contribution": "给出双线服务路径。",
            }
        ],
        "evidence_units": [
            {
                "evidence_id": unit.evidence_id,
                "disposition": "required",
                "reason": "包含实施路径。",
                "claims": [
                    {
                        "claim": "应构建线上+线下双线服务体系。",
                        "slot": "measure",
                        "kind": "recommendation",
                        "evidence_excerpt": (
                            "进一步打造科技成果示范区，构建“线上+线下”双线服务体系。"
                        ),
                        "scope": {},
                    }
                ],
            }
        ],
    }

    contract = parse_claim_contract(
        json.dumps(payload, ensure_ascii=False),
        group_id="topic-plus",
        refs=plus_refs,
    )

    assert r"\(^+\)" in contract.claims[0].evidence_excerpt


def test_claim_contract_recovers_sentence_split_across_adjacent_pdf_pages():
    split_refs = (
        {
            "ref_id": "ref-split",
            "type": "分析框架",
            "title": "同城化产业衔接",
            "evidence_blocks": [
                {
                    "block_id": "block-page-1",
                    "content": (
                        "因此，加快成德眉资同城化发展，加强区域内产业链、"
                        "创新链、供应链、价值链衔接配套，合力共"
                    ),
                    "page_number": 171,
                },
                {
                    "block_id": "block-page-2",
                    "content": (
                        "建跨区域产业生态圈，这是对城市发展方式和经济组织方式"
                        "的重大调整，不但有利于促进区域协同。"
                    ),
                    "page_number": 172,
                },
            ],
        },
    )
    unit = build_evidence_units(split_refs)[0]
    payload = {
        "canonical_question": "同城化如何重塑区域产业组织？",
        "members": [
            {
                "ref_id": "ref-split",
                "relation": "supports",
                "contribution": "说明跨区域产业生态圈的组织作用。",
            }
        ],
        "evidence_units": [
            {
                "evidence_id": unit.evidence_id,
                "disposition": "required",
                "reason": "包含产业组织逻辑。",
                "claims": [
                    {
                        "claim": "共建跨区域产业生态圈会调整经济组织方式。",
                        "slot": "cause",
                        "kind": "causal",
                        "evidence_excerpt": (
                            "加强区域内产业链、创新链、供应链、价值链衔接配套，"
                            "合力共建跨区域产业生态圈，这是对城市发展方式和"
                            "经济组织方式的重大调整。"
                        ),
                        "scope": {},
                    }
                ],
            }
        ],
    }

    contract = parse_claim_contract(
        json.dumps(payload, ensure_ascii=False),
        group_id="topic-split",
        refs=split_refs,
    )

    claim = contract.claims[0]
    assert claim.evidence_block_ids == ("block-page-1", "block-page-2")
    assert claim.page_numbers == (171, 172)
    assert "合力共\n\n建跨区域产业生态圈" in claim.evidence_excerpt


def test_claim_contract_rejects_missing_hard_anchor_and_sanitizes_scope():
    missing_anchor = json.loads(contract_response())
    missing_anchor["evidence_units"][0]["claims"][0]["claim"] = (
        "甲地区专项补贴依据《不存在政策》自2024年1月1日起实施。"
    )
    with pytest.raises(ContractError, match="missing hard anchors"):
        parse_claim_contract(
            json.dumps(missing_anchor, ensure_ascii=False),
            group_id="topic-test",
            refs=refs(),
        )

    invented_scope = json.loads(contract_response())
    invented_scope["evidence_units"][0]["claims"][0]["scope"] = {
        "region": "不存在地区"
    }
    contract = parse_claim_contract(
        json.dumps(invented_scope, ensure_ascii=False),
        group_id="topic-test",
        refs=refs(),
    )
    assert contract.claims[0].scope == {}

    factual_invention = json.loads(contract_response())
    factual_invention["evidence_units"][0]["claims"][0]["claim"] = (
        "不存在地区的专项补贴自2024年1月1日起实施。"
    )
    factual_invention["evidence_units"][0]["claims"][0]["scope"] = {
        "region": "不存在地区"
    }
    with pytest.raises(ContractError, match="claim contains a scope value absent"):
        parse_claim_contract(
            json.dumps(factual_invention, ensure_ascii=False),
            group_id="topic-test",
            refs=refs(),
        )


def test_claim_contract_keeps_supported_scope_values_from_mixed_list():
    payload = json.loads(contract_response())
    payload["evidence_units"][0]["claims"][0]["scope"] = {
        "region": ["甲地区", "不存在地区"]
    }
    contract = parse_claim_contract(
        json.dumps(payload, ensure_ascii=False),
        group_id="topic-test",
        refs=refs(),
    )
    assert contract.claims[0].scope == {"region": ["甲地区"]}


def test_claim_contract_allows_scope_resolved_in_same_ref_bundle():
    payload = json.loads(contract_response())
    payload["evidence_units"][0]["claims"][0]["claim"] = (
        "甲地区的该项补贴自2024年1月1日起实施。"
    )
    payload["evidence_units"][0]["claims"][0]["evidence_excerpt"] = (
        "自2024年1月1日起实施"
    )
    payload["evidence_units"][0]["claims"][0]["scope"] = {
        "region": "甲地区"
    }

    contract = parse_claim_contract(
        json.dumps(payload, ensure_ascii=False),
        group_id="topic-test",
        refs=refs(),
    )

    assert contract.claims[0].scope == {"region": "甲地区"}


def test_claim_excerpt_expands_within_one_real_block_for_missing_anchor():
    block_refs = (
        {
            "ref_id": "ref-list",
            "type": "分析框架",
            "title": "合作事项",
            "evidence": ["旧聚合证据不应优先。"],
            "evidence_blocks": [
                {
                    "block_id": "block-list",
                    "content": (
                        "会议审议了《工作机制》和《事项清单》。其中，"
                        "《事项清单》涵盖了交通基础设施等34个合作事项，"
                        "并分为共投共建、相向协作、共同争取三类。"
                    ),
                    "page_number": 12,
                }
            ],
        },
    )
    unit = build_evidence_units(block_refs)[0]
    response = json.dumps(
        {
            "canonical_question": "合作事项清单包含哪些内容？",
            "members": [
                {
                    "ref_id": "ref-list",
                    "relation": "supports",
                    "contribution": "说明事项数量和类别。",
                }
            ],
            "evidence_units": [
                {
                    "evidence_id": unit.evidence_id,
                    "disposition": "required",
                    "reason": "直接说明合作事项。",
                    "claims": [
                        {
                            "claim": "《事项清单》涵盖34个合作事项。",
                            "slot": "evidence",
                            "kind": "fact",
                            "evidence_excerpt": "涵盖了交通基础设施等34个合作事项",
                            "scope": {},
                        }
                    ],
                }
            ],
        },
        ensure_ascii=False,
    )

    contract = parse_claim_contract(
        response,
        group_id="topic-list",
        refs=block_refs,
    )

    claim = contract.claims[0]
    assert claim.evidence_excerpt == (
        "《事项清单》涵盖了交通基础设施等34个合作事项"
    )
    assert claim.evidence_block_ids == ("block-list",)
    assert claim.page_numbers == (12,)


def test_latex_percent_anchor_expands_from_body_year_not_policy_title_year():
    block_text = (
        "2023年，成渝地区电子信息先进制造集群成功入选工业和信息化部"
        "《国家先进制造业集群典型案例（2023年）》，集群1700余家规上企业"
        "营收规模超1.7万亿元，两地微型计算机产量超8000万台、占全国 "
        r"\(40\%\) 。"
    )
    metric_refs = (
        {
            "ref_id": "ref-metric",
            "type": "分析框架",
            "title": "电子信息产业集群",
            "evidence_blocks": [
                {
                    "block_id": "block-metric",
                    "content": block_text,
                    "page_number": 136,
                }
            ],
        },
    )
    unit = build_evidence_units(metric_refs)[0]
    excerpt = (
        "集群1700余家规上企业营收规模超1.7万亿元，两地微型计算机产量"
        r"超8000万台、占全国 \(40\%\) 。"
    )

    def response(claim: str) -> str:
        return json.dumps(
            {
                "canonical_question": "电子信息先进制造集群发展情况如何？",
                "members": [
                    {
                        "ref_id": "ref-metric",
                        "relation": "supports",
                        "contribution": "提供集群规模指标。",
                    }
                ],
                "evidence_units": [
                    {
                        "evidence_id": unit.evidence_id,
                        "disposition": "required",
                        "reason": "包含年度产业规模事实。",
                        "claims": [
                            {
                                "claim": claim,
                                "slot": "evidence",
                                "kind": "metric",
                                "evidence_excerpt": excerpt,
                                "scope": {},
                            }
                        ],
                    }
                ],
            },
            ensure_ascii=False,
        )

    contract = parse_claim_contract(
        response(
            "2023年，成渝地区电子信息先进制造集群1700余家规上企业营收规模"
            "超1.7万亿元，两地微型计算机产量超8000万台、占全国40%。"
        ),
        group_id="topic-metric",
        refs=metric_refs,
    )
    expanded = contract.claims[0].evidence_excerpt
    assert expanded.startswith("2023年，成渝地区电子信息先进制造集群")
    assert "《国家先进制造业集群典型案例（2023年）》" in expanded
    assert r"\(40\%\)" in expanded

    with pytest.raises(ContractError, match="unsupported temporal qualifier"):
        parse_claim_contract(
            response(
                "截至2023年，成渝地区电子信息先进制造集群1700余家规上企业"
                "营收规模超1.7万亿元，两地微型计算机产量超8000万台、占全国40%。"
            ),
            group_id="topic-metric",
            refs=metric_refs,
        )


def test_policy_title_year_cannot_support_bare_statistical_time_anchor():
    title_only_refs = (
        {
            "ref_id": "ref-title-year",
            "type": "分析框架",
            "title": "产业指标",
            "evidence_blocks": [
                {
                    "block_id": "block-title-year",
                    "content": (
                        "入选《国家先进制造业集群典型案例（2023年）》，"
                        "集群1700家规上企业营收增长。"
                    ),
                    "page_number": 3,
                }
            ],
        },
    )
    unit = build_evidence_units(title_only_refs)[0]
    response = json.dumps(
        {
            "canonical_question": "产业指标如何？",
            "members": [
                {
                    "ref_id": "ref-title-year",
                    "relation": "supports",
                    "contribution": "提供产业指标。",
                }
            ],
            "evidence_units": [
                {
                    "evidence_id": unit.evidence_id,
                    "disposition": "required",
                    "reason": "包含企业数量。",
                    "claims": [
                        {
                            "claim": "2023年集群有1700家规上企业。",
                            "slot": "evidence",
                            "kind": "metric",
                            "evidence_excerpt": "集群1700家规上企业营收增长",
                            "scope": {},
                        }
                    ],
                }
            ],
        },
        ensure_ascii=False,
    )

    with pytest.raises(ContractError, match="missing hard anchors.*2023年"):
        parse_claim_contract(
            response,
            group_id="topic-title-year",
            refs=title_only_refs,
        )


@pytest.mark.parametrize(
    ("blocks", "claim", "excerpt", "error"),
    [
        (
            [
                {"block_id": "anchor", "content": "《事项清单》另行发布。"},
                {"block_id": "fact", "content": "涵盖34个合作事项。"},
            ],
            "《事项清单》涵盖34个合作事项。",
            "涵盖34个合作事项",
            "missing hard anchors",
        ),
        (
            [
                {
                    "block_id": "stitched",
                    "content": "《事项清单》在前文发布，中间另有说明，涵盖34个合作事项。",
                }
            ],
            "《事项清单》涵盖34个合作事项。",
            "《事项清单》涵盖34个合作事项",
            "not verbatim evidence",
        ),
        (
            [
                {
                    "block_id": "long",
                    "content": "《事项清单》" + ("甲" * 241) + "涵盖34个合作事项。",
                }
            ],
            "《事项清单》涵盖34个合作事项。",
            "涵盖34个合作事项",
            "missing hard anchors",
        ),
        (
            [
                {"block_id": "missing", "content": "涵盖34个合作事项。"},
            ],
            "《不存在清单》涵盖34个合作事项。",
            "涵盖34个合作事项",
            "missing hard anchors",
        ),
    ],
)
def test_claim_excerpt_repair_never_crosses_or_stitches_evidence(
    blocks, claim, excerpt, error
):
    block_refs = (
        {
            "ref_id": "ref-unsafe",
            "type": "分析框架",
            "title": "合作事项",
            "evidence": ["旧聚合证据不应优先。"],
            "evidence_blocks": blocks,
        },
    )
    unit = build_evidence_units(block_refs)[0]
    response = json.dumps(
        {
            "canonical_question": "合作事项有哪些？",
            "members": [
                {
                    "ref_id": "ref-unsafe",
                    "relation": "supports",
                    "contribution": "说明合作事项。",
                }
            ],
            "evidence_units": [
                {
                    "evidence_id": unit.evidence_id,
                    "disposition": "required",
                    "reason": "说明合作事项。",
                    "claims": [
                        {
                            "claim": claim,
                            "slot": "evidence",
                            "kind": "fact",
                            "evidence_excerpt": excerpt,
                            "scope": {},
                        }
                    ],
                }
            ],
        },
        ensure_ascii=False,
    )

    with pytest.raises(ContractError, match=error):
        parse_claim_contract(
            response,
            group_id="topic-unsafe",
            refs=block_refs,
        )


def test_every_non_separate_member_must_contribute_a_claim():
    payload = json.loads(contract_response())
    payload["evidence_units"][-1] = {
        "evidence_id": build_evidence_units(refs())[-1].evidence_id,
        "disposition": "context_only",
        "reason": "错误地当作上下文。",
        "claims": [],
    }

    with pytest.raises(ContractError, match="must contribute"):
        parse_claim_contract(
            json.dumps(payload, ensure_ascii=False),
            group_id="topic-test",
            refs=refs(),
        )


def test_complete_coverage_passes_by_code_decision():
    contract = parsed_contract()
    sentence_catalog = build_draft_sentences(draft())
    schema = claim_coverage_json_schema(
        [item.claim_id for item in contract.claims],
        [item.sentence_id for item in sentence_catalog],
        sentence_catalog,
    )
    assert "decision" not in schema["properties"]
    row_excerpt = schema["properties"]["rows"]["items"]["properties"][
        "draft_excerpt"
    ]
    attribution_excerpt = schema["properties"]["sentence_attributions"][
        "items"
    ]["properties"]["draft_excerpt"]
    assert row_excerpt["enum"] == ["", *(item.text for item in sentence_catalog)]
    assert attribution_excerpt["enum"] == [item.text for item in sentence_catalog]
    assert "uniqueItems" not in json.dumps(schema)

    matrix = parse_claim_coverage(
        coverage_response(contract),
        contract=contract,
        draft=draft(),
    )

    assert matrix.decision == "pass"
    assert all(item.status == "covered" for item in matrix.rows)
    assert matrix.to_payload()["schema_version"] == "okfolio.claim-coverage.v2"


def test_supported_sentence_without_claim_ids_becomes_recompile_defect():
    contract = parsed_contract()
    payload = json.loads(coverage_response(contract))
    payload["sentence_attributions"][0]["status"] = "supported"
    payload["sentence_attributions"][0]["claim_ids"] = []

    matrix = parse_claim_coverage(
        json.dumps(payload, ensure_ascii=False),
        contract=contract,
        draft=draft(),
    )

    assert matrix.sentence_attributions[0].status == "unsupported"
    assert matrix.sentence_attributions[0].claim_ids == ()
    assert matrix.decision == "recompile"


def test_covered_claim_without_supported_sentence_becomes_omitted():
    contract = parsed_contract()
    target = contract.claims[0].claim_id
    payload = json.loads(coverage_response(contract))
    for attribution in payload["sentence_attributions"]:
        attribution["claim_ids"] = [
            claim_id
            for claim_id in attribution["claim_ids"]
            if claim_id != target
        ]

    matrix = parse_claim_coverage(
        json.dumps(payload, ensure_ascii=False),
        contract=contract,
        draft=draft(),
    )

    row = next(item for item in matrix.rows if item.claim_id == target)
    assert row.status == "omitted"
    assert row.draft_excerpt == ""
    assert matrix.decision == "recompile"


def test_covered_claim_can_be_supported_across_exact_draft_sentences():
    base = parsed_contract()
    claim = replace(
        base.claims[0],
        claim_id="claim-cross-sentence",
        claim=(
            "甲地区专项补贴自2024年1月1日起实施，"
            "补贴标准为30%。"
        ),
    )
    contract = replace(base, claims=(claim,))
    split_draft = {
        "title": "跨句事实",
        "description": "",
        "body": (
            "甲地区专项补贴自2024年1月1日起实施。"
            "补贴标准为30%。"
        ),
    }
    sentences = build_draft_sentences(split_draft)
    payload = {
        "rows": [
            {
                "claim_id": claim.claim_id,
                "status": "covered",
                "draft_excerpt": sentences[0].text,
                "finding": "该 claim 由连续两句共同表达。",
            }
        ],
        "sentence_attributions": [
            {
                "sentence_id": sentence.sentence_id,
                "status": "supported",
                "claim_ids": [claim.claim_id],
                "draft_excerpt": sentence.text,
                "finding": "该句承载 claim 的一部分。",
            }
            for sentence in sentences
        ],
        "unsupported_claims": [],
        "scope_violations": [],
    }

    matrix = parse_claim_coverage(
        json.dumps(payload, ensure_ascii=False),
        contract=contract,
        draft=split_draft,
    )

    assert matrix.decision == "pass"


@pytest.mark.parametrize(
    ("claim_text", "draft_sentence", "defect"),
    [
        (
            "《成渝地区联合工作机制》明确了协同安排。",
            "《工作机制》明确了协同安排。",
            "missing hard anchors",
        ),
        (
            "截至2023年，相关事项已经完成。",
            "2023年，相关事项已经完成。",
            "unsupported temporal qualifier",
        ),
    ],
)
def test_semantic_coverage_defect_downgrades_to_recompile(
    claim_text, draft_sentence, defect
):
    base = parsed_contract()
    claim = replace(
        base.claims[0],
        claim_id="claim-semantic-defect",
        claim=claim_text,
    )
    contract = replace(base, claims=(claim,))
    defect_draft = {
        "title": "覆盖不足",
        "description": "",
        "body": draft_sentence,
    }
    sentence = build_draft_sentences(defect_draft)[0]
    payload = {
        "rows": [
            {
                "claim_id": claim.claim_id,
                "status": "covered",
                "draft_excerpt": sentence.text,
                "finding": "模型认为草稿已覆盖。",
            }
        ],
        "sentence_attributions": [
            {
                "sentence_id": sentence.sentence_id,
                "status": "supported",
                "claim_ids": [claim.claim_id],
                "draft_excerpt": sentence.text,
                "finding": "模型将该句归因到 claim。",
            }
        ],
        "unsupported_claims": [],
        "scope_violations": [],
    }

    matrix = parse_claim_coverage(
        json.dumps(payload, ensure_ascii=False),
        contract=contract,
        draft=defect_draft,
    )

    assert matrix.decision == "recompile"
    assert matrix.rows[0].status == "omitted"
    assert matrix.rows[0].draft_excerpt == ""
    assert defect in matrix.rows[0].finding


def test_deterministic_gate_flags_new_numeric_and_policy_anchors():
    contract = parsed_contract()
    expanded = {
        **draft(),
        "body": (
            draft()["body"]
            + "另行补贴50%，并依据《不存在政策》建立专项资金。"
        ),
    }

    matrix = parse_claim_coverage(
        coverage_response(contract, draft_value=expanded),
        contract=contract,
        draft=expanded,
    )

    assert matrix.decision == "recompile"
    assert {item.draft_excerpt for item in matrix.unsupported_claims} == {
        "50%",
        "《不存在政策》",
    }


def test_sentence_audit_flags_unsupported_inference_even_when_model_supports_it():
    contract = parsed_contract()
    expanded = {
        **draft(),
        "body": (
            draft()["body"]
            + "会议确立了战略机制。"
            + "相关安排标志着双城经济圈建设进入实质性推进阶段。"
        ),
    }

    matrix = parse_claim_coverage(
        coverage_response(contract, draft_value=expanded),
        contract=contract,
        draft=expanded,
    )

    assert matrix.decision == "recompile"
    unsupported = {item.draft_excerpt for item in matrix.unsupported_claims}
    assert "会议确立了战略机制。" in unsupported
    assert "相关安排标志着双城经济圈建设进入实质性推进阶段。" in unsupported


def test_uncertain_strong_inference_becomes_recompile_not_human_review():
    base = parsed_contract()
    target = next(item for item in base.claims if "2024年1月1日" in item.claim)
    contract = replace(base, claims=(target,))
    inference_draft = {
        "title": "实施时间",
        "description": "",
        "body": (
            "甲地区专项补贴自2024年1月1日起实施，"
            "标志着政策进入实质性阶段。"
        ),
    }
    sentence = build_draft_sentences(inference_draft)[0]
    payload = {
        "rows": [
            {
                "claim_id": target.claim_id,
                "status": "covered",
                "draft_excerpt": sentence.text,
                "finding": "前半句覆盖实施时间。",
            }
        ],
        "sentence_attributions": [
            {
                "sentence_id": sentence.sentence_id,
                "status": "uncertain",
                "claim_ids": [target.claim_id],
                "draft_excerpt": sentence.text,
                "finding": "后半句是推断性判断，证据未明确支持。",
            }
        ],
        "unsupported_claims": [],
        "scope_violations": [],
    }

    matrix = parse_claim_coverage(
        json.dumps(payload, ensure_ascii=False),
        contract=contract,
        draft=inference_draft,
    )

    assert matrix.decision == "recompile"
    assert matrix.sentence_attributions[0].status == "unsupported"
    assert matrix.sentence_attributions[0].claim_ids == ()
    assert matrix.rows[0].status == "omitted"
    assert matrix.rows[0].draft_excerpt == ""
    assert {item.draft_excerpt for item in matrix.unsupported_claims} == {
        sentence.text
    }


@pytest.mark.parametrize(
    "sentence_text",
    [
        "该项安排是否适用于全部对象仍不明确。",
        "材料�标志着政策进入实质性阶段。",
    ],
)
def test_genuine_or_ocr_uncertainty_still_needs_human_review(sentence_text):
    base = parsed_contract()
    target = base.claims[0]
    contract = replace(base, claims=(target,))
    uncertain_draft = {
        "title": "待核事项",
        "description": "",
        "body": sentence_text,
    }
    sentence = build_draft_sentences(uncertain_draft)[0]
    payload = {
        "rows": [
            {
                "claim_id": target.claim_id,
                "status": "uncertain",
                "draft_excerpt": sentence.text,
                "finding": "无法确定是否覆盖。",
            }
        ],
        "sentence_attributions": [
            {
                "sentence_id": sentence.sentence_id,
                "status": "uncertain",
                "claim_ids": [target.claim_id],
                "draft_excerpt": sentence.text,
                "finding": "需要人工核对。",
            }
        ],
        "unsupported_claims": [],
        "scope_violations": [],
    }

    matrix = parse_claim_coverage(
        json.dumps(payload, ensure_ascii=False),
        contract=contract,
        draft=uncertain_draft,
    )

    assert matrix.decision == "human_review"
    assert matrix.sentence_attributions[0].status == "uncertain"


def test_sentence_audit_requires_exact_once_catalog_coverage():
    contract = parsed_contract()
    payload = json.loads(coverage_response(contract))
    payload["sentence_attributions"].pop()
    with pytest.raises(ContractError, match="audit every deterministic draft sentence"):
        parse_claim_coverage(
            json.dumps(payload, ensure_ascii=False),
            contract=contract,
            draft=draft(),
        )

    payload = json.loads(coverage_response(contract))
    payload["sentence_attributions"][0]["draft_excerpt"] = "草稿中的另一句话"
    with pytest.raises(ContractError, match="not verbatim text from the draft"):
        parse_claim_coverage(
            json.dumps(payload, ensure_ascii=False),
            contract=contract,
            draft=draft(),
        )


def test_model_view_excludes_anomalous_sentence_but_preserves_source_block():
    mixed_refs = (
        {
            "ref_id": "ref-anomaly",
            "type": "分析框架",
            "title": "产业进展",
            "evidence_blocks": [
                {
                    "block_id": "block-anomaly",
                    "content": (
                        f"产业集群{KNOWN_ANOMALY}，产业发展能级持续提升。"
                        "2023年后续干净事实为30%。"
                    ),
                    "page_number": 9,
                }
            ],
        },
    )
    default_unit = build_evidence_units(mixed_refs)[0]
    assert KNOWN_ANOMALY in default_unit.text
    assert default_unit.excluded_fragments == ()

    unit = build_evidence_units(
        mixed_refs,
        known_source_anomalies=(KNOWN_ANOMALY,),
    )[0]

    assert unit.text == "2023年后续干净事实为30%。"
    assert KNOWN_ANOMALY not in unit.text
    assert KNOWN_ANOMALY in unit.source_blocks[0].text
    assert unit.excluded_fragments[0].text == (
        f"产业集群{KNOWN_ANOMALY}，产业发展能级持续提升。"
    )
    assert unit.excluded_fragments[0].block_id == "block-anomaly"
    assert unit.excluded_fragments[0].page_number == 9
    assert unit.excluded_fragments[0].source_text_anomalies == (KNOWN_ANOMALY,)


def test_contract_parser_uses_same_frozen_anomaly_vocabulary():
    anomaly_refs = (
        {
            "ref_id": "ref-anomaly-contract",
            "type": "分析框架",
            "title": "合成源文异常",
            "evidence": [
                f"该句含{KNOWN_ANOMALY}，不应成为 claim。干净事实为30%。"
            ],
        },
    )
    unit = build_evidence_units(
        anomaly_refs,
        known_source_anomalies=(KNOWN_ANOMALY,),
    )[0]
    payload = {
        "canonical_question": "干净事实是什么？",
        "members": [
            {
                "ref_id": "ref-anomaly-contract",
                "relation": "supports",
                "contribution": "提供事实。",
            }
        ],
        "evidence_units": [
            {
                "evidence_id": unit.evidence_id,
                "disposition": "required",
                "reason": "回答问题。",
                "claims": [
                    {
                        "claim": f"该句含{KNOWN_ANOMALY}。",
                        "slot": "evidence",
                        "kind": "fact",
                        "evidence_excerpt": f"该句含{KNOWN_ANOMALY}",
                        "scope": {},
                    }
                ],
            }
        ],
    }

    with pytest.raises(ContractError, match="source-text anomaly"):
        parse_claim_contract(
            json.dumps(payload, ensure_ascii=False),
            group_id="group-anomaly-contract",
            refs=anomaly_refs,
            known_source_anomalies=(KNOWN_ANOMALY,),
        )


def test_source_text_anomaly_only_bundle_is_rejected_before_model():
    anomaly_refs = (
        {
            "ref_id": "ref-anomaly-only",
            "type": "分析框架",
            "title": "产业进展",
            "evidence_blocks": [
                {
                    "block_id": "block-anomaly-only",
                    "content": f"产业集群{KNOWN_ANOMALY}，产业发展能级持续提升。",
                    "page_number": 9,
                }
            ],
        },
    )

    with pytest.raises(ContractError, match="no clean model-visible text"):
        build_evidence_units(
            anomaly_refs,
            known_source_anomalies=(KNOWN_ANOMALY,),
        )


@pytest.mark.parametrize(
    ("fragment", "field"),
    [
        (KNOWN_ANOMALY, "source_text_anomalies"),
        ("�", "ocr_suspicions"),
        ("\ue000", "ocr_suspicions"),
    ],
)
def test_draft_source_or_ocr_warning_forces_human_review(fragment, field):
    contract = parsed_contract()
    warned_draft = {
        **draft(),
        "body": draft()["body"] + f"产业集群{fragment}，相关进展待核。",
    }
    matrix = parse_claim_coverage(
        coverage_response(contract, draft_value=warned_draft),
        contract=contract,
        draft=warned_draft,
        known_source_anomalies=(
            (fragment,) if field == "source_text_anomalies" else ()
        ),
    )

    assert matrix.decision == "human_review"
    assert any(getattr(item, field) == (fragment,) for item in matrix.sentence_attributions)


def test_omitted_claim_recompiles_and_uncertain_claim_needs_human_review():
    contract = parsed_contract()
    claim_id = contract.claims[0].claim_id

    omitted = parse_claim_coverage(
        coverage_response(contract, {claim_id: "omitted"}),
        contract=contract,
        draft=draft(),
    )
    uncertain = parse_claim_coverage(
        coverage_response(contract, {claim_id: "uncertain"}),
        contract=contract,
        draft=draft(),
    )

    assert omitted.decision == "recompile"
    assert uncertain.decision == "human_review"


def test_coverage_requires_every_claim_exactly_once():
    contract = parsed_contract()
    payload = json.loads(coverage_response(contract))
    payload["rows"].pop()

    with pytest.raises(ContractError, match="assess every claim"):
        parse_claim_coverage(
            json.dumps(payload, ensure_ascii=False),
            contract=contract,
            draft=draft(),
        )


def test_coverage_rejects_excerpt_not_found_in_draft():
    contract = parsed_contract()
    payload = json.loads(coverage_response(contract))
    payload["rows"][0]["draft_excerpt"] = "草稿中不存在的句子"

    with pytest.raises(ContractError, match="not verbatim text from the draft"):
        parse_claim_coverage(
            json.dumps(payload, ensure_ascii=False),
            contract=contract,
            draft=draft(),
        )


def test_coverage_missing_claim_hard_anchor_requests_recompile():
    contract = parsed_contract()
    weakened_draft = {
        **draft(),
        "body": draft()["body"].replace("2024年1月1日", "2024年"),
    }
    payload = json.loads(coverage_response(contract, draft_value=weakened_draft))

    matrix = parse_claim_coverage(
        json.dumps(payload, ensure_ascii=False),
        contract=contract,
        draft=weakened_draft,
    )

    assert matrix.decision == "recompile"
    target = next(item for item in contract.claims if "2024年1月1日" in item.claim)
    downgraded = next(
        item for item in matrix.rows if item.claim_id == target.claim_id
    )
    assert downgraded.status == "omitted"
    assert downgraded.draft_excerpt == ""
    assert "missing hard anchors" in downgraded.finding


def test_contradicted_claim_can_cite_the_wrong_draft_anchor_for_recompile():
    contract = parsed_contract()
    target = next(item for item in contract.claims if "30%" in item.claim)
    wrong_draft = {
        **draft(),
        "body": draft()["body"].replace("30%", "50%"),
    }
    payload = json.loads(coverage_response(contract, draft_value=wrong_draft))
    row = next(item for item in payload["rows"] if item["claim_id"] == target.claim_id)
    row.update(
        {
            "status": "contradicted",
            "draft_excerpt": next(
                item.text
                for item in build_draft_sentences(wrong_draft)
                if item.field == "body" and "50%" in item.text
            ),
            "finding": "草稿把证据中的30%写成了50%。",
        }
    )

    matrix = parse_claim_coverage(
        json.dumps(payload, ensure_ascii=False),
        contract=contract,
        draft=wrong_draft,
    )

    assert matrix.decision == "recompile"
    assert any(item.draft_excerpt == "50%" for item in matrix.unsupported_claims)


def test_separate_member_forces_human_review():
    payload = json.loads(contract_response(second_relation="separate"))
    payload["evidence_units"][-1] = {
        "evidence_id": build_evidence_units(refs())[-1].evidence_id,
        "disposition": "context_only",
        "reason": "该 Ref 不属于当前中心问题。",
        "claims": [],
    }
    contract = parse_claim_contract(
        json.dumps(payload, ensure_ascii=False),
        group_id="topic-test",
        refs=refs(),
    )

    matrix = parse_claim_coverage(
        coverage_response(contract),
        contract=contract,
        draft=draft(),
    )

    assert matrix.decision == "human_review"


def _claim_row_excerpt(claim, sentences) -> str:
    marker = "20%" if "20%" in claim.claim else "30%"
    return next(
        (item.text for item in sentences if marker in item.text),
        sentences[0].text,
    )


def _covered_batch(
    index: int,
    contract,
    sentences,
    *,
    claim_ids=None,
    unsupported=(),
    scope=(),
) -> ClaimCoverageBatch:
    by_id = {item.claim_id: item for item in contract.claims}
    selected = (
        tuple(claim_ids)
        if claim_ids is not None
        else tuple(item.claim_id for item in contract.claims)
    )
    rows = tuple(
        ClaimCoverageRow(
            claim_id=claim_id,
            status="covered",
            draft_excerpt=_claim_row_excerpt(by_id[claim_id], sentences),
            finding="逐条对照当前草稿。",
        )
        for claim_id in selected
    )
    attributions = tuple(
        DraftSentenceAttribution(
            sentence_id=sentence.sentence_id,
            status="supported",
            claim_ids=selected,
            draft_excerpt=sentence.text,
            finding="该句逐项归因到合同 claim。",
        )
        for sentence in sentences
    )
    return ClaimCoverageBatch(
        batch_index=index,
        rows=rows,
        sentence_attributions=attributions,
        unsupported_claims=tuple(unsupported),
        scope_violations=tuple(scope),
    )


def test_chunk_draft_sentences_is_deterministic_contiguous_and_bounded():
    long_draft = {
        "title": "长草稿",
        "description": "核心事实为30%。",
        "body": "核心事实为30%。" * 25,
    }
    sentences = build_draft_sentences(long_draft)
    assert len(sentences) == 26

    batches = chunk_draft_sentences(sentences, batch_size=12)
    assert [len(batch) for batch in batches] == [12, 12, 2]
    assert all(len(batch) <= 12 for batch in batches)
    joined = tuple(item for batch in batches for item in batch)
    assert [item.sentence_id for item in joined] == [
        item.sentence_id for item in sentences
    ]
    assert chunk_draft_sentences(sentences, batch_size=12) == batches

    single = chunk_draft_sentences(sentences, batch_size=1)
    assert len(single) == 26
    assert all(len(batch) == 1 for batch in single)
    assert [item.sentence_id for batch in single for item in batch] == [
        item.sentence_id for item in sentences
    ]

    with pytest.raises(ValueError, match="positive integer"):
        chunk_draft_sentences(sentences, batch_size=0)
    with pytest.raises(ValueError, match="positive integer"):
        chunk_draft_sentences(sentences, batch_size=-1)


def test_coverage_batch_schema_scopes_rows_and_attributions_to_the_batch():
    contract = parsed_contract()
    sentences = build_draft_sentences(draft())
    batch = sentences[:2]
    schema = claim_coverage_batch_json_schema(
        [item.claim_id for item in contract.claims],
        batch,
    )
    rows = schema["properties"]["rows"]
    assert rows["minItems"] == 0
    assert rows["maxItems"] == len(contract.claims)
    assert rows["items"]["properties"]["status"]["enum"] == [
        "contradicted",
        "covered",
        "uncertain",
    ]
    assert rows["items"]["properties"]["draft_excerpt"]["enum"] == [
        item.text for item in batch
    ]
    attributions = schema["properties"]["sentence_attributions"]
    assert attributions["minItems"] == len(batch)
    assert attributions["maxItems"] == len(batch)
    assert attributions["items"]["properties"]["sentence_id"]["enum"] == sorted(
        item.sentence_id for item in batch
    )
    assert attributions["items"]["properties"]["draft_excerpt"]["enum"] == [
        item.text for item in batch
    ]
    assert "omitted" not in json.dumps(rows["items"]["properties"]["status"])
    assert "decision" not in schema["properties"]


def _batch_attributions(batch_sentences, *, claim_ids=()) -> list[dict[str, object]]:
    return [
        {
            "sentence_id": sentence.sentence_id,
            "status": "supported" if claim_ids else "unsupported",
            "claim_ids": list(claim_ids),
            "draft_excerpt": sentence.text,
            "finding": "该句逐项归因到合同 claim。",
        }
        for sentence in batch_sentences
    ]


def test_parse_claim_coverage_batch_rejects_out_of_batch_sentence():
    contract = parsed_contract()
    sentences = build_draft_sentences(draft())
    batch = sentences[:2]
    outside = sentences[2]
    payload = {
        "rows": [],
        "sentence_attributions": [
            {
                "sentence_id": outside.sentence_id,
                "status": "unsupported",
                "claim_ids": [],
                "draft_excerpt": outside.text,
                "finding": "批外句子。",
            }
        ],
        "unsupported_claims": [],
        "scope_violations": [],
    }
    with pytest.raises(ContractError, match="not part of this coverage batch"):
        parse_claim_coverage_batch(
            json.dumps(payload, ensure_ascii=False),
            contract=contract,
            batch_index=0,
            batch_sentences=batch,
        )


def test_parse_claim_coverage_batch_rejects_duplicate_sentence_id():
    contract = parsed_contract()
    sentences = build_draft_sentences(draft())
    batch = sentences[:2]
    payload = {
        "rows": [],
        "sentence_attributions": _batch_attributions(batch),
        "unsupported_claims": [],
        "scope_violations": [],
    }
    payload["sentence_attributions"].append(
        dict(payload["sentence_attributions"][0])
    )
    with pytest.raises(ContractError, match="duplicate draft sentence_id"):
        parse_claim_coverage_batch(
            json.dumps(payload, ensure_ascii=False),
            contract=contract,
            batch_index=0,
            batch_sentences=batch,
        )


def test_parse_claim_coverage_batch_rejects_non_contract_claim_id():
    contract = parsed_contract()
    sentences = build_draft_sentences(draft())
    batch = sentences[:2]
    payload = {
        "rows": [
            {
                "claim_id": "claim-unknown",
                "status": "covered",
                "draft_excerpt": batch[0].text,
                "finding": "未知 claim。",
            }
        ],
        "sentence_attributions": _batch_attributions(batch),
        "unsupported_claims": [],
        "scope_violations": [],
    }
    with pytest.raises(ContractError, match="unknown coverage claim_id"):
        parse_claim_coverage_batch(
            json.dumps(payload, ensure_ascii=False),
            contract=contract,
            batch_index=0,
            batch_sentences=batch,
        )


def test_parse_claim_coverage_batch_rejects_non_verbatim_row_excerpt():
    contract = parsed_contract()
    sentences = build_draft_sentences(draft())
    batch = sentences[:2]
    payload = {
        "rows": [
            {
                "claim_id": contract.claims[0].claim_id,
                "status": "covered",
                "draft_excerpt": "草稿中不存在的句子",
                "finding": "非逐字。",
            }
        ],
        "sentence_attributions": _batch_attributions(batch),
        "unsupported_claims": [],
        "scope_violations": [],
    }
    with pytest.raises(ContractError, match="not verbatim text from the draft"):
        parse_claim_coverage_batch(
            json.dumps(payload, ensure_ascii=False),
            contract=contract,
            batch_index=0,
            batch_sentences=batch,
        )

    # A verbatim fragment is still rejected: the row excerpt must equal one
    # complete deterministic sentence of this batch.
    payload["rows"][0]["draft_excerpt"] = batch[0].text[:10]
    with pytest.raises(
        ContractError, match="must equal one deterministic sentence of this batch"
    ):
        parse_claim_coverage_batch(
            json.dumps(payload, ensure_ascii=False),
            contract=contract,
            batch_index=0,
            batch_sentences=batch,
        )


def test_parse_claim_coverage_batch_rejects_model_omitted_rows():
    contract = parsed_contract()
    sentences = build_draft_sentences(draft())
    batch = sentences[:2]
    payload = {
        "rows": [
            {
                "claim_id": contract.claims[0].claim_id,
                "status": "omitted",
                "draft_excerpt": "",
                "finding": "模型错误地输出 omitted。",
            }
        ],
        "sentence_attributions": _batch_attributions(batch),
        "unsupported_claims": [],
        "scope_violations": [],
    }
    with pytest.raises(ContractError, match="must not mark claims omitted"):
        parse_claim_coverage_batch(
            json.dumps(payload, ensure_ascii=False),
            contract=contract,
            batch_index=0,
            batch_sentences=batch,
        )


def test_parse_claim_coverage_batch_requires_exact_once_batch_coverage():
    contract = parsed_contract()
    sentences = build_draft_sentences(draft())
    batch = sentences[:2]
    payload = {
        "rows": [],
        "sentence_attributions": _batch_attributions(batch[:1]),
        "unsupported_claims": [],
        "scope_violations": [],
    }
    with pytest.raises(
        ContractError, match="must audit every batch sentence exactly once"
    ):
        parse_claim_coverage_batch(
            json.dumps(payload, ensure_ascii=False),
            contract=contract,
            batch_index=0,
            batch_sentences=batch,
        )


def test_merge_batches_is_exact_once_and_fills_omitted_claims():
    contract = parsed_contract()
    sentences = build_draft_sentences(draft())
    batch0 = _covered_batch(
        0,
        contract,
        sentences[:2],
        claim_ids=(
            contract.claims[0].claim_id,
            contract.claims[1].claim_id,
        ),
    )
    batch1 = _covered_batch(1, contract, sentences[2:], claim_ids=())

    matrix = merge_claim_coverage_batches(
        {0: batch0, 1: batch1},
        contract=contract,
        draft=draft(),
    )

    assert matrix.schema_version == "okfolio.claim-coverage.v2"
    assert len(matrix.rows) == len(contract.claims)
    assert len(set(item.claim_id for item in matrix.rows)) == len(contract.claims)
    missing = next(
        item for item in matrix.rows if item.claim_id == contract.claims[2].claim_id
    )
    assert missing.status == "omitted"
    assert missing.draft_excerpt == ""
    assert "补全为 omitted" in missing.finding
    # The code-completed omitted claim is a deterministic recompile signal.
    assert matrix.decision == "recompile"


def _with_rows(batch: ClaimCoverageBatch, rows) -> ClaimCoverageBatch:
    return ClaimCoverageBatch(
        batch_index=batch.batch_index,
        rows=tuple(rows),
        sentence_attributions=batch.sentence_attributions,
        unsupported_claims=batch.unsupported_claims,
        scope_violations=batch.scope_violations,
    )


@pytest.mark.parametrize(
    ("batch0_status", "batch1_status", "expected_status", "expected_excerpt"),
    [
        ("covered", "covered", "covered", None),  # tie keeps the earliest batch
        ("covered", "uncertain", "uncertain", None),
        ("covered", "contradicted", "contradicted", None),
        ("uncertain", "contradicted", "contradicted", None),
        ("contradicted", "uncertain", "contradicted", None),
    ],
)
def test_merge_converges_cross_batch_duplicate_claims_by_severity(
    batch0_status, batch1_status, expected_status, expected_excerpt
):
    contract = parsed_contract()
    sentences = build_draft_sentences(draft())
    target = contract.claims[0].claim_id
    batch0 = _with_rows(
        _covered_batch(0, contract, sentences[:2], claim_ids=(target,)),
        (
            ClaimCoverageRow(
                claim_id=target,
                status=batch0_status,
                draft_excerpt=sentences[0].text,
                finding="批0判断。",
            ),
        ),
    )
    batch1 = _with_rows(
        _covered_batch(1, contract, sentences[2:], claim_ids=(target,)),
        (
            ClaimCoverageRow(
                claim_id=target,
                status=batch1_status,
                draft_excerpt=sentences[2].text,
                finding="批1判断。",
            ),
        ),
    )

    matrix = merge_claim_coverage_batches(
        {0: batch0, 1: batch1},
        contract=contract,
        draft=draft(),
    )

    rows = [item for item in matrix.rows if item.claim_id == target]
    assert len(rows) == 1
    row = rows[0]
    assert row.status == expected_status
    if expected_excerpt is None:
        # The winning row keeps its own batch's verbatim excerpt: batch 0 for
        # ties and batch-0 wins, batch 1 when the later batch is more severe.
        expected_excerpt = (
            sentences[0].text
            if row.status == batch0_status
            else sentences[2].text
        )
    assert row.draft_excerpt == expected_excerpt


def test_merge_converges_equal_status_to_earliest_batch():
    contract = parsed_contract()
    sentences = build_draft_sentences(draft())
    target = contract.claims[0].claim_id
    batch0 = _with_rows(
        _covered_batch(0, contract, sentences[:2], claim_ids=(target,)),
        (
            ClaimCoverageRow(
                claim_id=target,
                status="covered",
                draft_excerpt=sentences[0].text,
                finding="批0覆盖。",
            ),
        ),
    )
    batch1 = _with_rows(
        _covered_batch(1, contract, sentences[2:], claim_ids=(target,)),
        (
            ClaimCoverageRow(
                claim_id=target,
                status="covered",
                draft_excerpt=sentences[2].text,
                finding="批1覆盖。",
            ),
        ),
    )

    matrix = merge_claim_coverage_batches(
        {1: batch1, 0: batch0},
        contract=contract,
        draft=draft(),
    )

    row = next(item for item in matrix.rows if item.claim_id == target)
    assert row.status == "covered"
    assert row.draft_excerpt == sentences[0].text
    assert row.finding == "批0覆盖。"


@pytest.mark.parametrize(
    ("first_status", "second_status", "expected_status", "expected_excerpt"),
    [
        ("covered", "covered", "covered", None),  # tie keeps earlier sentence
        ("covered", "uncertain", "uncertain", None),
        ("covered", "contradicted", "contradicted", None),
        ("uncertain", "contradicted", "contradicted", None),
        ("contradicted", "uncertain", "contradicted", None),
    ],
)
def test_parse_claim_coverage_batch_converges_duplicate_rows_by_severity(
    first_status, second_status, expected_status, expected_excerpt
):
    contract = parsed_contract()
    sentences = build_draft_sentences(draft())
    batch = sentences[:2]
    target = contract.claims[0].claim_id
    payload = {
        "rows": [
            {
                "claim_id": target,
                "status": first_status,
                "draft_excerpt": batch[0].text,
                "finding": "首句判断。",
            },
            {
                "claim_id": target,
                "status": second_status,
                "draft_excerpt": batch[1].text,
                "finding": "次句判断。",
            },
        ],
        "sentence_attributions": _batch_attributions(batch),
        "unsupported_claims": [],
        "scope_violations": [],
    }

    parsed = parse_claim_coverage_batch(
        json.dumps(payload, ensure_ascii=False),
        contract=contract,
        batch_index=0,
        batch_sentences=batch,
    )

    assert len(parsed.rows) == 1
    row = parsed.rows[0]
    assert row.status == expected_status
    if expected_excerpt is None:
        expected_excerpt = (
            batch[0].text if row.status == first_status else batch[1].text
        )
    assert row.draft_excerpt == expected_excerpt


def test_parse_claim_coverage_batch_tie_keeps_earlier_batch_sentence_row():
    contract = parsed_contract()
    sentences = build_draft_sentences(draft())
    batch = sentences[:2]
    target = contract.claims[0].claim_id
    payload = {
        "rows": [
            {
                "claim_id": target,
                "status": "covered",
                "draft_excerpt": batch[1].text,
                "finding": "后句覆盖。",
            },
            {
                "claim_id": target,
                "status": "covered",
                "draft_excerpt": batch[0].text,
                "finding": "先句覆盖。",
            },
        ],
        "sentence_attributions": _batch_attributions(batch),
        "unsupported_claims": [],
        "scope_violations": [],
    }

    parsed = parse_claim_coverage_batch(
        json.dumps(payload, ensure_ascii=False),
        contract=contract,
        batch_index=0,
        batch_sentences=batch,
    )

    assert len(parsed.rows) == 1
    assert parsed.rows[0].status == "covered"
    assert parsed.rows[0].draft_excerpt == batch[0].text
    assert parsed.rows[0].finding == "先句覆盖。"


def test_batch_internal_convergence_keeps_merged_matrix_exact_once():
    contract = parsed_contract()
    sentences = build_draft_sentences(draft())
    batch = sentences[:2]
    target = contract.claims[0].claim_id
    payload = {
        "rows": [
            {
                "claim_id": target,
                "status": "covered",
                "draft_excerpt": batch[0].text,
                "finding": "先句覆盖。",
            },
            {
                "claim_id": target,
                "status": "contradicted",
                "draft_excerpt": batch[1].text,
                "finding": "次句矛盾。",
            },
        ],
        "sentence_attributions": _batch_attributions(
            batch, claim_ids=(target,)
        ),
        "unsupported_claims": [],
        "scope_violations": [],
    }
    parsed = parse_claim_coverage_batch(
        json.dumps(payload, ensure_ascii=False),
        contract=contract,
        batch_index=0,
        batch_sentences=batch,
    )
    assert len(parsed.rows) == 1
    assert parsed.rows[0].status == "contradicted"

    tail = parse_claim_coverage_batch(
        json.dumps(
            {
                "rows": [],
                "sentence_attributions": _batch_attributions(sentences[2:]),
                "unsupported_claims": [],
                "scope_violations": [],
            },
            ensure_ascii=False,
        ),
        contract=contract,
        batch_index=1,
        batch_sentences=sentences[2:],
    )
    matrix = merge_claim_coverage_batches(
        {0: parsed, 1: tail},
        contract=contract,
        draft=draft(),
    )

    assert len(matrix.rows) == len(contract.claims)
    assert len(set(item.claim_id for item in matrix.rows)) == len(
        contract.claims
    )
    target_rows = [item for item in matrix.rows if item.claim_id == target]
    assert len(target_rows) == 1
    assert target_rows[0].status == "contradicted"
    assert target_rows[0].draft_excerpt == batch[1].text


def test_merge_batches_rejects_missing_sentence_attribution():
    contract = parsed_contract()
    sentences = build_draft_sentences(draft())
    batch0 = _covered_batch(0, contract, sentences[:2], claim_ids=())
    batch1 = _covered_batch(1, contract, sentences[2:], claim_ids=())
    batch1 = ClaimCoverageBatch(
        batch_index=1,
        rows=batch1.rows,
        sentence_attributions=(),
        unsupported_claims=batch1.unsupported_claims,
        scope_violations=batch1.scope_violations,
    )
    with pytest.raises(
        ContractError, match="must audit every deterministic draft sentence"
    ):
        merge_claim_coverage_batches(
            {0: batch0, 1: batch1},
            contract=contract,
            draft=draft(),
        )


def test_merge_batches_rejects_duplicate_sentence_across_batches():
    contract = parsed_contract()
    sentences = build_draft_sentences(draft())
    batch0 = _covered_batch(0, contract, sentences[:2], claim_ids=())
    batch1 = _covered_batch(1, contract, sentences[1:], claim_ids=())
    with pytest.raises(ContractError, match="more than one coverage batch"):
        merge_claim_coverage_batches(
            {0: batch0, 1: batch1},
            contract=contract,
            draft=draft(),
        )


def test_merge_is_independent_of_batch_completion_order():
    contract = parsed_contract()
    sentences = build_draft_sentences(draft())
    batch0 = _covered_batch(
        0,
        contract,
        sentences[:2],
        claim_ids=(
            contract.claims[0].claim_id,
            contract.claims[1].claim_id,
        ),
    )
    batch1 = _covered_batch(
        1, contract, sentences[2:], claim_ids=(contract.claims[2].claim_id,)
    )

    first = merge_claim_coverage_batches(
        {0: batch0, 1: batch1},
        contract=contract,
        draft=draft(),
    )
    second = merge_claim_coverage_batches(
        {1: batch1, 0: batch0},
        contract=contract,
        draft=draft(),
    )

    assert first.decision == "pass"
    assert first.to_payload() == second.to_payload()


def test_merged_matrix_flags_unsupported_inference_sentence():
    base = parsed_contract()
    contract = replace(base, claims=(base.claims[0],))
    inference_draft = {
        "title": "推断句",
        "description": "",
        "body": "会议确立了战略机制。",
    }
    sentence = build_draft_sentences(inference_draft)[0]
    batch = _covered_batch(
        0,
        contract,
        (sentence,),
        claim_ids=(base.claims[0].claim_id,),
    )

    matrix = merge_claim_coverage_batches(
        {0: batch},
        contract=contract,
        draft=inference_draft,
    )

    assert matrix.decision == "recompile"
    assert {item.draft_excerpt for item in matrix.unsupported_claims} == {
        sentence.text
    }


def test_merged_matrix_downgrades_covered_row_with_missing_numeric_anchor():
    base = parsed_contract()
    target = next(item for item in base.claims if "2024年1月1日" in item.claim)
    contract = replace(base, claims=(target,))
    weakened = {
        "title": "时间弱化",
        "description": "",
        "body": "甲地区专项补贴自2024年起实施。",
    }
    sentence = build_draft_sentences(weakened)[0]
    batch = _covered_batch(0, contract, (sentence,), claim_ids=(target.claim_id,))

    matrix = merge_claim_coverage_batches(
        {0: batch},
        contract=contract,
        draft=weakened,
    )

    assert matrix.decision == "recompile"
    assert matrix.rows[0].status == "omitted"
    assert matrix.rows[0].draft_excerpt == ""
    assert "missing hard anchors" in matrix.rows[0].finding


def test_merged_matrix_downgrades_covered_row_with_missing_temporal_qualifier():
    base = parsed_contract()
    claim = replace(
        base.claims[0],
        claim_id="claim-temporal-gate",
        claim="截至2023年，相关事项已经完成。",
    )
    contract = replace(base, claims=(claim,))
    defect_draft = {
        "title": "时间限定",
        "description": "",
        "body": "2023年，相关事项已经完成。",
    }
    sentence = build_draft_sentences(defect_draft)[0]
    batch = _covered_batch(0, contract, (sentence,), claim_ids=(claim.claim_id,))

    matrix = merge_claim_coverage_batches(
        {0: batch},
        contract=contract,
        draft=defect_draft,
    )

    assert matrix.decision == "recompile"
    assert matrix.rows[0].status == "omitted"
    assert "unsupported temporal qualifier" in matrix.rows[0].finding


def test_merged_matrix_carries_scope_violations_to_recompile():
    contract = parsed_contract()
    sentences = build_draft_sentences(draft())
    batch = _covered_batch(0, contract, sentences)
    batch = ClaimCoverageBatch(
        batch_index=0,
        rows=batch.rows,
        sentence_attributions=batch.sentence_attributions,
        unsupported_claims=batch.unsupported_claims,
        scope_violations=(
            ScopeViolation(
                claim_ids=(contract.claims[0].claim_id,),
                draft_excerpt=sentences[0].text,
                finding="地区范围被错误扩大。",
            ),
        ),
    )

    matrix = merge_claim_coverage_batches(
        {0: batch},
        contract=contract,
        draft=draft(),
    )

    assert matrix.decision == "recompile"
    assert matrix.scope_violations[0].claim_ids == (contract.claims[0].claim_id,)
    assert matrix.scope_violations[0].finding == "地区范围被错误扩大。"
