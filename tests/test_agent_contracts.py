import json

import pytest

from kmpro_wiki.agentwiki.agent_contracts import (
    AgentPolicy,
    parse_group_plan,
    parse_quality_audit,
    parse_source_plan,
    recover_group_plan,
)
from kmpro_wiki.agentwiki.contracts import ContractError


def test_agent_policy_bounds_public_control_knobs():
    assert AgentPolicy().max_recompile_attempts == 2

    with pytest.raises(ValueError, match="between 0 and 3"):
        AgentPolicy(max_recompile_attempts=4)


def test_source_plan_requires_real_heading_structure_and_hybrid_refine():
    with pytest.raises(ContractError, match="at least two"):
        parse_source_plan(
            json.dumps(
                {
                    "discovery_mode": "heading",
                    "refine_discovery": False,
                    "asset_policy": "auto",
                    "reason": "标题清楚",
                }
            ),
            structured_section_count=1,
            asset_count=0,
        )

    with pytest.raises(ContractError, match="requires refine"):
        parse_source_plan(
            json.dumps(
                {
                    "discovery_mode": "hybrid",
                    "refine_discovery": False,
                    "asset_policy": "auto",
                    "reason": "混合处理",
                }
            ),
            structured_section_count=3,
            asset_count=0,
        )


def test_group_plan_requires_exact_coverage_and_cross_article_joint_group():
    refs = {
        "r1": {"type": "分析框架", "article_id": "a"},
        "r2": {"type": "分析框架", "article_id": "b"},
    }
    response = json.dumps(
        {
            "groups": [
                {
                    "ref_ids": ["r1", "r2"],
                    "title": "联合概念",
                    "description": "联合摘要。",
                    "reason": "证据互补。",
                }
            ]
        },
        ensure_ascii=False,
    )

    groups = parse_group_plan(
        response,
        refs=refs,
        candidate_pairs={("r1", "r2")},
    )

    assert groups[0].ref_ids == ("r1", "r2")

    with pytest.raises(ContractError, match="omitted"):
        parse_group_plan(
            json.dumps(
                {
                    "groups": [
                        {
                            "ref_ids": ["r1"],
                            "title": "单概念",
                            "description": "摘要。",
                            "reason": "单独编译。",
                        }
                    ]
                },
                ensure_ascii=False,
            ),
            refs=refs,
            candidate_pairs={("r1", "r2")},
        )


def test_group_plan_recovery_demotes_every_group_involved_in_overlap():
    refs = {
        "r1": {
            "type": "分析框架",
            "article_id": "a",
            "title": "概念一",
            "description": "摘要一。",
        },
        "r2": {
            "type": "分析框架",
            "article_id": "b",
            "title": "概念二",
            "description": "摘要二。",
        },
        "r3": {
            "type": "分析框架",
            "article_id": "c",
            "title": "概念三",
            "description": "摘要三。",
        },
    }
    response = json.dumps(
        {
            "groups": [
                {
                    "ref_ids": ["r1", "r3"],
                    "title": "联合一",
                    "description": "联合摘要一。",
                    "reason": "候选关联。",
                },
                {
                    "ref_ids": ["r2", "r3"],
                    "title": "联合二",
                    "description": "联合摘要二。",
                    "reason": "重复归属。",
                },
            ]
        },
        ensure_ascii=False,
    )

    decisions, recovered = recover_group_plan(
        response,
        refs=refs,
        candidate_pairs={("r1", "r3"), ("r2", "r3")},
    )

    assert [item.ref_ids for item in decisions] == [("r1",), ("r2",), ("r3",)]
    assert recovered == ("r1", "r2", "r3")


def test_quality_pass_must_reach_threshold_and_recompile_needs_instruction():
    with pytest.raises(ContractError, match="threshold"):
        parse_quality_audit(
            '{"score":0.7,"decision":"pass","issues":[],"recompile_instructions":""}',
            0.8,
        )

    with pytest.raises(ContractError, match="requires"):
        parse_quality_audit(
            '{"score":0.7,"decision":"recompile","issues":["覆盖不足"],'
            '"recompile_instructions":""}',
            0.8,
        )
