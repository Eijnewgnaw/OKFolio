import pytest

from okfolio.agentwiki.global_cluster import (
    build_clusters,
    candidate_edges,
    validate_judgements,
)


def ref(ref_id: str, article_id: str, *, title: str, description: str, type: str = "政策建议") -> dict[str, str]:
    return {
        "ref_id": ref_id,
        "article_id": article_id,
        "title": title,
        "description": description,
        "type": type,
    }


def test_candidate_edges_are_cross_article_bounded_and_stateful():
    refs = [
        ref("a", "article-1", title="产业基金支持先进制造", description="建立产业基金支持先进制造企业。"),
        ref("b", "article-2", title="产业基金支持制造升级", description="以产业基金支持制造业升级。"),
        ref("c", "article-3", title="教育资源配置", description="优化基础教育资源配置。"),
    ]

    edges, states = candidate_edges(refs, top_k=1)

    assert [(edge.left_ref_id, edge.right_ref_id) for edge in edges] == [("a", "b")]
    assert states == {"a": "candidates", "b": "candidates", "c": "no_candidate"}


def test_cluster_only_merges_same_edges_and_keeps_related_separate():
    refs = [
        ref("a", "article-1", title="产业基金", description="支持制造业。"),
        ref("b", "article-2", title="产业基金", description="支持制造业。"),
        ref("c", "article-3", title="产业基金", description="支持制造业。"),
    ]
    judged = [
        {"edge_id": "a-b", "left_ref_id": "a", "right_ref_id": "b", "decision": "same", "reason": "同一方案"},
        {"edge_id": "a-c", "left_ref_id": "a", "right_ref_id": "c", "decision": "related", "reason": "相关但范围不同"},
    ]

    clusters = build_clusters(refs, judged)

    assert [cluster["ref_ids"] for cluster in clusters] == [["a", "b"], ["c"]]


def test_cluster_requires_direct_same_edge_to_representative_to_avoid_bridge_merge():
    refs = [
        ref("a", "article-1", title="产业基金", description="支持制造业。"),
        ref("b", "article-2", title="产业基金", description="支持制造业。"),
        ref("c", "article-3", title="产业基金", description="支持制造业。"),
    ]
    judged = [
        {"edge_id": "a-b", "left_ref_id": "a", "right_ref_id": "b", "decision": "same", "reason": "同一方案"},
        {"edge_id": "b-c", "left_ref_id": "b", "right_ref_id": "c", "decision": "same", "reason": "局部相似"},
    ]

    clusters = build_clusters(refs, judged)

    assert [cluster["ref_ids"] for cluster in clusters] == [["a", "b"], ["c"]]


def test_validate_judgements_requires_exact_candidate_coverage():
    candidates = [{"edge_id": "a-b"}, {"edge_id": "a-c"}]

    with pytest.raises(ValueError, match="cover exactly"):
        validate_judgements(candidates, [{"edge_id": "a-b", "decision": "same"}])
